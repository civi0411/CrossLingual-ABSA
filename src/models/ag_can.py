"""
src/models/ag_can.py
────────────────────
Model L1: Aspect-Guided Contextual Attention Network (AG-CAN)

Architecture Overview:
    1. Multilingual BERT (`mBERT`) fine-tuned representation layer -> token embeddings.
    2. Self-Attention Pooling over aspect category subword tokens to create a unified aspect query representation.
    3. Multi-Head Attention utilizing the aspect vector as Query and sequence embeddings as Keys/Values.
    4. Gated Residual connection fusing attention context and aspect projection.
    5. Layer Normalization followed by classification projection.

Cross-lingual and Few-shot Enhancements:
    - Gated Residual connections stabilize gradient propagation across diverse domain representations.
    - Self-Attention Pooling accommodates flexible aspect category lengths.
    - Layer Normalization prior to classification reduces internal covariate shift.
    - Xavier uniform initialization applied to projection weights for rapid convergence.
    - Label smoothing mitigates overconfidence when fine-tuning on limited target samples.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from pathlib import Path


class SelfAttentionPooling(nn.Module):
    """
    Self-Attention Pooling layer.

    Computes a weighted sum of sequence hidden states based on learnable attention scores.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, T, D), attention_mask: (B, T)
        scores = self.attention(hidden_states).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)        # (B, T)
        pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)  # (B, D)
        return pooled


class AspectGuidedMHA(nn.Module):
    """
    Aspect-Guided Multi-Head Attention layer.

    Uses the aspect category embedding as the Query vector (`Q`) to attend over sequence embeddings (`K`, `V`).

    Mathematical Formulation:
        Q = W_Q · aspect_embedding    (B, d_k)
        K = W_K · token_embeddings    (B, T, d_k)
        V = W_V · token_embeddings    (B, T, d_k)
        context = softmax(Q · Kᵀ / √d_k) · V  (B, d_k)
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads

        self.W_Q = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** 0.5

        # Initialize weights for better convergence (Xavier)
        self._init_weights()

    def _init_weights(self):
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        aspect_emb: torch.Tensor,          # (B, embed_dim)
        token_embs: torch.Tensor,           # (B, T, embed_dim)
        key_mask: torch.Tensor | None = None,  # (B, T), 1 for real tokens
    ) -> torch.Tensor:                     # (B, hidden_dim)
        B, T, _ = token_embs.size()
        H, d_k = self.num_heads, self.d_k

        # Project to hidden_dim then reshape for multi-head
        Q = self.W_Q(aspect_emb).view(B, H, d_k)                # (B, H, d_k)
        K = self.W_K(token_embs).view(B, T, H, d_k).permute(0, 2, 1, 3)  # (B, H, T, d_k)
        V = self.W_V(token_embs).view(B, T, H, d_k).permute(0, 2, 1, 3)  # (B, H, T, d_k)

        Q = Q.unsqueeze(2)                                      # (B, H, 1, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, 1, T)

        if key_mask is not None:
            # Mask padding tokens (mask=0 for padding)
            scores = scores.masked_fill(key_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)            # (B, H, 1, T)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)                 # (B, H, 1, d_k)
        context = context.squeeze(2).contiguous().view(B, -1)   # (B, hidden_dim)
        return self.W_O(context)                               # (B, hidden_dim)


class AGCAN(nn.Module):
    """
    Aspect-Guided Contextual Attention Network (AG-CAN) with Gated Residual Connections for ABSA.
    """

    def __init__(
        self,
        num_labels: int = 3,
        embedding_model: str = "bert-base-multilingual-cased",
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.3,
        label_smoothing: float = 0.1,
        unfreeze_last_n: int = 4,
    ):
        super().__init__()

        # Multilingual encoder: freeze most layers, fine-tune only the last
        # layers for cross-lingual/domain adaptation.
        self.embedder = AutoModel.from_pretrained(embedding_model)
        for p in self.embedder.parameters():
            p.requires_grad = False
        self._unfreeze_last_layers(unfreeze_last_n)
        embed_dim = self.embedder.config.hidden_size   # 768

        # Aspect pooling
        self.aspect_pooling = SelfAttentionPooling(embed_dim)

        # Aspect-guided MHA
        self.attention = AspectGuidedMHA(embed_dim, hidden_dim, num_heads, dropout)

        # Aspect projection for residual connection
        self.aspect_proj = nn.Linear(embed_dim, hidden_dim)

        # Gate for residual connection
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # Residual connection + LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # Label smoothing loss (helps with few-shot)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _unfreeze_last_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            return
        encoder = getattr(self.embedder, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            return
        for layer in layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    def _embed(self, input_ids, attention_mask) -> torch.Tensor:
        """Extract final hidden states from the backbone encoder (`mBERT`)."""
        return self.embedder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    def _embed_aspect(self, aspect_ids, aspect_mask) -> torch.Tensor:
        """Aggregate aspect category token embeddings via Self-Attention pooling."""
        hidden = self.embedder(
            input_ids=aspect_ids,
            attention_mask=aspect_mask,
        ).last_hidden_state
        return self.aspect_pooling(hidden, aspect_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_ids: torch.Tensor,
        aspect_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        # 1. Frozen mBERT embeddings
        token_embs = self._embed(input_ids, attention_mask)     # (B, T, 768)
        aspect_emb = self._embed_aspect(aspect_ids, aspect_mask)  # (B, 768)

        # 2. Aspect-guided attention
        context = self.attention(aspect_emb, token_embs, key_mask=attention_mask)  # (B, hidden_dim)

        # 3. Gated Residual connection + LayerNorm
        aspect_proj = self.aspect_proj(aspect_emb)
        gate = self.gate_proj(torch.cat([context, aspect_proj], dim=-1))
        out = self.layer_norm(gate * context + (1 - gate) * aspect_proj)

        # 4. Classify
        logits = self.classifier(self.dropout(out))

        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)
        return output

def build_agcan_model(cfg: dict) -> AGCAN:
    """
    Factory method to construct and initialize the AGCAN model from configuration parameters.

    Args:
        cfg (dict): Global configuration dictionary.

    Returns:
        AGCAN: Initialized AGCAN model instance.
    """
    paths = cfg.get("model_paths", {})
    model_path = paths.get("mbert", "pretrained_models/mbert")
    if not Path(model_path).exists():
        model_path = "bert-base-multilingual-cased"
    return AGCAN(
        num_labels=3,
        embedding_model=model_path,
        hidden_dim=cfg.get("hidden_dim", 256),
        num_heads=cfg.get("num_heads", 8),
        dropout=cfg.get("dropout", 0.3),
        label_smoothing=cfg.get("label_smoothing", 0.1),
        unfreeze_last_n=cfg.get("agcan_unfreeze_last_n", 4),
    )
