"""
src/models/xlmr.py
──────────────────
Model L2: XLM-RoBERTa (`XLM-R`) with Exact Masked Mean Pooling and Layer-wise Learning Rate Decay (`LLRD`).

Two Primary Improvements over Standard Fine-tuning:
    1. Exact Masked Mean Pooling:
       Instead of positional slicing (`[CLS]` + token slices, which risk misalignment when category strings
       are variable or shifted), we utilize a exact `category_mask` constructed at the dataset level.
       Only token positions corresponding to the aspect category contribute to the pooled aspect representation.
    2. Layer-wise Learning Rate Decay (`LLRD`):
       Lower encoder layers capture general multilingual syntax and morphology — preserved via smaller learning rates.
       Upper layers and the classification projection capture task-specific semantic representations — adapted via higher rates.

Additional Enhancements:
    - Cross-entropy label smoothing (`label_smoothing > 0`) to regularize few-shot training.
    - Pre-classifier dropout regularization.
    - Decoupled weight decay (no decay applied to biases or LayerNorm parameters).
    - Optional lower layer freezing for extreme low-resource adaptation scenarios (`freeze_lower_layers`).
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from pathlib import Path


class XLMRForABSA(nn.Module):
    """
    XLM-RoBERTa encoder equipped with aspect-aware classification head for Oracle Aspect Category Sentiment Classification.

    Input sequence: `[CLS] text [SEP] category [SEP]`
    Pooling strategy: `concat([CLS], masked_mean(category_tokens))`
    Output: Logits for 3 sentiment classes (`positive`, `negative`, `neutral`).
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size   # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _masked_mean_pool(
        self,
        hidden: torch.Tensor,        # (B, T, H)
        category_mask: torch.Tensor, # (B, T) — 1 at category positions
    ) -> torch.Tensor:               # (B, H)
        """
        Compute mathematically exact mean pooling exclusively over aspect category tokens.

        Uses the exact binary `category_mask` generated during dataset encoding to avoid slice boundaries.
        Zero-safe: clamps the denominator to prevent division by zero for implicit aspects (`term == "NULL"`).

        Args:
            hidden (torch.Tensor): Last hidden states from XLM-R of shape (batch_size, seq_len, hidden_size).
            category_mask (torch.Tensor): Binary mask of shape (batch_size, seq_len) with 1s at category tokens.

        Returns:
            torch.Tensor: Pooled aspect vector of shape (batch_size, hidden_size).
        """
        mask_exp = category_mask.unsqueeze(-1).float()           # (B, T, 1)
        sum_vec = (hidden * mask_exp).sum(dim=1)                # (B, H)
        count = mask_exp.sum(dim=1).clamp(min=1e-9)             # (B, 1)
        return sum_vec / count

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        category_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state                                       # (B, T, H)

        cls_vec = hidden[:, 0, :]                              # (B, H)
        aspect_vec = self._masked_mean_pool(hidden, category_mask)  # (B, H)
        combined = torch.cat([cls_vec, aspect_vec], dim=-1)    # (B, 2H)

        logits = self.classifier(self.dropout(combined))          # (B, 3)

        out = {"logits": logits}
        if labels is not None:
            out["loss"] = self.loss_fn(logits, labels)
        return out

    def freeze_lower_layers(self, num_layers: int = 6):
        """
        Freeze bottom encoder transformer layers to prevent catastrophic forgetting in few-shot settings.

        Args:
            num_layers (int): Number of lower transformer layers to freeze (`0` to `num_layers-1`).
        """
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                break


# ── LLRD optimizer ─────────────────────────────────────────────────────────────

def build_llrd_optimizer(
    model: XLMRForABSA,
    lr_embeddings: float = 1e-5,
    lr_encoder_low: float = 1.5e-5,
    lr_encoder_high: float = 2e-5,
    lr_classifier: float = 3e-5,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """
    Construct AdamW optimizer utilizing Layer-wise Learning Rate Decay (LLRD).

    Parameters are partitioned into embedding layers, lower/higher encoder transformer layers,
    and classification head layers with exponentially increasing learning rates.
    Weight decay is strictly excluded from bias vectors and LayerNorm parameters.

    Args:
        model (XLMRForABSA): The model instance to optimize.
        lr_embeddings (float): Learning rate for embedding parameters. Defaults to 1e-5.
        lr_encoder_low (float): Learning rate for layers 0-5. Defaults to 1.5e-5.
        lr_encoder_high (float): Learning rate for layers 6-11. Defaults to 2e-5.
        lr_classifier (float): Learning rate for classification head. Defaults to 3e-5.
        weight_decay (float): Weight decay coefficient. Defaults to 0.01.

    Returns:
        torch.optim.Optimizer: Configured `AdamW` optimizer instance.
    """
    no_decay = ["bias", "LayerNorm.weight"]

    def wd(name):
        return 0.0 if any(nd in name for nd in no_decay) else weight_decay

    groups = []

    # Embeddings
    for n, p in model.encoder.embeddings.named_parameters():
        if p.requires_grad:
            groups.append({"params": [p], "lr": lr_embeddings, "weight_decay": wd(n)})

    # Encoder layers
    for i, layer in enumerate(model.encoder.encoder.layer):
        lr = lr_encoder_low if i < 6 else lr_encoder_high
        for n, p in layer.named_parameters():
            if p.requires_grad:
                groups.append({"params": [p], "lr": lr, "weight_decay": wd(n)})

    # Classifier
    for n, p in model.classifier.named_parameters():
        if p.requires_grad:
            groups.append({"params": [p], "lr": lr_classifier, "weight_decay": wd(n)})

    return torch.optim.AdamW(groups)

def build_xlmr_model(cfg: dict) -> XLMRForABSA:
    """
    Factory method to construct and initialize the XLM-R model from configuration parameters.

    Args:
        cfg (dict): Global configuration dictionary.

    Returns:
        XLMRForABSA: Initialized XLMRForABSA model instance.
    """
    paths = cfg.get("model_paths", {})
    model_path = paths.get("xlmr", "pretrained_models/xlmr")
    if not Path(model_path).exists():
        model_path = "xlm-roberta-base"
    return XLMRForABSA(
        model_name=model_path,
        num_labels=3,
        dropout=cfg.get("dropout", 0.1),
        label_smoothing=cfg.get("label_smoothing", 0.1)
    )



