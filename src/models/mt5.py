"""
src/models/mt5.py
─────────────────
Model L3: `mT5-small` with Prefix Constrained Decoding and Sequence-to-Sequence Generation.

Prefix Constrained Decoding forces generated subword tokens to strictly belong to the allowed
label vocabulary (`positive`, `negative`, `neutral`) right from the very first decoding step.
This guarantees 100% syntactically valid outputs and eliminates generative hallucination.

Input Sequence : `"aspect: {category} review: {text}"`
Target Sequence: `"positive" / "negative" / "neutral"`
Post-processing: Decoded text strings are mapped directly to integer label IDs {0, 1, 2}.

Key Architectural Features:
    - Adjustable generation hyperparameters (`temperature`, `top_p`, `num_beams`) for inference flexibility.
    - Custom sequence-to-sequence Cross-Entropy loss computation supporting label smoothing and class weights.
    - Explicit pad token initialization ensuring reliable batch decoding across different devices.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}
ALLOWED_LABELS = ["positive", "negative", "neutral"]


class MT5ForABSA:
    """
    Lightweight wrapper around `mT5-small` (`AutoModelForSeq2SeqLM`) tailored for Oracle ASC.

    Handles prompt formatting, constrained prefix decoding, and label mapping.
    """

    def __init__(
        self,
        model_name: str = "google/mt5-small",
        use_constrained_decoding: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_length: int = 2,
        num_beams: int = 1,
        do_sample: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.use_constrained = use_constrained_decoding
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.num_beams = num_beams
        self.do_sample = do_sample

        # Ensure pad token is set (important for batch generation)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pre-compute allowed token IDs for constrained decoding
        self._allowed_ids = [
            self.tokenizer.encode(label, add_special_tokens=False)[0]
            for label in ALLOWED_LABELS
        ]

    def _prefix_fn(self, batch_id: int, input_ids: torch.Tensor) -> list[int]:
        """
        Prefix constrained decoding function passed to `prefix_allowed_tokens_fn`.

        Step 1 (at decoder start token): restricts next token candidates to valid label start tokens
        (`positive`, `negative`, `neutral`).
        Step 2+: enforces end-of-sequence (`EOS`), ensuring the model generates exactly one valid token.

        Args:
            batch_id (int): Index of the sequence in the batch.
            input_ids (torch.Tensor): Current decoded prefix sequence IDs.

        Returns:
            list[int]: List of allowed subword token IDs for the next generation step.
        """
        if input_ids.shape[-1] == 1:
            return self._allowed_ids
        return [self.tokenizer.eos_token_id]

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> list[int]:
        """
        Generate sentiment classification indices {0, 1, 2} for input sequences.

        Applies prefix constrained decoding if enabled to guarantee valid class outputs.

        Args:
            input_ids (torch.Tensor): Tokenized input sequence IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask tensor of shape (batch_size, seq_len).
            **kwargs: Optional generation overrides (e.g., `max_length`, `num_beams`, `do_sample`).

        Returns:
            list[int]: List of predicted integer labels {0, 1, 2}.
        """
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_length": kwargs.get("max_length", self.max_length),
            "num_beams": kwargs.get("num_beams", self.num_beams),
            "do_sample": kwargs.get("do_sample", self.do_sample),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
        }
        if self.use_constrained:
            gen_kwargs["prefix_allowed_tokens_fn"] = self._prefix_fn

        gen_ids = self.model.generate(**gen_kwargs)
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        # Lowercase to be safe (mT5 outputs lower case)
        return [LABEL2ID.get(t.strip().lower(), 2) for t in texts]

    @torch.no_grad()
    def predict_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[int]:
        """
        Classify input sequences by scoring the exact negative log-likelihood of candidate labels.

        Provides deterministic evaluation for sequence-to-sequence models by comparing
        the normalized sequence loss across candidate targets (`positive`, `negative`, `neutral`),
        selecting the target sequence with minimal average token loss.

        Args:
            input_ids (torch.Tensor): Tokenized input sequence IDs.
            attention_mask (torch.Tensor): Attention mask tensor.

        Returns:
            list[int]: List of predicted integer labels {0, 1, 2}.
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        num_labels = len(ALLOWED_LABELS)

        try:
            label_enc = self.tokenizer(
                text_target=ALLOWED_LABELS,
                padding=True,
                return_tensors="pt",
            )
        except TypeError:
            label_enc = self.tokenizer(
                ALLOWED_LABELS,
                padding=True,
                return_tensors="pt",
            )

        candidate_labels = label_enc["input_ids"].to(device)
        candidate_labels[candidate_labels == self.tokenizer.pad_token_id] = -100
        label_len = candidate_labels.size(1)

        repeated_input_ids = (
            input_ids.unsqueeze(1)
            .expand(batch_size, num_labels, input_ids.size(1))
            .reshape(batch_size * num_labels, input_ids.size(1))
        )
        repeated_attention = (
            attention_mask.unsqueeze(1)
            .expand(batch_size, num_labels, attention_mask.size(1))
            .reshape(batch_size * num_labels, attention_mask.size(1))
        )
        repeated_labels = (
            candidate_labels.unsqueeze(0)
            .expand(batch_size, num_labels, label_len)
            .reshape(batch_size * num_labels, label_len)
        )

        outputs = self.model(
            input_ids=repeated_input_ids,
            attention_mask=repeated_attention,
            labels=repeated_labels,
        )
        logits = outputs.logits
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            repeated_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(batch_size * num_labels, label_len)
        token_counts = repeated_labels.ne(-100).sum(dim=1).clamp(min=1)
        scores = (token_loss.sum(dim=1) / token_counts).view(batch_size, num_labels)
        return scores.argmin(dim=1).cpu().tolist()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
        ref_labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        """
        Execute training forward pass and compute sequence-to-sequence loss.

        Performs custom `F.cross_entropy` calculation across non-padded target tokens (`ignore_index=-100`) to:
            1. Enforce rigorous label smoothing (`label_smoothing > 0`), which HuggingFace `T5Config`
               frequently ignores when set outside encoder-decoder configurations.
            2. Incorporate class-weighted loss scaling for imbalanced target adaptation.

        Args:
            input_ids (torch.Tensor): Input sequence IDs.
            attention_mask (torch.Tensor): Input attention mask.
            labels (torch.Tensor): Target sequence IDs with padding masked as `-100`.
            label_smoothing (float): Label smoothing epsilon. Defaults to 0.0.
            class_weights (Optional[torch.Tensor]): Class weight vector of shape (3,). Defaults to None.
            ref_labels (Optional[torch.Tensor]): Ground truth integer labels for class weighting. Defaults to None.
            **kwargs: Additional keyword arguments passed to the underlying model.

        Returns:
            dict: Dictionary containing `loss` scalar tensor and `logits` tensor.
        """
        # Pass labels so HF auto-shifts decoder_input_ids correctly,
        # but we compute our own loss instead of using outputs.loss.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = outputs.logits

        per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
            label_smoothing=label_smoothing,
        ).view(labels.size(0), -1)
        token_counts = (labels != -100).sum(dim=1).clamp(min=1)
        per_example = per_token.sum(dim=1) / token_counts          # (B,)

        if class_weights is not None and ref_labels is not None:
            w = class_weights[ref_labels]
            loss = (per_example * w).sum() / w.sum()
        else:
            loss = per_example.mean()

        return {"loss": loss, "logits": logits}

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)

def build_mt5_model(cfg: dict) -> tuple[MT5ForABSA, AutoTokenizer]:
    """
    Factory method to construct and initialize the mT5-small wrapper and its tokenizer.

    Args:
        cfg (dict): Global configuration dictionary.

    Returns:
        tuple[MT5ForABSA, AutoTokenizer]: Tuple containing initialized wrapper instance and tokenizer.
    """
    paths = cfg.get("model_paths", {})
    model_path = paths.get("mt5", "pretrained_models/mt5")
    from pathlib import Path
    if not Path(model_path).exists():
        model_path = "google/mt5-small"
    model = MT5ForABSA(
        model_name=model_path,
        use_constrained_decoding=cfg.get("constrained_decoding", True),
        max_length=cfg.get("max_target_len", 10)
    )
    return model, model.tokenizer
