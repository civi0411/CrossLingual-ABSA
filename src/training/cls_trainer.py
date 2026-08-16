"""
src/training/cls_trainer.py
───────────────────────────
Unified Classification Trainer for L1 (AG-CAN) and L2 (XLM-R) models.

Data Flow and Forward Routing:
    - L1 AG-CAN: Batch provides `text_ids` + `aspect_ids` (dual-encoded independent tensors).
                 `_forward()` routes both input sequences to the model.
    - L2 XLM-R : Batch provides `input_ids` + `category_mask` (single merged tensor).
                 `_forward()` routes the merged tensor and exact subword category mask to the model.

Both architectures output classification logits of shape `(batch_size, 3)` corresponding to `{positive, negative, neutral}`.
"""

import torch
from transformers import get_linear_schedule_with_warmup

from src.training.base_cls_trainer import BaseTrainer
from src.models.xlmr import build_llrd_optimizer


class ClassificationTrainer(BaseTrainer):
    """
    Concrete trainer handling both L1 (AG-CAN) and L2 (XLM-R) classification architectures.

    Routing inside `_forward()` is dynamically resolved based on `model_name`.
    """

    def __init__(self, *args, warmup_ratio: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)

        # XLM-R and upgraded AG-CAN both benefit from warmup while fine-tuning.
        if self.model_name in ("xlmr", "ag_can"):
            total_steps = len(self.train_loader) * self.config.get("epochs", 10)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(total_steps * warmup_ratio),
                num_training_steps=total_steps,
            )
        else:
            self.scheduler = None

    # ── Forward routing ────────────────────────────────────────────────────────

    def _forward(self, batch: dict) -> dict:
        """Route input batch tensors to the appropriate model forward pass based on `model_name`."""
        if self.model_name == "ag_can":
            return self._forward_agcan(batch)
        elif self.model_name == "xlmr":
            return self._forward_xlmr(batch)
        raise ValueError(f"Unknown model: {self.model_name}")

    def _forward_agcan(self, batch: dict) -> dict:
        """
        Execute forward pass for L1 AG-CAN.

        Requires batch keys:
            - `input_ids`, `attention_mask` (for review sentence)
            - `aspect_ids`, `aspect_mask` (for aspect category)
        """
        return self.model(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            aspect_ids=batch["aspect_ids"].to(self.device),
            aspect_mask=batch["aspect_mask"].to(self.device),
        )

    def _forward_xlmr(self, batch: dict) -> dict:
        """
        Execute forward pass for L2 XLM-R.

        Requires batch keys:
            - `input_ids`, `attention_mask` (for merged text + category sequence)
            - `category_mask` (exact subword locations of category tokens)
        """
        return self.model(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            category_mask=batch["category_mask"].to(self.device),
        )

    # ── Predict ────────────────────────────────────────────────────────────────

    def _predict(self, batch: dict) -> list[int]:
        out = self._forward(batch)
        return out["logits"].argmax(dim=-1).cpu().tolist()


# ── Builder functions ──────────────────────────────────────────────────────────

def build_agcan_trainer(
    model,
    train_loader,
    val_loader,
    train_samples,
    config: dict,
    domain: str,
    setting: str,
    target_lang: str,
) -> ClassificationTrainer:
    """
    Construct and configure the trainer for AG-CAN (`L1`).

    Applies differential learning rates: fine-tunes top encoder layers (`embedder`) at a lower learning rate
    while training the aspect attention pooling and classification head at a higher learning rate.
    """
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("embedder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if encoder_params:
        param_groups.append({
            "params": encoder_params,
            "lr": config.get("agcan_encoder_lr", 2e-5),
            "weight_decay": config.get("weight_decay", 0.01),
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": config.get("agcan_head_lr", config.get("lr", 1e-3)),
            "weight_decay": config.get("weight_decay", 0.01),
        })

    optimizer = torch.optim.AdamW(param_groups)
    return ClassificationTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        train_samples=train_samples,
        config=config,
        domain=domain,
        model_name="ag_can",
        setting=setting,
        target_lang=target_lang,
    )


def build_xlmr_trainer(
    model,
    train_loader,
    val_loader,
    train_samples,
    config: dict,
    domain: str,
    setting: str,
    target_lang: str,
) -> ClassificationTrainer:
    """
    Construct and configure the trainer for XLM-R (`L2`) equipped with Layer-wise Learning Rate Decay (`LLRD`).

    Assigns exponentially scaled learning rates across embedding layers, bottom/top transformer blocks,
    and the final classification head.
    """
    optimizer = build_llrd_optimizer(
        model,
        lr_embeddings=config.get("lr_embeddings", 1e-5),
        lr_encoder_low=config.get("lr_encoder_low", 1.5e-5),
        lr_encoder_high=config.get("lr_encoder_high", 2e-5),
        lr_classifier=config.get("lr_classifier", 3e-5),
        weight_decay=config.get("weight_decay", 0.01),
    )
    return ClassificationTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        train_samples=train_samples,
        config=config,
        domain=domain,
        model_name="xlmr",
        setting=setting,
        target_lang=target_lang,
        warmup_ratio=config.get("warmup_ratio", 0.1),
    )
