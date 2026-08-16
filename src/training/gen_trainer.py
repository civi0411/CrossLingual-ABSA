"""
src/training/gen_trainer.py
───────────────────────────
Generative Sequence-to-Sequence Trainer for L3 (`mT5-small`).

Data Flow and Workflow:
    - Batch Structure:
        `input_ids`, `attention_mask`: Formatted prompt (`"aspect: {category} review: {text}"`)
        `labels`: Target token IDs (`"positive"`, `"negative"`, `"neutral"`) with padding masked as `-100`.
        `ref_labels`: Ground-truth integer class indices `{0, 1, 2}` used during validation metrics computation.
    - Training Pass: Executes teacher-forced forward pass with `labels` -> computes custom `F.cross_entropy` loss.
    - Evaluation Pass: Executes either deterministic `predict_labels` scoring or `generate` with prefix constrained decoding -> maps output strings to integer class indices `{0, 1, 2}` -> computes macro F1 score.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from src.data.ingest import assert_no_leakage
from src.utils.common import set_seed, get_device, save_checkpoint, ensure_dir, load_checkpoint

log = logging.getLogger(__name__)


class GenerativeTrainer:
    """
    Sequence-to-Sequence Generative Trainer tailored for `mT5-small` (`MT5ForABSA`).

    Integrates constrained prefix decoding during validation/inference, automatic mixed precision (`AMP`),
    class-weighted loss scaling across sequence targets, and checkpoint recovery.
    """

    def __init__(
        self,
        model,                          # MT5ForABSA wrapper
        train_loader,
        val_loader,
        train_samples: list[dict],
        config: dict,
        domain: str,
        setting: str,
        target_lang: str,
        resume_from: str | None = None,
    ):
        set_seed(config.get("seed", 42))
        self.device = get_device()

        self.model = model
        self.model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.domain = domain
        self.setting = setting
        self.target_lang = target_lang

        # Mixed precision
        self.use_amp = config.get("use_amp", True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            log.info("Mixed precision (AMP) enabled")
        else:
            self.scaler = None

        # Leakage guard
        allowed = ["en"] if setting == "s1" else ["en", target_lang]
        assert_no_leakage(train_samples, allowed, setting)

        # Class-weighted loss — synchronized across AG-CAN/XLM-R architectures
        labels_list = [s["label"] for s in train_samples]
        cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=labels_list)
        self.class_weights = torch.tensor(cw, dtype=torch.float).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("lr", 5e-4),
            weight_decay=config.get("weight_decay", 0.01),
        )

        # Scheduler
        total_steps = len(train_loader) * config.get("epochs", 10)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * config.get("warmup_ratio", 0.1)),
            num_training_steps=total_steps,
        )

        # Early stopping
        self.best_f1 = 0.0
        self.patience = config.get("early_stopping_patience", 3)
        self.patience_counter = 0
        self.grad_clip = config.get("grad_clip", 1.0)

        # Output directory
        self.out_dir = ensure_dir(
            Path(config.get("output_dir", "outputs/checkpoints"))
            / "mt5"
            / domain
            / setting
            / config.get("run_id", f"{target_lang}_seed{config.get('seed', 42)}")
        )

        # Resume if checkpoint provided
        self.start_epoch = 0
        if resume_from:
            ckpt = load_checkpoint(
                self.model, self.optimizer, resume_from, self.device, self.scaler
            )
            self.start_epoch = ckpt.get("epoch", 0)
            self.best_f1 = ckpt.get("score", 0.0)
            log.info(f"Resumed from {resume_from}, epoch {self.start_epoch}, best_f1={self.best_f1:.4f}")

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_valid = 0
        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            ref_labels = batch["ref_labels"].to(self.device)

            with autocast("cuda", enabled=self.use_amp):
                out = self.model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    label_smoothing=self.config.get("mt5_label_smoothing", 0.0),
                    class_weights=self.class_weights,
                    ref_labels=ref_labels,
                )
                loss = out["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                log.warning("NaN/Inf loss encountered at batch — skipping step (use_amp=%s).", self.use_amp)
                self.optimizer.zero_grad()
                continue

            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()
            n_valid += 1
        return total_loss / max(n_valid, 1)

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_metas = []
        for batch in loader:
            if self.config.get("mt5_eval_method", "label_scoring") == "label_scoring":
                pred_ids = self.model.predict_labels(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )
            else:
                pred_ids = self.model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )
            all_preds.extend(pred_ids)
            all_labels.extend(batch["ref_labels"].tolist())
            if "meta" in batch:
                all_metas.extend(batch["meta"])
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return {"macro_f1": macro_f1, "preds": all_preds, "labels": all_labels, "metas": all_metas}

    def train(self, epochs: int) -> dict:
        log.info(
            "Train mT5 | %s | %s | target=%s",
            self.domain, self.setting, self.target_lang,
        )
        history = {"train_loss": [], "val_f1": []}

        for epoch in range(self.start_epoch, epochs):
            train_loss = self._train_epoch()
            val_result = self.evaluate(self.val_loader)
            val_f1 = val_result["macro_f1"]
            history["train_loss"].append(train_loss)
            history["val_f1"].append(val_f1)
            log.info(
                "  Ep %d/%d | loss=%.4f | val_f1=%.4f",
                epoch + 1, epochs, train_loss, val_f1,
            )

            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    val_f1,
                    self.out_dir / "best.pt",
                    scaler=self.scaler,
                )
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    log.info("  Early stopping at epoch %d", epoch + 1)
                    break

            # Clear GPU cache after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log.info("  Best val F1: %.4f", self.best_f1)
        return history
