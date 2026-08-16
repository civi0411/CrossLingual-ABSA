"""
src/training/base_cls_trainer.py
────────────────────────────────
Abstract BaseTrainer class providing core standardized training infrastructure.

Executes standardized workflows including:
    - Deterministic seeding for reproducibility (`set_seed`)
    - Strict cross-lingual data leakage verification (`assert_no_leakage`)
    - Balanced class-weighted loss computation (`compute_class_weight`)
    - Automatic Mixed Precision (`AMP`) training with gradient scaling
    - Early stopping and gradient clipping
    - Checkpoint persistence and resume capabilities

Subclasses must implement abstract methods `_forward()` and `_predict()`.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from src.utils.common import set_seed, get_device, save_checkpoint, ensure_dir, load_checkpoint
from src.data.ingest import assert_no_leakage

log = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base trainer encapsulating shared training loops, evaluation workflows, and early stopping logic.

    All concrete classification subclasses must implement `_forward()` and `_predict()`.
    """

    def __init__(
        self,
        model,
        optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_samples: list[dict],
        config: dict,
        domain: str,
        model_name: str,
        setting: str,
        target_lang: str,
        scheduler=None,
        resume_from: str | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.domain = domain
        self.model_name = model_name
        self.setting = setting
        self.target_lang = target_lang

        # Reproducibility
        seed = config.get("seed", 42)
        set_seed(seed)

        # Device
        self.device = get_device()
        self.model.to(self.device)

        # Mixed precision
        self.use_amp = config.get("use_amp", True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            log.info("Mixed precision (AMP) enabled")
        else:
            self.scaler = None

        # Leakage guard
        allowed_langs = ["en"] if setting == "s1" else ["en", target_lang]
        assert_no_leakage(train_samples, allowed_langs, setting)

        # Class-weighted loss
        labels = [s["label"] for s in train_samples]
        class_weights = compute_class_weight(
            "balanced", classes=np.array([0, 1, 2]), y=labels
        )
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float).to(self.device),
            label_smoothing=config.get("label_smoothing", 0.0)
        )

        # Early stopping state
        self.best_f1 = 0.0
        self.patience = config.get("early_stopping_patience", 3)
        self.patience_counter = 0

        # Gradient clipping
        self.grad_clip = config.get("grad_clip", 1.0)

        # Output directory for checkpoints
        self.out_dir = ensure_dir(
            Path(config.get("output_dir", "outputs/checkpoints"))
            / model_name
            / domain
            / setting
            / config.get("run_id", f"{target_lang}_seed{seed}")
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

    @abstractmethod
    def _forward(self, batch: dict) -> dict:
        """Execute model forward pass and compute training loss. Must return a dict containing at least `'logits'`."""
        pass

    @abstractmethod
    def _predict(self, batch: dict) -> list[int]:
        """Execute evaluation forward pass and return a list of predicted integer class indices {0, 1, 2}."""
        pass

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with autocast("cuda", enabled=self.use_amp):
                out = self._forward(batch)
                labels = batch["labels"].to(self.device)
                loss = self.criterion(out["logits"], labels)

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

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_metas = []
        for batch in loader:
            batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items() if k != "meta"}
            preds = self._predict(batch_device)
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().tolist())
            if "meta" in batch:
                all_metas.extend(batch["meta"])
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return {"macro_f1": macro_f1, "preds": all_preds, "labels": all_labels, "metas": all_metas}

    def train(self, epochs: int) -> dict:
        log.info(
            "Train %s | %s | %s | target=%s",
            self.model_name, self.domain, self.setting, self.target_lang,
        )
        history = {"train_loss": [], "val_f1": []}

        for epoch in range(self.start_epoch, epochs):
            train_loss = self._train_epoch()
            val_result = self.evaluate(self.val_loader)
            val_f1 = val_result["macro_f1"]
            history["train_loss"].append(train_loss)
            history["val_f1"].append(val_f1)

            log.info(
                "  Epoch %d/%d | train_loss=%.4f | val_f1=%.4f",
                epoch + 1, epochs, train_loss, val_f1,
            )

            # Save if best
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

            # Optional: clear GPU cache after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log.info("  Best val F1: %.4f", self.best_f1)
        return history
