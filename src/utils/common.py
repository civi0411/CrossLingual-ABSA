"""
src/utils/common.py
───────────────────
Core shared utility functions: configuration management, deterministic seeding, device resolution, and checkpoint persistence.
"""
import os
import random
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import numpy as np

log = logging.getLogger(__name__)


class Config(dict):
    """Enables multi-level dot-attribute access for nested dictionary configurations loaded from YAML."""
    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
            return Config(v) if isinstance(v, dict) else v
        except KeyError:
            raise AttributeError(f"Config has no key '{k}'")

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f))


def ensure_dir(path: str | Path) -> Path:
    """Ensure the directory (and any necessary parent directories) exists on disk. Returns the `Path` instance."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preferred: str = "cuda") -> Any:
    import torch
    if preferred == "cuda" and torch.cuda.is_available():
        log.info("Device: GPU — %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        log.info("Device: Apple MPS")
        return torch.device("mps")
    log.info("Device: CPU")
    return torch.device("cpu")


def save_checkpoint(
    model: Any,
    optimizer: Optional[Any],
    epoch: int,
    score: float,
    filepath: str | Path,
    scaler: Optional[Any] = None,
    extra: Optional[Dict] = None,
) -> None:
    """
    Persist model checkpoints, optimizer states, gradient scaler states, and metadata to disk.

    Args:
        model (Any): Model instance to save.
        optimizer (Optional[Any]): Optimizer instance.
        epoch (int): Current epoch number.
        score (float): Validation metric score achieved at this checkpoint.
        filepath (str | Path): Destination file path.
        scaler (Optional[Any]): Mixed precision gradient scaler instance. Defaults to None.
        extra (Optional[Dict]): Additional metadata to bundle in the checkpoint payload. Defaults to None.
    """
    import torch
    ensure_dir(Path(filepath).parent)
    payload = {
        "epoch": epoch,
        "score": score,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if extra is not None:
        payload.update(extra)
    torch.save(payload, filepath)
    log.info("Checkpoint saved to %s (score=%.4f)", filepath, score)


def load_checkpoint(
    model: Any,
    optimizer: Optional[Any],
    filepath: str | Path,
    device: Any,
    scaler: Optional[Any] = None,
) -> Dict:
    """
    Restore model weights, optimizer states, and gradient scaler states from a saved checkpoint file.

    Args:
        model (Any): Target model instance to load weights into.
        optimizer (Optional[Any]): Optimizer instance to restore states into.
        filepath (str | Path): Source checkpoint file path.
        device (Any): PyTorch device mapping target.
        scaler (Optional[Any]): Gradient scaler instance to restore states into. Defaults to None.

    Returns:
        Dict: Parsed checkpoint dictionary payload containing epoch, score, and metadata.
    """
    import torch
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    log.info("Loaded checkpoint %s (epoch=%d, score=%.4f)", filepath, ckpt["epoch"], ckpt["score"])
    return ckpt