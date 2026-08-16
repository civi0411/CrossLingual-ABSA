#!/usr/bin/env python
"""
train.py — Unified CLI Entry Point for Cross-lingual ABSA Experiments.

Few-shot Evaluation Protocol:
    - `s1` (Zero-shot)  : Trained solely on English (`EN`), evaluated directly on target languages (`VI`, `DE`, `ZH`).
    - `s2` (Few-shot)   : English (`EN`) + `N` target-language training samples (`N` in {50, 100, 200}).
                          Each `N` represents an independent non-accumulating run initialized from `S1` weights.
                          -> REQUIRES the corresponding (`model`, `domain`, `target`) `S1` run to be completed first.
    - `s3` (Full-target): Trained on English (`EN`) + the complete target-language training split.
                          -> REQUIRES the corresponding (`model`, `domain`, `target`) `S1` run to be completed first.

Usage Examples:
    # Run zero-shot S1 benchmark across all models, domains, and targets
    python train.py --setting s1

    # Run S1 for a specific model, domain, and target language
    python train.py --setting s1 --models xlmr --domains restaurant --targets vi

    # Run few-shot S2 (requires completed S1 checkpoint)
    python train.py --setting s2 --n_values 50 100 200 --seeds 42 123 456

    # Run full-target S3 (requires completed S1 checkpoint)
    python train.py --setting s3

    # Execute all evaluation stages sequentially: S1 -> S2 -> S3
    python train.py --setting all

    # Execute S2 for a specific target sample count N and seed
    python train.py --setting s2 --n_values 50 --seeds 42 --models xlmr --domains restaurant --targets vi
"""

import argparse
import json
import logging
import sys
from itertools import product
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

import yaml

# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_logging(log_file: str = "outputs/logs/experiment.log", level: str = "INFO"):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, handlers=handlers)

log = logging.getLogger(__name__)


# ── Config loader ──────────────────────────────────────────────────────────────

def load_config(path: str = "config.yml") -> dict:
    """Load YAML configuration file into a flat dictionary populated with robust fallback defaults."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        log.warning("Config not found at %s — using built-in defaults.", path)
        return _default_config()

    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Flatten nested YAML structure into a flat dictionary for direct trainer access
    cfg = {}

    data = raw.get("data", {})
    cfg["processed_dir"]              = data.get("processed_dir", "data/processed")
    cfg["max_len"]                    = data.get("max_len", {"restaurant": 128, "phone": 160})
    cfg["min_tokens"]                 = data.get("min_tokens", 3)

    paths = raw.get("model_paths", {})
    cfg["model_paths"] = {
        "mbert": paths.get("mbert", "pretrained_models/mbert"),
        "xlmr":  paths.get("xlmr",  "pretrained_models/xlmr"),
        "mt5":   paths.get("mt5",   "pretrained_models/mt5"),
    }

    tr = raw.get("training", {})
    cfg["batch_size"]               = tr.get("batch_size", 16)
    cfg["epochs"]                   = tr.get("epochs", 20)
    cfg["early_stopping_patience"]  = tr.get("early_stopping_patience", 5)
    cfg["grad_clip"]                = tr.get("grad_clip", 1.0)
    cfg["use_amp"]                  = tr.get("use_amp", True)
    cfg["label_smoothing"]          = tr.get("label_smoothing", 0.1)

    # Flatten per-model hyperparameters with distinct prefixes
    agcan = tr.get("ag_can", {})
    cfg["lr"]           = agcan.get("lr", 1e-3)
    cfg["agcan_head_lr"] = agcan.get("head_lr", agcan.get("lr", 1e-3))
    cfg["agcan_encoder_lr"] = agcan.get("encoder_lr", 2e-5)
    cfg["agcan_unfreeze_last_n"] = agcan.get("unfreeze_last_n", 4)
    cfg["weight_decay"] = agcan.get("weight_decay", 0.01)
    cfg["hidden_dim"]   = agcan.get("hidden_dim", 256)
    cfg["num_heads"]    = agcan.get("num_heads", 8)
    cfg["dropout"]      = agcan.get("dropout", 0.3)

    xlmr = tr.get("xlmr", {})
    cfg["lr_embeddings"]   = xlmr.get("lr_embeddings",   1e-5)
    cfg["lr_encoder_low"]  = xlmr.get("lr_encoder_low",  1.5e-5)
    cfg["lr_encoder_high"] = xlmr.get("lr_encoder_high", 2e-5)
    cfg["lr_classifier"]   = xlmr.get("lr_classifier",   3e-5)
    cfg["warmup_ratio"]    = xlmr.get("warmup_ratio",     0.1)

    mt5 = tr.get("mt5", {})
    cfg["mt5_lr"]               = mt5.get("lr", 3e-4)
    cfg["mt5_label_smoothing"]  = mt5.get("label_smoothing", 0.1)
    cfg["mt5_eval_method"]      = mt5.get("eval_method", "label_scoring")
    cfg["max_target_len"]       = mt5.get("max_target_len", 5)
    cfg["constrained_decoding"] = mt5.get("constrained_decoding", True)

    evl = raw.get("evaluation", {})
    cfg["results_dir"]                    = evl.get("results_dir", "outputs/results")
    cfg["errors_dir"]                     = evl.get("errors_dir",  "outputs/errors")
    cfg["min_test_samples_per_category"]  = evl.get("min_test_samples_per_category", 5)

    cfg["output_dir"] = "outputs/checkpoints"

    lg = raw.get("logging", {})
    cfg["log_level"] = lg.get("level", "INFO")
    cfg["log_file"]  = lg.get("file",  "outputs/logs/experiment.log")

    return cfg


def _default_config() -> dict:
    return {
        "processed_dir": "data/processed",
        "max_len": {"restaurant": 128, "phone": 160},
        "min_tokens": 3,
        "model_paths": {
            "mbert": "pretrained_models/mbert",
            "xlmr":  "pretrained_models/xlmr",
            "mt5":   "pretrained_models/mt5",
        },
        "batch_size": 16, "epochs": 20,
        "early_stopping_patience": 5, "grad_clip": 1.0,
        "use_amp": True, "label_smoothing": 0.1,
        "lr": 1e-3, "agcan_head_lr": 1e-3, "agcan_encoder_lr": 2e-5,
        "agcan_unfreeze_last_n": 4, "weight_decay": 0.01,
        "hidden_dim": 256, "num_heads": 8, "dropout": 0.3,
        "lr_embeddings": 1e-5, "lr_encoder_low": 1.5e-5,
        "lr_encoder_high": 2e-5, "lr_classifier": 3e-5,
        "warmup_ratio": 0.1,
        "mt5_lr": 3e-4, "mt5_label_smoothing": 0.1,
        "mt5_eval_method": "label_scoring",
        "max_target_len": 5, "constrained_decoding": True,
        "results_dir": "outputs/results",
        "errors_dir":  "outputs/errors",
        "plots_dir": "outputs/plots",
        "min_test_samples_per_category": 5,
        "output_dir": "outputs/checkpoints",
        "log_level": "INFO", "log_file": "outputs/logs/experiment.log",
    }


# ── Prerequisite guard ─────────────────────────────────────────────────────────

def s1_result_exists(results_dir: Path, model: str, domain: str, target: str) -> bool:
    """
    Check whether the zero-shot (`s1`) evaluation artifact exists.

    Expected file naming convention: `{model}_{domain}_s1_{target}_0_42.json`
    """
    pattern = f"{model}_{domain}_s1_{target}_0_*.json"
    return any((results_dir / "s1").glob(pattern))


def require_s1(results_dir: Path, model: str, domain: str, target: str):
    """Abort execution immediately if prerequisite S1 results are missing."""
    if not s1_result_exists(results_dir, model, domain, target):
        log.error(
            "Missing prerequisite S1 result for (%s, %s, %s). "
            "Execute `python train.py --setting s1 --models %s --domains %s --targets %s` first.",
            model, domain, target, model, domain, target,
        )
        sys.exit(1)


# ── Result saver ───────────────────────────────────────────────────────────────

def save_metrics(
    metrics: dict,
    model: str, domain: str, setting: str,
    target: str, n: int, seed: int,
    out_dir: Path,
):
    setting_dir = out_dir / setting
    setting_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{model}_{domain}_{setting}_{target}_{n}_{seed}.json"
    path = setting_dir / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.info("Metrics -> %s", path)
    return path


def save_errors(result: dict, model: str, domain: str, setting: str, target: str, n: int, seed: int, cfg: dict):
    errors_dir = Path(cfg.get("errors_dir", "outputs/errors")) / setting
    errors_dir.mkdir(parents=True, exist_ok=True)
    n_val = n if setting == "s2" else 0
    error_filename = f"errors_{model}_{domain}_{setting}_{target}_{n_val}_{seed}.jsonl"
    error_filepath = errors_dir / error_filename

    preds = result["preds"]
    labels = result["labels"]
    metas = result.get("metas", [])

    errors = []
    for i in range(len(preds)):
        if preds[i] != labels[i]:
            meta_i = metas[i] if i < len(metas) else {}
            errors.append({
                "id": meta_i.get("id", ""),
                "lang": target,
                "domain": domain,
                "category": meta_i.get("category", ""),
                "true": labels[i],
                "pred": preds[i],
                "text": meta_i.get("text", ""),
                "is_implicit": meta_i.get("is_implicit", False)
            })

    with open(error_filepath, "w", encoding="utf-8") as f:
        for err in errors:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")
    log.info("Saved error cases to %s", error_filepath)


def save_history_plot(history: dict, model: str, domain: str, setting: str,
                      target: str, n: int, seed: int, plots_dir: Path):
    """Persist training convergence metrics to JSON and render epoch-wise loss/F1 plot as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_dir = plots_dir / setting
        plot_dir.mkdir(parents=True, exist_ok=True)
        fname = f"history_{model}_{domain}_{setting}_{target}_{n}_{seed}"

        # Save JSON
        with open(plot_dir / f"{fname}.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Plot
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Train Loss", color=color)
        ax1.plot(epochs, history["train_loss"], color=color, marker="o", label="Train Loss")
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Validation F1", color=color)
        ax2.plot(epochs, history["val_f1"], color=color, marker="x", label="Val F1")
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()
        plt.title(f"Training History: {model} ({domain}/{target} - {setting} N={n} Seed={seed})")
        plt.savefig(plot_dir / f"{fname}.png")
        plt.close()
        log.info("Saved training history and plot to %s", plot_dir)
    except Exception as e:
        log.warning("Could not save history plot: %s", e)


# ── Single-run core ────────────────────────────────────────────────────────────

def run_one(
    model_name: str,
    domain: str,
    setting: str,
    target: str,
    n: int,
    seed: int,
    cfg: dict,
) -> dict:
    """
    Execute a single experimental run.

    For `s2`: samples `n` target records independently initialized using `seed`.
    Returns evaluation metrics dictionary.
    """
    from src.utils.common import set_seed
    from src.data.ingest import build_datasets, load_domain

    set_seed(seed)
    run_id = f"{target}_n{n if setting == 's2' else 0}_seed{seed}"
    cfg = {**cfg, "seed": seed, "n": n, "run_id": run_id}

    log.info("▶ %s | %s | %s | target=%s | n=%s | seed=%s",
             model_name, domain, setting, target, n, seed)

    # ── Load data ──────────────────────────────────────────────────────────────
    processed_dir = Path(cfg["processed_dir"])
    required_files = [
        processed_dir / domain / lang / f"{split}.jsonl"
        for lang in ("en", target)
        for split in ("training", "val", "test")
    ]
    if any(not p.exists() for p in required_files):
        log.info("Processed files missing for %s/%s; preparing data from local raw files.", domain, target)
        load_domain(domain, cfg["processed_dir"])

    data = build_datasets(
        domain=domain,
        processed_dir=cfg["processed_dir"],
        setting=setting,
        target_lang=target,
        n_shot=n if setting == "s2" else None,
        seed=seed,
    )
    train_samples = data["training"]
    val_samples = data["val"]
    test_samples = data["test_target"]

    if model_name in ("ag_can", "xlmr"):
        metrics = _run_cls(model_name, domain, setting, target,
                           train_samples, val_samples, test_samples, cfg)
    else:
        metrics = _run_gen(domain, setting, target,
                           train_samples, val_samples, test_samples, cfg)

    actual_n = n
    if setting == "s2":
        actual_n = sum(1 for s in train_samples if s["lang"] == target)
    metrics["actual_n"] = actual_n

    log.info("✓ macro_f1=%.4f | acc=%.4f | actual_n=%d", metrics["macro_f1"], metrics["accuracy"], actual_n)
    return metrics


def _run_cls(model_name, domain, setting, target,
             train_samples, val_samples, test_samples, cfg):
    from src.data.dataset import build_cls_loaders
    from src.evaluation.metrics import compute_metrics
    from src.training.cls_trainer import build_agcan_trainer, build_xlmr_trainer

    train_loader, val_loader, test_loader = build_cls_loaders(
        train_samples, val_samples, test_samples, cfg, model_name
    )

    if model_name == "ag_can":
        from src.models.ag_can import build_agcan_model
        model = build_agcan_model(cfg)
        trainer = build_agcan_trainer(
            model=model,
            train_loader=train_loader, val_loader=val_loader,
            train_samples=train_samples,
            config=cfg, domain=domain,
            setting=setting, target_lang=target,
        )
    else:
        from src.models.xlmr import build_xlmr_model
        model = build_xlmr_model(cfg)
        trainer = build_xlmr_trainer(
            model=model,
            train_loader=train_loader, val_loader=val_loader,
            train_samples=train_samples,
            config=cfg, domain=domain,
            setting=setting, target_lang=target,
        )

    # Explicitly load S1 checkpoint for S2 and S3
    if setting in ["s2", "s3"]:
        import torch
        from src.utils.common import load_checkpoint, get_device
        # S1 is always trained once with seed=42
        s1_seed = 42 
        s1_ckpt_path = (
            Path(cfg.get("output_dir", "outputs/checkpoints"))
            / model_name / domain / "s1" / f"s1_seed{s1_seed}" / "best.pt"
        )
        if s1_ckpt_path.exists():
            log.info("Loading S1 checkpoint weights from %s", s1_ckpt_path)
            load_checkpoint(model, optimizer=None, filepath=s1_ckpt_path, device=get_device())
        else:
            log.warning("S1 checkpoint not found at %s. Training from scratch.", s1_ckpt_path)

    history = trainer.train(epochs=cfg["epochs"])
    n = cfg.get("n", 0)
    seed = cfg.get("seed", 42)
    save_history_plot(history, model_name, domain, setting, target, n, seed,
                      Path(cfg.get("plots_dir", "outputs/plots")))

    # Load best checkpoint from current run before evaluation
    best_ckpt_path = Path(trainer.out_dir) / "best.pt"
    if best_ckpt_path.exists():
        import torch
        from src.utils.common import load_checkpoint, get_device
        log.info("Loading best checkpoint from current run for evaluation: %s", best_ckpt_path)
        load_checkpoint(model, optimizer=None, filepath=best_ckpt_path, device=get_device())

    result = trainer.evaluate(test_loader)
    save_errors(result, model_name, domain, setting, target, cfg.get("n", 0), cfg.get("seed", 42), cfg)
    
    # Cleanup S2/S3 checkpoints to prevent disk overflow
    if setting in ["s2", "s3"] and best_ckpt_path.exists():
        best_ckpt_path.unlink()
        log.info("Deleted %s to save disk space.", best_ckpt_path)
        
    return compute_metrics(result["preds"], result["labels"])


def _run_gen(domain, setting, target,
             train_samples, val_samples, test_samples, cfg):
    from src.data.dataset import build_gen_loaders
    from src.models.mt5 import build_mt5_model
    from src.training.gen_trainer import GenerativeTrainer
    from src.evaluation.metrics import compute_metrics

    n = cfg.get("n", 0)
    seed = cfg.get("seed", 42)

    # Assign model-specific learning rate and disable AMP (T5/mT5 prone to NaN under FP16)
    gen_cfg = {**cfg, "lr": cfg.get("mt5_lr", 3e-4), "use_amp": False}

    model, tokenizer = build_mt5_model(gen_cfg)

    # Load S1 checkpoint for S2 / S3
    if setting in ["s2", "s3"]:
        import torch
        from src.utils.common import load_checkpoint, get_device
        # S1 is always trained once with seed=42
        s1_seed = 42 
        s1_ckpt_path = (
            Path(cfg.get("output_dir", "outputs/checkpoints"))
            / "mt5" / domain / "s1" / f"s1_seed{s1_seed}" / "best.pt"
        )
        if s1_ckpt_path.exists():
            log.info("Loading S1 checkpoint weights from %s", s1_ckpt_path)
            load_checkpoint(model, optimizer=None, filepath=s1_ckpt_path, device=get_device())
        else:
            if not cfg.get("mt5_allow_scratch_fallback", False):
                raise RuntimeError(
                    f"Missing S1 checkpoint at {s1_ckpt_path} for mT5 ({setting}/{domain}/{target}).\n"
                    f"Checkpoints may have been purged or not initialized.\n"
                    f"Re-run S1 baseline: python train.py --setting s1 --models mt5 --domains {domain} --targets {target}\n"
                    f"(To explicitly permit training from scratch, set `mt5_allow_scratch_fallback: true` in config.yml)"
                )
            log.warning("S1 checkpoint not found — training from scratch (permitted via config).")

    train_loader, val_loader, test_loader = build_gen_loaders(
        train_samples, val_samples, test_samples, tokenizer, gen_cfg
    )

    trainer = GenerativeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_samples=train_samples,
        config=gen_cfg,
        domain=domain,
        setting=setting,
        target_lang=target,
    )

    history = trainer.train(epochs=gen_cfg["epochs"])
    save_history_plot(history, "mt5", domain, setting, target, n, seed,
                      Path(cfg.get("plots_dir", "outputs/plots")))

    # Load best checkpoint from current run before evaluation
    best_ckpt_path = Path(trainer.out_dir) / "best.pt"
    if best_ckpt_path.exists():
        import torch
        from src.utils.common import load_checkpoint, get_device
        log.info("Loading best checkpoint from current run for evaluation: %s", best_ckpt_path)
        load_checkpoint(model, optimizer=None, filepath=best_ckpt_path, device=get_device())

    result = trainer.evaluate(test_loader)
    save_errors(result, "mt5", domain, setting, target, n, seed, cfg)

    # Cleanup S2/S3 checkpoints to prevent disk overflow
    if setting in ["s2", "s3"]:
        best_ckpt_path = Path(trainer.out_dir) / "best.pt"
        if best_ckpt_path.exists():
            best_ckpt_path.unlink()
            log.info("Deleted %s to save disk space.", best_ckpt_path)

    return compute_metrics(result["preds"], result["labels"])


# ── Setting runners ────────────────────────────────────────────────────────────

def run_s1_single(model_name: str, domain: str, targets: list[str], seed: int, cfg: dict, results_dir: Path):
    from src.utils.common import set_seed, load_checkpoint, get_device
    from src.data.ingest import build_datasets, load_split
    
    set_seed(seed)
    run_id = f"s1_seed{seed}"
    cfg = {**cfg, "seed": seed, "n": 0, "run_id": run_id}

    log.info("▶ %s | %s | s1 | target=ALL | seed=%s", model_name, domain, seed)

    # Build dataset using EN as the target temporarily to get EN train/val loaders
    data = build_datasets(
        domain=domain,
        processed_dir=cfg["processed_dir"],
        setting="s1",
        target_lang="en",
        n_shot=0,
        seed=seed,
    )
    train_samples = data["training"]
    val_samples = data["val"]
    
    if model_name in ("ag_can", "xlmr"):
        from src.data.dataset import build_cls_loaders
        from src.training.cls_trainer import build_agcan_trainer, build_xlmr_trainer
        train_loader, val_loader, _ = build_cls_loaders(train_samples, val_samples, data["test_target"], cfg, model_name)
        
        if model_name == "ag_can":
            from src.models.ag_can import build_agcan_model
            model = build_agcan_model(cfg)
            trainer = build_agcan_trainer(model, train_loader, val_loader, train_samples, cfg, domain, "s1", "en")
        else:
            from src.models.xlmr import build_xlmr_model
            model = build_xlmr_model(cfg)
            trainer = build_xlmr_trainer(model, train_loader, val_loader, train_samples, cfg, domain, "s1", "en")
            
        history = trainer.train(epochs=cfg["epochs"])
        save_history_plot(history, model_name, domain, "s1", "ALL", 0, seed, Path(cfg.get("plots_dir", "outputs/plots")))
        
        # Load best S1 checkpoint
        best_ckpt_path = Path(trainer.out_dir) / "best.pt"
        if best_ckpt_path.exists():
            load_checkpoint(model, optimizer=None, filepath=best_ckpt_path, device=get_device())
            
        # Loop over targets to evaluate
        from src.evaluation.metrics import compute_metrics
        for target in targets:
            test_samples = load_split(domain, target, "test", cfg["processed_dir"])
            _, _, test_loader = build_cls_loaders(train_samples, val_samples, test_samples, cfg, model_name)
            result = trainer.evaluate(test_loader)
            save_errors(result, model_name, domain, "s1", target, 0, seed, cfg)
            metrics = compute_metrics(result["preds"], result["labels"])
            metrics["actual_n"] = 0
            log.info("✓ %s | macro_f1=%.4f | acc=%.4f", target, metrics["macro_f1"], metrics["accuracy"])
            save_metrics(metrics, model_name, domain, "s1", target, 0, seed, results_dir)
            
    else: # mt5
        from src.data.dataset import build_gen_loaders
        from src.models.mt5 import build_mt5_model
        from src.training.gen_trainer import GenerativeTrainer
        from src.evaluation.metrics import compute_metrics
        
        gen_cfg = {**cfg, "lr": cfg.get("mt5_lr", 3e-4), "use_amp": False}
        model, tokenizer = build_mt5_model(gen_cfg)
        train_loader, val_loader, _ = build_gen_loaders(train_samples, val_samples, data["test_target"], tokenizer, gen_cfg)
        
        trainer = GenerativeTrainer(model, train_loader, val_loader, train_samples, gen_cfg, domain, "s1", "en")
        history = trainer.train(epochs=gen_cfg["epochs"])
        save_history_plot(history, "mt5", domain, "s1", "ALL", 0, seed, Path(cfg.get("plots_dir", "outputs/plots")))
        
        best_ckpt_path = Path(trainer.out_dir) / "best.pt"
        if best_ckpt_path.exists():
            load_checkpoint(model, optimizer=None, filepath=best_ckpt_path, device=get_device())
            
        for target in targets:
            test_samples = load_split(domain, target, "test", cfg["processed_dir"])
            _, _, test_loader = build_gen_loaders(train_samples, val_samples, test_samples, tokenizer, gen_cfg)
            result = trainer.evaluate(test_loader)
            save_errors(result, "mt5", domain, "s1", target, 0, seed, cfg)
            metrics = compute_metrics(result["preds"], result["labels"])
            metrics["actual_n"] = 0
            log.info("✓ %s | macro_f1=%.4f | acc=%.4f", target, metrics["macro_f1"], metrics["accuracy"])
            save_metrics(metrics, "mt5", domain, "s1", target, 0, seed, results_dir)

def run_s1(models, domains, targets, cfg, results_dir):
    """Execute S1 (Zero-shot benchmark). Trained once on English (`EN`), evaluated across all target languages."""
    for model, domain in product(models, domains):
        all_exist = all(s1_result_exists(results_dir, model, domain, t) for t in targets)
        if all_exist and not cfg.get("force", False):
            log.info("SKIP S1 (%s, %s) — results already exist for all targets.", model, domain)
            continue
        run_s1_single(model, domain, targets, seed=42, cfg=cfg, results_dir=results_dir)


def run_s2(models, domains, targets, n_values, seeds, cfg, results_dir):
    """
    Execute S2 (Few-shot transfer). Each (`model`, `domain`, `target`, `n`, `seed`) partition is an independent run.

    Sample budgets are non-accumulating (`N=50` and `N=100` are independently sampled experiments).
    Requires completed S1 checkpoint.
    """
    for model, domain, target in product(models, domains, targets):
        require_s1(results_dir, model, domain, target)

        for n, seed in product(n_values, seeds):
            out_path = results_dir / "s2" / f"{model}_{domain}_s2_{target}_{n}_{seed}.json"
            if out_path.exists() and not cfg.get("force", False):
                log.info("SKIP S2 (%s, %s, %s, n=%s, seed=%s) — result already exists.", model, domain, target, n, seed)
                continue
            metrics = run_one(model, domain, "s2", target, n=n, seed=seed, cfg=cfg)
            save_metrics(metrics, model, domain, "s2", target, n, seed, results_dir)


def run_s3(models, domains, targets, seeds, cfg, results_dir):
    """
    Execute S3 (Full-target transfer). Fine-tunes model on English plus complete target-language training split.
    Requires completed S1 checkpoint.
    """
    for model, domain, target in product(models, domains, targets):
        require_s1(results_dir, model, domain, target)

        for seed in seeds:
            out_path = results_dir / "s3" / f"{model}_{domain}_s3_{target}_0_{seed}.json"
            if out_path.exists() and not cfg.get("force", False):
                log.info("SKIP S3 (%s, %s, %s, seed=%s) — result already exists.", model, domain, target, seed)
                continue
            metrics = run_one(model, domain, "s3", target, n=0, seed=seed, cfg=cfg)
            save_metrics(metrics, model, domain, "s3", target, 0, seed, results_dir)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-lingual ABSA — train s1 / s2 / s3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--setting", required=True, choices=["s1", "s2", "s3", "all"],
        help="s1=zero-shot | s2=few-shot | s3=full-target | all=s1->s2->s3",
    )
    parser.add_argument(
        "--models", "--model", nargs="+", default=["ag_can", "xlmr", "mt5"],
        choices=["ag_can", "xlmr", "mt5"],
        dest="models",
        help="Target model architectures to run (default: all)",
    )
    parser.add_argument(
        "--domains", "--domain", nargs="+", default=["restaurant", "phone"],
        choices=["restaurant", "phone"],
        dest="domains",
        help="Target evaluation domains (default: all)",
    )
    parser.add_argument(
        "--targets", "--target", nargs="+", default=["vi", "de", "zh"],
        dest="targets",
        help="Target languages (`vi`, `de`, `zh`) (default: all)",
    )
    parser.add_argument(
        "--n_values", "--n", nargs="+", type=int, default=[50, 100, 200],
        dest="n_values",
        help="Few-shot target sample budgets N for S2, executed as independent runs (default: 50 100 200)",
    )
    parser.add_argument(
        "--seeds", "--seed", nargs="+", type=int, default=[42, 123, 456],
        dest="seeds",
        help="Random seeds for S2 independent sampling runs across seeds (default: 42 123 456)",
    )
    parser.add_argument(
        "--config", default="config.yml",
        help="Path to YAML configuration file (default: config.yml)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force retraining and overwrite existing results/checkpoints.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    cfg["force"] = args.force

    setup_logging(cfg["log_file"], cfg["log_level"])
    log.info("Config loaded from %s", args.config)
    log.info("Setting=%s | models=%s | domains=%s | targets=%s",
             args.setting, args.models, args.domains, args.targets)

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.setting == "s1":
        run_s1(args.models, args.domains, args.targets, cfg, results_dir)

    elif args.setting == "s2":
        run_s2(args.models, args.domains, args.targets,
               args.n_values, args.seeds, cfg, results_dir)

    elif args.setting == "s3":
        run_s3(args.models, args.domains, args.targets,
               args.seeds, cfg, results_dir)

    elif args.setting == "all":
        log.info("=== Executing Full Pipeline: S1 -> S2 -> S3 ===")
        run_s1(args.models, args.domains, args.targets, cfg, results_dir)
        run_s2(args.models, args.domains, args.targets,
               args.n_values, args.seeds, cfg, results_dir)
        run_s3(args.models, args.domains, args.targets,
               args.seeds, cfg, results_dir)

    log.info("=== All Pipeline Stages Completed Successfully ===")


if __name__ == "__main__":
    main()
