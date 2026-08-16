"""
src/evaluation/metrics.py
─────────────────────────
Evaluation metrics, statistical significance testing, global evaluation storage (`Evaluator`), and qualitative error taxonomy analysis (`ErrorAnalyzer`).

Classes:
    Evaluator    : Aggregates and persists evaluation results across models × domains × languages × transfer settings.
    ErrorAnalyzer: Categorizes prediction errors by qualitative linguistic phenomena (negation, intensifiers, cultural idioms).
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score

from src.utils.common import ensure_dir

log = logging.getLogger(__name__)

LABEL_NAMES = ["positive", "negative", "neutral"]


# ── Core metrics ───────────────────────────────────────────────────────────────

def compute_metrics(preds: list[int], labels: list[int]) -> dict:
    """
    Compute macro F1, per-class F1 (`positive`, `negative`, `neutral`), and overall accuracy.

    Args:
        preds (list[int]): Predicted integer labels.
        labels (list[int]): Ground truth integer labels.

    Returns:
        dict: Dictionary containing rounded evaluation metrics and total sample count.
    """
    per_class = f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "macro_f1":    round(f1_score(labels, preds, average="macro", zero_division=0), 4),
        "f1_positive": round(per_class[0], 4),
        "f1_negative": round(per_class[1], 4),
        "f1_neutral":  round(per_class[2], 4),
        "accuracy":    round(accuracy_score(labels, preds), 4),
        "n_samples":   len(labels),
    }


def slice_by(
    preds: list[int],
    labels: list[int],
    metas: list[dict],
    key: str,
    min_samples: int = 5,
) -> dict:
    """
    Compute macro F1 sliced by unique values of `metas[key]` (e.g., `'category'` or `'lang'`).

    Args:
        preds (list[int]): Predicted integer labels.
        labels (list[int]): Ground truth integer labels.
        metas (list[dict]): Sample metadata dictionaries.
        key (str): Metadata attribute key to slice by.
        min_samples (int): Minimum sample threshold required to report a reliable macro F1. Defaults to 5.

    Returns:
        dict: Dictionary mapping slice buckets to their evaluation metrics (`macro_f1`, sample count `n`, and `low_resource` flag).
    """
    buckets = defaultdict(lambda: ([], []))
    for p, l, m in zip(preds, labels, metas):
        buckets[m[key]][0].append(p)
        buckets[m[key]][1].append(l)
    out = {}
    for val, (ps, ls) in sorted(buckets.items()):
        if len(ls) < min_samples:
            out[val] = {"macro_f1": None, "n": len(ls), "low_resource": True}
        else:
            out[val] = {
                "macro_f1":    round(f1_score(ls, ps, average="macro", zero_division=0), 4),
                "n":           len(ls),
                "low_resource": False,
            }
    return out


# ── Statistical significance ───────────────────────────────────────────────────

def significance_test(
    scores_a: list[float],
    scores_b: list[float],
    method: str = "wilcoxon",
    alpha: float = 0.05,
) -> dict:
    """
    Perform statistical significance testing (`Wilcoxon signed-rank` or paired `t-test`) comparing two models (`one-tailed`).

    Requires at least 3 paired experimental runs (seeds) across identical evaluation conditions.

    Args:
        scores_a (list[float]): Evaluation scores from Model A across multiple runs.
        scores_b (list[float]): Evaluation scores from Model B across multiple runs.
        method (str): Statistical test methodology (`"wilcoxon"` or `"t_test"`). Defaults to `"wilcoxon"`.
        alpha (float): Significance threshold level. Defaults to 0.05.

    Returns:
        dict: Test summary including test statistic, `p_value`, and boolean `significant` flag.
    """
    if method == "wilcoxon":
        stat, p = stats.wilcoxon(scores_a, scores_b, alternative="greater")
    else:
        stat, p = stats.ttest_rel(scores_a, scores_b, alternative="greater")
    return {
        "method":      method,
        "statistic":   round(float(stat), 4),
        "p_value":     round(float(p), 4),
        "significant": p < alpha,
        "alpha":       alpha,
    }


# ── Throughput profiling (optional) ───────────────────────────────────────────

def profile_throughput(model, loader, device, n_batches: int = 20) -> float:
    """
    Profile inference throughput measured in samples processed per second.

    Supports models accepting keyword arguments `input_ids` and `attention_mask`.

    Args:
        model (Any): Model instance to profile.
        loader (DataLoader): Evaluation data loader.
        device (Any): Target computational device.
        n_batches (int): Maximum number of batches to profile. Defaults to 20.

    Returns:
        float: Estimated inference throughput (`samples/sec`).
    """
    import torch
    model.eval()
    total_samples = 0
    start = time.perf_counter()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # Generic forward; adjust if model expects extra args
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            total_samples += input_ids.size(0)

    elapsed = time.perf_counter() - start
    return round(total_samples / elapsed, 1)   # samples/sec


# ── Evaluator: aggregate all results ──────────────────────────────────────────

class Evaluator:
    """
    Global evaluation storage that aggregates and structures results across models × domains × settings × languages.

    Usage Example:
        >>> ev = Evaluator(config)
        >>> ev.add("xlmr", "restaurant", "vi", "s1", preds, labels, metas)
        >>> ev.add_run_f1("xlmr", "restaurant", "vi", "s1", macro_f1)
        >>> ev.save()
    """

    def __init__(self, config: dict):
        self.config   = config
        self.results  = {}   # [model][domain][lang][setting] -> metrics dict
        self.run_f1s  = {}   # [model][domain][lang][setting] -> list of F1 per seed
        self.errors   = {}
        self.out_dir  = ensure_dir(config.get("results_dir", "outputs/results"))
        self.err_dir  = ensure_dir(config.get("errors_dir",  "outputs/errors"))
        self.min_cat  = config.get("min_test_samples_per_category", 20)

    def add(
        self,
        model:   str,
        domain:  str,
        lang:    str,
        setting: str,
        preds:   list[int],
        labels:  list[int],
        metas:   list[dict],
    ) -> None:
        """Add evaluation results for a completed experimental partition (`model`, `domain`, `lang`, `setting`)."""
        overall = compute_metrics(preds, labels)
        per_cat = slice_by(preds, labels, metas, "category",
                           min_samples=self.min_cat if lang != "en" else 5)

        # Separate implicit vs explicit samples
        implicit_idx = [i for i, m in enumerate(metas) if m.get("is_implicit", False)]
        explicit_idx = [i for i, m in enumerate(metas) if not m.get("is_implicit", False)]

        def sub_metrics(idx):
            if not idx:
                return None
            return compute_metrics([preds[i] for i in idx],
                                   [labels[i] for i in idx])

        # Store
        self.results \
            .setdefault(model, {}) \
            .setdefault(domain, {}) \
            .setdefault(lang, {})[setting] = {
                "overall":      overall,
                "per_category": per_cat,
                "implicit":     sub_metrics(implicit_idx),
                "explicit":     sub_metrics(explicit_idx),
            }

        # Store errors for later analysis
        key = (model, domain, lang, setting)
        self.errors[key] = [
            {
                "id": metas[i]["id"],
                "lang": lang,
                "domain": domain,
                "category": metas[i]["category"],
                "true": labels[i],
                "pred": preds[i],
                "is_implicit": metas[i].get("is_implicit", False)
            }
            for i in range(len(preds)) if preds[i] != labels[i]
        ]

        log.info("Added %s|%s|%s|%s → macro_f1=%.4f  errors=%d",
                 model, domain, lang, setting, overall["macro_f1"],
                 len(self.errors[key]))

    def add_run_f1(self, model: str, domain: str, lang: str, setting: str, f1: float) -> None:
        """Record single-run macro F1 scores for subsequent statistical significance verification across seeds."""
        key = (model, domain, lang, setting)
        self.run_f1s.setdefault(key, []).append(f1)

    def cross_lingual_gap(self) -> dict:
        """Compute the cross-lingual transfer gap (`EN F1 - Target F1`) across all models, domains, and settings."""
        gaps = {}
        for model, domains in self.results.items():
            for domain, langs in domains.items():
                for lang, settings in langs.items():
                    if lang == "en":
                        continue
                    for setting, data in settings.items():
                        en_data = self.results.get(model, {}) \
                                           .get(domain, {}) \
                                           .get("en", {}) \
                                           .get(setting, {})
                        en_f1 = en_data.get("overall", {}).get("macro_f1")
                        tgt_f1 = data["overall"]["macro_f1"]
                        if en_f1 is not None and tgt_f1 is not None:
                            gaps.setdefault(model, {}) \
                                .setdefault(domain, {}) \
                                .setdefault(lang, {})[setting] = round(en_f1 - tgt_f1, 4)
        return gaps

    def save(self) -> None:
        """Persist aggregated results and qualitative error records to disk (`JSON` / `JSONL`), and log summary tables."""
        # Main results JSON
        out = {
            "results":            self.results,
            "cross_lingual_gap":  self.cross_lingual_gap(),
        }
        path = self.out_dir / "results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        log.info("Results → %s", path)

        # Error files
        for (model, domain, lang, setting), err_list in self.errors.items():
            err_path = self.err_dir / f"errors_{model}_{domain}_{lang}_{setting}.jsonl"
            with open(err_path, "w", encoding="utf-8") as f:
                for e in err_list:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")

        self._print_summary()

    def _print_summary(self) -> None:
        header = f"{'Model':10} {'Domain':12} {'Lang':4} {'Setting':12} {'MacroF1':>9}"
        log.info("\n" + header + "\n" + "─" * len(header))
        for model, domains in self.results.items():
            for domain, langs in domains.items():
                for lang, settings in langs.items():
                    for setting, data in settings.items():
                        f1 = data["overall"]["macro_f1"]
                        log.info("%-10s %-12s %-4s %-12s %9.4f",
                                 model, domain, lang, setting, f1)


# ── Error Analyzer: qualitative taxonomy ──────────────────────────────────────

# Heuristic keywords for error types
ERROR_TYPES = {
    "negation":    ["không", "chưa", "chẳng", "nicht", "kein", "keine", "no", "not"],
    "intensifier": ["hơi bị", "quá", "vcl", "cực", "rất", "sehr", "extrem"],
    "cultural":    ["tàm tạm", "ổn áp", "được lắm", "hơi chặt"],
}

class ErrorAnalyzer:
    """
    Qualitative linguistic error taxonomy analyzer.

    Tags misclassified prediction records with specific linguistic phenomena keywords (negation, intensifiers, cultural idioms)
    to generate detailed failure distributions (`e.g., "35% of misclassifications involve negation structures"`).
    """

    @staticmethod
    def tag_error(error: dict) -> str:
        """Assign a heuristic linguistic error taxonomy tag (`negation`, `intensifier`, `cultural`, `implicit`, or `other`) to a misclassified record."""
        text = error.get("text", "").lower()
        for etype, markers in ERROR_TYPES.items():
            if any(m in text for m in markers):
                return etype
        if error.get("is_implicit", False):
            return "implicit"
        return "other"

    @classmethod
    def analyze(cls, errors: list[dict]) -> dict:
        """
        Analyze a list of misclassified sample dictionaries and compute error taxonomy distributions and percentages.

        Args:
            errors (list[dict]): List of misclassified sample dictionaries containing `text` and optional `is_implicit`.

        Returns:
            dict: Mapping of error taxonomy tags to their occurrence counts and percentage shares.
        """
        counts = defaultdict(int)
        for e in errors:
            counts[cls.tag_error(e)] += 1
        total = len(errors)
        return {
            k: {"count": v, "pct": round(v/total*100, 1)}
            for k, v in sorted(counts.items(), key=lambda x: -x[1])
        }