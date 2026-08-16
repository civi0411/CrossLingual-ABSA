#!/usr/bin/env python
"""
scripts/eval.py
───────────────
Aggregates experimental evaluation results (`JSON` runs) into summary dataframes and generates benchmark plots.

Usage:
    # Full evaluation run (aggregates JSON -> summary.csv, then renders all figures)
    python scripts/eval.py

    # Skip aggregation and directly plot from existing summary.csv
    python scripts/eval.py --skip_aggregate

    # Specify custom output paths
    python scripts/eval.py --results_dir outputs/results \
                           --plots_dir   outputs/plots   \
                           --fig_dir     outputs/figures \
                           --errors_dir  outputs/errors

Generated Artifacts (`outputs/figures/`):
    Benchmark Plots:
        recovery_<domain>_<lang>.png      Cross-lingual F1 recovery curves across S1 -> S2 -> S3
        gap_matrix.png                    Zero-shot F1 heatmap matrix
        f1_comparison_<domain>.png        Grouped bar charts comparing S1 vs S3 performance
        error_taxonomy_bar/donut.png      Linguistic error taxonomy distributions

    Deep-Dive Diagnostic Plots:
        perclass_f1_<domain>.png          Zero-shot per-class F1 breakdown (Positive, Negative, Neutral)
        neutral_f1_recovery.png           Minority class (Neutral) F1 recovery curve vs few-shot budget N
        training_curves_panel.png         Convergence dynamics grid panel (loss + validation macro F1)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.visualization import (
    plot_error_taxonomy,
    plot_gap_matrix,
    plot_macro_f1_comparison,
    plot_neutral_f1_recovery,
    plot_perclass_f1,
    plot_recovery_curves,
    plot_training_curves_panel,
    plot_training_history,
)


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate JSON results → DataFrame + summary.csv
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_results(results_dir: Path, fig_dir: Path) -> pd.DataFrame:
    """
    Recursively scan `results_dir` for experimental run metrics (`*.json`).

    File naming convention: `{model}_{domain}_{setting}_{target}_{n}_{seed}.json`

    Returns:
        pd.DataFrame: Aggregated results dataframe with one row per `(model, domain, setting, target, n, seed)`.
    """
    records = []
    for f in sorted(results_dir.rglob("*.json")):
        parts = f.stem.split("_")
        if len(parts) < 6:
            continue

        seed_str = parts[-1]
        n_str    = parts[-2]
        target   = parts[-3]
        setting  = parts[-4]
        domain   = parts[-5]
        model    = "_".join(parts[:-5])
        n        = int(n_str) if setting == "s2" else 0

        with open(f) as fp:
            metrics = json.load(fp)

        records.append({
            "model":       model,
            "domain":      domain,
            "setting":     setting,
            "target":      target,
            "n":           n,
            "samples":     n,
            "seed":        int(seed_str),
            "macro_f1":    metrics.get("macro_f1",    0.0),
            "accuracy":    metrics.get("accuracy",    0.0),
            "f1_positive": metrics.get("f1_positive", 0.0),
            "f1_negative": metrics.get("f1_negative", 0.0),
            "f1_neutral":  metrics.get("f1_neutral",  0.0),
        })

        # Plot individual training curve if history is embedded in result JSON
        for key in ("history", None):
            if key and key in metrics:
                plot_training_history(
                    metrics[key], fig_dir, model,
                    f"{setting}_{domain}_{target}_seed{seed_str}",
                )
                break
            if key is None and "train_loss" in metrics and "val_f1" in metrics:
                plot_training_history(
                    {"train_loss": metrics["train_loss"], "val_f1": metrics["val_f1"]},
                    fig_dir, model,
                    f"{setting}_{domain}_{target}_seed{seed_str}",
                )

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# Statistical significance summary
# ══════════════════════════════════════════════════════════════════════════════

def _generate_metric_summary(df: pd.DataFrame, metric: str, out_name: str, results_dir: Path) -> None:
    records = []
    for (domain, setting, target, n), group in df.groupby(
            ["domain", "setting", "target", "n"]):
        ag_scores = group[group["model"] == "ag_can"][metric].values

        for model in group["model"].unique():
            m_scores  = group[group["model"] == model][metric].values
            mean_val  = float(np.mean(m_scores)) if len(m_scores) else 0.0
            std_val   = float(np.std(m_scores))  if len(m_scores) else 0.0
            score_str = f"{mean_val*100:.1f} ± {std_val*100:.1f}"

            p_value = 1.0
            if model != "ag_can" and len(ag_scores) > 1 and len(m_scores) > 1:
                _, p_value = stats.ttest_ind(ag_scores, m_scores, equal_var=False)

            records.append({
                "domain": domain, "setting": setting,
                "target": target, "n": n,
                "model": model,
                "mean_val": mean_val * 100, "std_val": std_val * 100,
                "formatted_score": score_str,
                "p_value_vs_agcan": p_value,
            })

    stats_df = pd.DataFrame(records)

    # Mark AG-CAN with * where it is significantly better than all baselines
    for (domain, setting, target, n), g in stats_df.groupby(
            ["domain", "setting", "target", "n"]):
        baselines  = g[g["model"] != "ag_can"]
        ag_can_row = g[g["model"] == "ag_can"]
        if not baselines.empty and not ag_can_row.empty:
            if baselines["p_value_vs_agcan"].max() < 0.05:
                idx = ag_can_row.index[0]
                stats_df.at[idx, "formatted_score"] += "*"

    if not stats_df.empty:
        pivot = stats_df.pivot_table(
            index=["domain", "setting", "target", "n"],
            columns="model",
            values="formatted_score",
            aggfunc="first",
        )
        out = results_dir / out_name
        pivot.to_csv(out)
        print(f"  {metric} summary → {out}")


def generate_statistical_summary(df: pd.DataFrame, results_dir: Path) -> None:
    """
    Compute mean and standard deviation (`mean ± std`) across seeds for each `(domain, setting, target, n)` partition.

    Performs Welch's independent t-test comparing each baseline against `AG-CAN` and flags statistically
    significant improvements (`p < 0.05`) with an asterisk (`*`). Persists formatted benchmark tables as CSVs.
    """
    _generate_metric_summary(df, "macro_f1", "benchmark_macro_f1.csv", results_dir)
    _generate_metric_summary(df, "accuracy", "benchmark_accuracy.csv", results_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir",    default="outputs/results",
                        help="Directory with JSON result files (default: outputs/results)")
    parser.add_argument("--plots_dir",      default="outputs/plots",
                        help="Directory with training history JSON/PNG (default: outputs/plots)")
    parser.add_argument("--fig_dir",        default="outputs/figures",
                        help="Output directory for figures (default: outputs/figures)")
    parser.add_argument("--errors_dir",     default="outputs/errors",
                        help="Directory with error JSONL files (default: outputs/errors)")
    parser.add_argument("--skip_aggregate", action="store_true",
                        help="Skip JSON aggregation and use existing summary.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir   = Path(args.plots_dir)
    fig_dir     = Path(args.fig_dir)
    errors_dir  = Path(args.errors_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load / build DataFrame ──────────────────────────────────────────
    summary_path = results_dir / "summary.csv"
    if args.skip_aggregate and summary_path.exists():
        print(f"Loading {summary_path} …")
        df = pd.read_csv(summary_path)
    else:
        print("Aggregating JSON result files …")
        df = aggregate_results(results_dir, fig_dir)
        if df.empty:
            print(f"No result files found in {results_dir}. Run training first.")
            return
        df.to_csv(summary_path, index=False)
        print(f"  {len(df)} records → {summary_path}")

    # ── 2. Existing figures ────────────────────────────────────────────────
    print("\n[1/7] Recovery curves (F1 vs N) …")
    plot_recovery_curves(df, fig_dir)

    print("[2/7] Gap matrix (zero-shot F1 heatmap) …")
    plot_gap_matrix(df, fig_dir)

    print("[3/7] Error taxonomy bar + donut …")
    plot_error_taxonomy(errors_dir, fig_dir)

    print("[4/7] F1 comparison: S1 vs S3 …")
    plot_macro_f1_comparison(df, fig_dir)

    # ── 3. New figures ─────────────────────────────────────────────────────
    print("[5/7] Per-class F1 grouped bar (Pos/Neg/Neu) …")
    plot_perclass_f1(df, fig_dir)

    print("[6/7] Neutral F1 recovery curve …")
    plot_neutral_f1_recovery(df, fig_dir)

    print("[7/7] Training convergence panel (history JSON) …")
    plot_training_curves_panel(plots_dir, fig_dir)

    # ── 4. Stats summary ───────────────────────────────────────────────────
    print("\n[+] Statistical significance summary …")
    generate_statistical_summary(df, results_dir)

    print(f"\nDone. All figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
