"""
src/evaluation/visualization.py
───────────────────────────────
Plotting utilities for cross-lingual aspect sentiment classification evaluation analysis.

Public API:
    plot_recovery_curves        – F1 recovery curves spanning S1 (Zero-shot) -> S2 (Few-shot) -> S3 (Full-data)
    plot_gap_matrix             – Heatmap illustrating cross-lingual macro F1 performance across zero-shot evaluation pairs
    plot_error_taxonomy         – Bar chart and donut chart visualizing qualitative linguistic error taxonomy distributions
    plot_macro_f1_comparison    – Grouped bar chart comparing S1 (Zero-shot) vs S3 (Full-data) across target languages
    plot_training_history       – Single-run convergence curve displaying training loss alongside validation macro F1
    plot_perclass_f1            – Detailed per-class F1 breakdown (Positive, Negative, Neutral) under S1 Zero-shot transfer
    plot_neutral_f1_recovery    – Minority class (Neutral) F1 recovery trajectory against few-shot target sample budget N
    plot_training_curves_panel  – 3×2 grid panel displaying training dynamics across architectures and domains
"""

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.metrics import ErrorAnalyzer

# ── Shared style ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams.update({
    'font.family':       'sans-serif',
    'axes.edgecolor':    '#333333',
    'axes.linewidth':    1.2,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# Shared palette — used across all new plots so figures look consistent
_MODEL_COLORS = {"xlmr": "#2196F3", "mt5": "#FF9800", "ag_can": "#4CAF50"}
_MODEL_LABELS = {"xlmr": "XLM-R",   "mt5": "mT5-small", "ag_can": "AG-CAN"}
_LANG_LABELS  = {"de": "DE (German)", "vi": "VI (Vietnamese)", "zh": "ZH (Chinese)"}
_CLASS_COLORS = {
    "f1_positive": "#27AE60",
    "f1_negative": "#E74C3C",
    "f1_neutral":  "#8E44AD",
}
_CLASS_LABELS = {
    "f1_positive": "Positive",
    "f1_negative": "Negative",
    "f1_neutral":  "Neutral",
}


def _avg(lst: list) -> float:
    """Compute safe mean float value (returning `0.0` for empty sequences)."""
    return float(np.mean(lst)) if lst else 0.0

def plot_recovery_curves(df: pd.DataFrame, fig_dir: Path):
    """
    Plot cross-lingual F1 recovery curves as a function of target sample budget (`N`).

    Compares few-shot recovery trajectories against full-data (`S3`) upper bounds across target languages.

    Args:
        df (pd.DataFrame): Evaluation summary dataframe containing `setting`, `model`, `domain`, `target`, `n`, and `macro_f1`.
        fig_dir (Path): Output directory path for generated figures.
    """
    plot_df = df[df['setting'].isin(['s1', 's2'])].copy()
    plot_df.loc[plot_df['setting'] == 's1', 'n'] = 0
    full_df = df[df['setting'] == 's3'].copy()
    full_avg = full_df.groupby(['model', 'domain', 'target'])['macro_f1'].mean().reset_index()

    for (domain, target), sub in plot_df.groupby(['domain', 'target']):
        plt.figure(figsize=(10, 6), dpi=200)
        
        # Plot with seaborn for shaded error bands and robust markers
        ax = sns.lineplot(
            data=sub, 
            x='n', y='macro_f1', hue='model', style='model',
            markers=True, dashes=False, linewidth=2.5, markersize=10, errorbar='sd'
        )
        
        # Add full-data horizontal lines
        colors = sns.color_palette("deep")
        models = sub['model'].unique()
        model_color_map = {m: colors[i % len(colors)] for i, m in enumerate(models)}
        
        for _, row in full_avg[(full_avg['domain'] == domain) & (full_avg['target'] == target)].iterrows():
            m = row['model']
            if m in model_color_map:
                ax.axhline(y=row['macro_f1'], linestyle='--', color=model_color_map[m], alpha=0.6, linewidth=2)
                ax.text(plot_df['n'].max() * 0.85, row['macro_f1'] + 0.005,
                         f"{m} full: {row['macro_f1']:.3f}", fontsize=10, color=model_color_map[m], fontweight='bold')
        
        # Handle the 0-shot plot smoothly
        ax.set_xscale('symlog', linthresh=10)
        ax.set_xticks([0, 50, 100, 200])
        ax.set_xticklabels(['0\n(Zero-shot)', '50', '100', '200'])
        
        plt.xlabel("Number of target samples (N)", fontweight='bold')
        plt.ylabel("Macro F1 Score", fontweight='bold')
        plt.title(f"Cross-lingual Recovery: {domain.capitalize()} (EN → {target.upper()})", fontsize=16, fontweight='bold', pad=15)
        
        plt.legend(title="Models", title_fontsize='12', fontsize='11', loc='lower right', frameon=True, shadow=True)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        
        out_path = fig_dir / f"recovery_{domain}_{target}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved {out_path}")

def plot_gap_matrix(df: pd.DataFrame, fig_dir: Path):
    """
    Plot zero-shot (`S1`) cross-lingual macro F1 performance matrix as an annotated heatmap.

    Args:
        df (pd.DataFrame): Evaluation summary dataframe.
        fig_dir (Path): Output directory path for generated figures.
    """
    zero = df[df['setting'] == 's1'].copy()
    if zero.empty:
        print("No zero-shot data.")
        return
    
    # Pivot: rows = (model, domain), columns = target languages
    pivot = zero.pivot_table(index=['model', 'domain'], columns='target', values='macro_f1')
    if pivot.empty:
        print("No data to plot.")
        return
    
    plt.figure(figsize=(10, 6), dpi=200)
    ax = sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', linewidths=1.5,
                cbar_kws={'label': 'Macro F1 Score', 'shrink': 0.8},
                annot_kws={"size": 13, "weight": "bold"})
    
    plt.title("Zero-shot Cross-lingual Performance (Higher is Better)", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Model & Domain", fontweight='bold')
    plt.xlabel("Target Language", fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold', rotation=0)
    plt.tight_layout()
    
    out_path = fig_dir / "gap_matrix.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out_path}")

def plot_error_taxonomy(errors_dir: Path, fig_dir: Path, top_k: int = 10):
    """
    Generate horizontal bar chart and donut chart representing linguistic error taxonomy distributions.

    Args:
        errors_dir (Path): Root directory containing misclassified sample JSONL logs (`errors_*.jsonl`).
        fig_dir (Path): Output directory path for generated figures.
        top_k (int): Number of top error taxonomy categories to display in the bar chart. Defaults to 10.
    """
    err_files = list(errors_dir.rglob("errors_*.jsonl"))
    if not err_files:
        print("No error files.")
        return
    counts = defaultdict(int)
    total = 0
    for f in err_files:
        with open(f, 'r', encoding='utf-8') as fp:
            for line in fp:
                err = json.loads(line)
                etype = ErrorAnalyzer.tag_error(err)
                counts[etype] += 1
                total += 1
    if total == 0:
        return
    df_err = pd.DataFrame([{'type': k, 'count': v, 'pct': v/total*100} for k, v in counts.items()])
    df_err = df_err.sort_values('count', ascending=False).head(top_k)
    
    # 1. Bar Chart
    plt.figure(figsize=(12, 7), dpi=200)
    ax = sns.barplot(data=df_err, y='type', x='pct', hue='type', palette='magma', legend=False)
    
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        if width > 0:
            ax.text(width + 0.5, p.get_y() + p.get_height() / 2.,
                    f'{width:.1f}%', ha="left", va="center", fontweight='bold', fontsize=12)
            
    plt.ylabel("Error Type", fontweight='bold')
    plt.xlabel("Percentage of Errors (%)", fontweight='bold')
    plt.title(f"Error Taxonomy Analysis (Total Errors: {total})", fontsize=16, fontweight='bold', pad=15)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    out_path_bar = fig_dir / "error_taxonomy_bar.png"
    plt.savefig(out_path_bar, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out_path_bar}")
    
    # 2. Donut Chart
    plt.figure(figsize=(8, 8), dpi=200)
    colors = sns.color_palette('magma', len(df_err))
    plt.pie(df_err['pct'], labels=df_err['type'], autopct='%1.1f%%', startangle=140, 
            colors=colors, textprops={'fontsize': 11, 'weight': 'bold'}, 
            wedgeprops={'linewidth': 3, 'edgecolor': 'white'})
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title("Error Distribution", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    out_path_donut = fig_dir / "error_taxonomy_donut.png"
    plt.savefig(out_path_donut, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out_path_donut}")

def plot_macro_f1_comparison(df: pd.DataFrame, fig_dir: Path):
    """
    Plot grouped bar chart comparing zero-shot (`S1`) vs full-data (`S3`) macro F1 scores across target languages.

    Args:
        df (pd.DataFrame): Evaluation summary dataframe.
        fig_dir (Path): Output directory path for generated figures.
    """
    comp_df = df[df['setting'].isin(['s1', 's3'])].copy()
    if comp_df.empty:
        return
    
    agg_df = comp_df.groupby(['model', 'domain', 'target', 'setting'])['macro_f1'].mean().reset_index()
    agg_df['setting'] = agg_df['setting'].map({'s1': 'Zero-shot (S1)', 's3': 'Full-data (S3)'})
    
    for domain, sub in agg_df.groupby('domain'):
        g = sns.catplot(
            data=sub, kind="bar",
            x="model", y="macro_f1", hue="setting", col="target",
            palette="muted", height=5, aspect=1.2, legend_out=False
        )
        g.set_axis_labels("Models", "Macro F1 Score", fontweight='bold')
        g.set_titles("Target: {col_name}", size=14, weight='bold')
        g.despine(left=True)
        
        # Style adjustments
        for ax in g.axes.flat:
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
            
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle(f"Performance Comparison: Zero-shot vs Full-data ({domain.capitalize()})", fontsize=16, fontweight='bold')
        
        out_path = fig_dir / f"f1_comparison_{domain}.png"
        g.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved {out_path}")

def plot_training_history(history: dict, fig_dir: Path, model_name: str, setting: str):
    """
    Plot single-run convergence curves tracking training loss alongside validation macro F1 across epochs.

    Args:
        history (dict): Training history dictionary containing `train_loss` and `val_f1` lists.
        fig_dir (Path): Output directory path for generated figures.
        model_name (str): Model architecture identifier (`"xlmr"`, `"mt5"`, or `"ag_can"`).
        setting (str): Experimental setting (`"s1"`, `"s2"`, or `"s3"`).
    """
    if not history or not history.get('train_loss'):
        print(f"No history found for {model_name} {setting}")
        return
        
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)
    
    color = 'tab:blue'
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss', color=color, fontweight='bold')
    ax1.plot(epochs, history['train_loss'], color=color, marker='o', linewidth=2.5, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Validation F1 Score', color=color, fontweight='bold')  
    ax2.plot(epochs, history['val_f1'], color=color, marker='s', linewidth=2.5, label='Validation F1')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f"Training Dynamics: {model_name} ({setting})", fontsize=16, fontweight='bold', pad=15)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', frameon=True)
    
    out_path = fig_dir / f"training_curve_{model_name}_{setting}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out_path}")




def plot_perclass_f1(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Plot grouped bar chart displaying zero-shot (`S1`) per-class F1 breakdown (`positive`, `negative`, `neutral`).

    Highlights class imbalance dynamics where minority class (`neutral`) performance can degrade despite
    relatively high overall macro F1 scores under cross-lingual zero-shot transfer.

    Args:
        df (pd.DataFrame): Evaluation summary dataframe.
        fig_dir (Path): Output directory path for generated figures.
    """
    s1      = df[df["setting"] == "s1"].copy()
    models  = [m for m in ["xlmr", "mt5", "ag_can"] if m in df["model"].unique()]
    langs   = [l for l in ["de", "vi", "zh"]         if l in df["target"].unique()]
    classes = ["f1_positive", "f1_negative", "f1_neutral"]

    for domain in ("restaurant", "phone"):
        sub = s1[s1["domain"] == domain]
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, len(langs), figsize=(5 * len(langs), 5.5),
                                 sharey=True)
        if len(langs) == 1:
            axes = [axes]

        fig.suptitle(
            f"Per-class F1 – Zero-shot (S1) | Domain: {domain.capitalize()}",
            fontsize=14, fontweight="bold", y=1.01,
        )

        x       = np.arange(len(models))
        width   = 0.24
        offsets = [-width, 0, width]

        for ax, lang in zip(axes, langs):
            for cls, offset in zip(classes, offsets):
                vals = [
                    _avg(sub[(sub["model"] == m) & (sub["target"] == lang)][cls].tolist())
                    for m in models
                ]
                bars = ax.bar(
                    x + offset, vals, width,
                    color=_CLASS_COLORS[cls], alpha=0.88,
                    edgecolor="white", linewidth=0.8,
                )
                for bar, v in zip(bars, vals):
                    if v > 0.03:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.012,
                            f"{v:.2f}",
                            ha="center", va="bottom",
                            fontsize=7, fontweight="bold", color="#333",
                        )

            ax.set_title(_LANG_LABELS.get(lang, lang), fontsize=12,
                         fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [_MODEL_LABELS.get(m, m) for m in models],
                fontsize=9, rotation=8,
            )
            ax.set_ylim(0, 1.12)
            ax.set_ylabel("F1 Score" if lang == langs[0] else "", fontsize=10)
            ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
            ax.set_axisbelow(True)

        handles = [
            mpatches.Patch(color=_CLASS_COLORS[c], label=_CLASS_LABELS[c])
            for c in classes
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.06))

        plt.tight_layout()
        out = fig_dir / f"perclass_f1_{domain}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved {out}")



def plot_neutral_f1_recovery(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Plot recovery trajectory of minority class (`neutral`) F1 against few-shot target sample budget (`N`).

    Illustrates how quickly each architecture adapts to the hardest sentiment class (~3-4% representation)
    when provided with incremental target-language supervision.

    Args:
        df (pd.DataFrame): Evaluation summary dataframe.
        fig_dir (Path): Output directory path for generated figures.
    """
    models = [m for m in ["xlmr", "mt5", "ag_can"] if m in df["model"].unique()]
    langs  = [l for l in ["de", "vi", "zh"]         if l in df["target"].unique()]

    def _get(setting, n_val, model, lang):
        pts = df[
            (df["setting"] == setting) & (df["n"] == n_val) &
            (df["model"] == model)    & (df["target"] == lang)
        ]["f1_neutral"].tolist()
        return _avg(pts) if pts else None

    fig, axes = plt.subplots(1, len(langs), figsize=(5 * len(langs), 5),
                             sharey=True)
    if len(langs) == 1:
        axes = [axes]

    fig.suptitle(
        "Neutral F1 Recovery Curve  (Hardest Class — ~3–4% of samples)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, lang in zip(axes, langs):
        for model in models:
            s1_val  = _get("s1", 0, model, lang)
            s3_val  = _get("s3", 0, model, lang)
            few_pts = [(n, _get("s2", n, model, lang)) for n in [50, 100, 200]]
            few_pts = [(n, v) for n, v in few_pts if v is not None]
            if s1_val is None:
                continue

            xs    = [0] + [n for n, _ in few_pts]
            ys    = [s1_val] + [v for _, v in few_pts]
            color = _MODEL_COLORS.get(model, "gray")

            ax.plot(xs, ys, "o-", color=color, linewidth=2.2,
                    markersize=7, label=_MODEL_LABELS.get(model, model), zorder=3)

            if s3_val is not None:
                ax.axhline(s3_val, linestyle="--", color=color,
                           alpha=0.5, linewidth=1.5)
                ax.text(202, s3_val + 0.008, "S3",
                        fontsize=7, color=color, fontweight="bold")

        ax.set_title(_LANG_LABELS.get(lang, lang), fontsize=12,
                     fontweight="bold")
        ax.set_xticks([0, 50, 100, 200])
        ax.set_xticklabels(["0\n(Zero-\nshot)", "50", "100", "200"], fontsize=8)
        ax.set_xlabel("Few-shot N", fontsize=10)
        ax.set_ylabel("Neutral F1" if lang == langs[0] else "", fontsize=10)
        ax.set_ylim(-0.02, 0.70)
        ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
        ax.set_axisbelow(True)
        if lang == langs[0]:
            ax.legend(fontsize=9, frameon=True, loc="upper left")

    plt.tight_layout()
    out = fig_dir / "neutral_f1_recovery.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")




def plot_training_curves_panel(
    plots_dir: Path,
    fig_dir:   Path,
    setting:   str = "s1",
    seed:      int = 42,
) -> None:
    """
    Generate a 3×2 grid panel showing training loss and validation macro F1 dynamics across architectures and domains.

    Loads training histories from persisted JSON artifacts produced during model training.
    Marks the optimal early stopping point (highest validation macro F1) on each subplot.

    Args:
        plots_dir (Path): Directory containing model training history JSON artifacts.
        fig_dir (Path): Output directory path for generated panel figure.
        setting (str): Transfer evaluation setting (`"s1"`, `"s2"`, or `"s3"`). Defaults to `"s1"`.
        seed (int): Random seed identifying the specific training run to visualize. Defaults to 42.
    """
    models  = ["xlmr", "mt5", "ag_can"]
    domains = ["restaurant", "phone"]

    fig, axes = plt.subplots(
        len(models), len(domains),
        figsize=(12, 10), sharex=False,
    )
    fig.suptitle(
        f"Training Convergence – {setting.upper()} Baseline (trained on English only)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for row_i, model in enumerate(models):
        for col_j, domain in enumerate(domains):
            ax    = axes[row_i][col_j]
            color = _MODEL_COLORS.get(model, "steelblue")

            pattern = str(
                plots_dir / setting
                / f"history_{model}_{domain}_{setting}_ALL_0_{seed}.json"
            )
            files = glob.glob(pattern)

            if not files:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color="gray")
                ax.set_title(
                    f"{_MODEL_LABELS.get(model, model)} | {domain.capitalize()}",
                    fontsize=10, fontweight="bold",
                )
                continue

            with open(files[0]) as f:
                history = json.load(f)

            train_loss = history.get("train_loss", [])
            val_f1     = history.get("val_f1", [])
            epochs     = list(range(1, len(train_loss) + 1))

            ax2 = ax.twinx()
            l1, = ax.plot(epochs, train_loss, "o-", color=color,
                          linewidth=2, markersize=4, alpha=0.9,
                          label="Train Loss")
            l2, = ax2.plot(epochs, val_f1, "s--", color="tomato",
                           linewidth=2, markersize=4, alpha=0.9,
                           label="Val Macro F1")

            best_ep = int(np.argmax(val_f1)) + 1
            best_f1 = float(max(val_f1))
            ax2.axvline(best_ep, color="tomato", linestyle=":",
                        alpha=0.55, linewidth=1.5)
            ax2.scatter([best_ep], [best_f1], color="tomato", s=60, zorder=5)
            ax2.annotate(
                f"Best\n{best_f1:.3f}",
                xy=(best_ep, best_f1),
                xytext=(best_ep + 0.5, best_f1 - 0.06),
                fontsize=7, color="tomato", fontweight="bold",
            )

            ax.set_title(
                f"{_MODEL_LABELS.get(model, model)} | {domain.capitalize()}",
                fontsize=10, fontweight="bold",
            )
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel("Train Loss", color=color, fontsize=9)
            ax2.set_ylabel("Val Macro F1", color="tomato", fontsize=9)
            ax.tick_params(axis="y", labelcolor=color,   labelsize=8)
            ax2.tick_params(axis="y", labelcolor="tomato", labelsize=8)
            ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
            ax.set_axisbelow(True)

            if row_i == 0 and col_j == 0:
                ax.legend(handles=[l1, l2], loc="upper right",
                          fontsize=8, frameon=True)

    plt.tight_layout()
    out = fig_dir / "training_curves_panel.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")