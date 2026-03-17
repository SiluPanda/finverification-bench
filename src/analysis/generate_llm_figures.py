"""Generate publication-quality figures for LLM evaluation results.

Produces:
  1. Model comparison bar chart (accuracy, precision, recall, F1, FPR)
  2. Detection rate by error category heatmap
  3. Detection rate by magnitude curve
  4. Strategy comparison grouped bar chart

Usage:
    python3 src/analysis/generate_llm_figures.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Colorblind-friendly palette
COLORS = {
    "Claude Sonnet 4": "#4e79a7",
    "MiniMax M2.5": "#e15759",
    "Rule-based": "#59a14f",
}

STRATEGY_COLORS = {
    "zero_shot": "#4e79a7",
    "few_shot": "#f28e2b",
    "cot": "#e15759",
}

STRATEGY_LABELS = {
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "cot": "Chain-of-Thought",
}

CATEGORY_LABELS = {
    "AE": "Arithmetic\n(AE)",
    "CL": "Cross-Stmt\n(CL)",
    "YOY": "Year-over-Year\n(YoY)",
    "MR": "Magnitude\n(MR)",
}

MAGNITUDE_ORDER = ["<1%", "1-5%", "5-20%", ">20%"]


def _apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def load_results() -> Dict[str, Dict[str, Any]]:
    """Load all result files. Returns dict keyed by 'model_strategy'."""
    results = {}
    for fp in sorted(RESULTS_DIR.glob("*_results.json")):
        name = fp.stem  # e.g. 'claude_zero_shot_results'
        with open(fp) as fh:
            data = json.load(fh)
        # Skip files with all errors (previous failed runs)
        if data.get("results"):
            has_real = any(r.get("parse_method") != "error" for r in data["results"])
            if has_real:
                results[name] = data
    return results


def plot_model_comparison(results: Dict[str, Dict[str, Any]], output_path: Path):
    """Bar chart comparing models across metrics, best strategy per model."""
    # Find best F1 strategy per model
    best_per_model: Dict[str, Dict[str, Any]] = {}
    for key, data in results.items():
        model = data.get("model", "Unknown")
        strategy = data.get("strategy", "unknown")
        metrics = data.get("metrics", {}).get("overall", {})
        f1 = metrics.get("f1", 0)

        if model not in best_per_model or f1 > best_per_model[model]["f1"]:
            best_per_model[model] = {
                "strategy": strategy,
                "f1": f1,
                "metrics": metrics,
            }

    if not best_per_model:
        logger.warning("No results to plot for model comparison")
        return

    # Add rule-based baseline
    rule_based_path = RESULTS_DIR / "rule_based_results.json"
    if rule_based_path.exists():
        with open(rule_based_path) as fh:
            rb_data = json.load(fh)
        rb_metrics = rb_data.get("metrics", {}).get("overall", {})
        if rb_metrics:
            best_per_model["Rule-based"] = {
                "strategy": "τ=0.01%",
                "f1": rb_metrics.get("f1", 0),
                "metrics": rb_metrics,
            }

    models = list(best_per_model.keys())
    metric_names = ["accuracy", "precision", "recall", "f1", "fpr"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "FPR"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        vals = [best_per_model[model]["metrics"].get(m, 0) for m in metric_names]
        strategy = best_per_model[model]["strategy"]
        color = COLORS.get(model, f"C{i}")
        bars = ax.bar(
            x + i * width - (len(models) - 1) * width / 2,
            vals, width * 0.9,
            label=f"{model} ({strategy})",
            color=color, edgecolor="white", linewidth=0.5,
        )
        # Value labels
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison on FinVerBench (Best Strategy per Model)")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved model comparison to %s", output_path)


def plot_strategy_comparison(results: Dict[str, Dict[str, Any]], output_path: Path):
    """Grouped bar chart: models × strategies."""
    # Group by model
    model_strategies: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for key, data in results.items():
        model = data.get("model", "Unknown")
        strategy = data.get("strategy", "unknown")
        metrics = data.get("metrics", {}).get("overall", {})
        model_strategies[model][strategy] = metrics

    if not model_strategies:
        return

    fig, axes = plt.subplots(1, len(model_strategies), figsize=(6 * len(model_strategies), 5),
                              sharey=True, squeeze=False)
    axes = axes.flatten()

    strategies_order = ["zero_shot", "few_shot", "cot"]
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Acc", "Prec", "Rec", "F1"]

    for idx, (model, strats) in enumerate(sorted(model_strategies.items())):
        ax = axes[idx]
        x = np.arange(len(metric_keys))
        width = 0.25

        for j, strat in enumerate(strategies_order):
            if strat not in strats:
                continue
            vals = [strats[strat].get(m, 0) for m in metric_keys]
            bars = ax.bar(
                x + j * width - width,
                vals, width * 0.9,
                label=STRATEGY_LABELS.get(strat, strat),
                color=STRATEGY_COLORS.get(strat, f"C{j}"),
                edgecolor="white", linewidth=0.5,
            )
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_title(model)
        ax.set_ylim(0, 110)
        if idx == 0:
            ax.set_ylabel("Score (%)")
        ax.legend(fontsize=8)

    fig.suptitle("Prompting Strategy Comparison", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved strategy comparison to %s", output_path)


def plot_category_heatmap(results: Dict[str, Dict[str, Any]], output_path: Path):
    """Heatmap: models × error categories detection rates."""
    # Collect all model/strategy combos
    entries = []
    for key, data in results.items():
        model = data.get("model", "Unknown")
        strategy = data.get("strategy", "unknown")
        by_cat = data.get("metrics", {}).get("by_category", {})
        label = f"{model}\n({STRATEGY_LABELS.get(strategy, strategy)})"
        entries.append((label, by_cat))

    if not entries:
        return

    categories = ["AE", "CL", "MR", "YOY"]
    cat_labels = ["Arithmetic\n(AE)", "Cross-Stmt\n(CL)", "Magnitude\n(MR)", "Year-over-Year\n(YoY)"]

    matrix = np.zeros((len(entries), len(categories)))
    row_labels = []

    for i, (label, by_cat) in enumerate(entries):
        row_labels.append(label)
        for j, cat in enumerate(categories):
            cat_data = by_cat.get(cat, {})
            matrix[i, j] = cat_data.get("detection_rate", 0)

    fig, ax = plt.subplots(figsize=(8, max(3, len(entries) * 0.8 + 1)))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # Annotate
    for i in range(len(entries)):
        for j in range(len(categories)):
            val = matrix[i, j]
            text_color = "white" if val < 30 or val > 80 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("Detection Rate by Error Category (%)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04, shrink=0.8)
    cbar.set_label("Detection Rate (%)")

    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved category heatmap to %s", output_path)


def plot_magnitude_curves(results: Dict[str, Dict[str, Any]], output_path: Path):
    """Line plot: detection rate vs. error magnitude for each model/strategy."""
    fig, ax = plt.subplots(figsize=(8, 5))

    line_styles = {
        "zero_shot": "-",
        "few_shot": "--",
        "cot": "-.",
    }
    markers = {
        "zero_shot": "o",
        "few_shot": "s",
        "cot": "^",
    }

    for key, data in sorted(results.items()):
        model = data.get("model", "Unknown")
        strategy = data.get("strategy", "unknown")
        by_mag = data.get("metrics", {}).get("by_magnitude", {})

        rates = []
        valid_mags = []
        for mag in MAGNITUDE_ORDER:
            mag_data = by_mag.get(mag, {})
            rate = mag_data.get("detection_rate")
            if rate is not None:
                rates.append(rate)
                valid_mags.append(mag)

        if not rates:
            continue

        color = COLORS.get(model, "gray")
        ls = line_styles.get(strategy, "-")
        marker = markers.get(strategy, "o")
        label = f"{model} ({STRATEGY_LABELS.get(strategy, strategy)})"

        ax.plot(range(len(valid_mags)), rates, marker=marker, markersize=6,
                linestyle=ls, linewidth=2, color=color, label=label, alpha=0.8)

    ax.set_xticks(range(len(MAGNITUDE_ORDER)))
    ax.set_xticklabels(MAGNITUDE_ORDER)
    ax.set_xlabel("Error Magnitude")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Detection Sensitivity by Error Magnitude")
    ax.set_ylim(-5, 105)
    ax.axhline(y=50, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(len(MAGNITUDE_ORDER) - 0.5, 52, "$m_{50}$", fontsize=9, color="gray")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved magnitude curves to %s", output_path)


def plot_precision_recall_tradeoff(results: Dict[str, Dict[str, Any]], output_path: Path):
    """Scatter plot: precision vs recall for all model/strategy combos."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for key, data in results.items():
        model = data.get("model", "Unknown")
        strategy = data.get("strategy", "unknown")
        metrics = data.get("metrics", {}).get("overall", {})

        prec = metrics.get("precision", 0)
        rec = metrics.get("recall", 0)
        color = COLORS.get(model, "gray")
        marker = {"zero_shot": "o", "few_shot": "s", "cot": "^"}.get(strategy, "D")

        label = f"{model} ({STRATEGY_LABELS.get(strategy, strategy)})"
        ax.scatter(rec, prec, c=color, marker=marker, s=120, label=label,
                   edgecolors="black", linewidth=0.5, zorder=5)

    # Add rule-based baseline
    rule_based_path = RESULTS_DIR / "rule_based_results.json"
    if rule_based_path.exists():
        with open(rule_based_path) as fh:
            rb_data = json.load(fh)
        rb_overall = rb_data.get("metrics", {}).get("overall", {})
        if rb_overall:
            ax.scatter(
                rb_overall.get("recall", 0), rb_overall.get("precision", 0),
                c=COLORS.get("Rule-based", "green"), marker="*", s=200,
                label="Rule-based (τ=0.01%)", edgecolors="black", linewidth=0.5, zorder=5,
            )

    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision-Recall Trade-off")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.axvline(x=50, color="gray", linestyle=":", linewidth=0.5, alpha=0.3)
    ax.axhline(y=50, color="gray", linestyle=":", linewidth=0.5, alpha=0.3)

    # Quadrant labels
    ax.text(75, 75, "Ideal", ha="center", fontsize=9, color="gray", alpha=0.5)
    ax.text(25, 75, "Conservative", ha="center", fontsize=9, color="gray", alpha=0.5)
    ax.text(75, 25, "Aggressive", ha="center", fontsize=9, color="gray", alpha=0.5)
    ax.text(25, 25, "Poor", ha="center", fontsize=9, color="gray", alpha=0.5)

    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved precision-recall tradeoff to %s", output_path)


def main():
    _apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()
    if not results:
        logger.error("No valid results found in %s", RESULTS_DIR)
        return 1

    logger.info("Loaded %d result files", len(results))
    for key in results:
        model = results[key].get("model", "?")
        strategy = results[key].get("strategy", "?")
        n = results[key].get("metrics", {}).get("overall", {}).get("total", 0)
        logger.info("  %s: %s / %s (n=%d)", key, model, strategy, n)

    plot_model_comparison(results, FIGURES_DIR / "figure_model_comparison.pdf")
    plot_strategy_comparison(results, FIGURES_DIR / "figure_strategy_comparison.pdf")
    plot_category_heatmap(results, FIGURES_DIR / "figure_category_heatmap.pdf")
    plot_magnitude_curves(results, FIGURES_DIR / "figure_magnitude_curves.pdf")
    plot_precision_recall_tradeoff(results, FIGURES_DIR / "figure_precision_recall.pdf")

    logger.info("All LLM figures generated in %s", FIGURES_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
