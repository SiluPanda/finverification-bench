"""Generate publication-quality figures for FinVerBench results.

Produces four figures:
  1. Detection rate vs. error magnitude (sigmoid-like psychometric curves)
  2. Detection rate by error category (grouped bar chart)
  3. Confusion heatmap of detection by error type x magnitude
  4. False positive rate comparison across models

All figures are saved as PDF to ``paper/figures/``.

Usage:
    python -m src.analysis.plot_results
    python -m src.analysis.plot_results --results-dir results/ --strategy cot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PDF generation
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from evaluation.metrics import (
    detection_metrics,
    false_positive_rate,
    localization_accuracy,
    per_category_detection_rates,
    per_magnitude_detection_rates,
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Academic publication style
# ---------------------------------------------------------------------------

# Color palette: colorblind-safe, print-friendly.
MODEL_COLORS = {
    "claude": "#1b9e77",
    "gpt-4": "#d95f02",
    "gpt-4o": "#7570b3",
    "llama": "#e7298a",
    "mistral": "#66a61e",
    "gemini": "#e6ab02",
}

CATEGORY_COLORS = {
    "AE": "#4e79a7",
    "CL": "#f28e2b",
    "YOY": "#e15759",
    "MR": "#76b7b2",
}

CATEGORY_ORDER = ["AE", "CL", "YOY", "MR"]
CATEGORY_LABELS = {
    "AE": "Arithmetic\n(AE)",
    "CL": "Cross-Stmt\n(CL)",
    "YOY": "Year-over-Year\n(YoY)",
    "MR": "Magnitude\n(MR)",
}

MAGNITUDE_ORDER = ["<1%", "1-5%", "5-10%", "10-20%", ">20%"]
# Numeric midpoints for plotting continuous magnitude curves.
MAGNITUDE_MIDPOINTS = {
    "<1%": 0.5,
    "1-5%": 3.0,
    "5-10%": 7.5,
    "10-20%": 15.0,
    ">20%": 25.0,
}


def _apply_publication_style() -> None:
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    })


def _get_model_color(model_name: str) -> str:
    """Return a color for the given model name."""
    model_lower = model_name.lower()
    for key, color in MODEL_COLORS.items():
        if key in model_lower:
            return color
    # Fallback: cycle through a default palette.
    fallback = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    idx = hash(model_name) % len(fallback)
    return fallback[idx]


def _short_model_name(model: str) -> str:
    """Shorten a model identifier for display."""
    name = model.split("/")[-1]
    # Truncate very long names.
    if len(name) > 25:
        name = name[:22] + "..."
    return name


# ---------------------------------------------------------------------------
# Data loading (mirrors analyze_results.py)
# ---------------------------------------------------------------------------

def _load_results(
    results_dir: Path,
    model_filter: Optional[str] = None,
    strategy_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load result JSON files from directory."""
    if not results_dir.is_dir():
        logger.warning("Results directory not found: %s", results_dir)
        return []

    loaded: List[Dict[str, Any]] = []
    for fp in sorted(results_dir.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        meta = data.get("metadata", {})

        if model_filter and model_filter.lower() not in meta.get("model", "").lower():
            continue
        if strategy_filter and meta.get("prompt_strategy") != strategy_filter:
            continue

        loaded.append(data)

    logger.info("Loaded %d result file(s)", len(loaded))
    return loaded


def _extract_preds_gts(
    result_file: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract (predictions, ground_truths) lists from a result file."""
    preds: List[Dict[str, Any]] = []
    gts: List[Dict[str, Any]] = []
    for entry in result_file.get("results", []):
        preds.append(entry.get("majority_prediction", {}))
        gts.append(entry.get("ground_truth", {}))
    return preds, gts


# ---------------------------------------------------------------------------
# Figure 1: Detection rate vs. error magnitude (psychometric curves)
# ---------------------------------------------------------------------------

def plot_magnitude_curves(
    result_files: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot detection rate vs. error magnitude for each model.

    Creates sigmoid-like psychometric curves with a dashed 50% line and a
    shaded materiality zone (0.5-5%).
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Shaded materiality zone.
    ax.axhspan(0, 100, xmin=0, xmax=1, alpha=0.0)  # placeholder
    ax.axvspan(0.5, 5.0, alpha=0.08, color="gray", label="Materiality zone")

    # 50% detection line.
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    for rf in result_files:
        meta = rf.get("metadata", {})
        model = meta.get("model", "?")
        strategy = meta.get("prompt_strategy", "?")
        label = f"{_short_model_name(model)} ({strategy})"

        preds, gts = _extract_preds_gts(rf)
        mag_rates = per_magnitude_detection_rates(preds, gts)

        x_vals = []
        y_vals = []
        for m in MAGNITUDE_ORDER:
            if m in mag_rates:
                x_vals.append(MAGNITUDE_MIDPOINTS[m])
                y_vals.append(mag_rates[m] * 100)

        if not x_vals:
            continue

        color = _get_model_color(model)
        ax.plot(
            x_vals, y_vals,
            marker="o", color=color, label=label,
            markeredgecolor="white", markeredgewidth=0.5,
        )

    ax.set_xlabel("Error Magnitude (%)")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Detection Rate vs. Error Magnitude")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 105)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend(loc="lower right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved Figure 1 (magnitude curves) to %s", output_path)


# ---------------------------------------------------------------------------
# Figure 2: Detection rate by error category (grouped bar chart)
# ---------------------------------------------------------------------------

def plot_category_bars(
    result_files: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot detection rate by error category as a grouped bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4))

    n_cats = len(CATEGORY_ORDER)
    n_models = len(result_files)
    if n_models == 0:
        plt.close(fig)
        return

    bar_width = 0.8 / max(n_models, 1)
    x_base = np.arange(n_cats)

    for model_idx, rf in enumerate(result_files):
        meta = rf.get("metadata", {})
        model = meta.get("model", "?")
        strategy = meta.get("prompt_strategy", "?")
        label = f"{_short_model_name(model)} ({strategy})"

        preds, gts = _extract_preds_gts(rf)
        cat_rates = per_category_detection_rates(preds, gts)

        y_vals = [cat_rates.get(c, 0.0) * 100 for c in CATEGORY_ORDER]
        offset = (model_idx - n_models / 2 + 0.5) * bar_width
        color = _get_model_color(model)

        bars = ax.bar(
            x_base + offset, y_vals, bar_width,
            label=label, color=color, edgecolor="white", linewidth=0.5,
        )
        # Value labels on top of bars.
        for bar, val in zip(bars, y_vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_xlabel("Error Category")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Detection Rate by Error Category")
    ax.set_xticks(x_base)
    ax.set_xticklabels([CATEGORY_LABELS.get(c, c) for c in CATEGORY_ORDER])
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved Figure 2 (category bars) to %s", output_path)


# ---------------------------------------------------------------------------
# Figure 3: Confusion heatmap (error type x magnitude)
# ---------------------------------------------------------------------------

def _compute_type_magnitude_matrix(
    predictions: Sequence[Dict[str, Any]],
    ground_truths: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build a (category x magnitude) detection-rate matrix.

    Returns (matrix, row_labels, col_labels) where matrix[i][j] is the
    detection rate for category i and magnitude bin j.
    """
    from evaluation.metrics import _magnitude_bin

    # Accumulate counts.
    counts: Dict[Tuple[str, str], List[bool]] = defaultdict(list)

    for pred, gt in zip(predictions, ground_truths):
        if not gt.get("has_error"):
            continue
        cat = gt.get("error_category")
        if cat is None:
            errors = gt.get("errors", [])
            cat = errors[0].get("error_category") if errors else None
        if cat is None:
            continue

        mag = gt.get("error_magnitude_pct")
        if mag is None:
            errors = gt.get("errors", [])
            mags = [abs(e.get("error_magnitude_pct", 0) or 0) for e in errors]
            mag = max(mags) if mags else 0.0
        bin_label = _magnitude_bin(mag)

        detected = bool(pred.get("has_error"))
        counts[(cat, bin_label)].append(detected)

    rows = CATEGORY_ORDER
    cols = MAGNITUDE_ORDER
    matrix = np.full((len(rows), len(cols)), np.nan)

    for i, cat in enumerate(rows):
        for j, mag in enumerate(cols):
            detections = counts.get((cat, mag))
            if detections:
                matrix[i, j] = sum(detections) / len(detections) * 100

    return matrix, rows, cols


def plot_confusion_heatmap(
    result_files: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot a heatmap of detection rate by error type x magnitude.

    If multiple result files exist, uses the first one (typically the primary
    model / strategy).
    """
    if not result_files:
        return

    # Use the first result file for the heatmap.
    rf = result_files[0]
    meta = rf.get("metadata", {})
    model = meta.get("model", "?")
    strategy = meta.get("prompt_strategy", "?")

    preds, gts = _extract_preds_gts(rf)
    matrix, row_labels, col_labels = _compute_type_magnitude_matrix(preds, gts)

    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Masked array for NaN cells.
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.RdYlGn  # red (low) to green (high)
    cmap.set_bad(color="lightgray")

    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # Axes.
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Error Magnitude")
    ax.set_ylabel("Error Category")
    ax.set_title(
        f"Detection Rate (%) — {_short_model_name(model)} ({strategy})"
    )

    # Annotate cells.
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 40 or val > 80 else "black"
                ax.text(
                    j, i, f"{val:.0f}",
                    ha="center", va="center", fontsize=9, color=text_color,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Detection Rate (%)")

    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved Figure 3 (heatmap) to %s", output_path)


# ---------------------------------------------------------------------------
# Figure 4: False positive rate comparison
# ---------------------------------------------------------------------------

def plot_fpr_comparison(
    result_files: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot false positive rate comparison across models (horizontal bar chart)."""
    if not result_files:
        return

    fig, ax = plt.subplots(figsize=(6, max(3, 0.6 * len(result_files) + 1)))

    labels: List[str] = []
    fprs: List[float] = []
    colors: List[str] = []

    for rf in result_files:
        meta = rf.get("metadata", {})
        model = meta.get("model", "?")
        strategy = meta.get("prompt_strategy", "?")
        label = f"{_short_model_name(model)} ({strategy})"

        preds, gts = _extract_preds_gts(rf)
        fpr = false_positive_rate(preds, gts) * 100

        labels.append(label)
        fprs.append(fpr)
        colors.append(_get_model_color(model))

    y_pos = np.arange(len(labels))
    bars = ax.barh(
        y_pos, fprs, height=0.5,
        color=colors, edgecolor="white", linewidth=0.5,
    )

    # Value labels.
    for bar, fpr_val in zip(bars, fprs):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{fpr_val:.1f}%", ha="left", va="center", fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("False Positive Rate (%)")
    ax.set_title("False Positive Rate on Clean Instances")
    ax.set_xlim(0, max(fprs) * 1.3 + 5 if fprs else 100)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved Figure 4 (FPR comparison) to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_figures(
    result_files: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Generate all four publication figures and return their paths."""
    _apply_publication_style()

    output_dir = output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []

    p1 = output_dir / "fig1_magnitude_curves.pdf"
    plot_magnitude_curves(result_files, p1)
    paths.append(p1)

    p2 = output_dir / "fig2_category_bars.pdf"
    plot_category_bars(result_files, p2)
    paths.append(p2)

    p3 = output_dir / "fig3_type_magnitude_heatmap.pdf"
    plot_confusion_heatmap(result_files, p3)
    paths.append(p3)

    p4 = output_dir / "fig4_fpr_comparison.pdf"
    plot_fpr_comparison(result_files, p4)
    paths.append(p4)

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures from FinVerBench results.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=f"Directory containing result JSON files (default: {RESULTS_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory for figures (default: {FIGURES_DIR}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter results to this model (substring match).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["zero_shot", "few_shot", "cot"],
        default=None,
        help="Filter results to this prompting strategy.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    result_files = _load_results(
        results_dir=args.results_dir or RESULTS_DIR,
        model_filter=args.model,
        strategy_filter=args.strategy,
    )
    if not result_files:
        logger.error("No result files found. Run evaluate_llm.py first.")
        return 1

    paths = generate_all_figures(
        result_files,
        output_dir=args.output_dir,
    )

    logger.info("Generated %d figures:", len(paths))
    for p in paths:
        logger.info("  %s", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
