"""Threshold sensitivity analysis for the FinVerBench rule-based verifier.

Sweeps the discrepancy threshold parameter across a wide range and measures
how detection performance (accuracy, precision, recall, F1, FPR) and
per-category / per-magnitude detection rates respond.  Identifies the
optimal threshold that maximises F1, the threshold regime where FPR = 0,
and the per-relationship-type noise floor on clean data.

Outputs:
  - paper/figures/figure_threshold_sensitivity.pdf  (3-panel figure)
  - results/threshold_analysis.json                 (full numeric results)
  - LaTeX table printed to stdout

Usage:
    PYTHONPATH=src python -m src.analysis.threshold_analysis
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from evaluation.rule_based_verifier import verify_statements, verify_and_predict
from evaluation.metrics import (
    detection_metrics,
    per_category_metrics,
    per_magnitude_metrics,
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "benchmark.json"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

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
# Constants
# ---------------------------------------------------------------------------
THRESHOLDS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

CATEGORY_ORDER = ["AE", "CL", "YOY", "MR"]
CATEGORY_LABELS = {
    "AE": "Arithmetic (AE)",
    "CL": "Cross-Statement (CL)",
    "YOY": "Year-over-Year (YoY)",
    "MR": "Magnitude (MR)",
}

MAGNITUDE_ORDER = ["<1%", "1-5%", "5-10%", "10-20%", ">20%"]

# Colorblind-friendly palette (Tableau 10 / Wong).
PALETTE_CAT = {
    "AE": "#4e79a7",
    "CL": "#f28e2b",
    "YOY": "#e15759",
    "MR": "#76b7b2",
}

PALETTE_MAG = {
    "<1%": "#bab0ac",
    "1-5%": "#76b7b2",
    "5-10%": "#f28e2b",
    "10-20%": "#e15759",
    ">20%": "#4e79a7",
}

# Metric line colors for Panel A.
METRIC_COLORS = {
    "Precision": "#4e79a7",
    "Recall": "#e15759",
    "F1": "#59a14f",
}


# ---------------------------------------------------------------------------
# Publication style (matching generate_figures.py)
# ---------------------------------------------------------------------------

def _apply_publication_style() -> None:
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_benchmark() -> List[Dict[str, Any]]:
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded %d benchmark instances from %s", len(data), BENCHMARK_PATH)
    return data


# ---------------------------------------------------------------------------
# Noise-floor analysis: discrepancies on clean data
# ---------------------------------------------------------------------------

def analyze_noise_floor(instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For every clean instance, run the verifier at threshold=0 and record
    which checks fail and by how much.  This reveals the inherent noise
    floor per relationship type."""
    clean = [inst for inst in instances if not inst["ground_truth"]["has_error"]]
    logger.info("Analyzing noise floor on %d clean instances", len(clean))

    # Collect discrepancies per check name.
    check_discrepancies: Dict[str, List[float]] = defaultdict(list)
    failing_checks_by_instance: List[Dict[str, Any]] = []

    for inst in clean:
        checks = verify_statements(inst["raw_statements"], threshold_pct=0.0)
        failures = [c for c in checks if not c.passed]
        for c in checks:
            check_discrepancies[c.check_name].append(c.discrepancy_pct)

        if failures:
            failing_checks_by_instance.append({
                "instance_id": inst["instance_id"],
                "company": inst.get("company", ""),
                "num_failing": len(failures),
                "checks": [
                    {
                        "name": f.check_name,
                        "discrepancy_pct": round(f.discrepancy_pct, 6),
                        "expected": f.expected_value,
                        "actual": f.actual_value,
                    }
                    for f in failures
                ],
            })

    # Summarise per check.
    check_summary: Dict[str, Dict[str, Any]] = {}
    for name, discs in sorted(check_discrepancies.items()):
        discs_nonzero = [d for d in discs if d > 0]
        check_summary[name] = {
            "num_instances": len(discs),
            "num_nonzero": len(discs_nonzero),
            "max_discrepancy_pct": round(max(discs), 6) if discs else 0.0,
            "mean_discrepancy_pct": round(
                sum(discs) / len(discs), 6
            ) if discs else 0.0,
            "mean_nonzero_pct": round(
                sum(discs_nonzero) / len(discs_nonzero), 6
            ) if discs_nonzero else 0.0,
        }

    # Overall: the maximum discrepancy seen on any check on any clean instance
    # tells us the minimum safe threshold for 0% FPR.
    all_discs = [d for dlist in check_discrepancies.values() for d in dlist]
    max_clean_disc = max(all_discs) if all_discs else 0.0

    return {
        "num_clean_instances": len(clean),
        "max_clean_discrepancy_pct": round(max_clean_disc, 6),
        "per_check_summary": check_summary,
        "failing_instances_at_zero_threshold": failing_checks_by_instance,
    }


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(
    instances: List[Dict[str, Any]],
    thresholds: List[float],
) -> List[Dict[str, Any]]:
    """Run the rule-based verifier at each threshold and collect metrics."""
    results = []

    for thresh in thresholds:
        logger.info("  threshold = %.4f%%", thresh)
        predictions = []
        ground_truths = []
        for inst in instances:
            pred = verify_and_predict(
                inst["raw_statements"], threshold_pct=thresh
            )
            predictions.append(pred)
            ground_truths.append(inst["ground_truth"])

        overall = detection_metrics(predictions, ground_truths)
        by_cat = per_category_metrics(predictions, ground_truths)
        by_mag = per_magnitude_metrics(predictions, ground_truths)

        results.append({
            "threshold_pct": thresh,
            "overall": overall,
            "by_category": by_cat,
            "by_magnitude": by_mag,
        })

    return results


# ---------------------------------------------------------------------------
# Find optimal threshold
# ---------------------------------------------------------------------------

def find_optimal(sweep: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Identify the threshold that maximises F1 score."""
    best = max(sweep, key=lambda r: r["overall"]["f1"])
    return {
        "optimal_threshold_pct": best["threshold_pct"],
        "f1": best["overall"]["f1"],
        "precision": best["overall"]["precision"],
        "recall": best["overall"]["recall"],
        "accuracy": best["overall"]["accuracy"],
        "fpr": best["overall"]["fpr"],
    }


def find_zero_fpr_threshold(sweep: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find the tightest threshold that still has FPR = 0."""
    zero_fpr = [r for r in sweep if r["overall"]["fpr"] == 0.0]
    if not zero_fpr:
        return {"threshold_pct": None, "note": "No threshold achieves FPR=0"}
    # Tightest = smallest threshold among those with FPR=0
    best = min(zero_fpr, key=lambda r: r["threshold_pct"])
    return {
        "threshold_pct": best["threshold_pct"],
        "f1": best["overall"]["f1"],
        "recall": best["overall"]["recall"],
        "precision": best["overall"]["precision"],
        "accuracy": best["overall"]["accuracy"],
        "fpr": best["overall"]["fpr"],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_threshold_sensitivity(
    sweep: List[Dict[str, Any]],
    optimal: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate the 3-panel threshold sensitivity figure."""
    _apply_publication_style()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax_a, ax_b, ax_c = axes

    thresholds = [r["threshold_pct"] for r in sweep]

    # ---------------------------------------------------------------
    # Panel A: Precision, Recall, F1 vs. threshold
    # ---------------------------------------------------------------
    for metric_name, color in METRIC_COLORS.items():
        values = [r["overall"][metric_name.lower()] for r in sweep]
        ax_a.plot(
            thresholds, values,
            marker="o", markersize=5, linewidth=1.8,
            color=color, label=metric_name, zorder=3,
        )

    # Mark optimal threshold.
    opt_thresh = optimal["optimal_threshold_pct"]
    opt_f1 = optimal["f1"]
    ax_a.axvline(
        opt_thresh, color="#999", linestyle="--", linewidth=0.9,
        zorder=1, label=f"Optimal ($\\tau$={opt_thresh}%)",
    )
    ax_a.scatter(
        [opt_thresh], [opt_f1], marker="*", s=150, color="#59a14f",
        edgecolors="black", linewidths=0.6, zorder=5,
    )

    ax_a.set_xscale("log")
    ax_a.set_xlabel("Threshold $\\tau$ (%)")
    ax_a.set_ylabel("Score")
    ax_a.set_title("(A) Precision, Recall, F1 vs. Threshold")
    ax_a.set_ylim(-0.02, 1.05)
    ax_a.legend(loc="lower right", framealpha=0.9)
    ax_a.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_a.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_a.set_xticks(thresholds)
    ax_a.set_xticklabels(
        [str(t) if t >= 1 else str(t) for t in thresholds],
        rotation=45, ha="right", fontsize=8,
    )
    ax_a.grid(axis="y", alpha=0.3, linewidth=0.5)

    # ---------------------------------------------------------------
    # Panel B: Detection rate by error category vs. threshold
    # ---------------------------------------------------------------
    for cat in CATEGORY_ORDER:
        rates = []
        for r in sweep:
            cat_data = r["by_category"].get(cat, {})
            rates.append(cat_data.get("detection_rate", cat_data.get("recall", 0.0)))
        ax_b.plot(
            thresholds, rates,
            marker="s", markersize=5, linewidth=1.8,
            color=PALETTE_CAT[cat], label=CATEGORY_LABELS[cat],
        )

    ax_b.set_xscale("log")
    ax_b.set_xlabel("Threshold $\\tau$ (%)")
    ax_b.set_ylabel("Detection Rate")
    ax_b.set_title("(B) Detection Rate by Error Category")
    ax_b.set_ylim(-0.02, 1.05)
    ax_b.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax_b.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_b.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_b.set_xticks(thresholds)
    ax_b.set_xticklabels(
        [str(t) if t >= 1 else str(t) for t in thresholds],
        rotation=45, ha="right", fontsize=8,
    )
    ax_b.grid(axis="y", alpha=0.3, linewidth=0.5)

    # ---------------------------------------------------------------
    # Panel C: Detection rate by magnitude range vs. threshold
    # ---------------------------------------------------------------
    for mag_label in MAGNITUDE_ORDER:
        rates = []
        for r in sweep:
            mag_data = r["by_magnitude"].get(mag_label, {})
            rates.append(mag_data.get("detection_rate", mag_data.get("recall", 0.0)))
        ax_c.plot(
            thresholds, rates,
            marker="D", markersize=5, linewidth=1.8,
            color=PALETTE_MAG[mag_label], label=mag_label,
        )

    ax_c.set_xscale("log")
    ax_c.set_xlabel("Threshold $\\tau$ (%)")
    ax_c.set_ylabel("Detection Rate")
    ax_c.set_title("(C) Detection Rate by Error Magnitude")
    ax_c.set_ylim(-0.02, 1.05)
    ax_c.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax_c.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_c.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_c.set_xticks(thresholds)
    ax_c.set_xticklabels(
        [str(t) if t >= 1 else str(t) for t in thresholds],
        rotation=45, ha="right", fontsize=8,
    )
    ax_c.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved threshold sensitivity figure to %s", output_path)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def print_latex_table(sweep: List[Dict[str, Any]], optimal: Dict[str, Any]) -> None:
    """Print a LaTeX-formatted table of key threshold results."""
    print()
    print("% ===== LaTeX Table: Threshold Sensitivity Analysis =====")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Rule-based verifier performance at different "
          r"discrepancy thresholds $\tau$. "
          r"Bold indicates the optimal F1 threshold.}")
    print(r"\label{tab:threshold_sensitivity}")
    print(r"\small")
    print(r"\begin{tabular}{rcccccc}")
    print(r"\toprule")
    print(r"$\tau$ (\%) & Accuracy & Precision & Recall & F1 & FPR "
          r"& Det.\ Rate \\")
    print(r"\midrule")

    opt_thresh = optimal["optimal_threshold_pct"]

    for r in sweep:
        t = r["threshold_pct"]
        o = r["overall"]
        # Detection rate = recall on error instances only
        det_rate = o["recall"]
        is_opt = (t == opt_thresh)
        bold = r"\textbf" if is_opt else ""

        # Format the threshold for display.
        if t == int(t):
            t_str = f"{int(t)}"
        else:
            t_str = f"{t}"

        if is_opt:
            row = (
                f"  {bold}{{{t_str}}} & {bold}{{{o['accuracy']:.4f}}} "
                f"& {bold}{{{o['precision']:.4f}}} "
                f"& {bold}{{{o['recall']:.4f}}} "
                f"& {bold}{{{o['f1']:.4f}}} "
                f"& {bold}{{{o['fpr']:.4f}}} "
                f"& {bold}{{{det_rate:.4f}}} \\\\"
            )
        else:
            row = (
                f"  {t_str} & {o['accuracy']:.4f} & {o['precision']:.4f} "
                f"& {o['recall']:.4f} & {o['f1']:.4f} & {o['fpr']:.4f} "
                f"& {det_rate:.4f} \\\\"
            )
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


def print_noise_floor_table(noise_floor: Dict[str, Any]) -> None:
    """Print a LaTeX table of noise-floor discrepancies per check type."""
    print()
    print("% ===== LaTeX Table: Noise Floor per Relationship Type =====")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Noise floor: discrepancy percentages observed on "
          r"\emph{clean} (error-free) instances per relationship type. "
          r"Any threshold below the max discrepancy risks false positives.}")
    print(r"\label{tab:noise_floor}")
    print(r"\small")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Relationship Check & $N$ & "
          r"$\bar{d}$ (\%) & $\bar{d}_{>0}$ (\%) & "
          r"$d_{\max}$ (\%) \\")
    print(r"\midrule")

    summary = noise_floor["per_check_summary"]
    # Sort by max discrepancy descending.
    for name in sorted(summary, key=lambda n: summary[n]["max_discrepancy_pct"],
                       reverse=True):
        s = summary[name]
        # Only show checks that were actually run on at least one instance.
        if s["num_instances"] == 0:
            continue
        # Shorten the check name for the table.
        short = name.replace("BS: ", "").replace("IS: ", "").replace("CFS: ", "")
        short = short.replace("Cross: ", "X: ").replace("YOY Prior ", "YoY ")
        if len(short) > 50:
            short = short[:47] + "..."
        print(
            f"  {short} & {s['num_instances']} "
            f"& {s['mean_discrepancy_pct']:.4f} "
            f"& {s['mean_nonzero_pct']:.4f} "
            f"& {s['max_discrepancy_pct']:.4f} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


# ---------------------------------------------------------------------------
# Summary findings
# ---------------------------------------------------------------------------

def print_findings(
    optimal: Dict[str, Any],
    zero_fpr: Dict[str, Any],
    noise_floor: Dict[str, Any],
    sweep: List[Dict[str, Any]],
) -> None:
    """Print the key findings as a readable summary."""
    print()
    print("=" * 72)
    print("  THRESHOLD SENSITIVITY ANALYSIS -- KEY FINDINGS")
    print("=" * 72)

    print()
    print("1. OPTIMAL THRESHOLD (maximises F1)")
    print(f"   Threshold:  {optimal['optimal_threshold_pct']}%")
    print(f"   F1:         {optimal['f1']:.4f}")
    print(f"   Precision:  {optimal['precision']:.4f}")
    print(f"   Recall:     {optimal['recall']:.4f}")
    print(f"   Accuracy:   {optimal['accuracy']:.4f}")
    print(f"   FPR:        {optimal['fpr']:.4f}")

    print()
    print("2. ZERO-FPR REGIME (tightest threshold with 0% false positives)")
    if zero_fpr.get("threshold_pct") is not None:
        print(f"   Threshold:  {zero_fpr['threshold_pct']}%")
        print(f"   F1:         {zero_fpr['f1']:.4f}")
        print(f"   Recall:     {zero_fpr['recall']:.4f}")
        print(f"   Precision:  {zero_fpr['precision']:.4f}")
        print(f"   Accuracy:   {zero_fpr['accuracy']:.4f}")
    else:
        print(f"   {zero_fpr.get('note', 'N/A')}")

    print()
    print("3. NOISE FLOOR ON CLEAN DATA")
    max_disc = noise_floor["max_clean_discrepancy_pct"]
    print(f"   Max discrepancy on any clean instance: {max_disc:.4f}%")
    print(f"   -> Any threshold >= {max_disc:.4f}% guarantees FPR = 0")

    # Show which checks contribute non-zero discrepancies.
    summary = noise_floor["per_check_summary"]
    nonzero_checks = {
        name: s for name, s in summary.items() if s["num_nonzero"] > 0
    }
    if nonzero_checks:
        print(f"   Checks with non-zero discrepancy on clean data "
              f"({len(nonzero_checks)}):")
        for name in sorted(nonzero_checks,
                           key=lambda n: nonzero_checks[n]["max_discrepancy_pct"],
                           reverse=True):
            s = nonzero_checks[name]
            print(f"     {name}")
            print(f"       instances with disc > 0: {s['num_nonzero']}/{s['num_instances']}")
            print(f"       mean (non-zero): {s['mean_nonzero_pct']:.4f}%")
            print(f"       max:             {s['max_discrepancy_pct']:.4f}%")
    else:
        print("   No checks have non-zero discrepancy on clean data.")
        print("   -> All relationships hold exactly; any threshold > 0 gives FPR = 0")

    print()
    print("4. TRADE-OFF SUMMARY")
    # Compare most aggressive vs. optimal vs. most relaxed
    for r in sweep:
        t = r["threshold_pct"]
        o = r["overall"]
        print(f"   tau={t:6.2f}%  |  Acc={o['accuracy']:.4f}  "
              f"Prec={o['precision']:.4f}  Rec={o['recall']:.4f}  "
              f"F1={o['f1']:.4f}  FPR={o['fpr']:.4f}  "
              f"TP={o['tp']}  FP={o['fp']}  TN={o['tn']}  FN={o['fn']}")

    print()
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logger.info("Starting threshold sensitivity analysis")

    # Load benchmark.
    instances = _load_benchmark()

    # Step 1: Noise-floor analysis on clean data.
    logger.info("Step 1: Noise-floor analysis on clean instances")
    noise_floor = analyze_noise_floor(instances)

    # Step 2: Threshold sweep.
    logger.info("Step 2: Sweeping %d thresholds", len(THRESHOLDS))
    sweep = sweep_thresholds(instances, THRESHOLDS)

    # Step 3: Find optimal and zero-FPR thresholds.
    optimal = find_optimal(sweep)
    zero_fpr = find_zero_fpr_threshold(sweep)

    # Step 4: Generate figure.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "figure_threshold_sensitivity.pdf"
    logger.info("Step 3: Generating figure -> %s", fig_path)
    plot_threshold_sensitivity(sweep, optimal, fig_path)

    # Step 5: Print findings and LaTeX tables.
    print_findings(optimal, zero_fpr, noise_floor, sweep)
    print_latex_table(sweep, optimal)
    print_noise_floor_table(noise_floor)

    # Step 6: Save full results to JSON.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "threshold_analysis.json"

    # Convert sweep by_category/by_magnitude to serialisable form.
    serialisable_sweep = []
    for r in sweep:
        serialisable_sweep.append({
            "threshold_pct": r["threshold_pct"],
            "overall": r["overall"],
            "by_category": r["by_category"],
            "by_magnitude": r["by_magnitude"],
        })

    output = {
        "metadata": {
            "analysis": "threshold_sensitivity",
            "thresholds_tested": THRESHOLDS,
            "num_instances": len(instances),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "optimal_threshold": optimal,
        "zero_fpr_threshold": zero_fpr,
        "noise_floor": noise_floor,
        "sweep_results": serialisable_sweep,
    }

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    logger.info("Saved detailed results to %s", results_path)

    print(f"\nResults saved to: {results_path}")
    print(f"Figure saved to:  {fig_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
