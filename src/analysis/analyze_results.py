"""Analyze FinVerBench evaluation results and generate summary tables.

Loads result files from ``results/``, computes all metrics defined in
``evaluation.metrics``, and prints formatted summary tables suitable for
copy-pasting into a LaTeX document.

Usage:
    python -m src.analysis.analyze_results
    python -m src.analysis.analyze_results --model claude-sonnet-4-20250514
    python -m src.analysis.analyze_results --strategy cot --error-type AE
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evaluation.metrics import (
    compute_all_metrics,
    detection_metrics,
    detection_threshold_m50,
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Category display order.
CATEGORY_ORDER = ["AE", "CL", "YOY", "MR"]
MAGNITUDE_ORDER = ["<1%", "1-5%", "5-10%", "10-20%", ">20%"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_result_file(filepath: Path) -> Dict[str, Any]:
    """Load a single result JSON file."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_all_results(
    results_dir: Optional[Path] = None,
    model_filter: Optional[str] = None,
    strategy_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load all result files, optionally filtered by model or strategy.

    Parameters
    ----------
    results_dir:
        Directory containing result JSON files.
    model_filter:
        If set, only load files whose model name contains this substring.
    strategy_filter:
        If set, only load files whose prompt strategy matches.

    Returns
    -------
    list[dict]
        Each dict is the full parsed result file.
    """
    results_dir = results_dir or RESULTS_DIR
    if not results_dir.is_dir():
        logger.warning("Results directory not found: %s", results_dir)
        return []

    loaded: List[Dict[str, Any]] = []
    for fp in sorted(results_dir.glob("*.json")):
        data = load_result_file(fp)
        meta = data.get("metadata", {})

        if model_filter and model_filter.lower() not in meta.get("model", "").lower():
            continue
        if strategy_filter and meta.get("prompt_strategy") != strategy_filter:
            continue

        data["_source_file"] = str(fp)
        loaded.append(data)

    logger.info("Loaded %d result file(s) from %s", len(loaded), results_dir)
    return loaded


# ---------------------------------------------------------------------------
# Prediction / ground-truth extraction
# ---------------------------------------------------------------------------

def _extract_predictions_and_gts(
    result_file: Dict[str, Any],
    error_type_filter: Optional[str] = None,
    magnitude_filter: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract parallel lists of (prediction, ground_truth) from a result file.

    Uses the majority-vote prediction across runs.

    Parameters
    ----------
    error_type_filter:
        If set, only include instances with this error category (e.g. "AE").
    magnitude_filter:
        If set, only include instances in this magnitude bin (e.g. "<1%").
    """
    predictions: List[Dict[str, Any]] = []
    ground_truths: List[Dict[str, Any]] = []

    for entry in result_file.get("results", []):
        gt = entry.get("ground_truth", {})

        # Apply error-type filter.
        if error_type_filter:
            cat = gt.get("error_category")
            if cat is None:
                errors = gt.get("errors", [])
                cat = errors[0].get("error_category") if errors else None
            if gt.get("has_error") and cat != error_type_filter:
                continue

        # Apply magnitude filter (only for error instances).
        if magnitude_filter and gt.get("has_error"):
            mag = gt.get("error_magnitude_pct")
            if mag is None:
                errors = gt.get("errors", [])
                mags = [abs(e.get("error_magnitude_pct", 0) or 0) for e in errors]
                mag = max(mags) if mags else 0.0
            from evaluation.metrics import _magnitude_bin
            if _magnitude_bin(mag) != magnitude_filter:
                continue

        pred = entry.get("majority_prediction", {})
        predictions.append(pred)
        ground_truths.append(gt)

    return predictions, ground_truths


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def _pct(value: float) -> str:
    """Format a float as a percentage string (e.g. 0.85 -> '85.0')."""
    return f"{value * 100:.1f}"


def generate_overall_table(
    result_files: List[Dict[str, Any]],
) -> str:
    """Generate the overall detection performance table (LaTeX tabular).

    Corresponds to Table 3 in the paper: Model, Prompt, Acc, F1, FPR, Loc. Acc.
    """
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Prompt} & \textbf{Acc.} & \textbf{F1} & \textbf{FPR} & \textbf{Loc. Acc.} \\",
        r"\midrule",
    ]

    for rf in result_files:
        meta = rf.get("metadata", {})
        model = meta.get("model", "?")
        strategy = meta.get("prompt_strategy", "?")

        preds, gts = _extract_predictions_and_gts(rf)
        if not preds:
            continue

        dm = detection_metrics(preds, gts)
        loc_acc = localization_accuracy(preds, gts)

        # Shorten model name for display.
        display_model = model.split("/")[-1]
        display_strategy = {
            "zero_shot": "Zero-shot",
            "few_shot": "Few-shot",
            "cot": "CoT",
        }.get(strategy, strategy)

        lines.append(
            f"{display_model} & {display_strategy} "
            f"& {_pct(dm['accuracy'])} & {_pct(dm['f1'])} "
            f"& {_pct(dm['fpr'])} & {_pct(loc_acc)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def generate_by_category_table(
    result_files: List[Dict[str, Any]],
) -> str:
    """Generate detection rate by error category table (LaTeX tabular).

    Corresponds to Table 4 in the paper: Model, AE, CL, YoY, MR.
    """
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{AE} & \textbf{CL} & \textbf{YoY} & \textbf{MR} \\",
        r"\midrule",
    ]

    for rf in result_files:
        meta = rf.get("metadata", {})
        model = meta.get("model", "?")
        strategy = meta.get("prompt_strategy", "?")

        preds, gts = _extract_predictions_and_gts(rf)
        if not preds:
            continue

        cat_rates = per_category_detection_rates(preds, gts)

        display_model = model.split("/")[-1]
        display_strategy = {
            "zero_shot": "0s", "few_shot": "fs", "cot": "CoT",
        }.get(strategy, strategy)

        cells = []
        for cat in CATEGORY_ORDER:
            rate = cat_rates.get(cat)
            cells.append(_pct(rate) if rate is not None else "--")

        lines.append(
            f"{display_model} ({display_strategy}) & "
            + " & ".join(cells) + " \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def generate_by_magnitude_table(
    result_files: List[Dict[str, Any]],
) -> str:
    """Generate detection rate by error magnitude table (LaTeX tabular)."""
    headers = " & ".join(
        [r"\textbf{Model}"] + [f"\\textbf{{{m}}}" for m in MAGNITUDE_ORDER]
    )
    lines = [
        r"\begin{tabular}{l" + "c" * len(MAGNITUDE_ORDER) + "}",
        r"\toprule",
        headers + r" \\",
        r"\midrule",
    ]

    for rf in result_files:
        meta = rf.get("metadata", {})
        model = meta.get("model", "?")
        strategy = meta.get("prompt_strategy", "?")

        preds, gts = _extract_predictions_and_gts(rf)
        if not preds:
            continue

        mag_rates = per_magnitude_detection_rates(preds, gts)

        display_model = model.split("/")[-1]
        display_strategy = {
            "zero_shot": "0s", "few_shot": "fs", "cot": "CoT",
        }.get(strategy, strategy)

        cells = []
        for m in MAGNITUDE_ORDER:
            rate = mag_rates.get(m)
            cells.append(_pct(rate) if rate is not None else "--")

        lines.append(
            f"{display_model} ({display_strategy}) & "
            + " & ".join(cells) + " \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def generate_m50_table(
    result_files: List[Dict[str, Any]],
) -> str:
    """Generate detection threshold m_50 table (LaTeX tabular)."""
    lines = [
        r"\begin{tabular}{llc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Prompt} & \textbf{$m_{50}$ (\%)} \\",
        r"\midrule",
    ]

    for rf in result_files:
        meta = rf.get("metadata", {})
        model = meta.get("model", "?")
        strategy = meta.get("prompt_strategy", "?")

        preds, gts = _extract_predictions_and_gts(rf)
        if not preds:
            continue

        m50 = detection_threshold_m50(preds, gts)

        display_model = model.split("/")[-1]
        display_strategy = {
            "zero_shot": "Zero-shot", "few_shot": "Few-shot", "cot": "CoT",
        }.get(strategy, strategy)

        m50_str = f"{m50:.1f}" if m50 is not None else r"$> 20$"
        lines.append(
            f"{display_model} & {display_strategy} & {m50_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_summary(
    result_files: List[Dict[str, Any]],
    error_type_filter: Optional[str] = None,
    magnitude_filter: Optional[str] = None,
) -> None:
    """Print a human-readable summary of all results to stdout."""
    print("=" * 72)
    print("FinVerBench Results Summary")
    print("=" * 72)

    for rf in result_files:
        meta = rf.get("metadata", {})
        model = meta.get("model", "?")
        strategy = meta.get("prompt_strategy", "?")

        preds, gts = _extract_predictions_and_gts(
            rf,
            error_type_filter=error_type_filter,
            magnitude_filter=magnitude_filter,
        )
        if not preds:
            print(f"\n  {model} / {strategy}: no matching instances")
            continue

        all_metrics = compute_all_metrics(preds, gts)

        print(f"\n--- {model} | {strategy} ---")
        print(f"  Instances evaluated: {len(preds)}")

        overall = all_metrics["overall"]
        print(f"  Accuracy:    {_pct(overall['accuracy'])}%")
        print(f"  Precision:   {_pct(overall['precision'])}%")
        print(f"  Recall:      {_pct(overall['recall'])}%")
        print(f"  F1:          {_pct(overall['f1'])}%")
        print(f"  FPR:         {_pct(overall['fpr'])}%")
        print(f"  Loc. Acc.:   {_pct(all_metrics['localization'])}%")
        print(f"  m_50:        {all_metrics['m50']}")

        print("  Detection by category:")
        for cat, cat_m in all_metrics.get("by_category", {}).items():
            print(f"    {cat}: {_pct(cat_m.get('detection_rate', 0.0))}%"
                  f" (n={cat_m.get('total', 0)})")

        print("  Detection by magnitude:")
        for mag, mag_m in all_metrics.get("by_magnitude", {}).items():
            print(f"    {mag}: {_pct(mag_m.get('detection_rate', 0.0))}%"
                  f" (n={mag_m.get('total', 0)})")

    print()
    print("=" * 72)
    print("LaTeX Tables")
    print("=" * 72)

    print("\n% --- Table: Overall Performance ---")
    print(generate_overall_table(result_files))

    print("\n% --- Table: Detection by Category ---")
    print(generate_by_category_table(result_files))

    print("\n% --- Table: Detection by Magnitude ---")
    print(generate_by_magnitude_table(result_files))

    print("\n% --- Table: Detection Threshold m_50 ---")
    print(generate_m50_table(result_files))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze FinVerBench results and generate summary tables.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=f"Directory containing result JSON files (default: {RESULTS_DIR}).",
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
    parser.add_argument(
        "--error-type",
        type=str,
        choices=["AE", "CL", "YOY", "MR"],
        default=None,
        help="Filter to a specific error category.",
    )
    parser.add_argument(
        "--magnitude",
        type=str,
        choices=MAGNITUDE_ORDER,
        default=None,
        help="Filter to a specific magnitude bin.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    result_files = load_all_results(
        results_dir=args.results_dir,
        model_filter=args.model,
        strategy_filter=args.strategy,
    )
    if not result_files:
        logger.error("No result files found.")
        return 1

    print_summary(
        result_files,
        error_type_filter=args.error_type,
        magnitude_filter=args.magnitude,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
