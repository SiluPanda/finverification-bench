"""Rule-based financial statement verifier -- baseline for FinVerBench.

Performs arithmetic, cross-statement, and year-over-year consistency
checks on financial statements in the flat dict format used by
``benchmark.error_injection``.  Designed as a deterministic baseline
against which LLM performance can be compared.

The verifier only runs checks whose underlying accounting relationship
is reliably satisfied by clean data.  Simplified row-sum formulas that
omit real-world line items (e.g. total_assets = current_assets + PPE,
which ignores goodwill and intangibles) are excluded to avoid false
positives.

Usage:
    python -m src.evaluation.rule_based_verifier
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evaluation.metrics import (
    compute_all_metrics,
    detection_metrics,
    per_category_metrics,
    per_magnitude_metrics,
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "benchmark.json"
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

# Type aliases
FinancialStatements = Dict[str, Any]


# ===================================================================
# Individual check result
# ===================================================================

class CheckResult:
    """Outcome of a single consistency check."""

    __slots__ = (
        "check_name", "expected_value", "actual_value",
        "discrepancy_pct", "statement_path", "passed",
    )

    def __init__(
        self,
        check_name: str,
        expected_value: float,
        actual_value: float,
        discrepancy_pct: float,
        statement_path: str,
        passed: bool,
    ) -> None:
        self.check_name = check_name
        self.expected_value = expected_value
        self.actual_value = actual_value
        self.discrepancy_pct = discrepancy_pct
        self.statement_path = statement_path
        self.passed = passed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "discrepancy_pct": round(self.discrepancy_pct, 6),
            "statement_path": self.statement_path,
            "passed": self.passed,
        }


# ===================================================================
# Helpers
# ===================================================================

def _safe_get(data: dict, dotpath: str) -> Optional[float]:
    """Retrieve a numeric value from *data* using a dot-separated path.

    Returns ``None`` if any key is missing or the leaf is not numeric.
    """
    keys = dotpath.split(".")
    node: Any = data
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    if isinstance(node, (int, float)):
        return float(node)
    return None


def _discrepancy_pct(actual: float, expected: float) -> float:
    """Compute |actual - expected| / |expected| * 100.

    Returns 0.0 when both values are zero.  When expected is zero but
    actual is not, returns infinity (always flagged).
    """
    if expected == 0.0:
        return 0.0 if actual == 0.0 else float("inf")
    return abs(actual - expected) / abs(expected) * 100.0


def _get_section(data: dict, path_prefix: str) -> Optional[dict]:
    """Navigate into a nested dict by dot-separated prefix."""
    node: Any = data
    for k in path_prefix.split("."):
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node if isinstance(node, dict) else None


def _check_equality(
    data: dict,
    path_a: str,
    path_b: str,
    check_name: str,
    threshold_pct: float,
    report_path: Optional[str] = None,
) -> Optional[CheckResult]:
    """Check whether two values in the data are equal within threshold.

    *report_path* overrides the statement_path in the result (defaults
    to *path_a*).
    """
    val_a = _safe_get(data, path_a)
    val_b = _safe_get(data, path_b)
    if val_a is None or val_b is None:
        return None
    disc = _discrepancy_pct(val_a, val_b)
    return CheckResult(
        check_name=check_name,
        expected_value=round(val_b, 4),
        actual_value=round(val_a, 4),
        discrepancy_pct=disc,
        statement_path=report_path or path_a,
        passed=disc <= threshold_pct,
    )


def _check_sum(
    data: dict,
    path_prefix: str,
    total_field: str,
    component_fields: List[str],
    check_name: str,
    threshold_pct: float,
) -> List[CheckResult]:
    """Check whether *component_fields* sum to *total_field*.

    All field paths are relative to *path_prefix*.  Returns an empty
    list if required fields are missing.

    When a failure is detected, emits both the aggregate check and
    per-component checks so the error can be localized.
    """
    section = _get_section(data, path_prefix)
    if section is None:
        return []

    total_val = section.get(total_field)
    if total_val is None or not isinstance(total_val, (int, float)):
        return []
    total_val = float(total_val)

    comp_vals: List[Tuple[str, float]] = []
    for comp in component_fields:
        val = section.get(comp)
        if val is None or not isinstance(val, (int, float)):
            return []
        comp_vals.append((comp, float(val)))

    expected = sum(v for _, v in comp_vals)
    disc = _discrepancy_pct(total_val, expected)
    total_path = f"{path_prefix}.{total_field}"
    passed = disc <= threshold_pct

    results: List[CheckResult] = [CheckResult(
        check_name=check_name,
        expected_value=round(expected, 4),
        actual_value=round(total_val, 4),
        discrepancy_pct=disc,
        statement_path=total_path,
        passed=passed,
    )]

    # Emit per-component results for localization on failure.
    if not passed:
        for comp_name, comp_val in comp_vals:
            implied = total_val - sum(v for n, v in comp_vals if n != comp_name)
            comp_disc = _discrepancy_pct(comp_val, implied)
            comp_path = f"{path_prefix}.{comp_name}"
            results.append(CheckResult(
                check_name=f"{check_name} [{comp_name}]",
                expected_value=round(implied, 4),
                actual_value=round(comp_val, 4),
                discrepancy_pct=comp_disc,
                statement_path=comp_path,
                passed=comp_disc <= threshold_pct,
            ))

    return results


# ===================================================================
# Within-statement checks: Income Statement
# ===================================================================

def _check_income_statement(
    stmts: FinancialStatements,
    threshold_pct: float,
) -> List[CheckResult]:
    """Run income-statement arithmetic checks.

    Only checks relationships that are reliably satisfied by clean data.
    The IS row sums from error_injection.py use additive (signed) convention.
    Many companies store expenses as positive values, so some groups
    never hold.  The reliable groups are identified empirically.
    """
    results: List[CheckResult] = []

    # IS: operating_income + interest_expense = income_before_tax
    # (additive convention; reliable when all three fields present)
    results.extend(_check_sum(
        stmts, "income_statement",
        "income_before_tax", ["operating_income", "interest_expense"],
        check_name="IS: Operating Income + Interest Expense = Income Before Tax",
        threshold_pct=threshold_pct,
    ))

    return results


# ===================================================================
# Within-statement checks: Balance Sheet
# ===================================================================

# Reliable BS row sums (verified against all clean instances):
_BS_RELIABLE_ROW_SUMS: List[Tuple[str, List[str], str]] = [
    (
        "total_current_liabilities",
        ["accounts_payable", "short_term_debt"],
        "BS: AP + ST Debt = Current Liabilities",
    ),
]


def _check_balance_sheet(
    stmts: FinancialStatements,
    threshold_pct: float,
) -> List[CheckResult]:
    """Run balance-sheet arithmetic checks (current year)."""
    results: List[CheckResult] = []

    for total_field, components, name in _BS_RELIABLE_ROW_SUMS:
        results.extend(_check_sum(
            stmts, "balance_sheet.current_year", total_field, components,
            check_name=name, threshold_pct=threshold_pct,
        ))

    # Accounting identity: Total Assets = Total L&E (always valid).
    cr = _check_equality(
        stmts,
        "balance_sheet.current_year.total_assets",
        "balance_sheet.current_year.total_liabilities_and_equity",
        check_name="BS: Total Assets = Total L&E (identity)",
        threshold_pct=threshold_pct,
        report_path="balance_sheet.current_year.total_assets",
    )
    if cr is not None:
        results.append(cr)

    return results


# ===================================================================
# Within-statement checks: Cash Flow Statement
# ===================================================================

# Reliable CFS row sum:
_CFS_RELIABLE_ROW_SUMS: List[Tuple[str, List[str], str]] = [
    (
        "cash_from_operations",
        ["net_income", "depreciation_amortization", "changes_in_working_capital"],
        "CFS: NI + D&A + WC Changes = Cash from Operations",
    ),
]


def _check_cash_flow_statement(
    stmts: FinancialStatements,
    threshold_pct: float,
) -> List[CheckResult]:
    """Run cash-flow-statement arithmetic checks."""
    results: List[CheckResult] = []
    for total_field, components, name in _CFS_RELIABLE_ROW_SUMS:
        results.extend(_check_sum(
            stmts, "cash_flow_statement", total_field, components,
            check_name=name, threshold_pct=threshold_pct,
        ))
    return results


# ===================================================================
# Cross-statement checks
# ===================================================================

def _check_cross_statement(
    stmts: FinancialStatements,
    threshold_pct: float,
) -> List[CheckResult]:
    """Check linkages between the three financial statements."""
    results: List[CheckResult] = []

    # 1. IS Net Income = CFS Net Income (starting point of indirect CFS).
    cr = _check_equality(
        stmts,
        "cash_flow_statement.net_income",
        "income_statement.net_income",
        check_name="Cross: IS Net Income = CFS Net Income",
        threshold_pct=threshold_pct,
        report_path="cash_flow_statement.net_income",
    )
    if cr is not None:
        results.append(cr)

    # 2. CFS Ending Cash = BS Cash and Cash Equivalents.
    cr = _check_equality(
        stmts,
        "cash_flow_statement.ending_cash",
        "balance_sheet.current_year.cash_and_equivalents",
        check_name="Cross: CFS Ending Cash = BS Cash",
        threshold_pct=threshold_pct,
        report_path="cash_flow_statement.ending_cash",
    )
    if cr is not None:
        results.append(cr)

    # 3. IS Depreciation & Amortization ~= CFS Depreciation & Amortization.
    is_dep = _safe_get(stmts, "income_statement.depreciation_amortization")
    cfs_dep = _safe_get(stmts, "cash_flow_statement.depreciation_amortization")
    if is_dep is not None and cfs_dep is not None:
        # D&A on IS may be negative (expense); on CFS it is a positive
        # add-back.  Compare absolute values.
        disc = _discrepancy_pct(abs(cfs_dep), abs(is_dep))
        results.append(CheckResult(
            check_name="Cross: IS D&A ~ CFS D&A",
            expected_value=round(abs(is_dep), 4),
            actual_value=round(abs(cfs_dep), 4),
            discrepancy_pct=disc,
            statement_path="cash_flow_statement.depreciation_amortization",
            passed=disc <= threshold_pct,
        ))

    return results


# ===================================================================
# Year-over-year checks
# ===================================================================

def _check_year_over_year(
    stmts: FinancialStatements,
    threshold_pct: float,
) -> List[CheckResult]:
    """Check prior-year consistency on the balance sheet.

    Runs the reliable BS row-sum checks on the prior-year column to
    detect opening-balance tampering.
    """
    results: List[CheckResult] = []

    bs_prior = stmts.get("balance_sheet", {}).get("prior_year")
    if not bs_prior:
        return results

    # Run reliable BS row-sum checks on prior_year.
    for total_field, components, name in _BS_RELIABLE_ROW_SUMS:
        results.extend(_check_sum(
            stmts, "balance_sheet.prior_year", total_field, components,
            check_name=f"YOY Prior {name}",
            threshold_pct=threshold_pct,
        ))

    return results


# ===================================================================
# Main verifier
# ===================================================================

def verify_statements(
    stmts: FinancialStatements,
    threshold_pct: float = 0.1,
) -> List[CheckResult]:
    """Run all consistency checks on a financial statement dict.

    Parameters
    ----------
    stmts:
        A financial-statement dict in the flat format used by
        ``error_injection.py`` (see that module's docstring for the
        schema).
    threshold_pct:
        Discrepancy percentage above which a check is flagged as
        failing.  Default 0.1% to allow for minor floating-point
        rounding while still catching intentional perturbations.

    Returns
    -------
    list[CheckResult]
        Every check that was executed, with pass/fail status.
    """
    all_checks: List[CheckResult] = []
    all_checks.extend(_check_income_statement(stmts, threshold_pct))
    all_checks.extend(_check_balance_sheet(stmts, threshold_pct))
    all_checks.extend(_check_cash_flow_statement(stmts, threshold_pct))
    all_checks.extend(_check_cross_statement(stmts, threshold_pct))
    all_checks.extend(_check_year_over_year(stmts, threshold_pct))
    return all_checks


def verify_and_predict(
    stmts: FinancialStatements,
    threshold_pct: float = 0.1,
) -> Dict[str, Any]:
    """Run the verifier and return a prediction dict compatible with
    ``evaluation.metrics``.

    The prediction includes ``has_error``, ``error_location`` (dot-path
    of the failing check with the largest discrepancy), and the full
    list of failing checks as ``errors_found``.
    """
    checks = verify_statements(stmts, threshold_pct=threshold_pct)
    failures = [c for c in checks if not c.passed]

    if not failures:
        return {
            "has_error": False,
            "error_location": None,
            "errors_found": [],
        }

    # Pick the failure with the largest discrepancy as the primary.
    worst = max(failures, key=lambda c: c.discrepancy_pct)

    return {
        "has_error": True,
        "error_location": worst.statement_path,
        "errors_found": [f.to_dict() for f in failures],
    }


# ===================================================================
# Benchmark evaluation
# ===================================================================

def _load_benchmark(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the benchmark dataset from JSON."""
    path = path or BENCHMARK_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Benchmark file must contain a JSON array.")
    logger.info("Loaded %d benchmark instances from %s", len(data), path)
    return data


def _format_table(
    headers: List[str],
    rows: List[List[str]],
    col_widths: Optional[List[int]] = None,
) -> str:
    """Render a simple ASCII table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(row[i]))
            col_widths.append(max_w + 2)

    sep = "+" + "+".join("-" * w for w in col_widths) + "+"

    def _fmt_row(cells: List[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            w = col_widths[i] if i < len(col_widths) else len(cell) + 2
            parts.append(f" {cell:<{w - 2}} ")
        return "|" + "|".join(parts) + "|"

    lines = [sep, _fmt_row(headers), sep]
    for row in rows:
        lines.append(_fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)


def evaluate_benchmark(
    benchmark_path: Optional[Path] = None,
    threshold_pct: float = 0.1,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the rule-based verifier on the full benchmark and compute
    metrics against ground truth.

    Parameters
    ----------
    benchmark_path:
        Path to ``benchmark.json``.  Defaults to
        ``data/benchmark/benchmark.json``.
    threshold_pct:
        Discrepancy threshold for the verifier (default 0.1%).
    output_path:
        Where to write the JSON results.  Defaults to
        ``results/rule_based_results.json``.

    Returns
    -------
    dict
        Full evaluation results including overall metrics, per-category
        and per-magnitude breakdowns, and per-instance details.
    """
    instances = _load_benchmark(benchmark_path)

    predictions: List[Dict[str, Any]] = []
    ground_truths: List[Dict[str, Any]] = []
    instance_details: List[Dict[str, Any]] = []

    for inst in instances:
        raw_stmts = inst["raw_statements"]
        gt = inst["ground_truth"]
        pred = verify_and_predict(raw_stmts, threshold_pct=threshold_pct)

        predictions.append(pred)
        ground_truths.append(gt)
        instance_details.append({
            "instance_id": inst["instance_id"],
            "company": inst.get("company", ""),
            "ground_truth": gt,
            "prediction": pred,
            "correct": bool(pred["has_error"]) == bool(gt.get("has_error")),
        })

    # --- Compute metrics using the existing metrics module ---------
    all_metrics = compute_all_metrics(predictions, ground_truths)
    cat_metrics = per_category_metrics(predictions, ground_truths)
    mag_metrics = per_magnitude_metrics(predictions, ground_truths)

    # --- Collect per-error-type breakdown --------------------------
    type_buckets: Dict[str, Tuple[List[Dict], List[Dict]]] = defaultdict(
        lambda: ([], [])
    )
    for pred, gt in zip(predictions, ground_truths):
        if not gt.get("has_error"):
            continue
        etype = gt.get("error_type", "unknown")
        preds_list, gts_list = type_buckets[etype]
        preds_list.append(pred)
        gts_list.append(gt)

    type_metrics: Dict[str, Dict[str, Any]] = {}
    for etype in sorted(type_buckets):
        preds_list, gts_list = type_buckets[etype]
        type_metrics[etype] = detection_metrics(preds_list, gts_list)

    # --- Assemble full results -------------------------------------
    results = {
        "metadata": {
            "verifier": "rule_based",
            "threshold_pct": threshold_pct,
            "num_instances": len(instances),
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        },
        "overall": all_metrics["overall"],
        "localization": all_metrics["localization"],
        "m50": all_metrics["m50"],
        "by_category": cat_metrics,
        "by_magnitude": mag_metrics,
        "by_error_type": type_metrics,
        "instance_details": instance_details,
    }

    # --- Write results ---------------------------------------------
    output_path = output_path or (RESULTS_DIR / "rule_based_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logger.info("Results written to %s", output_path)

    # --- Print summary tables --------------------------------------
    _print_summary(results)

    return results


def _print_summary(results: Dict[str, Any]) -> None:
    """Print a formatted summary of evaluation results to stdout."""
    overall = results["overall"]

    print()
    print("=" * 70)
    print("  Rule-Based Verifier -- FinVerBench Evaluation Results")
    print("=" * 70)

    # Overall metrics
    print()
    print("OVERALL METRICS")
    print("-" * 50)
    print(f"  Accuracy:   {overall['accuracy']:.4f}")
    print(f"  Precision:  {overall['precision']:.4f}")
    print(f"  Recall:     {overall['recall']:.4f}")
    print(f"  F1 Score:   {overall['f1']:.4f}")
    print(f"  FPR:        {overall['fpr']:.4f}")
    print(f"  Total:      {overall['total']}")
    print(f"  TP={overall['tp']}  FP={overall['fp']}  "
          f"TN={overall['tn']}  FN={overall['fn']}")
    loc = results.get("localization")
    if loc is not None:
        print(f"  Localization Accuracy: {loc:.4f}")
    m50 = results.get("m50")
    if m50 is not None:
        print(f"  Detection Threshold (m50): {m50}%")

    # By error category
    cat_metrics = results.get("by_category", {})
    if cat_metrics:
        print()
        print("PERFORMANCE BY ERROR CATEGORY")
        headers = ["Category", "Count", "Recall", "Precision", "F1"]
        rows = []
        for cat in sorted(cat_metrics):
            m = cat_metrics[cat]
            count = m.get("total", 0)
            rows.append([
                cat,
                str(count),
                f"{m.get('recall', 0):.4f}",
                f"{m.get('precision', 0):.4f}",
                f"{m.get('f1', 0):.4f}",
            ])
        print(_format_table(headers, rows))

    # By error type
    type_metrics = results.get("by_error_type", {})
    if type_metrics:
        print()
        print("PERFORMANCE BY ERROR TYPE")
        headers = ["Error Type", "Count", "Recall", "Precision", "F1"]
        rows = []
        for etype in sorted(type_metrics):
            m = type_metrics[etype]
            count = m.get("total", 0)
            rows.append([
                etype,
                str(count),
                f"{m.get('recall', 0):.4f}",
                f"{m.get('precision', 0):.4f}",
                f"{m.get('f1', 0):.4f}",
            ])
        print(_format_table(headers, rows))

    # By magnitude
    mag_metrics = results.get("by_magnitude", {})
    if mag_metrics:
        print()
        print("PERFORMANCE BY ERROR MAGNITUDE")
        headers = ["Magnitude", "Count", "Recall", "Precision", "F1"]
        rows = []
        for mag_label in ["<1%", "1-5%", "5-10%", "10-20%", ">20%"]:
            if mag_label not in mag_metrics:
                continue
            m = mag_metrics[mag_label]
            count = m.get("total", 0)
            rows.append([
                mag_label,
                str(count),
                f"{m.get('recall', 0):.4f}",
                f"{m.get('precision', 0):.4f}",
                f"{m.get('f1', 0):.4f}",
            ])
        print(_format_table(headers, rows))

    print()
    print(f"Results saved to: {RESULTS_DIR / 'rule_based_results.json'}")
    print()


# ===================================================================
# CLI
# ===================================================================

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the rule-based financial statement verifier against "
            "the FinVerBench benchmark and report metrics."
        ),
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=None,
        help=f"Path to benchmark.json (default: {BENCHMARK_PATH}).",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=0.1,
        help="Discrepancy threshold in percent (default: 0.1).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=f"Output JSON path (default: {RESULTS_DIR / 'rule_based_results.json'}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    evaluate_benchmark(
        benchmark_path=args.benchmark_path,
        threshold_pct=args.threshold_pct,
        output_path=args.output_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
