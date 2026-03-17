"""Build the complete benchmark dataset for financial-statement verification.

Reads processed financial statements from ``data/processed/``, generates
clean and error-injected instances, and writes the benchmark to
``data/benchmark/``.

Output layout
-------------
::

    data/benchmark/
        benchmark.json          # full dataset (list of BenchmarkInstance dicts)
        benchmark_stats.json    # summary statistics
        instances/
            <instance_id>.json  # one file per instance (optional granularity)

Each benchmark instance is a self-contained JSON document with:
- formatted financial statements (as human-readable text)
- a verification question
- machine-readable ground truth
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmark.error_injection import (
    InjectionResult,
    MultiInjectionResult,
    inject_error,
    inject_multiple_errors,
)
from benchmark.error_taxonomy import (
    ErrorCategory,
    ErrorSubtype,
    ErrorType,
    ERROR_REGISTRY,
)

logger = logging.getLogger(__name__)

# Type alias
FinancialStatements = Dict[str, Any]

# Default error configurations: (subtype, magnitude_pct) pairs to generate.
DEFAULT_SINGLE_ERROR_CONFIGS: List[Tuple[ErrorSubtype, float]] = [
    # Arithmetic
    (ErrorSubtype.AE_ROW_SUM, 5.0),
    (ErrorSubtype.AE_ROW_SUM, 10.0),
    (ErrorSubtype.AE_COLUMN_SUM, 5.0),
    (ErrorSubtype.AE_COLUMN_SUM, 10.0),
    # Cross-statement linkage
    (ErrorSubtype.CL_NET_INCOME_TO_RE, 5.0),
    (ErrorSubtype.CL_NET_INCOME_TO_CFS, 5.0),
    (ErrorSubtype.CL_ENDING_CASH, 5.0),
    (ErrorSubtype.CL_DEPRECIATION, 10.0),
    # Year-over-year
    (ErrorSubtype.YOY_OPENING_BALANCE, 5.0),
    (ErrorSubtype.YOY_COMPUTED_CHANGE, 5.0),
    # Magnitude / rounding at various levels
    (ErrorSubtype.MR_MINOR, 0.5),
    (ErrorSubtype.MR_MODERATE, 2.0),
    (ErrorSubtype.MR_SIGNIFICANT, 10.0),
    (ErrorSubtype.MR_EXTREME, 25.0),
]

# Multi-error combos for harder variants.
DEFAULT_MULTI_ERROR_CONFIGS: List[List[Tuple[ErrorSubtype, float]]] = [
    [
        (ErrorSubtype.AE_ROW_SUM, 5.0),
        (ErrorSubtype.CL_NET_INCOME_TO_CFS, 5.0),
    ],
    [
        (ErrorSubtype.CL_ENDING_CASH, 2.0),
        (ErrorSubtype.YOY_OPENING_BALANCE, 5.0),
        (ErrorSubtype.MR_MINOR, 0.5),
    ],
]

VERIFICATION_QUESTION = (
    "Are these financial statements internally consistent? "
    "Check all arithmetic totals, cross-statement linkages, "
    "and year-over-year continuity. Report any discrepancies found."
)


# ---------------------------------------------------------------------------
# Benchmark instance
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkInstance:
    """A single benchmark evaluation instance.

    Attributes
    ----------
    instance_id : str
        Unique identifier for this instance.
    company : str
        Company name or ticker.
    period : str
        Reporting period (e.g. ``"FY2024"``).
    formatted_statements : str
        The financial statements rendered as human-readable text, as they
        would appear in a filing.
    question : str
        The verification question posed to the model.
    ground_truth : dict
        Machine-readable answer key with error details (or ``has_error=False``).
    difficulty : str
        Qualitative difficulty label derived from error type metadata.
    raw_statements : dict
        The underlying structured JSON (for programmatic consumers).
    """

    instance_id: str
    company: str
    period: str
    formatted_statements: str
    question: str
    ground_truth: Dict[str, Any]
    difficulty: str
    raw_statements: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-friendly)."""
        return asdict(self)


@dataclass
class DatasetStatistics:
    """Summary statistics for the generated benchmark dataset.

    Attributes
    ----------
    total_instances : int
    clean_instances : int
    error_instances : int
    instances_per_company : dict[str, int]
    instances_per_error_type : dict[str, int]
    instances_per_category : dict[str, int]
    instances_per_difficulty : dict[str, int]
    magnitude_distribution : dict[str, int]
    """

    total_instances: int = 0
    clean_instances: int = 0
    error_instances: int = 0
    instances_per_company: Dict[str, int] = field(default_factory=dict)
    instances_per_error_type: Dict[str, int] = field(default_factory=dict)
    instances_per_category: Dict[str, int] = field(default_factory=dict)
    instances_per_difficulty: Dict[str, int] = field(default_factory=dict)
    magnitude_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Statement formatter
# ---------------------------------------------------------------------------

def _fmt_number(value: Any) -> str:
    """Format a numeric value with comma separators."""
    if isinstance(value, (int, float)):
        if value == int(value):
            return f"{int(value):,}"
        return f"{value:,.2f}"
    return str(value)


def _fmt_line(label: str, value: Any, indent: int = 2) -> str:
    """Format a single financial line item."""
    pad = " " * indent
    formatted_val = _fmt_number(value)
    # Right-align the number in a 15-char column.
    return f"{pad}{label:<40s} {formatted_val:>15s}"


def format_income_statement(is_data: Dict[str, Any]) -> str:
    """Render the income statement as formatted text."""
    lines = ["INCOME STATEMENT", "=" * 58]
    field_order = [
        ("Revenue", "revenue"),
        ("Cost of Goods Sold", "cost_of_goods_sold"),
        ("Gross Profit", "gross_profit"),
        ("Operating Expenses", "operating_expenses"),
        ("Depreciation & Amortization", "depreciation_amortization"),
        ("Operating Income", "operating_income"),
        ("Interest Expense", "interest_expense"),
        ("Income Before Tax", "income_before_tax"),
        ("Income Tax Expense", "income_tax_expense"),
        ("Net Income", "net_income"),
    ]
    separator_after = {"gross_profit", "operating_income", "income_before_tax"}
    for label, key in field_order:
        if key in is_data:
            lines.append(_fmt_line(label, is_data[key]))
            if key in separator_after:
                lines.append("  " + "-" * 56)
    return "\n".join(lines)


def format_balance_sheet(
    bs_data: Dict[str, Any],
) -> str:
    """Render the balance sheet as formatted text (current and prior year)."""
    lines = ["BALANCE SHEET", "=" * 78]

    current = bs_data.get("current_year", {})
    prior = bs_data.get("prior_year", {})
    has_prior = bool(prior)

    # Header row
    header = f"  {'':40s} {'Current':>15s}"
    if has_prior:
        header += f"  {'Prior':>15s}"
    lines.append(header)
    lines.append("  " + "-" * (56 if not has_prior else 76))

    field_order = [
        ("Cash and Equivalents", "cash_and_equivalents"),
        ("Accounts Receivable", "accounts_receivable"),
        ("Inventory", "inventory"),
        ("Total Current Assets", "total_current_assets"),
        ("Property, Plant & Equipment", "property_plant_equipment"),
        ("Total Assets", "total_assets"),
        ("", None),  # blank separator
        ("Accounts Payable", "accounts_payable"),
        ("Short-Term Debt", "short_term_debt"),
        ("Total Current Liabilities", "total_current_liabilities"),
        ("Long-Term Debt", "long_term_debt"),
        ("Total Liabilities", "total_liabilities"),
        ("Retained Earnings", "retained_earnings"),
        ("Total Equity", "total_equity"),
        ("Total Liabilities & Equity", "total_liabilities_and_equity"),
    ]
    separator_after = {
        "total_current_assets", "total_assets",
        "total_current_liabilities", "total_liabilities",
    }

    for label, key in field_order:
        if key is None:
            lines.append("")
            continue
        if key not in current:
            continue
        cur_val = _fmt_number(current[key])
        row = f"  {label:<40s} {cur_val:>15s}"
        if has_prior and key in prior:
            pri_val = _fmt_number(prior[key])
            row += f"  {pri_val:>15s}"
        lines.append(row)
        if key in separator_after:
            width = 56 if not has_prior else 76
            lines.append("  " + "-" * width)

    return "\n".join(lines)


def format_cash_flow_statement(cfs_data: Dict[str, Any]) -> str:
    """Render the cash flow statement as formatted text."""
    lines = ["CASH FLOW STATEMENT", "=" * 58]
    field_order = [
        ("Net Income", "net_income"),
        ("Depreciation & Amortization", "depreciation_amortization"),
        ("Changes in Working Capital", "changes_in_working_capital"),
        ("Cash from Operations", "cash_from_operations"),
        ("Capital Expenditures", "capital_expenditures"),
        ("Cash from Investing", "cash_from_investing"),
        ("Debt Repayment", "debt_repayment"),
        ("Dividends Paid", "dividends_paid"),
        ("Cash from Financing", "cash_from_financing"),
        ("Net Change in Cash", "net_change_in_cash"),
        ("Beginning Cash", "beginning_cash"),
        ("Ending Cash", "ending_cash"),
    ]
    separator_after = {
        "cash_from_operations", "cash_from_investing",
        "cash_from_financing", "net_change_in_cash",
    }
    for label, key in field_order:
        if key in cfs_data:
            lines.append(_fmt_line(label, cfs_data[key]))
            if key in separator_after:
                lines.append("  " + "-" * 56)
    return "\n".join(lines)


def format_statements(stmts: FinancialStatements) -> str:
    """Render a full set of financial statements as formatted text.

    The output mimics the layout of financial statements in an SEC filing
    or annual report, making it suitable as context for LLM evaluation.
    """
    parts: List[str] = []

    company = stmts.get("company", "Unknown Company")
    period = stmts.get("period", "")
    currency = stmts.get("currency", "USD")
    unit = stmts.get("unit", "")
    unit_note = f" (in {unit} of {currency})" if unit else f" ({currency})"

    parts.append(f"{company} — Financial Statements{unit_note}")
    parts.append(f"Period: {period}")
    parts.append("")

    if "income_statement" in stmts:
        parts.append(format_income_statement(stmts["income_statement"]))
        parts.append("")

    if "balance_sheet" in stmts:
        parts.append(format_balance_sheet(stmts["balance_sheet"]))
        parts.append("")

    if "cash_flow_statement" in stmts:
        parts.append(format_cash_flow_statement(stmts["cash_flow_statement"]))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Difficulty classification
# ---------------------------------------------------------------------------

def _classify_difficulty(ground_truth: Dict[str, Any]) -> str:
    """Assign a difficulty label based on the ground-truth metadata."""
    if not ground_truth.get("has_error"):
        return "baseline"

    # Multi-error is always hard.
    if ground_truth.get("error_count", 1) > 1:
        return "very_hard"

    error_type_str = ground_truth.get("error_type")
    if error_type_str:
        try:
            subtype = ErrorSubtype(error_type_str)
            et = ERROR_REGISTRY.get(subtype)
            if et:
                return et.detection_difficulty.value
        except ValueError:
            pass

    mag = abs(ground_truth.get("error_magnitude_pct", 0) or 0)
    if mag < 1:
        return "very_hard"
    if mag < 5:
        return "hard"
    if mag < 20:
        return "moderate"
    return "easy"


# ---------------------------------------------------------------------------
# Instance builders
# ---------------------------------------------------------------------------

def _make_instance_id(company: str, label: str) -> str:
    """Deterministic instance id: ``<company>__<label>``."""
    safe_company = company.lower().replace(" ", "_").replace(".", "")
    return f"{safe_company}__{label}"


def _build_clean_instance(
    stmts: FinancialStatements,
) -> BenchmarkInstance:
    """Create a clean (no-error) benchmark instance."""
    company = stmts.get("company", "unknown")
    period = stmts.get("period", "")
    ground_truth: Dict[str, Any] = {
        "has_error": False,
        "error_type": None,
        "error_category": None,
        "error_location": None,
        "error_magnitude_pct": None,
        "original_value": None,
        "modified_value": None,
        "description": None,
    }
    return BenchmarkInstance(
        instance_id=_make_instance_id(company, "clean"),
        company=company,
        period=period,
        formatted_statements=format_statements(stmts),
        question=VERIFICATION_QUESTION,
        ground_truth=ground_truth,
        difficulty=_classify_difficulty(ground_truth),
        raw_statements=stmts,
    )


def _build_single_error_instance(
    stmts: FinancialStatements,
    subtype: ErrorSubtype,
    magnitude_pct: float,
    index: int,
    seed: Optional[int] = None,
) -> Optional[BenchmarkInstance]:
    """Create one error-injected benchmark instance, or None on failure."""
    result = inject_error(stmts, subtype, magnitude_pct=magnitude_pct, seed=seed)
    if not result.error_injected:
        logger.warning(
            "Injection %s (%.1f%%) failed for company=%s — skipping.",
            subtype.value, magnitude_pct, stmts.get("company", "?"),
        )
        return None

    company = stmts.get("company", "unknown")
    period = stmts.get("period", "")
    gt = result.to_ground_truth()
    label = f"{subtype.value}_{magnitude_pct}pct_{index}"

    return BenchmarkInstance(
        instance_id=_make_instance_id(company, label),
        company=company,
        period=period,
        formatted_statements=format_statements(result.modified_statements),
        question=VERIFICATION_QUESTION,
        ground_truth=gt,
        difficulty=_classify_difficulty(gt),
        raw_statements=result.modified_statements,
    )


def _build_multi_error_instance(
    stmts: FinancialStatements,
    configs: List[Tuple[ErrorSubtype, float]],
    index: int,
    seed: Optional[int] = None,
) -> Optional[BenchmarkInstance]:
    """Create a multi-error benchmark instance."""
    result = inject_multiple_errors(stmts, configs, seed=seed)
    if result.error_count == 0:
        logger.warning(
            "Multi-error injection produced 0 errors for company=%s — skipping.",
            stmts.get("company", "?"),
        )
        return None

    company = stmts.get("company", "unknown")
    period = stmts.get("period", "")
    gt = result.to_ground_truth()
    codes = "_".join(s.value for s, _ in configs)
    label = f"multi_{codes}_{index}"

    return BenchmarkInstance(
        instance_id=_make_instance_id(company, label),
        company=company,
        period=period,
        formatted_statements=format_statements(result.modified_statements),
        question=VERIFICATION_QUESTION,
        ground_truth=gt,
        difficulty=_classify_difficulty(gt),
        raw_statements=result.modified_statements,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_statistics(instances: List[BenchmarkInstance]) -> DatasetStatistics:
    """Compute summary statistics over the full benchmark dataset."""
    stats = DatasetStatistics()
    stats.total_instances = len(instances)

    company_counts: Counter[str] = Counter()
    error_type_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    magnitude_buckets: Counter[str] = Counter()

    for inst in instances:
        company_counts[inst.company] += 1
        difficulty_counts[inst.difficulty] += 1
        gt = inst.ground_truth

        if not gt.get("has_error"):
            stats.clean_instances += 1
            continue

        stats.error_instances += 1

        # Handle multi-error ground truths.
        errors = gt.get("errors", [gt])
        for err in errors:
            etype = err.get("error_type")
            ecat = err.get("error_category")
            if etype:
                error_type_counts[etype] += 1
            if ecat:
                category_counts[ecat] += 1

            mag = abs(err.get("error_magnitude_pct") or 0)
            if mag < 1:
                magnitude_buckets["<1%"] += 1
            elif mag < 5:
                magnitude_buckets["1-5%"] += 1
            elif mag < 20:
                magnitude_buckets["5-20%"] += 1
            else:
                magnitude_buckets[">20%"] += 1

    stats.instances_per_company = dict(company_counts)
    stats.instances_per_error_type = dict(error_type_counts)
    stats.instances_per_category = dict(category_counts)
    stats.instances_per_difficulty = dict(difficulty_counts)
    stats.magnitude_distribution = dict(magnitude_buckets)
    return stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_processed_statements(
    processed_dir: str | Path,
) -> List[FinancialStatements]:
    """Load all ``*.json`` files from *processed_dir*.

    Each file is expected to contain a single financial-statement document
    conforming to the schema described in :mod:`benchmark.error_injection`.
    """
    processed_dir = Path(processed_dir)
    if not processed_dir.is_dir():
        raise FileNotFoundError(
            f"Processed data directory not found: {processed_dir}"
        )

    statements: List[FinancialStatements] = []
    for fp in sorted(processed_dir.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Support both single-doc and list-of-docs files.
        if isinstance(data, list):
            statements.extend(data)
        else:
            statements.append(data)
    return statements


def build_benchmark_dataset(
    processed_dir: str | Path = "data/processed",
    output_dir: str | Path = "data/benchmark",
    single_error_configs: Optional[List[Tuple[ErrorSubtype, float]]] = None,
    multi_error_configs: Optional[List[List[Tuple[ErrorSubtype, float]]]] = None,
    seed: int = 42,
    write_individual_files: bool = True,
) -> Tuple[List[BenchmarkInstance], DatasetStatistics]:
    """Build and persist the complete benchmark dataset.

    Parameters
    ----------
    processed_dir:
        Path to the directory containing processed financial-statement
        JSON files.
    output_dir:
        Path where the benchmark dataset will be written.
    single_error_configs:
        List of ``(ErrorSubtype, magnitude_pct)`` pairs.  Each is applied
        once per company.  Defaults to ``DEFAULT_SINGLE_ERROR_CONFIGS``.
    multi_error_configs:
        List of multi-error combos (each combo is a list of
        ``(ErrorSubtype, magnitude_pct)``).  Defaults to
        ``DEFAULT_MULTI_ERROR_CONFIGS``.
    seed:
        Base RNG seed for reproducibility.
    write_individual_files:
        If ``True``, also write one JSON file per instance under
        ``output_dir/instances/``.

    Returns
    -------
    tuple[list[BenchmarkInstance], DatasetStatistics]
        The generated instances and summary statistics.
    """
    if single_error_configs is None:
        single_error_configs = DEFAULT_SINGLE_ERROR_CONFIGS
    if multi_error_configs is None:
        multi_error_configs = DEFAULT_MULTI_ERROR_CONFIGS

    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)

    all_statements = load_processed_statements(processed_dir)
    if not all_statements:
        logger.warning("No processed statements found in %s", processed_dir)

    logger.info(
        "Loaded %d financial statement(s) from %s",
        len(all_statements), processed_dir,
    )

    instances: List[BenchmarkInstance] = []

    for stmt_idx, stmts in enumerate(all_statements):
        company = stmts.get("company", f"company_{stmt_idx}")

        # 1. Clean instance.
        instances.append(_build_clean_instance(stmts))

        # 2. Single-error instances.
        for cfg_idx, (subtype, mag) in enumerate(single_error_configs):
            inst_seed = seed + stmt_idx * 1000 + cfg_idx
            inst = _build_single_error_instance(
                stmts, subtype, mag, index=cfg_idx, seed=inst_seed,
            )
            if inst is not None:
                instances.append(inst)

        # 3. Multi-error instances.
        for multi_idx, combo in enumerate(multi_error_configs):
            inst_seed = seed + stmt_idx * 1000 + 900 + multi_idx
            inst = _build_multi_error_instance(
                stmts, combo, index=multi_idx, seed=inst_seed,
            )
            if inst is not None:
                instances.append(inst)

    stats = _compute_statistics(instances)

    # ---- Persist ----------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = output_dir / "benchmark.json"
    with open(benchmark_path, "w", encoding="utf-8") as fh:
        json.dump(
            [inst.to_dict() for inst in instances],
            fh,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Wrote %d instances to %s", len(instances), benchmark_path)

    stats_path = output_dir / "benchmark_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, ensure_ascii=False)
    logger.info("Wrote statistics to %s", stats_path)

    if write_individual_files:
        instances_dir = output_dir / "instances"
        instances_dir.mkdir(parents=True, exist_ok=True)
        for inst in instances:
            fp = instances_dir / f"{inst.instance_id}.json"
            with open(fp, "w", encoding="utf-8") as fh:
                json.dump(inst.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info(
            "Wrote %d individual instance files to %s",
            len(instances), instances_dir,
        )

    return instances, stats
