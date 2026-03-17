"""Analyze the FinVerBench dataset for complexity and diversity metrics.

Computes per-company and cross-company statistics, error injection
coverage rates, and generates a publication-quality 2-panel figure.

Usage:
    PYTHONPATH=src python src/analysis/dataset_analysis.py
"""

from __future__ import annotations

import copy
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from benchmark.error_injection import (
    _IS_ROW_SUMS,
    _BS_ROW_SUMS,
    _CFS_ROW_SUMS,
    inject_error,
)
from benchmark.error_taxonomy import ErrorSubtype

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERTED_DIR = PROJECT_ROOT / "data" / "converted"
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
# Sector mapping (ticker -> sector)
# ---------------------------------------------------------------------------
TICKER_SECTOR: Dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "META": "Technology", "NVDA": "Technology", "ADBE": "Technology",
    "ORCL": "Technology", "CRM": "Technology", "CSCO": "Technology",
    "INTC": "Technology", "AVGO": "Technology", "TXN": "Technology",
    # Healthcare
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
    "LLY": "Healthcare", "ABT": "Healthcare", "MRK": "Healthcare",
    "TMO": "Healthcare", "AMGN": "Healthcare",
    # Consumer
    "AMZN": "Consumer", "COST": "Consumer", "HD": "Consumer",
    "LOW": "Consumer", "NKE": "Consumer", "KO": "Consumer",
    "PEP": "Consumer", "PG": "Consumer", "WMT": "Consumer",
    "MCD": "Consumer", "PM": "Consumer",
    # Financial
    "JPM": "Financial", "BAC": "Financial", "V": "Financial",
    "BRK.B": "Financial", "GS": "Financial", "MA": "Financial",
    # Industrial
    "HON": "Industrial", "CAT": "Industrial", "BA": "Industrial",
    "GE": "Industrial", "UPS": "Industrial",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "NEE": "Energy",
    # Telecom
    "VZ": "Telecom", "T": "Telecom", "CMCSA": "Telecom",
    "DIS": "Telecom",
}

SECTOR_ORDER = [
    "Technology", "Healthcare", "Consumer", "Financial",
    "Industrial", "Energy", "Telecom",
]

SECTOR_COLORS = {
    "Technology": "#4e79a7",
    "Healthcare": "#e15759",
    "Consumer": "#f28e2b",
    "Financial": "#b07aa1",
    "Industrial": "#76b7b2",
    "Energy": "#edc948",
    "Telecom": "#59a14f",
}

# ---------------------------------------------------------------------------
# Error subtypes and their human-readable labels
# ---------------------------------------------------------------------------
ERROR_SUBTYPE_LABELS = {
    "AE_ROW_SUM": "Row Sum (AE)",
    "AE_COLUMN_SUM": "Column Sum (AE)",
    "CL_NET_INCOME_TO_RE": "NI -> Ret. Earnings (CL)",
    "CL_NET_INCOME_TO_CFS": "NI -> Cash Flow (CL)",
    "CL_ENDING_CASH": "Ending Cash (CL)",
    "CL_DEPRECIATION": "Depreciation (CL)",
    "YOY_OPENING_BALANCE": "Opening Balance (YoY)",
    "YOY_COMPUTED_CHANGE": "Computed Change (YoY)",
    "MR_MINOR": "Minor <1% (MR)",
    "MR_MODERATE": "Moderate 1-5% (MR)",
    "MR_SIGNIFICANT": "Significant 5-20% (MR)",
    "MR_EXTREME": "Extreme >20% (MR)",
}

ERROR_SUBTYPE_ORDER = list(ERROR_SUBTYPE_LABELS.keys())

CATEGORY_COLORS = {
    "AE": "#4e79a7",
    "CL": "#f28e2b",
    "YOY": "#e15759",
    "MR": "#76b7b2",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_converted_data() -> List[Dict[str, Any]]:
    """Load all *.json files from data/converted/."""
    companies: List[Dict[str, Any]] = []
    for fp in sorted(CONVERTED_DIR.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        companies.append(data)
    return companies


# ---------------------------------------------------------------------------
# Per-company statistics
# ---------------------------------------------------------------------------

def _count_line_items(section: Dict[str, Any]) -> int:
    """Count numeric fields in a flat dict section."""
    return sum(1 for v in section.values() if isinstance(v, (int, float)))


def _count_checkable_relationships(data: Dict[str, Any]) -> int:
    """Count how many accounting relationships have both sides present."""
    count = 0

    # Income statement row sums
    is_data = data.get("income_statement", {})
    for total_field, components in _IS_ROW_SUMS:
        if total_field in is_data and all(c in is_data for c in components):
            count += 1

    # Balance sheet row sums (current year)
    bs = data.get("balance_sheet", {}).get("current_year", {})
    for total_field, components in _BS_ROW_SUMS:
        if total_field in bs and all(c in bs for c in components):
            count += 1

    # Cash flow statement row sums
    cfs = data.get("cash_flow_statement", {})
    for total_field, components in _CFS_ROW_SUMS:
        if total_field in cfs and all(c in cfs for c in components):
            count += 1

    # Cross-statement linkages
    # NI: IS <-> CFS
    if "net_income" in is_data and "net_income" in cfs:
        count += 1
    # NI: IS <-> BS retained earnings
    if "net_income" in is_data and "retained_earnings" in bs:
        count += 1
    # Ending cash: CFS <-> BS
    if "ending_cash" in cfs and "cash_and_equivalents" in bs:
        count += 1
    # Depreciation: IS <-> CFS
    if "depreciation_amortization" in is_data and "depreciation_amortization" in cfs:
        count += 1
    # BS equation: assets = liabilities + equity
    if all(k in bs for k in ["total_assets", "total_liabilities_and_equity"]):
        count += 1

    return count


def compute_per_company_stats(
    companies: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute statistics for each company."""
    results = []
    for data in companies:
        ticker = data.get("ticker", "???")
        company_name = data.get("company", "Unknown")
        period = data.get("period", "")
        unit = data.get("unit", "millions")
        sector = TICKER_SECTOR.get(ticker, "Unknown")

        # Revenue and total assets
        revenue_raw = data.get("income_statement", {}).get("revenue")
        total_assets_raw = data.get("balance_sheet", {}).get(
            "current_year", {}
        ).get("total_assets")

        # Convert to billions (data is in millions)
        scale = 1e3 if unit == "millions" else 1.0
        revenue_B = revenue_raw / scale if revenue_raw is not None else None
        total_assets_B = (
            total_assets_raw / scale if total_assets_raw is not None else None
        )

        # Line items per statement
        is_data = data.get("income_statement", {})
        bs_cy = data.get("balance_sheet", {}).get("current_year", {})
        bs_py = data.get("balance_sheet", {}).get("prior_year", {})
        cfs = data.get("cash_flow_statement", {})

        is_items = _count_line_items(is_data)
        bs_cy_items = _count_line_items(bs_cy)
        bs_py_items = _count_line_items(bs_py)
        cfs_items = _count_line_items(cfs)
        total_items = is_items + bs_cy_items + bs_py_items + cfs_items

        # Checkable relationships
        n_checkable = _count_checkable_relationships(data)

        # Statement completeness
        has_is = bool(is_data)
        has_bs = bool(bs_cy)
        has_cfs = bool(cfs)
        has_all_three = has_is and has_bs and has_cfs
        has_prior_year = bool(bs_py)

        results.append({
            "ticker": ticker,
            "company": company_name,
            "period": period,
            "sector": sector,
            "revenue_billions": round(revenue_B, 2) if revenue_B is not None else None,
            "total_assets_billions": (
                round(total_assets_B, 2) if total_assets_B is not None else None
            ),
            "is_line_items": is_items,
            "bs_cy_line_items": bs_cy_items,
            "bs_py_line_items": bs_py_items,
            "cfs_line_items": cfs_items,
            "total_line_items": total_items,
            "checkable_relationships": n_checkable,
            "has_income_statement": has_is,
            "has_balance_sheet": has_bs,
            "has_cash_flow_statement": has_cfs,
            "has_all_three_statements": has_all_three,
            "has_prior_year_data": has_prior_year,
        })

    return results


# ---------------------------------------------------------------------------
# Cross-company statistics
# ---------------------------------------------------------------------------

def compute_cross_company_stats(
    per_company: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate statistics across all companies."""
    revenues = [c["revenue_billions"] for c in per_company
                if c["revenue_billions"] is not None]
    assets = [c["total_assets_billions"] for c in per_company
              if c["total_assets_billions"] is not None]
    line_items = [c["total_line_items"] for c in per_company]
    checkable = [c["checkable_relationships"] for c in per_company]

    # Fiscal year distribution
    fy_counter: Counter[str] = Counter()
    for c in per_company:
        fy_counter[c["period"]] += 1

    # Completeness counts
    n_prior_year = sum(1 for c in per_company if c["has_prior_year_data"])
    n_all_three = sum(1 for c in per_company if c["has_all_three_statements"])

    # Sector distribution
    sector_counter: Counter[str] = Counter()
    for c in per_company:
        sector_counter[c["sector"]] += 1

    return {
        "n_companies": len(per_company),
        "revenue_billions": {
            "min": round(min(revenues), 2),
            "max": round(max(revenues), 2),
            "mean": round(float(np.mean(revenues)), 2),
            "std": round(float(np.std(revenues)), 2),
            "median": round(float(np.median(revenues)), 2),
        },
        "total_assets_billions": {
            "min": round(min(assets), 2),
            "max": round(max(assets), 2),
            "mean": round(float(np.mean(assets)), 2),
            "std": round(float(np.std(assets)), 2),
            "median": round(float(np.median(assets)), 2),
        },
        "line_items_per_company": {
            "min": min(line_items),
            "max": max(line_items),
            "mean": round(float(np.mean(line_items)), 1),
            "std": round(float(np.std(line_items)), 1),
        },
        "checkable_relationships_per_company": {
            "min": min(checkable),
            "max": max(checkable),
            "mean": round(float(np.mean(checkable)), 1),
            "std": round(float(np.std(checkable)), 1),
        },
        "fiscal_year_distribution": dict(fy_counter.most_common()),
        "companies_with_prior_year_data": n_prior_year,
        "companies_with_all_three_statements": n_all_three,
        "sector_distribution": {
            s: sector_counter.get(s, 0) for s in SECTOR_ORDER
        },
    }


# ---------------------------------------------------------------------------
# Error injection coverage
# ---------------------------------------------------------------------------

def _can_inject(
    data: Dict[str, Any],
    subtype: ErrorSubtype,
) -> bool:
    """Check whether an error subtype can be injected into this company's data.

    Rather than duplicating the precondition logic from each injector, we
    attempt injection on a deep copy and check whether error_injected is True.
    """
    result = inject_error(copy.deepcopy(data), subtype, magnitude_pct=5.0, seed=0)
    return result.error_injected


def compute_error_coverage(
    companies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """For each error subtype, compute how many companies support injection."""
    subtypes = list(ErrorSubtype)
    coverage: Dict[str, Dict[str, Any]] = {}

    n_companies = len(companies)

    for subtype in subtypes:
        can_count = 0
        cannot_tickers: List[str] = []
        for data in companies:
            if _can_inject(data, subtype):
                can_count += 1
            else:
                cannot_tickers.append(data.get("ticker", "?"))

        rate = can_count / n_companies if n_companies > 0 else 0.0
        coverage[subtype.value] = {
            "companies_covered": can_count,
            "companies_skipped": n_companies - can_count,
            "coverage_rate": round(rate, 4),
            "skipped_tickers": cannot_tickers,
        }

    return coverage


# ---------------------------------------------------------------------------
# Publication-quality figure
# ---------------------------------------------------------------------------

def _apply_publication_style() -> None:
    """Configure matplotlib for publication-quality output."""
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


def generate_figure(
    per_company: List[Dict[str, Any]],
    error_coverage: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate the 2-panel dataset diversity figure.

    Panel A: Scatter of revenue vs total assets (log-log), colored by sector.
    Panel B: Bar chart of error injection coverage rate by error subtype.
    """
    _apply_publication_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: Revenue vs Total Assets scatter (log-log) -----------------
    for sector in SECTOR_ORDER:
        xs, ys, labels = [], [], []
        for c in per_company:
            if c["sector"] != sector:
                continue
            if c["revenue_billions"] is None or c["total_assets_billions"] is None:
                continue
            xs.append(c["revenue_billions"])
            ys.append(c["total_assets_billions"])
            labels.append(c["ticker"])

        if not xs:
            continue

        color = SECTOR_COLORS[sector]
        ax1.scatter(
            xs, ys, label=sector, color=color,
            s=60, alpha=0.85, edgecolors="white", linewidth=0.5, zorder=3,
        )

        # Annotate each point with the ticker
        for x, y, lab in zip(xs, ys, labels):
            ax1.annotate(
                lab, (x, y),
                textcoords="offset points", xytext=(5, 4),
                fontsize=6, color=color, alpha=0.9,
            )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Revenue ($ Billions)")
    ax1.set_ylabel("Total Assets ($ Billions)")
    ax1.set_title("(A) Company Size Distribution by Sector")
    ax1.legend(
        loc="upper left", framealpha=0.9, fontsize=8,
        title="Sector", title_fontsize=9,
    )
    ax1.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Panel B: Error injection coverage rate by subtype ------------------
    subtypes_to_plot = ERROR_SUBTYPE_ORDER
    rates = [error_coverage[s]["coverage_rate"] * 100 for s in subtypes_to_plot]
    bar_labels = [ERROR_SUBTYPE_LABELS[s] for s in subtypes_to_plot]

    # Color bars by error category
    bar_colors = []
    for s in subtypes_to_plot:
        cat_prefix = s.split("_")[0]
        bar_colors.append(CATEGORY_COLORS.get(cat_prefix, "#aaa"))

    y_pos = np.arange(len(subtypes_to_plot))
    bars = ax2.barh(
        y_pos, rates, height=0.65, color=bar_colors,
        edgecolor="white", linewidth=0.5,
    )

    # Value labels at end of bars
    for bar, rate in zip(bars, rates):
        ax2.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.0f}%",
            ha="left", va="center", fontsize=8, fontweight="bold",
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(bar_labels, fontsize=8)
    ax2.set_xlabel("Coverage Rate (%)")
    ax2.set_title("(B) Error Injection Coverage by Subtype")
    ax2.set_xlim(0, 115)
    ax2.invert_yaxis()

    # Category legend
    cat_patches = [
        mpatches.Patch(color=CATEGORY_COLORS[c], label=lbl)
        for c, lbl in [
            ("AE", "Arithmetic (AE)"),
            ("CL", "Cross-Linkage (CL)"),
            ("YOY", "Year-over-Year (YoY)"),
            ("MR", "Magnitude (MR)"),
        ]
    ]
    ax2.legend(
        handles=cat_patches, loc="lower right", framealpha=0.9,
        fontsize=7, title="Category", title_fontsize=8,
    )

    fig.tight_layout(w_pad=3.0)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved figure to %s", output_path)


# ---------------------------------------------------------------------------
# Print summary for the paper
# ---------------------------------------------------------------------------

def print_paper_statistics(
    per_company: List[Dict[str, Any]],
    cross_stats: Dict[str, Any],
    error_coverage: Dict[str, Dict[str, Any]],
) -> None:
    """Print key statistics suitable for inclusion in the paper."""
    sep = "=" * 72

    print(f"\n{sep}")
    print("  KEY STATISTICS FOR PAPER")
    print(sep)

    n = cross_stats["n_companies"]
    print(f"\nDataset size: {n} companies from the S&P 500")

    # Sector breakdown
    sd = cross_stats["sector_distribution"]
    sector_parts = [f"{s}: {sd[s]}" for s in SECTOR_ORDER if sd.get(s, 0) > 0]
    print(f"Sector breakdown: {', '.join(sector_parts)}")

    # Revenue range
    rv = cross_stats["revenue_billions"]
    print(
        f"\nRevenue range: ${rv['min']}B - ${rv['max']}B "
        f"(mean=${rv['mean']}B, median=${rv['median']}B, std=${rv['std']}B)"
    )

    # Total assets range
    ta = cross_stats["total_assets_billions"]
    print(
        f"Total assets range: ${ta['min']}B - ${ta['max']}B "
        f"(mean=${ta['mean']}B, median=${ta['median']}B, std=${ta['std']}B)"
    )

    # Span ratio (max/min) for revenue and assets
    rev_span = rv["max"] / rv["min"] if rv["min"] > 0 else float("inf")
    asset_span = ta["max"] / ta["min"] if ta["min"] > 0 else float("inf")
    print(f"Revenue span ratio (max/min): {rev_span:.1f}x")
    print(f"Total assets span ratio (max/min): {asset_span:.1f}x")

    # Line items
    li = cross_stats["line_items_per_company"]
    print(
        f"\nLine items per company: {li['min']}-{li['max']} "
        f"(mean={li['mean']}, std={li['std']})"
    )

    # Checkable relationships
    cr = cross_stats["checkable_relationships_per_company"]
    print(
        f"Checkable relationships per company: {cr['min']}-{cr['max']} "
        f"(mean={cr['mean']}, std={cr['std']})"
    )

    # Completeness
    n_all3 = cross_stats["companies_with_all_three_statements"]
    n_py = cross_stats["companies_with_prior_year_data"]
    print(f"\nCompanies with all 3 statements: {n_all3}/{n} ({n_all3/n*100:.0f}%)")
    print(f"Companies with prior-year data: {n_py}/{n} ({n_py/n*100:.0f}%)")

    # Fiscal year distribution
    fy = cross_stats["fiscal_year_distribution"]
    fy_str = ", ".join(f"{k}: {v}" for k, v in sorted(fy.items()))
    print(f"Fiscal year distribution: {fy_str}")

    # Error coverage
    print(f"\n{'─' * 72}")
    print("  ERROR INJECTION COVERAGE")
    print(f"{'─' * 72}")
    print(f"{'Error Subtype':<28s}  {'Covered':>8s}  {'Skipped':>8s}  {'Rate':>8s}")
    print(f"{'─' * 56}")

    full_coverage = []
    partial_coverage = []
    for subtype_val in ERROR_SUBTYPE_ORDER:
        info = error_coverage[subtype_val]
        covered = info["companies_covered"]
        skipped = info["companies_skipped"]
        rate = info["coverage_rate"] * 100
        print(f"{subtype_val:<28s}  {covered:>8d}  {skipped:>8d}  {rate:>7.1f}%")
        if rate == 100.0:
            full_coverage.append(subtype_val)
        else:
            partial_coverage.append((subtype_val, rate))

    print(f"\nFull coverage (100%): {len(full_coverage)} subtypes")
    if partial_coverage:
        print("Partial coverage:")
        for st, rate in partial_coverage:
            tickers = error_coverage[st]["skipped_tickers"]
            print(f"  {st}: {rate:.1f}% (skipped: {', '.join(tickers)})")

    # Overall coverage
    total_possible = len(ERROR_SUBTYPE_ORDER) * n
    total_covered = sum(
        error_coverage[s]["companies_covered"] for s in ERROR_SUBTYPE_ORDER
    )
    overall_rate = total_covered / total_possible * 100 if total_possible > 0 else 0
    print(f"\nOverall injection coverage: {total_covered}/{total_possible} "
          f"({overall_rate:.1f}%)")

    # Total checkable relationships across all companies
    total_checkable = sum(c["checkable_relationships"] for c in per_company)
    print(f"Total checkable relationships across dataset: {total_checkable}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logger.info("Loading converted financial data from %s", CONVERTED_DIR)
    companies = load_converted_data()
    logger.info("Loaded %d companies", len(companies))

    # 1. Per-company statistics
    logger.info("Computing per-company statistics...")
    per_company = compute_per_company_stats(companies)

    # 2. Cross-company statistics
    logger.info("Computing cross-company statistics...")
    cross_stats = compute_cross_company_stats(per_company)

    # 3. Error injection coverage
    logger.info("Computing error injection coverage (testing each subtype x company)...")
    error_coverage = compute_error_coverage(companies)

    # 4. Generate figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURES_DIR / "figure_dataset_diversity.pdf"
    logger.info("Generating figure -> %s", figure_path)
    generate_figure(per_company, error_coverage, figure_path)

    # 5. Print paper statistics
    print_paper_statistics(per_company, cross_stats, error_coverage)

    # 6. Save results to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "dataset_analysis.json"
    results = {
        "per_company": per_company,
        "cross_company": cross_stats,
        "error_injection_coverage": error_coverage,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logger.info("Saved results to %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
