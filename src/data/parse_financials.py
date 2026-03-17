"""
Parse raw SEC EDGAR XBRL JSON into structured financial statements.

Reads the company-facts JSON files produced by fetch_filings.py (in data/raw/)
and extracts Balance Sheet, Income Statement, and Cash Flow Statement line
items.  The output is one clean JSON file per company in data/processed/.

Usage:
    python -m src.data.parse_financials                 # parse all raw files
    python -m src.data.parse_financials --ticker AAPL   # parse one company
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

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
# XBRL concept → readable label mapping
#
# Keys are the XBRL tag names (without namespace prefix) as they appear in
# the EDGAR company-facts JSON under facts → us-gaap.
# ---------------------------------------------------------------------------

BALANCE_SHEET_CONCEPTS: Dict[str, str] = {
    "Assets": "Total Assets",
    "Liabilities": "Total Liabilities",
    "StockholdersEquity": "Total Stockholders Equity",
    "LiabilitiesAndStockholdersEquity": "Total Liabilities and Equity",
    "CashAndCashEquivalentsAtCarryingValue": "Cash and Cash Equivalents",
    "ShortTermInvestments": "Short-Term Investments",
    "AccountsReceivableNetCurrent": "Accounts Receivable",
    "InventoryNet": "Inventory",
    "Inventory": "Inventory",
    "AssetsCurrent": "Total Current Assets",
    "PropertyPlantAndEquipmentNet": "Property, Plant and Equipment (net)",
    "Goodwill": "Goodwill",
    "IntangibleAssetsNetExcludingGoodwill": "Intangible Assets (net)",
    "LiabilitiesCurrent": "Total Current Liabilities",
    "LongTermDebt": "Long-Term Debt",
    "LongTermDebtNoncurrent": "Long-Term Debt (non-current)",
    "RetainedEarningsAccumulatedDeficit": "Retained Earnings (Accumulated Deficit)",
    "CommonStockSharesOutstanding": "Common Shares Outstanding",
    "CommonStockSharesAuthorized": "Common Shares Authorized",
}

INCOME_STATEMENT_CONCEPTS: Dict[str, str] = {
    "Revenues": "Total Revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue from Contracts",
    "SalesRevenueNet": "Net Sales Revenue",
    "CostOfGoodsAndServicesSold": "Cost of Goods Sold",
    "CostOfRevenue": "Cost of Revenue",
    "CostOfGoodsSold": "Cost of Goods Sold",
    "GrossProfit": "Gross Profit",
    "OperatingExpenses": "Operating Expenses",
    "SellingGeneralAndAdministrativeExpense": "SG&A Expense",
    "ResearchAndDevelopmentExpense": "R&D Expense",
    "OperatingIncomeLoss": "Operating Income (Loss)",
    "InterestExpense": "Interest Expense",
    "IncomeTaxExpenseBenefit": "Income Tax Expense",
    "NetIncomeLoss": "Net Income (Loss)",
    "EarningsPerShareBasic": "Earnings Per Share (Basic)",
    "EarningsPerShareDiluted": "Earnings Per Share (Diluted)",
    "WeightedAverageNumberOfShareOutstandingBasicAndDiluted": (
        "Weighted Avg Shares Outstanding"
    ),
    "WeightedAverageNumberOfDilutedSharesOutstanding": (
        "Weighted Avg Diluted Shares Outstanding"
    ),
}

CASH_FLOW_CONCEPTS: Dict[str, str] = {
    "NetCashProvidedByUsedInOperatingActivities": (
        "Cash from Operating Activities"
    ),
    "NetCashProvidedByUsedInInvestingActivities": (
        "Cash from Investing Activities"
    ),
    "NetCashProvidedByUsedInFinancingActivities": (
        "Cash from Financing Activities"
    ),
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect": (
        "Net Change in Cash"
    ),
    "CashAndCashEquivalentsPeriodIncreaseDecrease": (
        "Net Change in Cash (legacy)"
    ),
    "DepreciationDepletionAndAmortization": "Depreciation & Amortization",
    "DepreciationAndAmortization": "Depreciation & Amortization",
    "PaymentsToAcquirePropertyPlantAndEquipment": "Capital Expenditures",
    "PaymentsOfDividends": "Dividends Paid",
    "PaymentsOfDividendsCommonStock": "Common Dividends Paid",
    "PaymentsForRepurchaseOfCommonStock": "Share Repurchases",
    "ProceedsFromIssuanceOfLongTermDebt": "Proceeds from Debt Issuance",
    "RepaymentsOfLongTermDebt": "Repayments of Long-Term Debt",
}

# Group all concept maps by statement type.
STATEMENT_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "balance_sheet": BALANCE_SHEET_CONCEPTS,
    "income_statement": INCOME_STATEMENT_CONCEPTS,
    "cash_flow_statement": CASH_FLOW_CONCEPTS,
}

# XBRL period types that belong to each statement.
# Balance sheet items are "instant" (point-in-time);
# income-statement and cash-flow items are "duration" (over a period).
EXPECTED_PERIOD_TYPE: Dict[str, str] = {
    "balance_sheet": "instant",
    "income_statement": "duration",
    "cash_flow_statement": "duration",
}


# ---------------------------------------------------------------------------
# Fact extraction helpers
# ---------------------------------------------------------------------------

def _extract_facts_for_concept(
    us_gaap: Dict[str, Any],
    concept: str,
) -> List[Dict[str, Any]]:
    """Return the list of USD facts (or shares) for a single XBRL concept.

    The EDGAR JSON nests facts under:
        facts → us-gaap → <Concept> → units → <unit> → [list of facts]

    Each fact dict has keys like: val, end, start (for duration), fy, fp,
    form, filed, accn, frame.

    We prefer USD units; fall back to 'shares' for per-share or share-count
    items; then try 'USD/shares' for EPS.
    """
    concept_data = us_gaap.get(concept)
    if concept_data is None:
        return []

    units = concept_data.get("units", {})

    # Priority: USD → USD/shares → shares → pure (for ratios)
    for unit_key in ("USD", "USD/shares", "shares", "pure"):
        if unit_key in units:
            return units[unit_key]

    # Fall back to the first available unit.
    for unit_key, facts in units.items():
        return facts

    return []


def _is_annual_10k(fact: Dict[str, Any]) -> bool:
    """Return True if the fact comes from a 10-K (annual) filing."""
    form = fact.get("form", "")
    fp = fact.get("fp", "")
    return form == "10-K" or fp == "FY"


def _is_quarterly_10q(fact: Dict[str, Any]) -> bool:
    """Return True if the fact comes from a 10-Q (quarterly) filing."""
    form = fact.get("form", "")
    return form == "10-Q"


def _period_key(fact: Dict[str, Any]) -> str:
    """Return a sortable string that identifies this fact's reporting period.

    For instant facts (balance-sheet): the 'end' date.
    For duration facts (income/cash-flow): 'start—end'.
    """
    end = fact.get("end", "")
    start = fact.get("start", "")
    if start:
        return f"{start}--{end}"
    return end


def _fiscal_label(fact: Dict[str, Any]) -> str:
    """Return a human-readable fiscal-period label like 'FY2023' or 'Q2 2023'."""
    fy = fact.get("fy")
    fp = fact.get("fp", "")

    if fy is None:
        # Fall back to the end-date year.
        end = fact.get("end", "")
        fy = end[:4] if len(end) >= 4 else "????"

    if fp == "FY":
        return f"FY{fy}"
    if fp.startswith("Q"):
        return f"{fp} {fy}"
    return f"{fp}{fy}"


def _deduplicate_facts(
    facts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove duplicate facts for the same period, keeping the latest filing.

    EDGAR often has multiple filings for the same period (amendments, restated
    figures).  We keep the one filed most recently.
    """
    by_period: Dict[str, Dict[str, Any]] = {}
    for f in facts:
        key = _period_key(f)
        existing = by_period.get(key)
        if existing is None or f.get("filed", "") > existing.get("filed", ""):
            by_period[key] = f
    # Return sorted by period key (chronological).
    return [by_period[k] for k in sorted(by_period)]


# ---------------------------------------------------------------------------
# Statement-level extraction
# ---------------------------------------------------------------------------

def _extract_statement(
    us_gaap: Dict[str, Any],
    concept_map: Dict[str, str],
    period_type: str,
    annual_only: bool = True,
) -> Dict[str, Any]:
    """Extract a single financial statement from the XBRL facts.

    Returns a dict structured as:
        {
            "line_items": {
                "<readable_label>": {
                    "xbrl_concept": "<Concept>",
                    "periods": {
                        "<fiscal_label>": {
                            "value": ...,
                            "end": "...",
                            "start": "...",  # for duration items
                            "filed": "...",
                            "form": "...",
                        },
                        ...
                    }
                },
                ...
            }
        }
    """
    line_items: Dict[str, Any] = {}

    for concept, label in concept_map.items():
        raw_facts = _extract_facts_for_concept(us_gaap, concept)
        if not raw_facts:
            continue

        # Filter to annual (10-K / FY) facts only when requested.
        if annual_only:
            raw_facts = [f for f in raw_facts if _is_annual_10k(f)]
        if not raw_facts:
            continue

        deduped = _deduplicate_facts(raw_facts)

        periods: Dict[str, Any] = {}
        for fact in deduped:
            fl = _fiscal_label(fact)
            entry: Dict[str, Any] = {"value": fact.get("val")}

            # Include date boundaries.
            if fact.get("end"):
                entry["end"] = fact["end"]
            if fact.get("start"):
                entry["start"] = fact["start"]

            entry["filed"] = fact.get("filed", "")
            entry["form"] = fact.get("form", "")

            periods[fl] = entry

        if periods:
            # If the same readable label is already present (e.g. two XBRL
            # concepts map to "Inventory"), keep whichever has more data.
            if label in line_items:
                existing_count = len(line_items[label].get("periods", {}))
                if len(periods) <= existing_count:
                    continue

            line_items[label] = {
                "xbrl_concept": concept,
                "periods": periods,
            }

    return {"line_items": line_items}


# ---------------------------------------------------------------------------
# Company-level parsing
# ---------------------------------------------------------------------------

def parse_company(raw_json: Dict[str, Any]) -> Dict[str, Any]:
    """Parse one company's raw EDGAR JSON into structured statements.

    Returns a dict with metadata and three statement sections.
    """
    # Metadata embedded in the EDGAR response.
    cik = raw_json.get("cik", "")
    entity_name = raw_json.get("entityName", "")

    us_gaap = raw_json.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        logger.warning(
            "No us-gaap facts for CIK %s (%s) — skipping.",
            cik,
            entity_name,
        )
        return {
            "metadata": {
                "cik": cik,
                "entity_name": entity_name,
                "total_line_items": 0,
                "fiscal_years": [],
            },
            "balance_sheet": {"line_items": {}},
            "income_statement": {"line_items": {}},
            "cash_flow_statement": {"line_items": {}},
        }

    result: Dict[str, Any] = {
        "metadata": {
            "cik": cik,
            "entity_name": entity_name,
        },
    }

    for stmt_key, concept_map in STATEMENT_DEFINITIONS.items():
        period_type = EXPECTED_PERIOD_TYPE[stmt_key]
        result[stmt_key] = _extract_statement(
            us_gaap,
            concept_map,
            period_type,
            annual_only=True,
        )

    # Attach summary statistics.
    total_items = sum(
        len(result[s]["line_items"]) for s in STATEMENT_DEFINITIONS
    )
    fiscal_years = set()
    for s in STATEMENT_DEFINITIONS:
        for item in result[s]["line_items"].values():
            fiscal_years.update(item.get("periods", {}).keys())

    result["metadata"]["total_line_items"] = total_items
    result["metadata"]["fiscal_years"] = sorted(fiscal_years)

    return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_raw_json(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load and return a raw JSON file, or None on error."""
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read %s: %s", filepath, exc)
        return None


def save_processed_json(
    data: Dict[str, Any],
    ticker: str,
    output_dir: Path,
) -> Path:
    """Write a processed JSON file and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{ticker}.json"
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Saved %s (%d bytes)", filepath, filepath.stat().st_size)
    return filepath


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def parse_all(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    tickers: Optional[List[str]] = None,
) -> Tuple[int, int]:
    """Parse all (or selected) raw JSON files and write processed output.

    Returns (success_count, skip_or_fail_count).
    """
    raw_files = sorted(raw_dir.glob("*.json"))
    if not raw_files:
        logger.warning("No raw JSON files found in %s", raw_dir)
        return 0, 0

    if tickers:
        upper = {t.upper() for t in tickers}
        raw_files = [f for f in raw_files if f.stem.upper() in upper]

    success, failed = 0, 0

    for filepath in raw_files:
        ticker = filepath.stem
        logger.info("Parsing %s ...", ticker)

        raw = load_raw_json(filepath)
        if raw is None:
            failed += 1
            continue

        parsed = parse_company(raw)
        total = parsed["metadata"]["total_line_items"]
        years = len(parsed["metadata"]["fiscal_years"])
        logger.info(
            "  %s: %d line items across %d fiscal years",
            ticker,
            total,
            years,
        )

        save_processed_json(parsed, ticker, output_dir)
        success += 1

    return success, failed


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse raw EDGAR XBRL JSON into structured financial statements.",
    )
    parser.add_argument(
        "--ticker",
        nargs="+",
        metavar="SYM",
        help="Parse only these ticker(s).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help=f"Directory containing raw JSON files (default: {RAW_DATA_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help=f"Output directory (default: {PROCESSED_DATA_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    ok, fail = parse_all(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        tickers=args.ticker,
    )
    logger.info("Done. %d parsed, %d skipped/failed.", ok, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
