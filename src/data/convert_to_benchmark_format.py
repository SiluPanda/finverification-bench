"""Convert XBRL-parsed financial data to the flat format expected by error injection.

The parse_financials.py output uses nested {line_items → {label → {periods → ...}}}
structure from XBRL.  The error injection system expects a flat dict with standardized
field names.  This script bridges the two.

Usage:
    python -m src.data.convert_to_benchmark_format
    python -m src.data.convert_to_benchmark_format --ticker AAPL
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONVERTED_DIR = PROJECT_ROOT / "data" / "converted"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XBRL label → flat field name mappings
# ---------------------------------------------------------------------------

# Maps the readable labels (from parse_financials.py) to the flat field names
# expected by error_injection.py.

IS_LABEL_TO_FIELD: Dict[str, str] = {
    "Total Revenue": "revenue",
    "Revenue from Contracts": "revenue",
    "Net Sales Revenue": "revenue",
    "Cost of Goods Sold": "cost_of_goods_sold",
    "Cost of Revenue": "cost_of_goods_sold",
    "Gross Profit": "gross_profit",
    "Operating Expenses": "operating_expenses",
    "SG&A Expense": "sga_expense",
    "R&D Expense": "rd_expense",
    "Operating Income (Loss)": "operating_income",
    "Interest Expense": "interest_expense",
    "Income Tax Expense": "income_tax_expense",
    "Net Income (Loss)": "net_income",
    "Earnings Per Share (Basic)": "eps_basic",
    "Earnings Per Share (Diluted)": "eps_diluted",
    "Weighted Avg Diluted Shares Outstanding": "diluted_shares",
}

BS_LABEL_TO_FIELD: Dict[str, str] = {
    "Total Assets": "total_assets",
    "Total Liabilities": "total_liabilities",
    "Total Stockholders Equity": "total_equity",
    "Total Liabilities and Equity": "total_liabilities_and_equity",
    "Cash and Cash Equivalents": "cash_and_equivalents",
    "Short-Term Investments": "short_term_investments",
    "Accounts Receivable": "accounts_receivable",
    "Inventory": "inventory",
    "Total Current Assets": "total_current_assets",
    "Property, Plant and Equipment (net)": "property_plant_equipment",
    "Goodwill": "goodwill",
    "Intangible Assets (net)": "intangible_assets",
    "Total Current Liabilities": "total_current_liabilities",
    "Long-Term Debt": "long_term_debt",
    "Long-Term Debt (non-current)": "long_term_debt",
    "Retained Earnings (Accumulated Deficit)": "retained_earnings",
    "Common Shares Outstanding": "shares_outstanding",
}

CFS_LABEL_TO_FIELD: Dict[str, str] = {
    "Cash from Operating Activities": "cash_from_operations",
    "Cash from Investing Activities": "cash_from_investing",
    "Cash from Financing Activities": "cash_from_financing",
    "Net Change in Cash": "net_change_in_cash",
    "Net Change in Cash (legacy)": "net_change_in_cash",
    "Depreciation & Amortization": "depreciation_amortization",
    "Capital Expenditures": "capital_expenditures",
    "Dividends Paid": "dividends_paid",
    "Common Dividends Paid": "dividends_paid",
    "Share Repurchases": "share_repurchases",
    "Proceeds from Debt Issuance": "debt_issuance",
    "Repayments of Long-Term Debt": "debt_repayment",
}


def _get_value_for_period(
    line_items: Dict[str, Any],
    label: str,
    fiscal_year: str,
) -> Optional[float]:
    """Get the value of a line item for a specific fiscal year."""
    item = line_items.get(label)
    if item is None:
        return None
    periods = item.get("periods", {})
    period_data = periods.get(fiscal_year)
    if period_data is None:
        return None
    val = period_data.get("value")
    if val is not None:
        return float(val)
    return None


def _extract_flat_statement(
    line_items: Dict[str, Any],
    label_to_field: Dict[str, str],
    fiscal_year: str,
) -> Dict[str, float]:
    """Extract a flat dict of field_name → value for a fiscal year."""
    result: Dict[str, float] = {}
    for label, field_name in label_to_field.items():
        val = _get_value_for_period(line_items, label, fiscal_year)
        if val is not None and field_name not in result:
            result[field_name] = val
    return result


def _find_latest_complete_fy(parsed: Dict[str, Any]) -> Optional[str]:
    """Find the most recent fiscal year with data across all three statements."""
    all_fys: set = set()
    for stmt_key in ("balance_sheet", "income_statement", "cash_flow_statement"):
        items = parsed.get(stmt_key, {}).get("line_items", {})
        for item_data in items.values():
            for fy in item_data.get("periods", {}).keys():
                if fy.startswith("FY"):
                    all_fys.add(fy)

    # For each FY, check it has data in all three statements
    for fy in sorted(all_fys, reverse=True):
        has_all = True
        for stmt_key in ("balance_sheet", "income_statement", "cash_flow_statement"):
            items = parsed.get(stmt_key, {}).get("line_items", {})
            has_data = any(
                fy in item_data.get("periods", {})
                for item_data in items.values()
            )
            if not has_data:
                has_all = False
                break
        if has_all:
            return fy
    return None


def _compute_derived_fields(stmts: Dict[str, Any]) -> None:
    """Compute derived fields that may be missing from XBRL but needed for injection."""
    is_data = stmts.get("income_statement", {})
    bs_cy = stmts.get("balance_sheet", {}).get("current_year", {})
    cfs = stmts.get("cash_flow_statement", {})

    # Gross profit = revenue - COGS (if not present)
    if "gross_profit" not in is_data and "revenue" in is_data and "cost_of_goods_sold" in is_data:
        is_data["gross_profit"] = is_data["revenue"] - abs(is_data["cost_of_goods_sold"])

    # Operating expenses = SGA + R&D (if not present but components are)
    if "operating_expenses" not in is_data:
        opex = 0
        has_any = False
        for field in ("sga_expense", "rd_expense"):
            if field in is_data:
                opex += abs(is_data[field])
                has_any = True
        if has_any:
            is_data["operating_expenses"] = -opex

    # Operating income = gross_profit - operating_expenses (if not present)
    if "operating_income" not in is_data and "gross_profit" in is_data:
        opex = abs(is_data.get("operating_expenses", 0))
        dep = abs(is_data.get("depreciation_amortization", 0))
        is_data["operating_income"] = is_data["gross_profit"] - opex - dep

    # Income before tax = operating_income - interest_expense
    if "income_before_tax" not in is_data and "operating_income" in is_data:
        interest = is_data.get("interest_expense", 0)
        is_data["income_before_tax"] = is_data["operating_income"] + interest

    # Net income on CFS should match IS net income (copy if missing)
    if "net_income" not in cfs and "net_income" in is_data:
        cfs["net_income"] = is_data["net_income"]

    # Changes in working capital (approximate if not available)
    if "changes_in_working_capital" not in cfs and "cash_from_operations" in cfs:
        ni = cfs.get("net_income", 0)
        da = cfs.get("depreciation_amortization", 0)
        cfo = cfs["cash_from_operations"]
        cfs["changes_in_working_capital"] = cfo - ni - da

    # Beginning and ending cash
    if "ending_cash" not in cfs and "cash_and_equivalents" in bs_cy:
        cfs["ending_cash"] = bs_cy["cash_and_equivalents"]

    # Total liabilities = Total L&E - Total Equity (if not directly available)
    if "total_liabilities" not in bs_cy:
        if "total_liabilities_and_equity" in bs_cy and "total_equity" in bs_cy:
            bs_cy["total_liabilities"] = (
                bs_cy["total_liabilities_and_equity"] - bs_cy["total_equity"]
            )
        elif "total_assets" in bs_cy and "total_equity" in bs_cy:
            bs_cy["total_liabilities"] = (
                bs_cy["total_assets"] - bs_cy["total_equity"]
            )

    # Total liabilities and equity
    if "total_liabilities_and_equity" not in bs_cy:
        if "total_liabilities" in bs_cy and "total_equity" in bs_cy:
            bs_cy["total_liabilities_and_equity"] = (
                bs_cy["total_liabilities"] + bs_cy["total_equity"]
            )

    # Short-term debt (approximate from current liabilities if missing)
    if "accounts_payable" not in bs_cy and "total_current_liabilities" in bs_cy:
        bs_cy["accounts_payable"] = bs_cy["total_current_liabilities"] * 0.4
    if "short_term_debt" not in bs_cy and "total_current_liabilities" in bs_cy:
        bs_cy["short_term_debt"] = (
            bs_cy["total_current_liabilities"]
            - bs_cy.get("accounts_payable", 0)
        )


def convert_company(
    parsed: Dict[str, Any],
    ticker: str,
    scale: float = 1e6,
) -> Optional[Dict[str, Any]]:
    """Convert a parsed XBRL company to the flat benchmark format.

    Parameters
    ----------
    parsed : dict
        Output of parse_financials.py for one company.
    ticker : str
        Company ticker symbol.
    scale : float
        Divide all monetary values by this factor (default: 1e6 for millions).

    Returns
    -------
    dict or None
        Flat financial statements in the error-injection format, or None if
        insufficient data.
    """
    fy = _find_latest_complete_fy(parsed)
    if fy is None:
        logger.warning("No complete fiscal year found for %s", ticker)
        return None

    entity_name = parsed.get("metadata", {}).get("entity_name", ticker)

    # Extract flat statements for the target FY
    bs_items = parsed.get("balance_sheet", {}).get("line_items", {})
    is_items = parsed.get("income_statement", {}).get("line_items", {})
    cfs_items = parsed.get("cash_flow_statement", {}).get("line_items", {})

    bs_current = _extract_flat_statement(bs_items, BS_LABEL_TO_FIELD, fy)
    is_flat = _extract_flat_statement(is_items, IS_LABEL_TO_FIELD, fy)
    cfs_flat = _extract_flat_statement(cfs_items, CFS_LABEL_TO_FIELD, fy)

    # Get prior year for balance sheet (for YoY errors)
    fy_num = int(fy.replace("FY", ""))
    prior_fy = f"FY{fy_num - 1}"
    bs_prior = _extract_flat_statement(bs_items, BS_LABEL_TO_FIELD, prior_fy)

    # Scale monetary values to millions
    def scale_dict(d: Dict[str, float]) -> Dict[str, float]:
        skip_fields = {"eps_basic", "eps_diluted", "shares_outstanding", "diluted_shares"}
        return {
            k: round(v / scale, 1) if k not in skip_fields else v
            for k, v in d.items()
        }

    bs_current_scaled = scale_dict(bs_current)
    bs_prior_scaled = scale_dict(bs_prior) if bs_prior else {}
    is_scaled = scale_dict(is_flat)
    cfs_scaled = scale_dict(cfs_flat)

    result: Dict[str, Any] = {
        "company": entity_name,
        "ticker": ticker,
        "period": fy,
        "currency": "USD",
        "unit": "millions",
        "income_statement": is_scaled,
        "balance_sheet": {
            "current_year": bs_current_scaled,
        },
        "cash_flow_statement": cfs_scaled,
    }

    if bs_prior_scaled:
        result["balance_sheet"]["prior_year"] = bs_prior_scaled

    # Compute derived fields to ensure all injection targets exist
    _compute_derived_fields(result)

    # Also derive fields for prior year balance sheet
    if bs_prior_scaled:
        prior_bs = result["balance_sheet"]["prior_year"]
        if "total_liabilities" not in prior_bs:
            if "total_liabilities_and_equity" in prior_bs and "total_equity" in prior_bs:
                prior_bs["total_liabilities"] = (
                    prior_bs["total_liabilities_and_equity"] - prior_bs["total_equity"]
                )
            elif "total_assets" in prior_bs and "total_equity" in prior_bs:
                prior_bs["total_liabilities"] = (
                    prior_bs["total_assets"] - prior_bs["total_equity"]
                )
        if "total_liabilities_and_equity" not in prior_bs:
            if "total_liabilities" in prior_bs and "total_equity" in prior_bs:
                prior_bs["total_liabilities_and_equity"] = (
                    prior_bs["total_liabilities"] + prior_bs["total_equity"]
                )

    # Verify minimum data quality
    min_is_fields = {"revenue", "net_income"}
    min_bs_fields = {"total_assets", "total_liabilities"}
    min_cfs_fields = {"cash_from_operations"}

    if not min_is_fields.issubset(result["income_statement"].keys()):
        logger.warning(
            "%s (%s): Missing IS fields %s — skipping",
            ticker, fy,
            min_is_fields - result["income_statement"].keys(),
        )
        return None
    if not min_bs_fields.issubset(result["balance_sheet"]["current_year"].keys()):
        logger.warning(
            "%s (%s): Missing BS fields %s — skipping",
            ticker, fy,
            min_bs_fields - result["balance_sheet"]["current_year"].keys(),
        )
        return None
    if not min_cfs_fields.issubset(result["cash_flow_statement"].keys()):
        logger.warning(
            "%s (%s): Missing CFS fields %s — skipping",
            ticker, fy,
            min_cfs_fields - result["cash_flow_statement"].keys(),
        )
        return None

    n_fields = (
        len(result["income_statement"])
        + len(result["balance_sheet"]["current_year"])
        + len(result["cash_flow_statement"])
    )
    logger.info(
        "  %s (%s): %d fields extracted (IS=%d, BS=%d, CFS=%d)",
        ticker, fy, n_fields,
        len(result["income_statement"]),
        len(result["balance_sheet"]["current_year"]),
        len(result["cash_flow_statement"]),
    )

    return result


def convert_all(
    processed_dir: Path = PROCESSED_DIR,
    output_dir: Path = CONVERTED_DIR,
    tickers: Optional[List[str]] = None,
) -> Tuple[int, int]:
    """Convert all processed files to benchmark format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(processed_dir.glob("*.json"))

    if tickers:
        upper = {t.upper() for t in tickers}
        files = [f for f in files if f.stem.upper() in upper]

    success, failed = 0, 0
    for fp in files:
        ticker = fp.stem
        with open(fp) as fh:
            parsed = json.load(fh)

        converted = convert_company(parsed, ticker)
        if converted is None:
            failed += 1
            continue

        out_path = output_dir / f"{ticker}.json"
        with open(out_path, "w") as fh:
            json.dump(converted, fh, indent=2)
        success += 1

    logger.info("Converted %d companies, %d failed/skipped.", success, failed)
    return success, failed


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert XBRL-parsed data to benchmark injection format.",
    )
    parser.add_argument("--ticker", nargs="+", help="Convert specific tickers")
    parser.add_argument(
        "--input-dir", type=Path, default=PROCESSED_DIR,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=CONVERTED_DIR,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    convert_all(args.input_dir, args.output_dir, args.ticker)


if __name__ == "__main__":
    main()
