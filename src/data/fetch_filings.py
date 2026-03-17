"""
Fetch structured financial data from SEC EDGAR XBRL API.

Uses the company facts endpoint to pull all reported XBRL facts for a set
of S&P 500 companies and saves the raw JSON responses to data/raw/.

SEC EDGAR API docs:
  https://www.sec.gov/edgar/sec-api-documentation

Usage:
    python -m src.data.fetch_filings            # fetch all companies
    python -m src.data.fetch_filings --ticker AAPL MSFT  # fetch specific tickers
    python -m src.data.fetch_filings --max 5    # fetch first 5 only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# SEC EDGAR configuration
# ---------------------------------------------------------------------------
EDGAR_BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
USER_AGENT = "FinVerBench research@finverbench.org"
REQUEST_INTERVAL = 0.1  # 10 requests/second max → 100 ms between requests

# ---------------------------------------------------------------------------
# S&P 500 subset — 50 major companies with their CIK numbers.
# CIKs are zero-padded to 10 digits as required by the EDGAR API.
# ---------------------------------------------------------------------------
SP500_COMPANIES: List[Dict[str, str]] = [
    {"ticker": "AAPL",  "cik": "0000320193", "name": "Apple Inc."},
    {"ticker": "MSFT",  "cik": "0000789019", "name": "Microsoft Corporation"},
    {"ticker": "AMZN",  "cik": "0001018724", "name": "Amazon.com Inc."},
    {"ticker": "GOOGL", "cik": "0001652044", "name": "Alphabet Inc."},
    {"ticker": "META",  "cik": "0001326801", "name": "Meta Platforms Inc."},
    {"ticker": "BRK.B", "cik": "0001067983", "name": "Berkshire Hathaway Inc."},
    {"ticker": "JNJ",   "cik": "0000200406", "name": "Johnson & Johnson"},
    {"ticker": "V",     "cik": "0001403161", "name": "Visa Inc."},
    {"ticker": "JPM",   "cik": "0000019617", "name": "JPMorgan Chase & Co."},
    {"ticker": "PG",    "cik": "0000080424", "name": "Procter & Gamble Company"},
    {"ticker": "UNH",   "cik": "0000731766", "name": "UnitedHealth Group Inc."},
    {"ticker": "HD",    "cik": "0000354950", "name": "The Home Depot Inc."},
    {"ticker": "MA",    "cik": "0001141391", "name": "Mastercard Inc."},
    {"ticker": "NVDA",  "cik": "0001045810", "name": "NVIDIA Corporation"},
    {"ticker": "DIS",   "cik": "0001744489", "name": "The Walt Disney Company"},
    {"ticker": "BAC",   "cik": "0000070858", "name": "Bank of America Corporation"},
    {"ticker": "XOM",   "cik": "0000034088", "name": "Exxon Mobil Corporation"},
    {"ticker": "PFE",   "cik": "0000078003", "name": "Pfizer Inc."},
    {"ticker": "CSCO",  "cik": "0000858877", "name": "Cisco Systems Inc."},
    {"ticker": "KO",    "cik": "0000021344", "name": "The Coca-Cola Company"},
    {"ticker": "PEP",   "cik": "0000077476", "name": "PepsiCo Inc."},
    {"ticker": "TMO",   "cik": "0000097745", "name": "Thermo Fisher Scientific Inc."},
    {"ticker": "COST",  "cik": "0000909832", "name": "Costco Wholesale Corporation"},
    {"ticker": "ABT",   "cik": "0000001800", "name": "Abbott Laboratories"},
    {"ticker": "CRM",   "cik": "0001108524", "name": "Salesforce Inc."},
    {"ticker": "AVGO",  "cik": "0001649338", "name": "Broadcom Inc."},
    {"ticker": "NKE",   "cik": "0000320187", "name": "NIKE Inc."},
    {"ticker": "MRK",   "cik": "0000310158", "name": "Merck & Co. Inc."},
    {"ticker": "WMT",   "cik": "0000104169", "name": "Walmart Inc."},
    {"ticker": "CVX",   "cik": "0000093410", "name": "Chevron Corporation"},
    {"ticker": "LLY",   "cik": "0000059478", "name": "Eli Lilly and Company"},
    {"ticker": "ADBE",  "cik": "0000796343", "name": "Adobe Inc."},
    {"ticker": "ORCL",  "cik": "0001341439", "name": "Oracle Corporation"},
    {"ticker": "CMCSA", "cik": "0001166691", "name": "Comcast Corporation"},
    {"ticker": "ACN",   "cik": "0001281761", "name": "Accenture plc"},
    {"ticker": "INTC",  "cik": "0000050863", "name": "Intel Corporation"},
    {"ticker": "VZ",    "cik": "0000732712", "name": "Verizon Communications Inc."},
    {"ticker": "T",     "cik": "0000732717", "name": "AT&T Inc."},
    {"ticker": "MCD",   "cik": "0000063908", "name": "McDonald's Corporation"},
    {"ticker": "TXN",   "cik": "0000097476", "name": "Texas Instruments Inc."},
    {"ticker": "HON",   "cik": "0000773840", "name": "Honeywell International Inc."},
    {"ticker": "NEE",   "cik": "0000753308", "name": "NextEra Energy Inc."},
    {"ticker": "UPS",   "cik": "0001090727", "name": "United Parcel Service Inc."},
    {"ticker": "PM",    "cik": "0001413329", "name": "Philip Morris International Inc."},
    {"ticker": "LOW",   "cik": "0000060667", "name": "Lowe's Companies Inc."},
    {"ticker": "GS",    "cik": "0000886982", "name": "The Goldman Sachs Group Inc."},
    {"ticker": "CAT",   "cik": "0000018230", "name": "Caterpillar Inc."},
    {"ticker": "BA",    "cik": "0000012927", "name": "The Boeing Company"},
    {"ticker": "AMGN",  "cik": "0000318154", "name": "Amgen Inc."},
    {"ticker": "GE",    "cik": "0000040554", "name": "General Electric Company"},
]

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
# Core logic
# ---------------------------------------------------------------------------

def _build_session() -> requests.Session:
    """Return a requests.Session with the required SEC User-Agent header."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    })
    return session


def fetch_company_facts(
    session: requests.Session,
    cik: str,
    ticker: str,
    name: str,
) -> Optional[dict]:
    """Fetch the full XBRL company-facts JSON for a single company.

    Returns the parsed JSON dict on success, or None on failure.
    """
    url = EDGAR_BASE_URL.format(cik=cik)
    logger.info("Fetching %s (%s) from %s", ticker, name, url)

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as exc:
        logger.error(
            "HTTP %s for %s (%s): %s",
            exc.response.status_code if exc.response is not None else "???",
            ticker,
            cik,
            exc,
        )
    except requests.exceptions.ConnectionError as exc:
        logger.error("Connection error for %s (%s): %s", ticker, cik, exc)
    except requests.exceptions.Timeout:
        logger.error("Timeout for %s (%s)", ticker, cik)
    except requests.exceptions.RequestException as exc:
        logger.error("Request failed for %s (%s): %s", ticker, cik, exc)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON for %s (%s): %s", ticker, cik, exc)

    return None


def save_raw_json(data: dict, ticker: str, output_dir: Path) -> Path:
    """Write raw JSON to *output_dir*/<ticker>.json and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{ticker}.json"
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Saved %s (%d bytes)", filepath, filepath.stat().st_size)
    return filepath


def resolve_companies(
    tickers: Optional[List[str]] = None,
    max_companies: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Return the list of companies to fetch, filtered by CLI args."""
    companies = SP500_COMPANIES

    if tickers:
        upper = {t.upper() for t in tickers}
        companies = [c for c in companies if c["ticker"] in upper]
        missing = upper - {c["ticker"] for c in companies}
        if missing:
            logger.warning("Tickers not in hardcoded list: %s", missing)

    if max_companies is not None:
        companies = companies[:max_companies]

    return companies


def fetch_all(
    companies: List[Dict[str, str]],
    output_dir: Path = RAW_DATA_DIR,
) -> Tuple[int, int]:
    """Fetch company facts for every company in *companies*.

    Returns (success_count, failure_count).
    """
    session = _build_session()
    success, failure = 0, 0
    last_request_time = 0.0

    for company in companies:
        # Enforce rate limit: at least REQUEST_INTERVAL seconds between requests.
        elapsed = time.monotonic() - last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)

        last_request_time = time.monotonic()
        data = fetch_company_facts(
            session,
            cik=company["cik"],
            ticker=company["ticker"],
            name=company["name"],
        )

        if data is not None:
            save_raw_json(data, company["ticker"], output_dir)
            success += 1
        else:
            failure += 1

    return success, failure


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch SEC EDGAR XBRL company facts for S&P 500 companies.",
    )
    parser.add_argument(
        "--ticker",
        nargs="+",
        metavar="SYM",
        help="Fetch only these ticker(s). E.g. --ticker AAPL MSFT",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        dest="max_companies",
        help="Fetch at most N companies (useful for testing).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help=f"Directory for raw JSON files (default: {RAW_DATA_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    companies = resolve_companies(
        tickers=args.ticker,
        max_companies=args.max_companies,
    )

    if not companies:
        logger.error("No companies to fetch.")
        return 1

    logger.info("Fetching %d companies...", len(companies))
    ok, fail = fetch_all(companies, output_dir=args.output_dir)
    logger.info("Done. %d succeeded, %d failed.", ok, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
