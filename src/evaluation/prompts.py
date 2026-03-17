"""Prompt templates for LLM financial statement verification.

Defines three prompting strategies:
  1. Zero-shot  -- direct instruction with no examples
  2. Few-shot   -- two demonstration examples (one clean, one error-injected)
  3. Chain-of-thought (CoT) -- step-by-step verification procedure

Each public function accepts the formatted financial statement text and returns
the full prompt string ready for submission to an LLM.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Shared response-format instructions
# ---------------------------------------------------------------------------

_RESPONSE_FORMAT = """\
Respond in the following JSON format (do NOT include any text outside the JSON):
{
  "has_error": true or false,
  "error_location": "dot-path to the erroneous field, e.g. income_statement.revenue" or null,
  "explanation": "brief explanation of what is inconsistent and why"
}"""


# ---------------------------------------------------------------------------
# Few-shot demonstration examples
# ---------------------------------------------------------------------------

_FEWSHOT_CLEAN_STATEMENTS = """\
Acme Corp — Financial Statements (in millions of USD)
Period: FY2024

INCOME STATEMENT
==========================================================
  Revenue                                          1,000
  Cost of Goods Sold                                -600
  Gross Profit                                       400
  --------------------------------------------------------
  Operating Expenses                                -200
  Depreciation & Amortization                        -50
  Operating Income                                   150
  --------------------------------------------------------
  Interest Expense                                   -10
  Income Before Tax                                  140
  --------------------------------------------------------
  Income Tax Expense                                 -35
  Net Income                                         105

BALANCE SHEET
==============================================================================
                                            Current          Prior
  --------------------------------------------------------------------------
  Cash and Equivalents                          250            220
  Accounts Receivable                           120            110
  Inventory                                      80             70
  Total Current Assets                          450            400
  --------------------------------------------------------------------------
  Property, Plant & Equipment                   500            480
  Total Assets                                  950            880
  --------------------------------------------------------------------------

  Accounts Payable                               90             80
  Short-Term Debt                                60             50
  Total Current Liabilities                     150            130
  --------------------------------------------------------------------------
  Long-Term Debt                                200            230
  Total Liabilities                             350            360
  Retained Earnings                             500            420
  Total Equity                                  600            520
  Total Liabilities & Equity                    950            880

CASH FLOW STATEMENT
==========================================================
  Net Income                                         105
  Depreciation & Amortization                         50
  Changes in Working Capital                         -20
  Cash from Operations                               135
  --------------------------------------------------------
  Capital Expenditures                               -60
  Cash from Investing                                -60
  --------------------------------------------------------
  Debt Repayment                                     -30
  Dividends Paid                                     -15
  Cash from Financing                                -45
  --------------------------------------------------------
  Net Change in Cash                                  30
  Beginning Cash                                     220
  Ending Cash                                        250"""

_FEWSHOT_CLEAN_RESPONSE = """\
{
  "has_error": false,
  "error_location": null,
  "explanation": "All arithmetic totals are correct. Net income (105) matches between IS and CFS. Ending cash (250) matches between CFS and BS. Total assets equal total liabilities and equity (950). Prior-year balances are consistent with current-year opening positions."
}"""

_FEWSHOT_ERROR_STATEMENTS = """\
BetaCo Inc — Financial Statements (in millions of USD)
Period: FY2024

INCOME STATEMENT
==========================================================
  Revenue                                          2,000
  Cost of Goods Sold                              -1,100
  Gross Profit                                       900
  --------------------------------------------------------
  Operating Expenses                                -400
  Depreciation & Amortization                       -100
  Operating Income                                   400
  --------------------------------------------------------
  Interest Expense                                   -25
  Income Before Tax                                  375
  --------------------------------------------------------
  Income Tax Expense                                 -94
  Net Income                                         281

BALANCE SHEET
==============================================================================
                                            Current          Prior
  --------------------------------------------------------------------------
  Cash and Equivalents                          500            420
  Accounts Receivable                           250            220
  Inventory                                     150            130
  Total Current Assets                          900            770
  --------------------------------------------------------------------------
  Property, Plant & Equipment                 1,100          1,050
  Total Assets                                2,000          1,820
  --------------------------------------------------------------------------

  Accounts Payable                              180            160
  Short-Term Debt                               120            100
  Total Current Liabilities                     300            260
  --------------------------------------------------------------------------
  Long-Term Debt                                400            450
  Total Liabilities                             700            710
  Retained Earnings                           1,100            910
  Total Equity                                1,300          1,110
  Total Liabilities & Equity                  2,000          1,820

CASH FLOW STATEMENT
==========================================================
  Net Income                                         295
  Depreciation & Amortization                        100
  Changes in Working Capital                         -30
  Cash from Operations                               365
  --------------------------------------------------------
  Capital Expenditures                              -150
  Cash from Investing                               -150
  --------------------------------------------------------
  Debt Repayment                                     -50
  Dividends Paid                                     -85
  Cash from Financing                               -135
  --------------------------------------------------------
  Net Change in Cash                                  80
  Beginning Cash                                     420
  Ending Cash                                        500"""

_FEWSHOT_ERROR_RESPONSE = """\
{
  "has_error": true,
  "error_location": "cash_flow_statement.net_income",
  "explanation": "Net income on the cash flow statement is 295, but net income on the income statement is 281. These two figures must agree (CFS starts with IS net income under the indirect method). The CFS net income appears to have been overstated by 14 (approximately 5%). Additionally, the CFS cash from operations (365) does not equal 295 + 100 + (-30) = 365 only because net income was independently modified, but the cross-statement linkage between IS and CFS is broken."
}"""


# ---------------------------------------------------------------------------
# Zero-shot prompt
# ---------------------------------------------------------------------------

def build_zero_shot_prompt(formatted_statements: str) -> str:
    """Build a zero-shot verification prompt.

    Parameters
    ----------
    formatted_statements:
        Human-readable financial statements (the output of
        ``dataset_builder.format_statements``).

    Returns
    -------
    str
        Complete prompt string.
    """
    return f"""\
You are an expert financial auditor. Your task is to verify whether the
following financial statements are internally consistent.

Check for:
- Arithmetic errors (do line items sum to their stated totals?)
- Cross-statement linkage errors (does net income match between IS and CFS?
  does ending cash on CFS match the balance sheet? etc.)
- Year-over-year inconsistencies (does the prior-year ending balance equal the
  current-year opening balance?)
- Magnitude / rounding errors (are any values suspiciously different from what
  the arithmetic implies?)

FINANCIAL STATEMENTS:
{formatted_statements}

{_RESPONSE_FORMAT}"""


# ---------------------------------------------------------------------------
# Few-shot prompt
# ---------------------------------------------------------------------------

def build_few_shot_prompt(formatted_statements: str) -> str:
    """Build a few-shot verification prompt with two demonstration examples.

    The first example is a clean statement set (no errors); the second contains
    a cross-statement linkage error between the income statement and the cash
    flow statement.

    Parameters
    ----------
    formatted_statements:
        Human-readable financial statements to verify.

    Returns
    -------
    str
        Complete prompt string.
    """
    return f"""\
You are an expert financial auditor. Your task is to verify whether financial
statements are internally consistent. Below are two worked examples followed
by a new set of statements for you to verify.

--- EXAMPLE 1 (clean statements) ---

FINANCIAL STATEMENTS:
{_FEWSHOT_CLEAN_STATEMENTS}

YOUR ANALYSIS:
{_FEWSHOT_CLEAN_RESPONSE}

--- EXAMPLE 2 (statements with an error) ---

FINANCIAL STATEMENTS:
{_FEWSHOT_ERROR_STATEMENTS}

YOUR ANALYSIS:
{_FEWSHOT_ERROR_RESPONSE}

--- YOUR TASK ---

Now verify the following financial statements using the same approach.

FINANCIAL STATEMENTS:
{formatted_statements}

{_RESPONSE_FORMAT}"""


# ---------------------------------------------------------------------------
# Chain-of-thought prompt
# ---------------------------------------------------------------------------

def build_cot_prompt(formatted_statements: str) -> str:
    """Build a chain-of-thought verification prompt.

    Instructs the model to work through each class of accounting relationship
    step by step before reaching a conclusion.  This mirrors the systematic
    verification procedures used by human auditors.

    Parameters
    ----------
    formatted_statements:
        Human-readable financial statements to verify.

    Returns
    -------
    str
        Complete prompt string.
    """
    return f"""\
You are an expert financial auditor performing a systematic verification of
financial statements. Work through each check step by step, showing your
calculations, before reaching a final conclusion.

FINANCIAL STATEMENTS:
{formatted_statements}

STEP-BY-STEP VERIFICATION:

Step 1 — Income Statement Arithmetic
  - Does Revenue + COGS = Gross Profit?
  - Does Gross Profit + Operating Expenses + D&A = Operating Income?
  - Does Operating Income + Interest Expense = Income Before Tax?
  - Does Income Before Tax + Tax Expense = Net Income?

Step 2 — Balance Sheet Arithmetic
  - Does Cash + AR + Inventory = Total Current Assets?
  - Does Total Current Assets + PP&E = Total Assets?
  - Does AP + Short-Term Debt = Total Current Liabilities?
  - Does Total Current Liabilities + Long-Term Debt = Total Liabilities?
  - Does Total Liabilities + Total Equity = Total Liabilities & Equity?
  - Does Total Assets = Total Liabilities & Equity?

Step 3 — Cash Flow Statement Arithmetic
  - Does Net Income + D&A + Working Capital Changes = Cash from Operations?
  - Do subtotals (operations + investing + financing) = Net Change in Cash?
  - Does Beginning Cash + Net Change in Cash = Ending Cash?

Step 4 — Cross-Statement Linkages
  - Does Net Income on IS = Net Income on CFS?
  - Does Ending Cash on CFS = Cash on BS (current year)?
  - Does D&A on IS = D&A add-back on CFS?
  - Is the change in Retained Earnings = Net Income - Dividends?

Step 5 — Year-over-Year Consistency
  - Do prior-year values match expectations for a comparative presentation?
  - Is Beginning Cash on CFS = prior-year Ending Cash on BS?

After completing all checks, provide your final answer.

{_RESPONSE_FORMAT}"""


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

STRATEGY_BUILDERS = {
    "zero_shot": build_zero_shot_prompt,
    "few_shot": build_few_shot_prompt,
    "cot": build_cot_prompt,
}


def build_prompt(
    strategy: str,
    formatted_statements: str,
) -> str:
    """Build a prompt using the named strategy.

    Parameters
    ----------
    strategy:
        One of ``"zero_shot"``, ``"few_shot"``, or ``"cot"``.
    formatted_statements:
        Human-readable financial statements to verify.

    Returns
    -------
    str
        Complete prompt string.

    Raises
    ------
    ValueError
        If *strategy* is not recognised.
    """
    builder = STRATEGY_BUILDERS.get(strategy)
    if builder is None:
        valid = ", ".join(sorted(STRATEGY_BUILDERS))
        raise ValueError(
            f"Unknown prompt strategy {strategy!r}. Valid: {valid}"
        )
    return builder(formatted_statements)
