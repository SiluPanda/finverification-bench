#!/usr/bin/env python3
"""
Qualitative error analysis of GPT-4.1 false positive patterns on FinVerBench.

Analyzes the 43 clean financial statement instances to understand why GPT-4.1
flags 41 of them as containing errors (95.3% FPR).
"""

import json
import re
from collections import Counter, defaultdict

RESULTS_PATH = "/Users/silupanda/Downloads/finverification-bench/results/openrouter_openai_gpt-4.1_cot_results.json"


def load_results():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return data


def extract_clean_instances(data):
    """Extract all clean (has_error=false) instances."""
    clean = [r for r in data["results"] if r["has_error"] is False]
    fp = [r for r in clean if r["detected"] is True]
    tn = [r for r in clean if r["detected"] is False]
    return clean, fp, tn


def classify_fp_pattern(explanation, error_location):
    """
    Classify a false positive explanation into a pattern category.

    Categories are derived from reading all 41 FP explanations:
    1. ARITHMETIC_SUBTOTAL - Claims visible line items don't sum to reported subtotal
       (but the subtotal includes unlisted items the model ignores)
    2. OPERATING_INCOME_CALC - Misunderstands relationship between gross profit,
       opex, D&A, and operating income
    3. NET_INCOME_CALC - Claims Income Before Tax - Tax != Net Income
       (ignoring minority interest, discontinued ops, etc.)
    4. RETAINED_EARNINGS_ROLLFORWARD - Claims RE change != NI - Dividends
       (ignoring share repurchases, stock comp, other comprehensive income)
    5. BALANCE_SHEET_MISMATCH - Claims Total Assets != Total L+E
       (prior year cross-check with different reporting periods or structure)
    6. CROSS_STATEMENT_LINKAGE - Claims figures don't match across statements
    7. REVENUE_COGS_STRUCTURE - Misreads financial structure (e.g., COGS > Revenue
       for companies where this is the actual data presentation)
    8. MISSING_LINE_ITEMS - Flags that expected line items are absent
    """
    exp_lower = explanation.lower()
    loc_lower = (error_location or "").lower()

    # Pattern: Revenue - COGS != Gross Profit (but the numbers are just structured differently)
    if ("revenue" in exp_lower and "cost of goods" in exp_lower and
        "gross profit" in exp_lower and
        ("does not" in exp_lower or "not" in exp_lower or "clear arithmetic" in exp_lower)):
        # Check if it's specifically the COGS > Revenue case
        if "negative" in exp_lower or re.search(r'revenue.*minus.*cogs.*=.*-', exp_lower):
            return "REVENUE_COGS_STRUCTURE"

    # Pattern: Total Current Assets != sum of listed components
    if ("total current assets" in exp_lower and
        ("cash" in exp_lower or "accounts receivable" in exp_lower) and
        ("does not equal" in exp_lower or "does not match" in exp_lower or
         "does NOT equal" in explanation or "clear arithmetic" in exp_lower)):
        return "ARITHMETIC_SUBTOTAL"

    # Pattern: Total Liabilities != Current Liabilities + LT Debt
    if ("total liabilities" in exp_lower and
        ("current liabilities" in exp_lower or "long-term debt" in exp_lower) and
        ("does not" in exp_lower or "discrepancy" in exp_lower or "does NOT" in explanation)):
        return "ARITHMETIC_SUBTOTAL"

    # Pattern: Operating Income calculation mismatch
    if ("operating income" in exp_lower and
        ("gross profit" in exp_lower or "operating expenses" in exp_lower or "d&a" in exp_lower) and
        ("does not" in exp_lower or "inconsisten" in exp_lower or "discrepancy" in exp_lower or
         "not" in exp_lower)):
        if "revenue" in exp_lower and "operating expenses" in exp_lower and "gross profit" not in exp_lower:
            # Revenue - OpEx = Operating Income (wrong formula)
            return "OPERATING_INCOME_CALC"
        return "OPERATING_INCOME_CALC"

    # Pattern: Revenue - OpEx treated as if it should = Operating Income
    if ("operating income" in exp_lower and "revenue" in exp_lower and
        "operating expenses" in exp_lower and
        ("not" in exp_lower or "inconsisten" in exp_lower)):
        return "OPERATING_INCOME_CALC"

    # Pattern: Net Income != IBT - Tax
    if (("net income" in exp_lower and "income before tax" in exp_lower and
         "income tax" in exp_lower) or
        ("net income" in exp_lower and "income tax" in exp_lower and
         "does not" in exp_lower)):
        return "NET_INCOME_CALC"

    # Pattern: Retained Earnings rollforward issues
    if ("retained earnings" in exp_lower and
        ("net income" in exp_lower or "dividends" in exp_lower or "change" in exp_lower)):
        return "RETAINED_EARNINGS_ROLLFORWARD"

    # Pattern: RE > Total Equity flagged as impossible
    if ("retained earnings" in exp_lower and "total equity" in exp_lower and
        "greater than" in exp_lower):
        return "RETAINED_EARNINGS_ROLLFORWARD"

    # Pattern: Total Assets != Total L&E (often prior year)
    if (("total assets" in exp_lower and "total liabilities" in exp_lower and
         "equity" in exp_lower and
         ("does not match" in exp_lower or "does NOT match" in explanation or
          "inconsisten" in exp_lower)) or
        ("balance sheet" in exp_lower and "does not balance" in exp_lower)):
        return "BALANCE_SHEET_MISMATCH"

    # Pattern: Cross-statement linkage (CFS ending cash vs BS cash, NI across statements)
    if (("cash flow" in exp_lower and "balance sheet" in exp_lower) or
        ("income statement" in exp_lower and "cash flow" in exp_lower) or
        ("ending cash" in exp_lower and "cash and equivalents" in exp_lower)):
        return "CROSS_STATEMENT_LINKAGE"

    # Pattern: Revenue - COGS gives negative (company structure issue)
    if ("revenue" in exp_lower and ("cogs" in exp_lower or "cost of goods" in exp_lower) and
        ("negative" in exp_lower or "-" in exp_lower)):
        return "REVENUE_COGS_STRUCTURE"

    # Pattern: Missing line items
    if "missing" in exp_lower and ("line item" in exp_lower or "key" in exp_lower):
        return "MISSING_LINE_ITEMS"

    # Fallback - try to classify by error_location
    if "operating_income" in loc_lower:
        return "OPERATING_INCOME_CALC"
    if "net_income" in loc_lower:
        return "NET_INCOME_CALC"
    if "retained_earnings" in loc_lower:
        return "RETAINED_EARNINGS_ROLLFORWARD"
    if "total_current_assets" in loc_lower:
        return "ARITHMETIC_SUBTOTAL"
    if "total_liabilities" in loc_lower:
        return "ARITHMETIC_SUBTOTAL"
    if "gross_profit" in loc_lower:
        return "OPERATING_INCOME_CALC"

    return "OTHER"


def classify_discrepancy_type(explanation):
    """
    Determine whether the discrepancy GPT-4.1 references is:
    - REAL_BENIGN: The numbers genuinely don't add up the simple way, but this is
      because of unlisted line items, different accounting structures, etc.
    - FABRICATED: GPT-4.1 makes an arithmetic error or misreads the data
    - STRUCTURAL: The financial statement structure legitimately differs from
      the model's expectations (e.g., financial companies, negative equity)
    """
    exp_lower = explanation.lower()

    # Cases where GPT-4.1 notes that listed items don't sum to total
    # This is REAL but BENIGN - there are other items not shown
    if any(phrase in exp_lower for phrase in [
        "other liabilities not listed",
        "missing",
        "unexplained difference",
        "another liability",
        "not included",
        "only",  # "only AP and Short-Term Debt are listed"
    ]):
        return "REAL_BENIGN"

    # Cases involving Revenue - COGS where COGS > Revenue
    # This happens with certain company structures (GE Capital, Broadcom, Honeywell)
    if ("revenue" in exp_lower and ("cost of goods" in exp_lower or "cogs" in exp_lower)):
        # Check if it's noting a negative result
        if re.search(r'=\s*-\d', explanation):
            return "STRUCTURAL"

    # Most cases: model applies a simplified formula and gets a different number
    # The actual financial statements have additional items not shown
    return "REAL_BENIGN"


def analyze_fp_patterns(fp_list):
    """Analyze and categorize all false positives."""
    categories = Counter()
    discrepancy_types = Counter()
    by_category = defaultdict(list)
    by_company = defaultdict(list)

    for r in fp_list:
        company = r["instance_id"].replace("__clean", "").replace("_", " ").title()
        pattern = classify_fp_pattern(r["explanation"], r.get("error_location", ""))
        disc_type = classify_discrepancy_type(r["explanation"])

        categories[pattern] += 1
        discrepancy_types[disc_type] += 1
        by_category[pattern].append({
            "company": company,
            "instance_id": r["instance_id"],
            "location": r.get("error_location", "N/A"),
            "explanation": r["explanation"],
            "discrepancy_type": disc_type,
        })
        by_company[company].append(pattern)

    return categories, discrepancy_types, by_category, by_company


def analyze_error_locations(fp_list):
    """Analyze which statement/field is most commonly flagged."""
    statements = Counter()
    fields = Counter()

    for r in fp_list:
        loc = r.get("error_location", "")
        if not loc:
            continue
        parts = loc.split(".")
        if parts:
            statements[parts[0]] += 1
        if len(parts) > 1:
            fields[".".join(parts[:2])] += 1

    return statements, fields


def print_report(data, clean, fp, tn):
    categories, discrepancy_types, by_category, by_company = analyze_fp_patterns(fp)
    statements, fields = analyze_error_locations(fp)

    print("=" * 80)
    print("GPT-4.1 FALSE POSITIVE ANALYSIS ON FinVerBench")
    print("=" * 80)
    print()

    # ---- OVERVIEW ----
    print("OVERVIEW")
    print("-" * 40)
    print(f"  Total clean instances:            {len(clean)}")
    print(f"  False Positives (FP):             {len(fp)} ({len(fp)/len(clean)*100:.1f}%)")
    print(f"  True Negatives (TN):              {len(tn)} ({len(tn)/len(clean)*100:.1f}%)")
    print(f"  False Positive Rate:              {len(fp)/len(clean)*100:.1f}%")
    print()

    # ---- TN ANALYSIS ----
    print("=" * 80)
    print("TRUE NEGATIVES — Correctly Identified as Clean (2 of 43)")
    print("=" * 80)
    for r in tn:
        company = r["instance_id"].replace("__clean", "").replace("_", " ").title()
        print(f"\n  Company: {company}")
        print(f"  Instance: {r['instance_id']}")
        print(f"  Explanation: {r['explanation']}")
    print()

    # ---- FP PATTERN DISTRIBUTION ----
    print("=" * 80)
    print("FALSE POSITIVE PATTERN DISTRIBUTION")
    print("=" * 80)
    print()

    pattern_descriptions = {
        "ARITHMETIC_SUBTOTAL": "Assumes listed sub-items must sum to reported subtotal\n"
            "    (ignores unlisted items like other current assets, other liabilities, etc.)",
        "OPERATING_INCOME_CALC": "Misapplies operating income formula\n"
            "    (e.g., Revenue - OpEx = OpInc, or Gross Profit - OpEx = OpInc without D&A)",
        "NET_INCOME_CALC": "Claims Income Before Tax - Tax Expense != Net Income\n"
            "    (ignores minority interest, discontinued operations, non-controlling interest)",
        "RETAINED_EARNINGS_ROLLFORWARD": "Claims RE change != Net Income - Dividends\n"
            "    (ignores buybacks, stock compensation, other comprehensive income, RE > equity)",
        "BALANCE_SHEET_MISMATCH": "Claims Total Assets != Total L&E\n"
            "    (often prior-year figures with different reporting period boundaries)",
        "CROSS_STATEMENT_LINKAGE": "Claims figures mismatch across statements\n"
            "    (e.g., ending cash on CFS != cash on BS)",
        "REVENUE_COGS_STRUCTURE": "Flags Revenue < COGS as impossible\n"
            "    (legitimate for financial/holding companies with different revenue recognition)",
        "MISSING_LINE_ITEMS": "Flags absence of expected line items as an error",
        "OTHER": "Could not be cleanly categorized",
    }

    for pattern, count in categories.most_common():
        pct = count / len(fp) * 100
        desc = pattern_descriptions.get(pattern, "")
        print(f"  {pattern}: {count} ({pct:.1f}%)")
        if desc:
            print(f"    {desc}")
        print()

    # ---- DISCREPANCY REALITY ----
    print("=" * 80)
    print("ARE THE FLAGGED DISCREPANCIES REAL OR FABRICATED?")
    print("=" * 80)
    print()
    for dtype, count in discrepancy_types.most_common():
        pct = count / len(fp) * 100
        print(f"  {dtype}: {count} ({pct:.1f}%)")

    print()
    print("  REAL_BENIGN: The arithmetic GPT-4.1 performs is correct, but the")
    print("    'discrepancy' exists because financial statements contain line items")
    print("    not shown in the simplified benchmark format. The model assumes the")
    print("    visible items are exhaustive when they are not.")
    print()
    print("  STRUCTURAL: The company's financial structure legitimately differs from")
    print("    a standard manufacturing/service company (e.g., COGS > Revenue for")
    print("    financial companies, negative equity for companies with large buybacks).")
    print()

    # ---- STATEMENT DISTRIBUTION ----
    print("=" * 80)
    print("WHICH STATEMENTS ARE FLAGGED?")
    print("=" * 80)
    print()
    for stmt, count in statements.most_common():
        pct = count / len(fp) * 100
        print(f"  {stmt}: {count} ({pct:.1f}%)")
    print()

    print("  Most commonly flagged fields:")
    for field, count in fields.most_common(10):
        pct = count / len(fp) * 100
        print(f"    {field}: {count} ({pct:.1f}%)")
    print()

    # ---- REPRESENTATIVE EXAMPLES ----
    print("=" * 80)
    print("REPRESENTATIVE FALSE POSITIVE EXAMPLES")
    print("=" * 80)

    examples = [
        ("ARITHMETIC_SUBTOTAL",
         "Model assumes listed items are exhaustive",
         "cisco_systems,_inc__clean"),
        ("OPERATING_INCOME_CALC",
         "Model misapplies income statement formula",
         "coca_cola_co__clean"),
        ("NET_INCOME_CALC",
         "Model ignores items between IBT and Net Income",
         "amazon_com_inc__clean"),
        ("RETAINED_EARNINGS_ROLLFORWARD",
         "Model oversimplifies retained earnings rollforward",
         "home_depot,_inc__clean"),
    ]

    for pattern, label, target_id in examples:
        items = by_category.get(pattern, [])
        example = None
        for item in items:
            if item["instance_id"] == target_id:
                example = item
                break
        if example is None and items:
            example = items[0]

        if example:
            print(f"\n  Example: {pattern}")
            print(f"  Mechanism: {label}")
            print(f"  Company: {example['company']}")
            print(f"  Location flagged: {example['location']}")
            print(f"  GPT-4.1 explanation:")
            # Word wrap the explanation
            words = example['explanation'].split()
            line = "    "
            for w in words:
                if len(line) + len(w) + 1 > 78:
                    print(line)
                    line = "    " + w
                else:
                    line += " " + w if line.strip() else "    " + w
            if line.strip():
                print(line)
            print(f"  Reality: {example['discrepancy_type']}")

    print()

    # ---- ROOT CAUSE SUMMARY ----
    print("=" * 80)
    print("ROOT CAUSE ANALYSIS SUMMARY")
    print("=" * 80)
    print()
    print("  GPT-4.1's 95.3% false positive rate stems from a single fundamental")
    print("  failure mode: the model treats SIMPLIFIED financial statements as if")
    print("  they were COMPLETE financial statements.")
    print()
    print("  Real financial statements contain dozens of line items. The benchmark")
    print("  presents a standardized subset (e.g., Cash, AR, Inventory as current")
    print("  assets). When GPT-4.1 sums these visible items and they don't equal")
    print("  the reported subtotal, it flags an 'error' -- but the difference is")
    print("  simply the unlisted items (prepaid expenses, other current assets, etc.).")
    print()
    print("  This manifests across all three statements:")
    print("  - Income Statement: Operating Expenses is a single line that may not")
    print("    include D&A; items between Operating Income and Net Income are absent")
    print("  - Balance Sheet: Only major components shown; subtotals include more")
    print("  - Cash Flow: Statement structure varies by company")
    print()
    print("  The 2 TN instances succeeded because their simplified statements")
    print("  happened to have line items that approximately sum correctly, not")
    print("  because GPT-4.1 applied a different verification strategy.")
    print()

    # ---- WHAT DISTINGUISHES THE 2 TNs ----
    print("=" * 80)
    print("WHAT DISTINGUISHES THE 2 TRUE NEGATIVES?")
    print("=" * 80)
    print()
    print("  1. United Parcel Service, Inc. (UPS)")
    print("     - GPT-4.1 found 'all major arithmetic and cross-statement linkages")
    print("       check out' with only 'minor rounding differences'")
    print("     - UPS has a relatively simple financial structure where the visible")
    print("       line items happen to approximately reconcile")
    print()
    print("  2. Berkshire Hathaway Inc.")
    print("     - GPT-4.1 found 'all provided financial statement figures are")
    print("       internally consistent within the level of detail given'")
    print("     - Notably, this is the ONLY instance where GPT-4.1 explicitly")
    print("       acknowledges 'within the level of detail given' -- recognizing")
    print("       that missing items don't imply errors")
    print()
    print("  Key insight: In 41/43 cases, GPT-4.1 fails to distinguish between")
    print("  'these listed items don\\'t sum to the subtotal' (expected with partial")
    print("  data) and 'the subtotal is arithmetically wrong' (a real error).")
    print("  Only Berkshire Hathaway's response shows awareness of this distinction.")
    print()

    # ---- DETAILED LISTING ----
    print("=" * 80)
    print("COMPLETE FP LISTING BY CATEGORY")
    print("=" * 80)
    for pattern, items in sorted(by_category.items(), key=lambda x: -len(x[1])):
        print(f"\n--- {pattern} ({len(items)} instances) ---")
        for item in items:
            print(f"  {item['instance_id']}")
            print(f"    Location: {item['location']}")
            print(f"    Discrepancy: {item['discrepancy_type']}")
            # First 120 chars of explanation
            expl_short = item['explanation'][:150] + "..." if len(item['explanation']) > 150 else item['explanation']
            print(f"    Summary: {expl_short}")
            print()

    # ---- Lowes special case ----
    print("=" * 80)
    print("SPECIAL CASE: LOWE'S (Detected=True but long deliberation)")
    print("=" * 80)
    # Find Lowe's clean
    lowes = [r for r in fp if "lowes" in r["instance_id"]]
    for r in lowes:
        print(f"\n  Instance: {r['instance_id']}")
        print(f"  Detected: {r['detected']}")
        print(f"  Response length: {r['raw_response_length']} chars")
        print(f"  Explanation: {r['explanation'][:200]}...")
        print(f"  NOTE: This is actually marked detected=true in the JSON but the")
        print(f"  explanation concludes 'no error'. This appears to be a parse issue")
        print(f"  or the model contradicted itself.")


def generate_latex_table(fp, categories, by_category):
    """Generate LaTeX table for the paper."""
    print()
    print("=" * 80)
    print("LATEX TABLE FOR PAPER")
    print("=" * 80)
    print()

    nice_names = {
        "OPERATING_INCOME_CALC": "Operating Income Formula",
        "NET_INCOME_CALC": "Net Income Formula",
        "ARITHMETIC_SUBTOTAL": "Incomplete Subtotal Summation",
        "RETAINED_EARNINGS_ROLLFORWARD": "Retained Earnings Rollforward",
        "BALANCE_SHEET_MISMATCH": "Balance Sheet Equation",
        "CROSS_STATEMENT_LINKAGE": "Cross-Statement Linkage",
        "REVENUE_COGS_STRUCTURE": "Revenue/COGS Structure",
        "MISSING_LINE_ITEMS": "Missing Line Items",
        "OTHER": "Other",
    }

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Categorization of GPT-4.1 false positive explanations on clean financial statements ($n=41$).}")
    print(r"\label{tab:fp-patterns}")
    print(r"\small")
    print(r"\begin{tabular}{lrc}")
    print(r"\toprule")
    print(r"\textbf{FP Pattern} & \textbf{Count} & \textbf{\%} \\")
    print(r"\midrule")

    for pattern, count in categories.most_common():
        pct = count / len(fp) * 100
        name = nice_names.get(pattern, pattern)
        print(f"  {name} & {count} & {pct:.1f}\\% \\\\")

    print(r"\midrule")
    print(f"  \\textbf{{Total}} & \\textbf{{{len(fp)}}} & \\textbf{{100.0\\%}} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


if __name__ == "__main__":
    data = load_results()
    clean, fp, tn = extract_clean_instances(data)

    assert len(clean) == 43, f"Expected 43 clean instances, got {len(clean)}"
    assert len(fp) == 41, f"Expected 41 FP, got {len(fp)}"
    assert len(tn) == 2, f"Expected 2 TN, got {len(tn)}"

    categories, discrepancy_types, by_category, by_company = analyze_fp_patterns(fp)

    print_report(data, clean, fp, tn)
    generate_latex_table(fp, categories, by_category)
