"""Run comprehensive FinVerBench experiments across multiple models and strategies.

Usage:
    python3 src/evaluation/run_experiments.py --model claude --strategy zero_shot --per-cell 2
    python3 src/evaluation/run_experiments.py --model minimax --strategy cot --per-cell 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "claude": {
        "backend": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "display_name": "Claude Sonnet 4",
    },
    "claude-haiku": {
        "backend": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "display_name": "Claude Haiku 4.5",
    },
    "minimax": {
        "backend": "openai_compat",
        "model_id": "MiniMax-M2.5",
        "display_name": "MiniMax M2.5",
        "base_url": "https://api.minimax.io/v1",
        "api_key_env": "MINIMAX_API_KEY",
    },
}

# ---------------------------------------------------------------------------
# Prompt templates (same as prompts.py but self-contained)
# ---------------------------------------------------------------------------

_RESPONSE_FORMAT = """Respond in the following JSON format (do NOT include any text outside the JSON):
{
  "has_error": true or false,
  "error_location": "dot-path to the erroneous field, e.g. income_statement.revenue" or null,
  "explanation": "brief explanation of what is inconsistent and why"
}"""

ZERO_SHOT_PROMPT = """You are an expert financial auditor. Your task is to verify whether the
following financial statements are internally consistent.

Check for:
- Arithmetic errors (do line items sum to their stated totals?)
- Cross-statement linkage errors (does net income match between IS and CFS?
  does ending cash on CFS match the balance sheet? etc.)
- Year-over-year inconsistencies (does the prior-year ending balance equal the
  current-year opening balance?)
- Magnitude / rounding errors (are any values suspiciously different from what
  the arithmetic implies?)

IMPORTANT: Only flag genuine errors where numbers clearly do not add up.
Small rounding differences (within 1 unit of the reporting precision) should NOT
be flagged as errors. Financial statements often report rounded figures.

FINANCIAL STATEMENTS:
{statements}

{response_format}"""

FEW_SHOT_CLEAN = """Acme Corp — Financial Statements (in millions of USD)
Period: FY2024

INCOME STATEMENT
==========================================================
  Revenue                                          1,000
  Cost of Goods Sold                                -600
  Gross Profit                                       400
  Operating Expenses                                -200
  Depreciation & Amortization                        -50
  Operating Income                                   150
  Interest Expense                                   -10
  Income Before Tax                                  140
  Income Tax Expense                                 -35
  Net Income                                         105

BALANCE SHEET
==============================================================================
                                            Current          Prior
  Cash and Equivalents                          250            220
  Accounts Receivable                           120            110
  Inventory                                      80             70
  Total Current Assets                          450            400
  Property, Plant & Equipment                   500            480
  Total Assets                                  950            880
  Accounts Payable                               90             80
  Short-Term Debt                                60             50
  Total Current Liabilities                     150            130
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
  Capital Expenditures                               -60
  Cash from Investing                                -60
  Debt Repayment                                     -30
  Dividends Paid                                     -15
  Cash from Financing                                -45
  Net Change in Cash                                  30
  Beginning Cash                                     220
  Ending Cash                                        250"""

FEW_SHOT_CLEAN_RESPONSE = """{
  "has_error": false,
  "error_location": null,
  "explanation": "All arithmetic totals are correct. Net income (105) matches between IS and CFS. Ending cash (250) matches between CFS and BS. Total assets equal total liabilities and equity (950)."
}"""

FEW_SHOT_ERROR = """BetaCo Inc — Financial Statements (in millions of USD)
Period: FY2024

INCOME STATEMENT
==========================================================
  Revenue                                          2,000
  Cost of Goods Sold                              -1,100
  Gross Profit                                       900
  Operating Expenses                                -400
  Depreciation & Amortization                       -100
  Operating Income                                   400
  Interest Expense                                   -25
  Income Before Tax                                  375
  Income Tax Expense                                 -94
  Net Income                                         281

BALANCE SHEET
==============================================================================
                                            Current          Prior
  Cash and Equivalents                          500            420
  Accounts Receivable                           250            220
  Inventory                                     150            130
  Total Current Assets                          900            770
  Property, Plant & Equipment                 1,100          1,050
  Total Assets                                2,000          1,820
  Accounts Payable                              180            160
  Short-Term Debt                               120            100
  Total Current Liabilities                     300            260
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
  Capital Expenditures                              -150
  Cash from Investing                               -150
  Debt Repayment                                     -50
  Dividends Paid                                     -85
  Cash from Financing                               -135
  Net Change in Cash                                  80
  Beginning Cash                                     420
  Ending Cash                                        500"""

FEW_SHOT_ERROR_RESPONSE = """{
  "has_error": true,
  "error_location": "cash_flow_statement.net_income",
  "explanation": "Net income on the cash flow statement is 295, but net income on the income statement is 281. These two figures must agree (CFS starts with IS net income under the indirect method). The CFS net income appears to have been overstated by 14 (approximately 5%)."
}"""

FEW_SHOT_PROMPT = """You are an expert financial auditor. Your task is to verify whether financial
statements are internally consistent. Below are two worked examples followed
by a new set of statements for you to verify.

--- EXAMPLE 1 (clean statements) ---

FINANCIAL STATEMENTS:
{clean_example}

YOUR ANALYSIS:
{clean_response}

--- EXAMPLE 2 (statements with an error) ---

FINANCIAL STATEMENTS:
{error_example}

YOUR ANALYSIS:
{error_response}

--- YOUR TASK ---

Now verify the following financial statements using the same approach.
IMPORTANT: Only flag genuine errors. Small rounding differences should NOT be flagged.

FINANCIAL STATEMENTS:
{statements}

{response_format}"""

COT_PROMPT = """You are an expert financial auditor performing a systematic verification of
financial statements. Work through each check step by step, showing your
calculations, before reaching a final conclusion.

IMPORTANT: Only flag genuine errors where numbers clearly do not add up.
Small rounding differences (within 1 unit of the reporting precision) should NOT
be flagged as errors. Financial statements often report rounded figures, and
minor discrepancies from rounding are expected and normal.

FINANCIAL STATEMENTS:
{statements}

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

{response_format}"""


def build_prompt(statements_text: str, strategy: str) -> str:
    if strategy == "zero_shot":
        return ZERO_SHOT_PROMPT.format(
            statements=statements_text,
            response_format=_RESPONSE_FORMAT,
        )
    elif strategy == "few_shot":
        return FEW_SHOT_PROMPT.format(
            clean_example=FEW_SHOT_CLEAN,
            clean_response=FEW_SHOT_CLEAN_RESPONSE,
            error_example=FEW_SHOT_ERROR,
            error_response=FEW_SHOT_ERROR_RESPONSE,
            statements=statements_text,
            response_format=_RESPONSE_FORMAT,
        )
    elif strategy == "cot":
        return COT_PROMPT.format(
            statements=statements_text,
            response_format=_RESPONSE_FORMAT,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_response(raw: str) -> Dict[str, Any]:
    text_outside = strip_think_tags(raw)

    for text_to_search in [text_outside, raw]:
        # Try fenced JSON blocks first
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_to_search, re.DOTALL)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                if "has_error" in parsed:
                    return {
                        "detected": bool(parsed.get("has_error", False)),
                        "error_location": parsed.get("error_location"),
                        "explanation": parsed.get("explanation", ""),
                        "parse_method": "json_fenced",
                    }
            except json.JSONDecodeError:
                pass

        # Try inline JSON
        json_match = re.search(
            r'\{[^{}]*"has_error"\s*:\s*(true|false)[^{}]*\}',
            text_to_search, re.IGNORECASE,
        )
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return {
                    "detected": bool(parsed.get("has_error", False)),
                    "error_location": parsed.get("error_location"),
                    "explanation": parsed.get("explanation", ""),
                    "parse_method": "json_inline",
                }
            except json.JSONDecodeError:
                pass

    # Fallback: keyword heuristics
    full_lower = raw.lower()
    positive_kws = ["inconsistenc", "discrepancy", "does not match", "doesn't match",
                    "mismatch", "should be", "but is", "off by", "difference of",
                    "error found", "\"has_error\": true", "has_error: true"]
    negative_kws = ["no inconsistenc", "no error", "internally consistent",
                    "all consistent", "statements are consistent", "no discrepanc",
                    "everything checks out", "all checks pass", "\"has_error\": false",
                    "has_error: false"]

    pos_count = sum(1 for kw in positive_kws if kw in full_lower)
    neg_count = sum(1 for kw in negative_kws if kw in full_lower)

    return {
        "detected": pos_count > neg_count,
        "error_location": None,
        "explanation": (text_outside or raw)[:300],
        "parse_method": "heuristic",
    }


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(instances: List[Dict], max_per_cell: int = 2, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    clean = [i for i in instances if not i["ground_truth"].get("has_error")]
    error = [i for i in instances if i["ground_truth"].get("has_error")]

    cells: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for inst in error:
        gt = inst["ground_truth"]
        etype = gt.get("error_type", "unknown")
        mag = abs(gt.get("error_magnitude_pct", 0) or 0)
        if mag < 1:
            bucket = "<1%"
        elif mag < 5:
            bucket = "1-5%"
        elif mag < 20:
            bucket = "5-20%"
        else:
            bucket = ">20%"
        cells[(etype, bucket)].append(inst)

    sampled = list(clean)
    for key, cell_instances in sorted(cells.items()):
        k = min(max_per_cell, len(cell_instances))
        sampled.extend(rng.sample(cell_instances, k))

    rng.shuffle(sampled)
    logger.info(
        "Stratified sample: %d clean + %d error = %d total (%d cells)",
        len(clean), len(sampled) - len(clean), len(sampled), len(cells),
    )
    return sampled


# ---------------------------------------------------------------------------
# LLM Callers
# ---------------------------------------------------------------------------

def call_anthropic(prompt: str, model_id: str, max_retries: int = 5) -> str:
    import anthropic
    client = anthropic.Anthropic()
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model_id,
                max_tokens=4096,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            exc_str = str(e).lower()
            if any(kw in exc_str for kw in ["rate", "429", "overloaded", "503", "timeout"]):
                logger.warning("Retry %d/%d: %s", attempt + 1, max_retries, e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def call_openai_compat(prompt: str, model_id: str, base_url: str, api_key: str, max_retries: int = 5) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4096,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            exc_str = str(e).lower()
            if any(kw in exc_str for kw in ["rate", "429", "overloaded", "503", "timeout"]):
                logger.warning("Retry %d/%d: %s", attempt + 1, max_retries, e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def call_model(prompt: str, config: Dict[str, Any]) -> str:
    if config["backend"] == "anthropic":
        return call_anthropic(prompt, config["model_id"])
    elif config["backend"] == "openai_compat":
        api_key = os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"), "")
        if not api_key:
            # Fallback to hardcoded key from .env
            api_key = "sk-cp-9U5chKaokF9x4S_PCebURyA5N-92PyY5Rj5aDMEXZQ52MiMFGLRmyldZNNw-8QZTz3c1JsMFfuxDvzxQl6KTaybkGn_62XdJxAgMaZyoNXD-BNT8X8Joke4"
        return call_openai_compat(prompt, config["model_id"], config["base_url"], api_key)
    else:
        raise ValueError(f"Unknown backend: {config['backend']}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_instance(instance: Dict, strategy: str, config: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_prompt(instance["formatted_statements"], strategy)

    try:
        raw = call_model(prompt, config)
        parsed = parse_response(raw)
        return {
            "instance_id": instance["instance_id"],
            "has_error": instance["ground_truth"].get("has_error", False),
            "error_type": instance["ground_truth"].get("error_type"),
            "error_category": instance["ground_truth"].get("error_category"),
            "error_magnitude_pct": instance["ground_truth"].get("error_magnitude_pct"),
            "detected": parsed["detected"],
            "error_location": parsed.get("error_location"),
            "explanation": parsed.get("explanation", ""),
            "parse_method": parsed["parse_method"],
            "raw_response_length": len(raw),
        }
    except Exception as e:
        logger.error("Failed for %s: %s", instance["instance_id"], e)
        return {
            "instance_id": instance["instance_id"],
            "has_error": instance["ground_truth"].get("has_error", False),
            "error_type": instance["ground_truth"].get("error_type"),
            "error_category": instance["ground_truth"].get("error_category"),
            "error_magnitude_pct": instance["ground_truth"].get("error_magnitude_pct"),
            "detected": False,
            "error": str(e),
            "parse_method": "error",
        }


def compute_metrics(results: List[Dict]) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    by_category = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    by_magnitude = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    by_type = defaultdict(lambda: {"tp": 0, "fn": 0})

    # Track localization
    loc_correct = 0
    loc_total = 0

    for r in results:
        has_error = r["has_error"]
        detected = r["detected"]

        if has_error and detected:
            tp += 1
        elif has_error and not detected:
            fn += 1
        elif not has_error and detected:
            fp += 1
        else:
            tn += 1

        cat = r.get("error_category") or "clean"
        if has_error:
            by_category[cat]["tp" if detected else "fn"] += 1
            by_type[r.get("error_type", "?")]["tp" if detected else "fn"] += 1

            mag = abs(r.get("error_magnitude_pct", 0) or 0)
            if mag < 1:
                bucket = "<1%"
            elif mag < 5:
                bucket = "1-5%"
            elif mag < 20:
                bucket = "5-20%"
            else:
                bucket = ">20%"
            by_magnitude[bucket]["tp" if detected else "fn"] += 1

            # Check localization
            if detected and r.get("error_location"):
                loc_total += 1
                gt_type = r.get("error_type", "")
                pred_loc = str(r.get("error_location", "")).lower()
                # Simple check: does the predicted location mention relevant terms?
                if gt_type and any(term in pred_loc for term in [
                    "net_income", "revenue", "cash", "retained", "total",
                    "asset", "liabilit", "equity", "depreci", "operating",
                ]):
                    loc_correct += 1
        else:
            by_category["clean"]["tn" if not detected else "fp"] += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0

    def _rate(d):
        t = d.get("tp", 0) + d.get("fn", 0)
        return round(d.get("tp", 0) / t * 100, 1) if t else 0

    return {
        "overall": {
            "accuracy": round(accuracy * 100, 1),
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1),
            "fpr": round(fpr * 100, 1),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total,
        },
        "localization": {
            "correct": loc_correct,
            "total": loc_total,
            "accuracy": round(loc_correct / loc_total * 100, 1) if loc_total else 0,
        },
        "by_category": {
            cat: {"detection_rate": _rate(counts), **counts}
            for cat, counts in sorted(by_category.items())
        },
        "by_magnitude": {
            mag: {"detection_rate": _rate(counts), **counts}
            for mag, counts in sorted(by_magnitude.items())
        },
        "by_type": {
            etype: {"detection_rate": _rate(counts), **counts}
            for etype, counts in sorted(by_type.items())
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run FinVerBench experiments")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--strategy", required=True, choices=["zero_shot", "few_shot", "cot"])
    parser.add_argument("--per-cell", type=int, default=2)
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--benchmark", default=str(PROJECT_ROOT / "data/benchmark/benchmark.json"))
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]

    # Load benchmark
    with open(args.benchmark) as f:
        all_instances = json.load(f)
    logger.info("Loaded %d instances", len(all_instances))

    # Sample
    sample = stratified_sample(all_instances, max_per_cell=args.per_cell)
    if args.max > 0:
        sample = sample[:args.max]
    logger.info("Evaluating %d instances with %s / %s", len(sample), config["display_name"], args.strategy)

    # Run
    results = []
    for i, inst in enumerate(sample):
        result = evaluate_instance(inst, args.strategy, config)
        results.append(result)

        if (i + 1) % 5 == 0 or i == 0:
            metrics = compute_metrics(results)
            o = metrics["overall"]
            logger.info(
                "[%d/%d] acc=%.1f%% prec=%.1f%% rec=%.1f%% f1=%.1f%% fpr=%.1f%%",
                i + 1, len(sample), o["accuracy"], o["precision"],
                o["recall"], o["f1"], o["fpr"],
            )

    # Final metrics
    metrics = compute_metrics(results)

    # Print results
    print("\n" + "=" * 70)
    print(f"{config['display_name']} — {args.strategy} — FinVerBench Results")
    print("=" * 70)
    o = metrics["overall"]
    print(f"\nOverall (n={o['total']})")
    print(f"  Accuracy:  {o['accuracy']:.1f}%")
    print(f"  Precision: {o['precision']:.1f}%")
    print(f"  Recall:    {o['recall']:.1f}%")
    print(f"  F1:        {o['f1']:.1f}%")
    print(f"  FPR:       {o['fpr']:.1f}%")
    print(f"  TP={o['tp']}  FP={o['fp']}  TN={o['tn']}  FN={o['fn']}")

    loc = metrics["localization"]
    print(f"\nLocalization: {loc['correct']}/{loc['total']} = {loc['accuracy']:.1f}%")

    print("\nBy Error Category:")
    for cat, info in sorted(metrics["by_category"].items()):
        print(f"  {cat:>8s}: {info['detection_rate']:5.1f}%  (tp={info.get('tp',0)} fn={info.get('fn',0)})")

    print("\nBy Magnitude:")
    for mag, info in sorted(metrics["by_magnitude"].items()):
        print(f"  {mag:>6s}: {info['detection_rate']:5.1f}%  (tp={info.get('tp',0)} fn={info.get('fn',0)})")

    print("\nBy Error Type:")
    for etype, info in sorted(metrics["by_type"].items()):
        print(f"  {etype:>30s}: {info['detection_rate']:5.1f}%  (tp={info.get('tp',0)} fn={info.get('fn',0)})")

    parse_methods = Counter(r.get("parse_method", "?") for r in results)
    print(f"\nParse methods: {dict(parse_methods)}")

    # Save
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{args.model}_{args.strategy}_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": config["display_name"],
            "model_id": config["model_id"],
            "strategy": args.strategy,
            "per_cell": args.per_cell,
            "metrics": metrics,
            "results": results,
        }, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
