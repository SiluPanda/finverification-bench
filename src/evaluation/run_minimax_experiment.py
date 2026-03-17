"""Run FinVerBench evaluation using MiniMax M2.5 via OpenAI-compatible API.

Usage:
    PYTHONPATH=src python -m evaluation.run_minimax_experiment [--max N] [--strategy zero_shot|cot]
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

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get(
    "MINIMAX_API_KEY",
    "sk-cp-9U5chKaokF9x4S_PCebURyA5N-92PyY5Rj5aDMEXZQ52MiMFGLRmyldZNNw-8QZTz3c1JsMFfuxDvzxQl6KTaybkGn_62XdJxAgMaZyoNXD-BNT8X8Joke4",
)
BASE_URL = "https://api.minimax.io/v1"
MODEL = "MiniMax-M2.5"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ZERO_SHOT_PROMPT = """You are a financial auditor. Examine the following financial statements for internal consistency.

{statements}

Determine whether there are any numerical errors or inconsistencies. Check if all line items add up to their stated totals, and if values that should match across statements actually match.

IMPORTANT: You MUST provide your final answer as JSON (outside any thinking/reasoning tags):
{{"has_error": true/false, "error_location": "field_name or null", "explanation": "brief explanation"}}"""

COT_PROMPT = """You are a financial auditor. Systematically verify the internal consistency of these financial statements by checking each relationship step by step.

{statements}

Check these relationships:
1. Do all income statement line items sum correctly? (Revenue - COGS = Gross Profit, etc.)
2. Does the balance sheet balance? (Assets = Liabilities + Equity)
3. Does net income on the income statement match net income on the cash flow statement?
4. Does ending cash on the cash flow statement match cash on the balance sheet?
5. Do operating + investing + financing = net change in cash?
6. Are prior year comparative figures consistent?

For each check, state the values and whether they match.

IMPORTANT: You MUST provide your final answer as JSON (outside any thinking/reasoning tags):
{{"has_error": true/false, "error_location": "field_name or null", "explanation": "brief explanation"}}"""


def build_prompt(statements_text: str, strategy: str) -> str:
    if strategy == "cot":
        return COT_PROMPT.format(statements=statements_text)
    return ZERO_SHOT_PROMPT.format(statements=statements_text)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_response(raw: str) -> Dict[str, Any]:
    """Extract detection decision from model response."""
    # First try content OUTSIDE think tags
    text_outside = strip_think_tags(raw)

    # Try JSON outside think tags first (preferred)
    for text_to_search in [text_outside, raw]:  # try outside, then full response
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
                    "parse_method": "json" if text_to_search == text_outside else "json_in_think",
                }
            except json.JSONDecodeError:
                pass

    # Fallback: keyword heuristics on FULL response (including think content)
    full_lower = raw.lower()
    detected = False

    # Positive signals (error found)
    positive_kws = ["inconsistenc", "discrepancy", "does not match", "doesn't match",
                    "mismatch", "should be", "but is", "off by", "difference of"]
    negative_kws = ["no inconsistenc", "no error", "internally consistent",
                    "all consistent", "statements are consistent", "no discrepanc",
                    "everything checks out", "all checks pass"]

    pos_count = sum(1 for kw in positive_kws if kw in full_lower)
    neg_count = sum(1 for kw in negative_kws if kw in full_lower)

    detected = pos_count > neg_count

    return {
        "detected": detected,
        "error_location": None,
        "explanation": (text_outside or raw)[:200],
        "parse_method": "heuristic",
    }


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(instances: List[Dict], max_per_cell: int = 3, seed: int = 42) -> List[Dict]:
    """Select a stratified sample: all clean + N per (error_type, magnitude_bucket)."""
    rng = random.Random(seed)

    clean = [i for i in instances if not i["ground_truth"].get("has_error")]
    error = [i for i in instances if i["ground_truth"].get("has_error")]

    # Group by (error_type, magnitude_bucket)
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

    sampled = list(clean)  # all clean instances
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
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_instance(
    client: OpenAI,
    instance: Dict,
    strategy: str,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Evaluate a single benchmark instance."""
    prompt = build_prompt(instance["formatted_statements"], strategy)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content or ""
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
                "model": response.model,
            }
        except Exception as e:
            logger.warning("Attempt %d failed for %s: %s", attempt + 1, instance["instance_id"], e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
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
    """Compute all evaluation metrics from results."""
    tp = fp = tn = fn = 0
    by_category = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    by_magnitude = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    by_type = defaultdict(lambda: {"tp": 0, "fn": 0})

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
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "total": total,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0, help="Max instances (0=all sampled)")
    parser.add_argument("--strategy", default="zero_shot", choices=["zero_shot", "cot"])
    parser.add_argument("--per-cell", type=int, default=3, help="Instances per (type, magnitude) cell")
    parser.add_argument("--benchmark", default=str(PROJECT_ROOT / "data/benchmark/benchmark.json"))
    args = parser.parse_args()

    # Load benchmark
    with open(args.benchmark) as f:
        all_instances = json.load(f)
    logger.info("Loaded %d instances", len(all_instances))

    # Sample
    sample = stratified_sample(all_instances, max_per_cell=args.per_cell)
    if args.max > 0:
        sample = sample[:args.max]
    logger.info("Evaluating %d instances with strategy=%s", len(sample), args.strategy)

    # Init client
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Run evaluation
    results = []
    for i, inst in enumerate(sample):
        result = evaluate_instance(client, inst, args.strategy)
        results.append(result)

        # Progress
        if (i + 1) % 10 == 0 or i == 0:
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
    print(f"MiniMax M2.5 — {args.strategy} — FinVerBench Results")
    print("=" * 70)
    o = metrics["overall"]
    print(f"\nOverall (n={o['total']})")
    print(f"  Accuracy:  {o['accuracy']:.1f}%")
    print(f"  Precision: {o['precision']:.1f}%")
    print(f"  Recall:    {o['recall']:.1f}%")
    print(f"  F1:        {o['f1']:.1f}%")
    print(f"  FPR:       {o['fpr']:.1f}%")
    print(f"  TP={o['tp']}  FP={o['fp']}  TN={o['tn']}  FN={o['fn']}")

    print("\nBy Error Category:")
    for cat, info in sorted(metrics["by_category"].items()):
        print(f"  {cat:>6s}: {info['detection_rate']:5.1f}%")

    print("\nBy Magnitude:")
    for mag, info in sorted(metrics["by_magnitude"].items()):
        print(f"  {mag:>6s}: {info['detection_rate']:5.1f}%")

    print("\nBy Error Type:")
    for etype, info in sorted(metrics["by_type"].items()):
        print(f"  {etype:>25s}: {info['detection_rate']:5.1f}%")

    # Parse method distribution
    parse_methods = Counter(r.get("parse_method", "?") for r in results)
    print(f"\nParse methods: {dict(parse_methods)}")

    # Save results
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"minimax_m25_{args.strategy}.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL,
            "strategy": args.strategy,
            "metrics": metrics,
            "results": results,
        }, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
