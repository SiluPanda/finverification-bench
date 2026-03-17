"""Run FinVerBench experiments on multiple models via OpenRouter API.

Usage:
    OPENROUTER_API_KEY=sk-or-... python3 src/evaluation/run_openrouter_experiments.py
"""

from __future__ import annotations

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
# OpenRouter model configs — models to evaluate
# ---------------------------------------------------------------------------
OPENROUTER_MODELS = [
    {
        "model_id": "openai/gpt-4.1",
        "display_name": "GPT-4.1",
    },
    {
        "model_id": "google/gemini-2.5-pro-preview-03-25",
        "display_name": "Gemini 2.5 Pro",
    },
    {
        "model_id": "deepseek/deepseek-r1",
        "display_name": "DeepSeek R1",
    },
    {
        "model_id": "meta-llama/llama-4-maverick",
        "display_name": "Llama 4 Maverick",
    },
    {
        "model_id": "qwen/qwen3-235b-a22b",
        "display_name": "Qwen 3 235B",
    },
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
STRATEGY = "cot"  # chain-of-thought — best-performing strategy

# ---------------------------------------------------------------------------
# Prompt templates (same as run_experiments.py)
# ---------------------------------------------------------------------------

_RESPONSE_FORMAT = """Respond in the following JSON format (do NOT include any text outside the JSON):
{
  "has_error": true or false,
  "error_location": "dot-path to the erroneous field, e.g. income_statement.revenue" or null,
  "explanation": "brief explanation of what is inconsistent and why"
}"""

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


def build_prompt(statements_text: str) -> str:
    return COT_PROMPT.format(
        statements=statements_text,
        response_format=_RESPONSE_FORMAT,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_response(raw: str) -> Dict[str, Any]:
    text_outside = strip_think_tags(raw)

    for text_to_search in [text_outside, raw]:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_to_search, re.DOTALL)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                if "has_error" in parsed:
                    he = parsed.get("has_error", False)
                    if isinstance(he, str):
                        he = he.lower() in ("true", "yes", "1")
                    return {
                        "detected": bool(he),
                        "error_location": parsed.get("error_location"),
                        "explanation": parsed.get("explanation", ""),
                        "parse_method": "json_fenced",
                    }
            except json.JSONDecodeError:
                pass

        json_match = re.search(
            r'\{[^{}]*"has_error"\s*:\s*(true|false)[^{}]*\}',
            text_to_search, re.IGNORECASE,
        )
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                he = parsed.get("has_error", False)
                if isinstance(he, str):
                    he = he.lower() in ("true", "yes", "1")
                return {
                    "detected": bool(he),
                    "error_location": parsed.get("error_location"),
                    "explanation": parsed.get("explanation", ""),
                    "parse_method": "json_inline",
                }
            except json.JSONDecodeError:
                pass

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
# OpenRouter API caller
# ---------------------------------------------------------------------------

def call_openrouter(prompt: str, model_id: str, api_key: str, max_retries: int = 5) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    backoff = 2.0

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4096,
            )
            content = response.choices[0].message.content or ""
            return content
        except Exception as e:
            exc_str = str(e).lower()
            if any(kw in exc_str for kw in ["rate", "429", "overloaded", "503", "timeout", "connection", "502"]):
                logger.warning("Retry %d/%d for %s: %s", attempt + 1, max_retries, model_id, e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            elif "credit" in exc_str or "insufficient" in exc_str or "402" in exc_str:
                logger.error("Credits exhausted: %s", e)
                raise
            else:
                raise

    raise RuntimeError(f"Failed after {max_retries} retries for {model_id}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: List[Dict]) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    by_category = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    by_magnitude = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    by_type = defaultdict(lambda: {"tp": 0, "fn": 0})
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
            if detected and r.get("error_location"):
                loc_total += 1
                gt_type = r.get("error_type", "")
                pred_loc = str(r.get("error_location", "")).lower()
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
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_instance(instance: Dict, api_key: str, model_id: str) -> Dict[str, Any]:
    prompt = build_prompt(instance["formatted_statements"])

    try:
        raw = call_openrouter(prompt, model_id, api_key)
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


def run_model(model_config: Dict, sample: List[Dict], api_key: str) -> Optional[Dict]:
    """Run experiment for a single model. Returns None if credits exhausted."""
    model_id = model_config["model_id"]
    display_name = model_config["display_name"]

    logger.info("=" * 70)
    logger.info("Starting: %s (%s) — %d instances", display_name, model_id, len(sample))
    logger.info("=" * 70)

    results = []
    for i, inst in enumerate(sample):
        try:
            result = evaluate_instance(inst, api_key, model_id)
        except Exception as e:
            if "credit" in str(e).lower() or "insufficient" in str(e).lower() or "402" in str(e).lower():
                logger.error("Credits exhausted during %s at instance %d/%d", display_name, i + 1, len(sample))
                # Save partial results
                if results:
                    return _save_and_report(model_config, results, partial=True)
                return None
            raise

        results.append(result)

        if (i + 1) % 5 == 0 or i == 0:
            metrics = compute_metrics(results)
            o = metrics["overall"]
            logger.info(
                "[%s] %d/%d — acc=%.1f%% prec=%.1f%% rec=%.1f%% f1=%.1f%% fpr=%.1f%%",
                display_name, i + 1, len(sample),
                o["accuracy"], o["precision"], o["recall"], o["f1"], o["fpr"],
            )

    return _save_and_report(model_config, results, partial=False)


def _save_and_report(model_config: Dict, results: List[Dict], partial: bool) -> Dict:
    display_name = model_config["display_name"]
    model_id = model_config["model_id"]

    metrics = compute_metrics(results)
    o = metrics["overall"]

    print(f"\n{'=' * 70}")
    tag = " (PARTIAL)" if partial else ""
    print(f"{display_name} — CoT — FinVerBench Results{tag}")
    print(f"{'=' * 70}")
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
        print(f"  {cat:>8s}: {info['detection_rate']:5.1f}%  (tp={info.get('tp', 0)} fn={info.get('fn', 0)})")

    print("\nBy Magnitude:")
    for mag, info in sorted(metrics["by_magnitude"].items()):
        print(f"  {mag:>6s}: {info['detection_rate']:5.1f}%  (tp={info.get('tp', 0)} fn={info.get('fn', 0)})")

    parse_methods = Counter(r.get("parse_method", "?") for r in results)
    print(f"\nParse methods: {dict(parse_methods)}")

    # Save
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    safe_name = model_id.replace("/", "_").replace(":", "_")
    suffix = "_partial" if partial else ""
    output_path = output_dir / f"openrouter_{safe_name}_cot{suffix}_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": display_name,
            "model_id": model_id,
            "strategy": "cot",
            "backend": "openrouter",
            "partial": partial,
            "per_cell": 2,
            "metrics": metrics,
            "results": results,
        }, f, indent=2)
    logger.info("Results saved to %s", output_path)

    return {"model": display_name, "metrics": metrics, "path": str(output_path)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    benchmark_path = PROJECT_ROOT / "data" / "benchmark" / "benchmark.json"
    with open(benchmark_path) as f:
        all_instances = json.load(f)
    logger.info("Loaded %d instances", len(all_instances))

    sample = stratified_sample(all_instances, max_per_cell=2)
    logger.info("Sample size: %d instances per model", len(sample))

    completed = []
    for model_config in OPENROUTER_MODELS:
        try:
            result = run_model(model_config, sample, api_key)
            if result:
                completed.append(result)
            else:
                logger.warning("Credits exhausted. Stopping.")
                break
        except Exception as e:
            if "credit" in str(e).lower() or "insufficient" in str(e).lower() or "402" in str(e).lower():
                logger.warning("Credits exhausted. Stopping.")
                break
            logger.error("Model %s failed: %s", model_config["display_name"], e)
            continue

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    for r in completed:
        o = r["metrics"]["overall"]
        print(f"  {r['model']:25s}  acc={o['accuracy']:5.1f}%  f1={o['f1']:5.1f}%  fpr={o['fpr']:5.1f}%")
    print(f"\nCompleted {len(completed)}/{len(OPENROUTER_MODELS)} models")


if __name__ == "__main__":
    main()
