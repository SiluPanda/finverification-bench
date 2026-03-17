"""Run FinVerBench experiments using Claude Code CLI for Opus 4.6 and Sonnet 4.6.

Programmatically invokes the `claude` CLI with `-p` (print mode) to send
prompts and collect responses. This avoids needing a separate API key since
it uses the user's existing Claude Code subscription.

Usage:
    python3 src/evaluation/run_claude_cli_experiments.py
    python3 src/evaluation/run_claude_cli_experiments.py --model opus
    python3 src/evaluation/run_claude_cli_experiments.py --model sonnet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import tempfile
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
CLAUDE_CLI_MODELS = {
    "opus": {
        "cli_model": "opus",
        "display_name": "Claude Opus 4.6",
        "model_id": "claude-opus-4-6",
    },
    "sonnet": {
        "cli_model": "sonnet",
        "display_name": "Claude Sonnet 4.6",
        "model_id": "claude-sonnet-4-6",
    },
}

STRATEGY = "cot"

# ---------------------------------------------------------------------------
# Prompt template (CoT)
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
# Claude CLI caller
# ---------------------------------------------------------------------------

def call_claude_cli(prompt: str, model_alias: str, max_retries: int = 3) -> str:
    """Call Claude Code CLI with -p flag and return the text response.

    Writes the prompt to a temp file and pipes it to avoid shell escaping issues.
    Uses --tools "" to disable all tools (pure LLM response).
    Uses --no-session-persistence to avoid cluttering session history.
    """
    backoff = 5.0

    for attempt in range(max_retries):
        try:
            # Write prompt to a temp file to avoid argument length limits
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(prompt)
                prompt_file = f.name

            try:
                # Pipe the prompt via stdin
                with open(prompt_file, "r") as pf:
                    result = subprocess.run(
                        [
                            "claude",
                            "-p",
                            "--model", model_alias,
                            "--tools", "",
                            "--no-session-persistence",
                            "--output-format", "text",
                        ],
                        stdin=pf,
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minute timeout per call
                    )
            finally:
                os.unlink(prompt_file)

            if result.returncode != 0:
                error_msg = result.stderr.strip() or f"Exit code {result.returncode}"
                if any(kw in error_msg.lower() for kw in ["rate", "overloaded", "timeout"]):
                    raise RuntimeError(f"Retryable: {error_msg}")
                raise RuntimeError(f"Claude CLI error: {error_msg}")

            response = result.stdout.strip()
            if not response:
                raise RuntimeError("Empty response from Claude CLI")

            return response

        except (subprocess.TimeoutExpired, RuntimeError) as e:
            if attempt < max_retries - 1:
                logger.warning("Retry %d/%d for %s: %s", attempt + 1, max_retries, model_alias, e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            else:
                raise

    raise RuntimeError(f"Failed after {max_retries} retries")


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

def evaluate_instance(instance: Dict, model_alias: str) -> Dict[str, Any]:
    prompt = build_prompt(instance["formatted_statements"])

    try:
        raw = call_claude_cli(prompt, model_alias)
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


def run_model(model_key: str, sample: List[Dict]) -> Dict:
    config = CLAUDE_CLI_MODELS[model_key]
    display_name = config["display_name"]
    model_alias = config["cli_model"]

    logger.info("=" * 70)
    logger.info("Starting: %s (claude -p --model %s) — %d instances",
                display_name, model_alias, len(sample))
    logger.info("=" * 70)

    results = []
    for i, inst in enumerate(sample):
        result = evaluate_instance(inst, model_alias)
        results.append(result)

        if (i + 1) % 5 == 0 or i == 0:
            metrics = compute_metrics(results)
            o = metrics["overall"]
            logger.info(
                "[%s] %d/%d — acc=%.1f%% prec=%.1f%% rec=%.1f%% f1=%.1f%% fpr=%.1f%%",
                display_name, i + 1, len(sample),
                o["accuracy"], o["precision"], o["recall"], o["f1"], o["fpr"],
            )

        # Save checkpoint every 10 instances
        if (i + 1) % 10 == 0:
            _save_results(config, results, checkpoint=True)

    return _save_results(config, results, checkpoint=False)


def _save_results(config: Dict, results: List[Dict], checkpoint: bool = False) -> Dict:
    display_name = config["display_name"]
    model_id = config["model_id"]

    metrics = compute_metrics(results)
    o = metrics["overall"]

    if not checkpoint:
        print(f"\n{'=' * 70}")
        print(f"{display_name} — CoT — FinVerBench Results")
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

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    safe_name = model_id.replace("/", "_").replace(":", "_")
    suffix = "_checkpoint" if checkpoint else ""
    output_path = output_dir / f"claude_cli_{safe_name}_cot{suffix}_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": display_name,
            "model_id": model_id,
            "strategy": "cot",
            "backend": "claude_cli",
            "per_cell": 2,
            "num_completed": len(results),
            "metrics": metrics,
            "results": results,
        }, f, indent=2)
    logger.info("Results saved to %s (%d instances)", output_path, len(results))

    return {"model": display_name, "metrics": metrics, "path": str(output_path)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run FinVerBench via Claude CLI")
    parser.add_argument("--model", choices=["opus", "sonnet", "both"], default="both",
                        help="Which model to run (default: both)")
    parser.add_argument("--per-cell", type=int, default=2)
    parser.add_argument("--max", type=int, default=0, help="Max instances (0 = all)")
    args = parser.parse_args()

    # Verify claude CLI is available
    try:
        result = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=10)
        logger.info("Claude CLI version: %s", result.stdout.strip())
    except Exception as e:
        logger.error("Claude CLI not available: %s", e)
        sys.exit(1)

    benchmark_path = PROJECT_ROOT / "data" / "benchmark" / "benchmark.json"
    with open(benchmark_path) as f:
        all_instances = json.load(f)
    logger.info("Loaded %d instances", len(all_instances))

    sample = stratified_sample(all_instances, max_per_cell=args.per_cell)
    if args.max > 0:
        sample = sample[:args.max]
    logger.info("Sample size: %d instances per model", len(sample))

    models_to_run = []
    if args.model in ("opus", "both"):
        models_to_run.append("opus")
    if args.model in ("sonnet", "both"):
        models_to_run.append("sonnet")

    completed = []
    for model_key in models_to_run:
        result = run_model(model_key, sample)
        completed.append(result)

    # Summary
    print(f"\n{'=' * 70}")
    print("CLAUDE CLI EXPERIMENT SUMMARY")
    print(f"{'=' * 70}")
    for r in completed:
        o = r["metrics"]["overall"]
        print(f"  {r['model']:25s}  acc={o['accuracy']:5.1f}%  f1={o['f1']:5.1f}%  fpr={o['fpr']:5.1f}%")


if __name__ == "__main__":
    main()
