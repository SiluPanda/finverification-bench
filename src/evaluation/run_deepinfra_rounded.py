"""Run rounding ablation experiment on open-weight models via DeepInfra API.

Uses the ROUNDED benchmark data (errors rounded to whole numbers) to test
whether models rely on decimal anomaly cues for error detection.
"""

from __future__ import annotations
import json, logging, os, re, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

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
Step 2 — Balance Sheet Arithmetic
Step 3 — Cash Flow Statement Arithmetic
Step 4 — Cross-Statement Linkages
Step 5 — Year-over-Year Consistency

After completing all checks, provide your final answer.

{response_format}"""


def build_prompt(statements_text: str) -> str:
    return COT_PROMPT.format(statements=statements_text, response_format=_RESPONSE_FORMAT)


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
                    if isinstance(he, str): he = he.lower() in ("true", "yes", "1")
                    return {"detected": bool(he), "error_location": parsed.get("error_location"),
                            "explanation": parsed.get("explanation", ""), "parse_method": "json_fenced"}
            except json.JSONDecodeError: pass
        json_match = re.search(r'\{[^{}]*"has_error"\s*:\s*(true|false)[^{}]*\}', text_to_search, re.IGNORECASE)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                he = parsed.get("has_error", False)
                if isinstance(he, str): he = he.lower() in ("true", "yes", "1")
                return {"detected": bool(he), "error_location": parsed.get("error_location"),
                        "explanation": parsed.get("explanation", ""), "parse_method": "json_inline"}
            except json.JSONDecodeError: pass
    full_lower = raw.lower()
    pos = ["inconsistenc", "discrepancy", "does not match", "mismatch", "error found", "\"has_error\": true"]
    neg = ["no inconsistenc", "no error", "internally consistent", "\"has_error\": false"]
    return {"detected": sum(1 for k in pos if k in full_lower) > sum(1 for k in neg if k in full_lower),
            "error_location": None, "explanation": (text_outside or raw)[:300], "parse_method": "heuristic"}


def call_deepinfra(prompt, model_id, api_key, max_retries=5):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=4096)
            return response.choices[0].message.content or ""
        except Exception as e:
            exc_str = str(e).lower()
            if any(kw in exc_str for kw in ["rate", "429", "overloaded", "503", "timeout", "502"]):
                logger.warning("Retry %d/%d: %s", attempt + 1, max_retries, e)
                time.sleep(backoff); backoff = min(backoff * 2, 60.0)
            elif "credit" in exc_str or "402" in exc_str:
                raise
            else: raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def compute_metrics(results):
    tp = fp = tn = fn = 0
    by_cat = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    loc_correct = loc_total = 0
    for r in results:
        he, det = r["has_error"], r["detected"]
        if he and det: tp += 1
        elif he and not det: fn += 1
        elif not he and det: fp += 1
        else: tn += 1
        cat = r.get("error_category") or "clean"
        if he:
            by_cat[cat]["tp" if det else "fn"] += 1
            if det and r.get("error_location"):
                loc_total += 1
                if any(t in str(r.get("error_location","")).lower() for t in ["net_income","revenue","cash","retained","total","asset","liabilit","equity","depreci","operating"]):
                    loc_correct += 1
        else:
            by_cat["clean"]["tn" if not det else "fp"] += 1
    total = tp+fp+tn+fn
    acc = (tp+tn)/total*100 if total else 0
    prec = tp/(tp+fp)*100 if (tp+fp) else 0
    rec = tp/(tp+fn)*100 if (tp+fn) else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    fpr = fp/(fp+tn)*100 if (fp+tn) else 0
    def _rate(d):
        t = d.get("tp",0)+d.get("fn",0); return round(d.get("tp",0)/t*100,1) if t else 0
    return {"overall": {"accuracy": round(acc,1), "precision": round(prec,1), "recall": round(rec,1),
                        "f1": round(f1,1), "fpr": round(fpr,1), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total},
            "localization": {"correct": loc_correct, "total": loc_total,
                            "accuracy": round(loc_correct/loc_total*100,1) if loc_total else 0},
            "by_category": {c: {"detection_rate": _rate(v), **v} for c, v in sorted(by_cat.items())}}


def run_model(model_id, display_name, all_instances, api_key, output_path, checkpoint_path):
    """Run evaluation for a single model, with checkpoint support."""
    # Load checkpoint if exists
    results = []
    completed_ids = set()
    if checkpoint_path.exists():
        checkpoint = json.load(open(checkpoint_path))
        results = checkpoint.get("results", [])
        completed_ids = {r["instance_id"] for r in results}
        logger.info("Resumed from checkpoint: %d/%d done", len(results), len(all_instances))

    for i, inst in enumerate(all_instances):
        if inst["instance_id"] in completed_ids:
            continue
        prompt = build_prompt(inst["formatted_statements"])
        try:
            raw = call_deepinfra(prompt, model_id, api_key)
            parsed = parse_response(raw)
            results.append({"instance_id": inst["instance_id"],
                "has_error": inst["ground_truth"].get("has_error", False),
                "error_type": inst["ground_truth"].get("error_type"),
                "error_category": inst["ground_truth"].get("error_category"),
                "error_magnitude_pct": inst["ground_truth"].get("error_magnitude_pct"),
                "detected": parsed["detected"], "error_location": parsed.get("error_location"),
                "explanation": parsed.get("explanation", ""), "parse_method": parsed["parse_method"],
                "raw_response_length": len(raw)})
        except Exception as e:
            if "credit" in str(e).lower() or "402" in str(e).lower():
                logger.error("Credits exhausted at %d/%d for %s", i, len(all_instances), display_name)
                break
            logger.error("Failed %s: %s", inst["instance_id"], e)
            results.append({"instance_id": inst["instance_id"],
                "has_error": inst["ground_truth"].get("has_error", False),
                "error_type": inst["ground_truth"].get("error_type"),
                "error_category": inst["ground_truth"].get("error_category"),
                "error_magnitude_pct": inst["ground_truth"].get("error_magnitude_pct"),
                "detected": False, "error": str(e), "parse_method": "error"})

        # Progress logging
        done = len(results)
        if done % 5 == 0 or done == 1 or done == len(all_instances):
            m = compute_metrics(results)
            o = m["overall"]
            logger.info("[%s] %d/%d acc=%.1f%% rec=%.1f%% fpr=%.1f%%", display_name, done, len(all_instances), o["accuracy"], o["recall"], o["fpr"])

        # Save checkpoint every 10 instances
        if done % 10 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({"model": display_name, "model_id": model_id, "results": results}, f, indent=2)

    # Final save
    if results:
        metrics = compute_metrics(results)
        o = metrics["overall"]
        print(f"\n{'='*60}\n{display_name} — Rounded CoT (n={o['total']})\n{'='*60}")
        print(f"  Acc={o['accuracy']}  Prec={o['precision']}  Rec={o['recall']}  F1={o['f1']}  FPR={o['fpr']}")
        print(f"  TP={o['tp']} FP={o['fp']} TN={o['tn']} FN={o['fn']}")
        with open(output_path, "w") as f:
            json.dump({"model": display_name, "model_id": model_id, "strategy": "cot",
                      "backend": "deepinfra", "ablation": "rounded",
                      "metrics": metrics, "results": results}, f, indent=2)
        logger.info("Saved: %s", output_path)
    return results


def main():
    api_key = os.environ.get("DEEPINFRA_API_KEY", "")
    if not api_key:
        logger.error("DEEPINFRA_API_KEY not set"); sys.exit(1)

    # Load the rounded benchmark data
    rounded_path = PROJECT_ROOT / "data" / "benchmark" / "benchmark_rounded.json"
    with open(rounded_path) as f:
        all_instances = json.load(f)
    logger.info("Loaded %d rounded instances", len(all_instances))

    models = [
        {"model_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
         "display_name": "Llama 4 Maverick",
         "output": "deepinfra_llama_rounded_cot_results.json",
         "checkpoint": "deepinfra_llama_rounded_cot_checkpoint.json"},
        {"model_id": "Qwen/Qwen3-235B-A22B-Instruct-2507",
         "display_name": "Qwen 3 235B",
         "output": "deepinfra_qwen_rounded_cot_results.json",
         "checkpoint": "deepinfra_qwen_rounded_cot_checkpoint.json"},
    ]

    results_dir = PROJECT_ROOT / "results"
    for cfg in models:
        logger.info("=" * 60)
        logger.info("Starting: %s (rounded ablation)", cfg["display_name"])
        run_model(
            cfg["model_id"], cfg["display_name"], all_instances, api_key,
            results_dir / cfg["output"], results_dir / cfg["checkpoint"])


if __name__ == "__main__":
    main()
