"""Run additional FinVerBench evaluations on new models via DeepInfra."""

from __future__ import annotations
import json, logging, os, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.run_deepinfra_experiments import (
    build_prompt, parse_response, call_deepinfra, compute_metrics, stratified_sample
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

NEW_MODELS = [
    {"model_id": "deepseek-ai/DeepSeek-R1-0528", "display_name": "DeepSeek R1"},
    {"model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "display_name": "Llama 4 Scout"},
    {"model_id": "google/gemma-3-27b-it", "display_name": "Gemma 3 27B"},
]


def evaluate_model(model_id, display_name, sample, api_key):
    logger.info("=" * 60)
    logger.info("Starting: %s (%s)", display_name, model_id)
    results = []
    errors = 0
    for i, inst in enumerate(sample):
        prompt = build_prompt(inst["formatted_statements"])
        try:
            raw = call_deepinfra(prompt, model_id, api_key)
            parsed = parse_response(raw)
            results.append({
                "instance_id": inst["instance_id"],
                "has_error": inst["ground_truth"].get("has_error", False),
                "error_type": inst["ground_truth"].get("error_type"),
                "error_category": inst["ground_truth"].get("error_category"),
                "error_magnitude_pct": inst["ground_truth"].get("error_magnitude_pct"),
                "detected": parsed["detected"],
                "error_location": parsed.get("error_location"),
                "explanation": parsed.get("explanation", ""),
                "parse_method": parsed["parse_method"],
                "raw_response_length": len(raw),
            })
        except Exception as e:
            if "credit" in str(e).lower() or "402" in str(e).lower():
                logger.error("Credits exhausted at %d/%d for %s", i, len(sample), display_name)
                break
            logger.error("Failed %s: %s", inst["instance_id"], e)
            errors += 1
            results.append({
                "instance_id": inst["instance_id"],
                "has_error": inst["ground_truth"].get("has_error", False),
                "error_type": inst["ground_truth"].get("error_type"),
                "error_category": inst["ground_truth"].get("error_category"),
                "error_magnitude_pct": inst["ground_truth"].get("error_magnitude_pct"),
                "detected": False,
                "error": str(e),
                "parse_method": "error",
            })

        if (i + 1) % 10 == 0 or i == 0 or i == len(sample) - 1:
            m = compute_metrics(results)
            o = m["overall"]
            logger.info(
                "[%s] %d/%d (err=%d) acc=%.1f%% rec=%.1f%% fpr=%.1f%%",
                display_name, i + 1, len(sample), errors,
                o["accuracy"], o["recall"], o["fpr"],
            )

    if results:
        metrics = compute_metrics(results)
        o = metrics["overall"]
        print(f"\n{'=' * 60}")
        print(f"{display_name} — CoT (n={o['total']}, api_errors={errors})")
        print(f"{'=' * 60}")
        print(f"  Acc={o['accuracy']}  Prec={o['precision']}  Rec={o['recall']}  F1={o['f1']}  FPR={o['fpr']}")
        print(f"  TP={o['tp']} FP={o['fp']} TN={o['tn']} FN={o['fn']}")

        safe = model_id.replace("/", "_").replace(":", "_")
        out = PROJECT_ROOT / "results" / f"deepinfra_{safe}_cot_results.json"
        with open(out, "w") as f:
            json.dump({
                "model": display_name,
                "model_id": model_id,
                "strategy": "cot",
                "metrics": metrics,
                "results": results,
                "api_errors": errors,
            }, f, indent=2)
        logger.info("Saved: %s", out)
        return metrics
    return None


def main():
    api_key = os.environ.get("DEEPINFRA_API_KEY", "")
    if not api_key:
        logger.error("DEEPINFRA_API_KEY not set")
        sys.exit(1)

    with open(PROJECT_ROOT / "data/benchmark/benchmark.json") as f:
        all_instances = json.load(f)

    sample = stratified_sample(all_instances, max_per_cell=2)
    logger.info("Sample: %d instances", len(sample))

    # Run specific model if passed as argument, otherwise run all
    if len(sys.argv) > 1:
        model_idx = int(sys.argv[1])
        cfg = NEW_MODELS[model_idx]
        evaluate_model(cfg["model_id"], cfg["display_name"], sample, api_key)
    else:
        for cfg in NEW_MODELS:
            evaluate_model(cfg["model_id"], cfg["display_name"], sample, api_key)


if __name__ == "__main__":
    main()
