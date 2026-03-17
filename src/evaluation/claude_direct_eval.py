"""Direct Claude evaluation via Claude Code agent invocations.

This script prepares benchmark instances for evaluation. Each instance
is written to a file that can be processed by a Claude Code agent,
which provides the LLM's judgment. Results are collected and analyzed.

Usage:
    python3 src/evaluation/claude_direct_eval.py --prepare
    python3 src/evaluation/claude_direct_eval.py --analyze
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "benchmark.json"
EVAL_DIR = PROJECT_ROOT / "results" / "claude_direct"


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
    return sampled


def prepare():
    """Prepare instances for Claude Code agent evaluation."""
    with open(BENCHMARK_PATH) as f:
        all_instances = json.load(f)

    sample = stratified_sample(all_instances, max_per_cell=2)
    print(f"Total sample: {len(sample)} instances")
    print(f"  Clean: {sum(1 for i in sample if not i['ground_truth'].get('has_error'))}")
    print(f"  Error: {sum(1 for i in sample if i['ground_truth'].get('has_error'))}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Write batch files for agent processing (10 instances per batch)
    batch_size = 10
    for batch_idx in range(0, len(sample), batch_size):
        batch = sample[batch_idx:batch_idx + batch_size]
        batch_data = []
        for inst in batch:
            batch_data.append({
                "instance_id": inst["instance_id"],
                "formatted_statements": inst["formatted_statements"],
                "ground_truth": inst["ground_truth"],
            })
        batch_path = EVAL_DIR / f"batch_{batch_idx // batch_size:03d}.json"
        with open(batch_path, "w") as f:
            json.dump(batch_data, f, indent=2)
        print(f"  Wrote {len(batch)} instances to {batch_path}")

    # Also write the full sample for reference
    sample_data = [{
        "instance_id": inst["instance_id"],
        "ground_truth": inst["ground_truth"],
    } for inst in sample]
    with open(EVAL_DIR / "sample_index.json", "w") as f:
        json.dump(sample_data, f, indent=2)


def analyze():
    """Analyze collected Claude evaluation results."""
    results_path = EVAL_DIR / "all_results.json"
    if not results_path.exists():
        print(f"No results found at {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    tp = fp = tn = fn = 0
    by_category = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    by_magnitude = defaultdict(lambda: {"tp": 0, "fn": 0})

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

    print(f"\nClaude Sonnet 4 — Direct Evaluation Results")
    print(f"{'='*60}")
    print(f"Total instances: {total}")
    print(f"  Accuracy:  {accuracy*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall:    {recall*100:.1f}%")
    print(f"  F1:        {f1*100:.1f}%")
    print(f"  FPR:       {fpr*100:.1f}%")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    def _rate(d):
        t = d.get("tp", 0) + d.get("fn", 0)
        return d.get("tp", 0) / t * 100 if t else 0

    print(f"\nBy Category:")
    for cat in sorted(by_category.keys()):
        info = by_category[cat]
        print(f"  {cat:>8s}: {_rate(info):5.1f}%  tp={info.get('tp',0)} fn={info.get('fn',0)} tn={info.get('tn',0)} fp={info.get('fp',0)}")

    print(f"\nBy Magnitude:")
    for mag in ["<1%", "1-5%", "5-20%", ">20%"]:
        if mag in by_magnitude:
            info = by_magnitude[mag]
            print(f"  {mag:>6s}: {_rate(info):5.1f}%  tp={info.get('tp',0)} fn={info.get('fn',0)}")

    # Save metrics
    metrics = {
        "model": "Claude Sonnet 4",
        "model_id": "claude-sonnet-4-20250514",
        "strategy": "cot",
        "metrics": {
            "overall": {
                "accuracy": round(accuracy * 100, 1),
                "precision": round(precision * 100, 1),
                "recall": round(recall * 100, 1),
                "f1": round(f1 * 100, 1),
                "fpr": round(fpr * 100, 1),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total,
            },
            "by_category": {
                cat: {"detection_rate": _rate(counts), **counts}
                for cat, counts in sorted(by_category.items())
            },
            "by_magnitude": {
                mag: {"detection_rate": _rate(counts), **counts}
                for mag, counts in sorted(by_magnitude.items())
            },
        },
        "results": results,
    }
    output_path = PROJECT_ROOT / "results" / "claude_cot_results.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        prepare()
    elif args.analyze:
        analyze()
    else:
        parser.print_help()
