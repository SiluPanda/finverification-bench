#!/usr/bin/env python3
"""
Comprehensive pairwise statistical analysis across all 9 models in the
FinVerBench benchmark.

Outputs:
  - Per-model accuracy summary
  - Pairwise McNemar's tests (exact binomial) with discordant pair counts
  - Pairwise Cohen's kappa (inter-model agreement)
  - Fleiss' kappa (all-model agreement)
  - Instance-level error analysis (universally easy/hard, unique strengths)
  - Category-level disagreement analysis
"""

import json
import os
import itertools
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest  # exact binomial for McNemar
from statsmodels.stats.inter_rater import fleiss_kappa

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

MODEL_FILES = {
    "Claude Sonnet 4":   "claude_cot_results.json",
    "Claude Opus 4.6":   "claude_cli_claude-opus-4-6_cot_results.json",
    "Claude Sonnet 4.6": "claude_cli_claude-sonnet-4-6_cot_results.json",
    "GPT-4.1":           "openrouter_openai_gpt-4.1_cot_results.json",
    "DeepSeek V3.2":     "deepinfra_deepseek-ai_DeepSeek-V3.2_cot_results.json",
    "DeepSeek R1":       "deepinfra_deepseek-ai_DeepSeek-R1-0528_cot_results.json",
    "Qwen 3 235B":       "deepinfra_Qwen_Qwen3-235B-A22B-Instruct-2507_cot_results.json",
    "Gemini 2.5 Pro":    "openrouter_google_gemini-2.5-pro-preview-03-25_cot_results.json",
    "MiniMax M2.5":      "minimax_cot_results.json",
    "Llama 4 Maverick":  "deepinfra_meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8_cot_results.json",
    "Llama 4 Scout":     "deepinfra_meta-llama_Llama-4-Scout-17B-16E-Instruct_cot_results.json",
    "Gemma 3 27B":       "deepinfra_google_gemma-3-27b-it_cot_results.json",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def normalize_id(iid):
    """Normalize instance IDs: NFKC normalization + normalize whitespace and quotes."""
    s = unicodedata.normalize("NFKC", iid)
    s = s.replace("\xa0", " ")          # non-breaking space -> space
    s = s.replace("\u2019", "'")         # right single quotation mark -> ASCII apostrophe
    s = s.replace("\u2018", "'")         # left single quotation mark -> ASCII apostrophe
    return s


def load_all_models():
    """Return {model_name: {instance_id: {'has_error': bool, 'detected': bool,
    'correct': bool, 'error_category': str|None}}}"""
    models = {}
    for name, fname in MODEL_FILES.items():
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        with open(path) as f:
            data = json.load(f)
        instances = {}
        for r in data["results"]:
            iid = normalize_id(r["instance_id"])
            has_error = r["has_error"]
            detected = r["detected"]
            correct = (has_error == detected)
            instances[iid] = {
                "has_error": has_error,
                "detected": detected,
                "correct": correct,
                "error_category": r.get("error_category"),
                "error_type": r.get("error_type"),
                "error_magnitude_pct": r.get("error_magnitude_pct"),
            }
        models[name] = instances
    return models


def get_shared_ids(models):
    """Return sorted list of instance IDs shared across ALL models."""
    all_ids = [set(v.keys()) for v in models.values()]
    shared = set.intersection(*all_ids)
    return sorted(shared)


# ---------------------------------------------------------------------------
# McNemar's test (exact binomial for small counts)
# ---------------------------------------------------------------------------

def mcnemar_exact(correct_a, correct_b):
    """
    Exact McNemar's test using the binomial distribution.

    Returns (chi2_approx, p_value, b_count, c_count) where:
      b = number of instances A correct & B wrong
      c = number of instances A wrong & B correct
    """
    assert len(correct_a) == len(correct_b)
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    n = b + c
    if n == 0:
        return 0.0, 1.0, b, c
    # Exact two-sided binomial test under H0: p=0.5
    p_value = binomtest(b, n, 0.5).pvalue
    # Chi-squared approximation (with continuity correction) for reference
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    return chi2, p_value, b, c


# ---------------------------------------------------------------------------
# Cohen's kappa (two raters)
# ---------------------------------------------------------------------------

def cohens_kappa(pred_a, pred_b):
    """Cohen's kappa between two binary prediction vectors."""
    assert len(pred_a) == len(pred_b)
    n = len(pred_a)
    # Observed agreement
    agree = sum(1 for a, b in zip(pred_a, pred_b) if a == b)
    po = agree / n
    # Expected agreement by chance
    a1 = sum(pred_a) / n
    a0 = 1 - a1
    b1 = sum(pred_b) / n
    b0 = 1 - b1
    pe = a1 * b1 + a0 * b0
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


# ---------------------------------------------------------------------------
# Fleiss' kappa (multiple raters)
# ---------------------------------------------------------------------------

def compute_fleiss_kappa(models, shared_ids):
    """
    Fleiss' kappa treating each model as a rater assigning each instance
    to {correct, incorrect}.
    """
    n_subjects = len(shared_ids)
    n_raters = len(models)
    # Build the rating matrix: rows = instances, cols = [n_correct, n_incorrect]
    table = np.zeros((n_subjects, 2), dtype=int)
    model_names = list(models.keys())
    for i, iid in enumerate(shared_ids):
        n_correct = sum(1 for m in model_names if models[m][iid]["correct"])
        table[i, 0] = n_correct
        table[i, 1] = n_raters - n_correct
    kappa = fleiss_kappa(table, method="fleiss")
    return kappa


# ---------------------------------------------------------------------------
# Instance-level analysis
# ---------------------------------------------------------------------------

def instance_analysis(models, shared_ids):
    """Identify universally easy/hard instances and unique strengths."""
    model_names = list(models.keys())
    n_models = len(model_names)

    all_correct = []
    all_wrong = []
    sonnet4_only = []  # Only Claude Sonnet 4 correct
    per_instance_correct_count = {}

    for iid in shared_ids:
        corrects = [m for m in model_names if models[m][iid]["correct"]]
        wrongs = [m for m in model_names if not models[m][iid]["correct"]]
        per_instance_correct_count[iid] = len(corrects)

        if len(corrects) == n_models:
            all_correct.append(iid)
        elif len(corrects) == 0:
            all_wrong.append(iid)

        if (len(corrects) == 1 and corrects[0] == "Claude Sonnet 4"):
            sonnet4_only.append(iid)

    return all_correct, all_wrong, sonnet4_only, per_instance_correct_count


def category_disagreement(models, shared_ids):
    """For error-containing instances, compute per-category model disagreement."""
    model_names = list(models.keys())
    # We use the ground-truth category from the first model that has it
    category_map = {}
    for iid in shared_ids:
        for m in model_names:
            cat = models[m][iid].get("error_category")
            if cat:
                category_map[iid] = cat
                break

    # Only look at instances with errors (has_error=True)
    cat_stats = defaultdict(lambda: {"total": 0, "disagreements": 0, "per_model_correct": Counter()})
    for iid in shared_ids:
        has_error = models[model_names[0]][iid]["has_error"]
        if not has_error:
            continue
        cat = category_map.get(iid, "unknown")
        corrects = [m for m in model_names if models[m][iid]["correct"]]
        n_correct = len(corrects)
        cat_stats[cat]["total"] += 1
        # Disagreement = not all agree
        if 0 < n_correct < len(model_names):
            cat_stats[cat]["disagreements"] += 1
        for m in corrects:
            cat_stats[cat]["per_model_correct"][m] += 1

    return dict(cat_stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    models = load_all_models()
    model_names = list(models.keys())
    shared_ids = get_shared_ids(models)
    n = len(shared_ids)
    print(f"Loaded {len(models)} models, {n} shared instances.\n")

    # ---- Per-model accuracy ------------------------------------------------
    print("=" * 80)
    print("PER-MODEL ACCURACY SUMMARY")
    print("=" * 80)
    header = f"{'Model':<22} {'Accuracy':>8}  {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}  {'Prec':>6} {'Rec':>6} {'F1':>6}"
    print(header)
    print("-" * len(header))
    for m in model_names:
        tp = sum(1 for iid in shared_ids if models[m][iid]["has_error"] and models[m][iid]["detected"])
        tn = sum(1 for iid in shared_ids if not models[m][iid]["has_error"] and not models[m][iid]["detected"])
        fp = sum(1 for iid in shared_ids if not models[m][iid]["has_error"] and models[m][iid]["detected"])
        fn = sum(1 for iid in shared_ids if models[m][iid]["has_error"] and not models[m][iid]["detected"])
        acc = (tp + tn) / n
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{m:<22} {acc:>7.1%}  {tp:>4} {tn:>4} {fp:>4} {fn:>4}  {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")
    print()

    # ---- Pairwise McNemar's test -------------------------------------------
    print("=" * 80)
    print("PAIRWISE McNEMAR'S TEST (exact binomial)")
    print("=" * 80)
    print(f"{'Model A':<22} {'Model B':<22} {'b(A>B)':>6} {'c(B>A)':>6} {'chi2':>8} {'p-value':>10} {'Sig':>5}")
    print("-" * 82)

    mcnemar_results = {}
    for a, b in itertools.combinations(model_names, 2):
        correct_a = [models[a][iid]["correct"] for iid in shared_ids]
        correct_b = [models[b][iid]["correct"] for iid in shared_ids]
        chi2, p, b_count, c_count = mcnemar_exact(correct_a, correct_b)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{a:<22} {b:<22} {b_count:>6} {c_count:>6} {chi2:>8.3f} {p:>10.4f} {sig:>5}")
        mcnemar_results[(a, b)] = {"chi2": chi2, "p": p, "b": b_count, "c": c_count}
    print()
    print("  b(A>B) = instances A correct, B wrong")
    print("  c(B>A) = instances B correct, A wrong")
    print("  Significance: * p<0.05, ** p<0.01, *** p<0.001")
    print()

    # ---- Pairwise Cohen's kappa --------------------------------------------
    print("=" * 80)
    print("PAIRWISE COHEN'S KAPPA (prediction agreement)")
    print("=" * 80)
    # Print as a matrix
    short_names = {m: m[:16] for m in model_names}
    w = 17
    print(f"{'':>{w}}", end="")
    for m in model_names:
        print(f"{short_names[m]:>{w}}", end="")
    print()

    kappa_matrix = {}
    for a in model_names:
        print(f"{short_names[a]:>{w}}", end="")
        for b in model_names:
            if a == b:
                print(f"{'1.000':>{w}}", end="")
                kappa_matrix[(a, b)] = 1.0
            else:
                pred_a = [models[a][iid]["detected"] for iid in shared_ids]
                pred_b = [models[b][iid]["detected"] for iid in shared_ids]
                k = cohens_kappa(pred_a, pred_b)
                kappa_matrix[(a, b)] = k
                print(f"{k:>{w}.3f}", end="")
        print()
    print()
    print("  Interpretation: <0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate,")
    print("                  0.61-0.80 substantial, 0.81-1.00 almost perfect")
    print()

    # ---- Fleiss' kappa -----------------------------------------------------
    print("=" * 80)
    print("FLEISS' KAPPA (all-model agreement on correctness)")
    print("=" * 80)
    fk = compute_fleiss_kappa(models, shared_ids)
    print(f"  Fleiss' kappa = {fk:.4f}")
    if fk < 0.20:
        interp = "slight agreement"
    elif fk < 0.40:
        interp = "fair agreement"
    elif fk < 0.60:
        interp = "moderate agreement"
    elif fk < 0.80:
        interp = "substantial agreement"
    else:
        interp = "almost perfect agreement"
    print(f"  Interpretation: {interp}")
    print()

    # ---- Fleiss' kappa on raw predictions -----------------------------------
    print("=" * 80)
    print("FLEISS' KAPPA (all-model agreement on predictions: error_detected)")
    print("=" * 80)
    n_subjects = len(shared_ids)
    n_raters = len(model_names)
    table_pred = np.zeros((n_subjects, 2), dtype=int)
    for i, iid in enumerate(shared_ids):
        n_detected = sum(1 for m in model_names if models[m][iid]["detected"])
        table_pred[i, 0] = n_detected        # detected=True
        table_pred[i, 1] = n_raters - n_detected  # detected=False
    fk_pred = fleiss_kappa(table_pred, method="fleiss")
    print(f"  Fleiss' kappa = {fk_pred:.4f}")
    print()

    # ---- Instance-level analysis -------------------------------------------
    print("=" * 80)
    print("INSTANCE-LEVEL ANALYSIS")
    print("=" * 80)
    all_correct, all_wrong, sonnet4_only, per_instance_counts = instance_analysis(models, shared_ids)

    print(f"\n  Instances ALL {len(model_names)} models get RIGHT:  {len(all_correct)} / {n}")
    print(f"  Instances ALL {len(model_names)} models get WRONG:  {len(all_wrong)} / {n}")
    print(f"  Instances ONLY Claude Sonnet 4 gets right: {len(sonnet4_only)}")

    # Distribution of "how many models get it right"
    count_dist = Counter(per_instance_counts.values())
    print(f"\n  Distribution of per-instance correct model count:")
    for k in sorted(count_dist.keys()):
        bar = "#" * count_dist[k]
        print(f"    {k:>2} models correct: {count_dist[k]:>3}  {bar}")

    # List universally wrong instances
    if all_wrong:
        print(f"\n  Universally WRONG instances ({len(all_wrong)}):")
        for iid in all_wrong:
            info = models[model_names[0]][iid]
            cat = info.get("error_category", "N/A")
            has_err = info["has_error"]
            label = "has_error" if has_err else "clean"
            print(f"    - {iid}  [{label}, cat={cat}]")

    # List Sonnet 4-only instances
    if sonnet4_only:
        print(f"\n  Instances ONLY Claude Sonnet 4 gets right ({len(sonnet4_only)}):")
        for iid in sonnet4_only:
            info = models["Claude Sonnet 4"][iid]
            cat = info.get("error_category", "N/A")
            has_err = info["has_error"]
            label = "has_error" if has_err else "clean"
            print(f"    - {iid}  [{label}, cat={cat}]")

    # Unique correct per model (instances only that model gets right)
    print(f"\n  Unique correct instances per model (only that model gets it right):")
    for m in model_names:
        unique = []
        for iid in shared_ids:
            corrects = [mm for mm in model_names if models[mm][iid]["correct"]]
            if len(corrects) == 1 and corrects[0] == m:
                unique.append(iid)
        print(f"    {m:<22}: {len(unique):>2} instances")
    print()

    # ---- Category-level disagreement ---------------------------------------
    print("=" * 80)
    print("ERROR CATEGORY DISAGREEMENT ANALYSIS")
    print("(Only error-containing instances)")
    print("=" * 80)
    cat_stats = category_disagreement(models, shared_ids)
    print(f"\n  {'Category':<28} {'Total':>5} {'Disagree':>8} {'Rate':>6}")
    print(f"  {'-'*28} {'-----':>5} {'--------':>8} {'------':>6}")
    for cat in sorted(cat_stats.keys(), key=lambda c: cat_stats[c]["disagreements"] / max(cat_stats[c]["total"], 1), reverse=True):
        s = cat_stats[cat]
        rate = s["disagreements"] / s["total"] if s["total"] > 0 else 0
        print(f"  {cat:<28} {s['total']:>5} {s['disagreements']:>8} {rate:>6.1%}")

    # Detailed per-model recall by category
    print(f"\n  Per-model detection rate by error category:")
    cats = sorted(cat_stats.keys())
    header = f"  {'Category':<28}"
    for m in model_names:
        header += f" {m[:10]:>10}"
    print(header)
    print(f"  {'-'*28}" + " ----------" * len(model_names))
    for cat in cats:
        s = cat_stats[cat]
        line = f"  {cat:<28}"
        for m in model_names:
            rate = s["per_model_correct"][m] / s["total"] if s["total"] > 0 else 0
            line += f" {rate:>9.0%} "
            # ^-- note: "correct" for error instances means detected=True
        print(line)
    print()

    # ---- Significant pairs summary -----------------------------------------
    print("=" * 80)
    print("SIGNIFICANT PAIRWISE DIFFERENCES (p < 0.05)")
    print("=" * 80)
    sig_pairs = [(a, b, v) for (a, b), v in mcnemar_results.items() if v["p"] < 0.05]
    sig_pairs.sort(key=lambda x: x[2]["p"])
    if sig_pairs:
        for a, b, v in sig_pairs:
            direction = f"{a} > {b}" if v["b"] > v["c"] else f"{b} > {a}"
            print(f"  {a} vs {b}: p={v['p']:.4f}, direction: {direction}  (b={v['b']}, c={v['c']})")
    else:
        print("  No significant pairwise differences at p < 0.05.")
    print()

    # ---- Bonferroni correction ---------------------------------------------
    n_comparisons = len(mcnemar_results)
    bonferroni_threshold = 0.05 / n_comparisons
    print(f"  Bonferroni-corrected threshold (alpha=0.05, {n_comparisons} comparisons): {bonferroni_threshold:.5f}")
    sig_bonf = [(a, b, v) for (a, b), v in mcnemar_results.items() if v["p"] < bonferroni_threshold]
    sig_bonf.sort(key=lambda x: x[2]["p"])
    if sig_bonf:
        print(f"  Significant after Bonferroni correction ({len(sig_bonf)}):")
        for a, b, v in sig_bonf:
            direction = f"{a} > {b}" if v["b"] > v["c"] else f"{b} > {a}"
            print(f"    {a} vs {b}: p={v['p']:.6f}, direction: {direction}")
    else:
        print("  No pairs survive Bonferroni correction.")
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
