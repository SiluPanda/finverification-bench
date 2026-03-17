"""
McNemar's test comparing Claude Sonnet 4 vs GPT-4.1 on FinVerBench (108 instances).
Both models evaluated on the same instances with the same chain-of-thought prompt.
"""

import json
import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import cohen_kappa_score

# Load results
with open("/Users/silupanda/Downloads/finverification-bench/results/claude_cot_results.json") as f:
    claude_data = json.load(f)

with open("/Users/silupanda/Downloads/finverification-bench/results/openrouter_openai_gpt-4.1_cot_results.json") as f:
    gpt_data = json.load(f)

import unicodedata

def normalize_id(s):
    """Normalize unicode characters (e.g., non-breaking spaces, curly quotes) to ASCII equivalents."""
    s = unicodedata.normalize("NFKD", s)
    # Replace curly/smart apostrophes with straight apostrophe
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u02bc", "'")
    return s

claude_results = {normalize_id(r["instance_id"]): r for r in claude_data["results"]}
gpt_results = {normalize_id(r["instance_id"]): r for r in gpt_data["results"]}

# Verify same instances
assert set(claude_results.keys()) == set(gpt_results.keys()), \
    f"Instance sets differ! Only in Claude: {set(claude_results.keys()) - set(gpt_results.keys())}, Only in GPT: {set(gpt_results.keys()) - set(claude_results.keys())}"
print(f"Total instances: {len(claude_results)}")

# Build per-instance correctness
# Correct means: detected == has_error (i.e., correctly classified)
all_ids = sorted(claude_results.keys())

claude_correct = []
gpt_correct = []
claude_detected = []
gpt_detected = []
has_error_list = []

for iid in all_ids:
    cr = claude_results[iid]
    gr = gpt_results[iid]

    has_error = cr["has_error"]
    assert has_error == gr["has_error"], f"Ground truth mismatch for {iid}"

    c_det = cr["detected"]
    g_det = gr["detected"]

    c_correct = (c_det == has_error)
    g_correct = (g_det == has_error)

    claude_correct.append(c_correct)
    gpt_correct.append(g_correct)
    claude_detected.append(c_det)
    gpt_detected.append(g_det)
    has_error_list.append(has_error)

claude_correct = np.array(claude_correct)
gpt_correct = np.array(gpt_correct)
claude_detected = np.array(claude_detected)
gpt_detected = np.array(gpt_detected)
has_error_list = np.array(has_error_list)

# ============================================================
# 1. Overall McNemar's Test (all 108 instances)
# ============================================================
print("\n" + "="*60)
print("OVERALL McNEMAR'S TEST (N=108)")
print("="*60)

# 2x2 contingency table
both_correct = np.sum(claude_correct & gpt_correct)
claude_only = np.sum(claude_correct & ~gpt_correct)  # Claude right, GPT wrong
gpt_only = np.sum(~claude_correct & gpt_correct)      # GPT right, Claude wrong
both_wrong = np.sum(~claude_correct & ~gpt_correct)

print(f"\nContingency Table:")
print(f"                        GPT-4.1 Correct    GPT-4.1 Wrong")
print(f"Claude Correct          {both_correct:>10d}          {claude_only:>10d}")
print(f"Claude Wrong            {gpt_only:>10d}          {both_wrong:>10d}")
print(f"\nDiscordant pairs: b={claude_only} (Claude right, GPT wrong), c={gpt_only} (GPT right, Claude wrong)")
print(f"Total discordant: {claude_only + gpt_only}")

# McNemar's with continuity correction
table = np.array([[both_correct, claude_only],
                   [gpt_only, both_wrong]])

result_cc = mcnemar(table, exact=False, correction=True)
print(f"\nMcNemar's test (with continuity correction):")
print(f"  Chi-squared statistic: {result_cc.statistic:.4f}")
print(f"  p-value: {result_cc.pvalue:.6e}")

# McNemar's exact test (binomial test on discordant pairs)
# Under H0, b ~ Binomial(b+c, 0.5)
n_discordant = claude_only + gpt_only
if n_discordant > 0:
    exact_p = binomtest(claude_only, n_discordant, 0.5).pvalue
    print(f"\nMcNemar's exact test (binomial):")
    print(f"  n_discordant = {n_discordant}, b = {claude_only}")
    print(f"  p-value: {exact_p:.6e}")
else:
    print("\nNo discordant pairs - exact test not applicable")

# Also compute using scipy exact mcnemar
result_exact = mcnemar(table, exact=True)
print(f"\nMcNemar's exact test (statsmodels):")
print(f"  statistic: {result_exact.statistic}")
print(f"  p-value: {result_exact.pvalue:.6e}")

# Cohen's kappa between the two models' predictions
kappa = cohen_kappa_score(claude_detected, gpt_detected)
print(f"\nCohen's kappa (agreement between models' predictions): {kappa:.4f}")

# Also compute kappa on correctness
kappa_correct = cohen_kappa_score(claude_correct.astype(int), gpt_correct.astype(int))
print(f"Cohen's kappa (agreement on correctness): {kappa_correct:.4f}")

# Accuracy summary
print(f"\nClaude accuracy: {np.mean(claude_correct)*100:.1f}% ({np.sum(claude_correct)}/{len(claude_correct)})")
print(f"GPT-4.1 accuracy: {np.mean(gpt_correct)*100:.1f}% ({np.sum(gpt_correct)}/{len(gpt_correct)})")

# ============================================================
# 2. CLEAN instances only (McNemar on FPR)
# ============================================================
print("\n" + "="*60)
print("CLEAN INSTANCES ONLY (N=43) - FPR Comparison")
print("="*60)

clean_mask = ~has_error_list
n_clean = np.sum(clean_mask)
print(f"Number of clean instances: {n_clean}")

# For clean instances, "correct" = not detected (true negative)
claude_clean_correct = ~claude_detected[clean_mask]  # TN
gpt_clean_correct = ~gpt_detected[clean_mask]        # TN

bc = np.sum(claude_clean_correct & gpt_clean_correct)      # both TN
co = np.sum(claude_clean_correct & ~gpt_clean_correct)     # Claude TN, GPT FP
go = np.sum(~claude_clean_correct & gpt_clean_correct)     # Claude FP, GPT TN
bw = np.sum(~claude_clean_correct & ~gpt_clean_correct)    # both FP

print(f"\nContingency Table (correct = true negative):")
print(f"                        GPT-4.1 TN         GPT-4.1 FP")
print(f"Claude TN               {bc:>10d}          {co:>10d}")
print(f"Claude FP               {go:>10d}          {bw:>10d}")
print(f"\nDiscordant pairs: b={co} (Claude TN/GPT FP), c={go} (Claude FP/GPT TN)")

print(f"\nClaude FPR: {np.sum(claude_detected[clean_mask])}/{n_clean} = {np.mean(claude_detected[clean_mask])*100:.1f}%")
print(f"GPT-4.1 FPR: {np.sum(gpt_detected[clean_mask])}/{n_clean} = {np.mean(gpt_detected[clean_mask])*100:.1f}%")

table_clean = np.array([[bc, co], [go, bw]])

if co + go > 0:
    result_clean_cc = mcnemar(table_clean, exact=False, correction=True)
    result_clean_exact = mcnemar(table_clean, exact=True)
    print(f"\nMcNemar's test (continuity correction):")
    print(f"  Chi-squared: {result_clean_cc.statistic:.4f}")
    print(f"  p-value: {result_clean_cc.pvalue:.6e}")
    print(f"\nMcNemar's exact test:")
    print(f"  p-value: {result_clean_exact.pvalue:.6e}")
else:
    print("\nNo discordant pairs on clean instances")

# ============================================================
# 3. ERROR instances only (McNemar on Recall)
# ============================================================
print("\n" + "="*60)
print("ERROR INSTANCES ONLY (N=65) - Recall Comparison")
print("="*60)

error_mask = has_error_list
n_error = np.sum(error_mask)
print(f"Number of error instances: {n_error}")

# For error instances, "correct" = detected (true positive)
claude_err_correct = claude_detected[error_mask]   # TP
gpt_err_correct = gpt_detected[error_mask]         # TP

bc_e = np.sum(claude_err_correct & gpt_err_correct)        # both TP
co_e = np.sum(claude_err_correct & ~gpt_err_correct)       # Claude TP, GPT FN
go_e = np.sum(~claude_err_correct & gpt_err_correct)       # Claude FN, GPT TP
bw_e = np.sum(~claude_err_correct & ~gpt_err_correct)      # both FN

print(f"\nContingency Table (correct = true positive):")
print(f"                        GPT-4.1 TP         GPT-4.1 FN")
print(f"Claude TP               {bc_e:>10d}          {co_e:>10d}")
print(f"Claude FN               {go_e:>10d}          {bw_e:>10d}")
print(f"\nDiscordant pairs: b={co_e} (Claude TP/GPT FN), c={go_e} (Claude FN/GPT TP)")

print(f"\nClaude Recall: {np.sum(claude_detected[error_mask])}/{n_error} = {np.mean(claude_detected[error_mask])*100:.1f}%")
print(f"GPT-4.1 Recall: {np.sum(gpt_detected[error_mask])}/{n_error} = {np.mean(gpt_detected[error_mask])*100:.1f}%")

table_error = np.array([[bc_e, co_e], [go_e, bw_e]])

if co_e + go_e > 0:
    result_err_cc = mcnemar(table_error, exact=False, correction=True)
    result_err_exact = mcnemar(table_error, exact=True)
    print(f"\nMcNemar's test (continuity correction):")
    print(f"  Chi-squared: {result_err_cc.statistic:.4f}")
    print(f"  p-value: {result_err_cc.pvalue:.6e}")
    print(f"\nMcNemar's exact test:")
    print(f"  p-value: {result_err_exact.pvalue:.6e}")
else:
    print("\nNo discordant pairs on error instances")

# ============================================================
# 4. Detail: which instances are discordant?
# ============================================================
print("\n" + "="*60)
print("DISCORDANT PAIR DETAILS")
print("="*60)

print("\n--- Claude correct, GPT-4.1 wrong ---")
for iid in all_ids:
    cr = claude_results[iid]
    gr = gpt_results[iid]
    c_ok = (cr["detected"] == cr["has_error"])
    g_ok = (gr["detected"] == gr["has_error"])
    if c_ok and not g_ok:
        label = "CLEAN" if not cr["has_error"] else f"ERROR ({cr['error_category']}, {cr['error_type']})"
        print(f"  {iid} [{label}]")

print("\n--- GPT-4.1 correct, Claude wrong ---")
for iid in all_ids:
    cr = claude_results[iid]
    gr = gpt_results[iid]
    c_ok = (cr["detected"] == cr["has_error"])
    g_ok = (gr["detected"] == gr["has_error"])
    if not c_ok and g_ok:
        label = "CLEAN" if not cr["has_error"] else f"ERROR ({cr['error_category']}, {cr['error_type']})"
        print(f"  {iid} [{label}]")

# ============================================================
# 5. Summary table for paper
# ============================================================
print("\n" + "="*60)
print("SUMMARY FOR PAPER")
print("="*60)

print(f"""
Overall (N=108):
  Claude Sonnet 4 accuracy: {np.mean(claude_correct)*100:.1f}%
  GPT-4.1 accuracy: {np.mean(gpt_correct)*100:.1f}%
  Contingency: a={both_correct}, b={claude_only}, c={gpt_only}, d={both_wrong}
  McNemar chi2 (cc): {result_cc.statistic:.2f}, p = {result_cc.pvalue:.2e}
  McNemar exact p: {result_exact.pvalue:.2e}
  Cohen's kappa (predictions): {kappa:.3f}

Clean instances (N={n_clean}, FPR comparison):
  Claude FPR: {np.mean(claude_detected[clean_mask])*100:.1f}%
  GPT-4.1 FPR: {np.mean(gpt_detected[clean_mask])*100:.1f}%
  Discordant: b={co}, c={go}
  McNemar exact p: {result_clean_exact.pvalue:.2e}

Error instances (N={n_error}, Recall comparison):
  Claude Recall: {np.mean(claude_detected[error_mask])*100:.1f}%
  GPT-4.1 Recall: {np.mean(gpt_detected[error_mask])*100:.1f}%
  Discordant: b={co_e}, c={go_e}
  McNemar exact p: {result_err_exact.pvalue:.2e}
""")
