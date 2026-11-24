"""
Run: python generate_result_graphs.py
Saves all result graphs to RESULT/graphs/
"""
import os, json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "RESULT/graphs"
os.makedirs(OUT, exist_ok=True)

SYS   = ["Adaptive", "Normal LLM", "Standard RAG", "Self-Reflection", "Fixed Threshold"]
SCOLS = ["#2196f3", "#9e9e9e", "#4caf50", "#ff9800", "#9c27b0"]

def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {name}.png")


# ── 1. Tier 1 — Accuracy & Hallucination Rate ────────────────────────────────
acc   = [0.294, 0.252, 0.360, 0.256, 0.262]
halluc= [0.509, 0.748, 0.640, 0.744, 0.738]

x = np.arange(len(SYS))
w = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w/2, acc,   w, label="Accuracy ↑",          color=SCOLS, alpha=0.9)
b2 = ax.bar(x + w/2, halluc, w, label="Hallucination Rate ↓", color=SCOLS, alpha=0.4, hatch="//")
ax.set_xticks(x); ax.set_xticklabels(SYS, fontsize=9)
ax.set_ylim(0, 1); ax.set_ylabel("Score")
ax.set_title("Tier 1 — Answer Quality: Accuracy vs Hallucination Rate")
ax.legend(); ax.axhline(0, color="black", linewidth=0.5)
for bar in b1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.2f}", ha="center", fontsize=8)
for bar in b2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.2f}", ha="center", fontsize=8)
save("tier1_accuracy_hallucination")


# ── 2. Tier 2 — Calibration (ECE, Brier, AUROC) ─────────────────────────────
t2_sys  = ["Adaptive", "Fixed Threshold"]
t2_cols = ["#2196f3", "#9c27b0"]
ece    = [0.476, 0.519]
brier  = [0.430, 0.468]
auroc  = [0.657, 0.623]

x2 = np.arange(len(t2_sys))
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for ax, vals, title, better in zip(axes,
        [ece, brier, auroc],
        ["ECE ↓ (lower=better)", "Brier Score ↓", "AUROC ↑ (higher=better)"],
        ["low", "low", "high"]):
    bars = ax.bar(t2_sys, vals, color=t2_cols, alpha=0.85)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 1)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
fig.suptitle("Tier 2 — Uncertainty Calibration", fontsize=12, fontweight="bold")
save("tier2_calibration")


# ── 3. Tier 3 — Action Accuracy & Unnecessary Retrieval ─────────────────────
act_acc = [0.42, 0.17, 0.40, 0.17, 0.15]
unnec   = [0.004, 0.000, 0.173, 0.000, 0.039]

x = np.arange(len(SYS))
fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w/2, act_acc, w, label="Action Accuracy ↑", color=SCOLS, alpha=0.9)
b2 = ax.bar(x + w/2, unnec,   w, label="Unnecessary Retrieval ↓", color=SCOLS, alpha=0.4, hatch="//")
ax.set_xticks(x); ax.set_xticklabels(SYS, fontsize=9)
ax.set_ylim(0, 0.7); ax.set_ylabel("Rate")
ax.set_title("Tier 3 — Decision Quality: Action Accuracy vs Unnecessary Retrieval")
ax.legend()
for bar in b1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{bar.get_height():.2f}", ha="center", fontsize=8)
for bar in b2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{bar.get_height():.3f}", ha="center", fontsize=8)
save("tier3_decision_quality")


# ── 4. Tier 3 — Per-action P/R/F1 for Adaptive ──────────────────────────────
actions = ["answer", "retrieve", "verify", "clarify", "abstain"]
prec  = [0.40, 0.57, 0.37, 0.37, 0.31]
rec   = [0.90, 0.29, 0.31, 0.55, 0.16]
f1    = [0.55, 0.39, 0.34, 0.44, 0.21]
acols = ["#4caf50","#2196f3","#ff9800","#9c27b0","#f44336"]

x = np.arange(len(actions))
w3 = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w3, prec, w3, label="Precision", color=acols, alpha=0.9)
ax.bar(x,      rec,  w3, label="Recall",    color=acols, alpha=0.6)
ax.bar(x + w3, f1,   w3, label="F1",        color=acols, alpha=0.4, hatch="//")
ax.set_xticks(x); ax.set_xticklabels([a.upper() for a in actions])
ax.set_ylim(0, 0.8); ax.set_ylabel("Score")
ax.set_title("Tier 3 — Adaptive Agent: Per-Action Precision / Recall / F1")
handles = [mpatches.Patch(alpha=0.9, label="Precision", color="#888"),
           mpatches.Patch(alpha=0.6, label="Recall",    color="#888"),
           mpatches.Patch(alpha=0.4, label="F1",        color="#888", hatch="//")]
ax.legend(handles=handles)
save("tier3_per_action_prf")


# ── 5. Tier 4 — Efficiency ───────────────────────────────────────────────────
llm_calls = [6.53, 1.00, 1.00, 2.00, 7.08]
latency   = [2.356, 0.874, 1.271, 3.198, 4.347]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
bars1 = ax1.bar(SYS, llm_calls, color=SCOLS, alpha=0.85)
ax1.set_title("Avg LLM Calls ↓"); ax1.set_ylabel("Calls")
ax1.set_xticklabels(SYS, rotation=15, ha="right", fontsize=8)
for bar in bars1: ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f"{bar.get_height():.2f}", ha="center", fontsize=8)

bars2 = ax2.bar(SYS, latency, color=SCOLS, alpha=0.85)
ax2.set_title("Avg Latency (s) ↓"); ax2.set_ylabel("Seconds")
ax2.set_xticklabels(SYS, rotation=15, ha="right", fontsize=8)
for bar in bars2: ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f"{bar.get_height():.2f}", ha="center", fontsize=8)

fig.suptitle("Tier 4 — Efficiency", fontsize=12, fontweight="bold")
save("tier4_efficiency")


# ── 6. Efficiency Frontier — Accuracy vs LLM Calls ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for i, (s, c, acc_v, llm) in enumerate(zip(SYS, SCOLS, acc, llm_calls)):
    ax.scatter(llm, acc_v, color=c, s=180, zorder=5)
    ax.annotate(s, (llm, acc_v), textcoords="offset points", xytext=(8, 4), fontsize=9, color=c)
ax.set_xlabel("Avg LLM Calls (lower = cheaper)")
ax.set_ylabel("Accuracy (higher = better)")
ax.set_title("Efficiency Frontier — Accuracy vs Computational Cost")
ax.set_xlim(0, 10); ax.set_ylim(0, 0.8)
ax.axvline(6.53, color="#2196f3", linestyle="--", alpha=0.3)
ax.axhline(0.294, color="#2196f3", linestyle="--", alpha=0.3)
save("efficiency_frontier")


# ── 7. Threshold Sweep ───────────────────────────────────────────────────────
with open("data/processed/threshold_sweep_results.json") as f:
    sweep = json.load(f)
thresholds = [r["threshold"] for r in sweep]
sw_acc     = [r["accuracy"] for r in sweep]
sw_ret     = [r["retrieval_rate"] for r in sweep]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, sw_acc, "o-", color="#2196f3", linewidth=2, label="Accuracy ↑")
ax.plot(thresholds, sw_ret, "s--", color="#ff9800", linewidth=2, label="Retrieval Rate ↓")
ax.axvline(0.1, color="#2196f3", linestyle=":", alpha=0.4, label="Best accuracy threshold (0.1)")
ax.set_xlabel("Uncertainty Threshold"); ax.set_ylabel("Score")
ax.set_title("Threshold Sweep — Reliability vs Cost Trade-off")
ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.2)
save("threshold_sweep")


# ── 8. Lambda Sweep ──────────────────────────────────────────────────────────
with open("data/processed/lambda_sweep_results.json") as f:
    lsweep = json.load(f)
lambdas  = [r["lambda"] for r in lsweep]
lm_acc   = [r["action_accuracy"] for r in lsweep]
lm_ret   = [r["retrieve_rate"] for r in lsweep]
lm_ver   = [r["verify_rate"] for r in lsweep]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(lambdas, lm_acc, "o-", color="#2196f3", linewidth=2, label="Action Accuracy ↑")
ax.plot(lambdas, lm_ret, "s--", color="#ff9800", linewidth=2, label="Retrieve Rate")
ax.plot(lambdas, lm_ver, "^:", color="#4caf50", linewidth=2, label="Verify Rate")
ax.axvline(0.05, color="#2196f3", linestyle=":", alpha=0.4, label="Optimal λ=0.05")
ax.set_xlabel("Lambda (cost penalty)"); ax.set_ylabel("Rate")
ax.set_title("Lambda Sweep — Cost Penalty vs Action Accuracy")
ax.set_ylim(0, 0.6); ax.legend(); ax.grid(alpha=0.2)
save("lambda_sweep")


# ── 9. Ablation Study ────────────────────────────────────────────────────────
with open("data/processed/ablation_results.json") as f:
    ablation = json.load(f)

labels_map = {
    "full_model":               "Full Model",
    "remove_uncertainty":       "− Uncertainty",
    "remove_evidence_coverage": "− Evidence Coverage",
    "remove_contradiction":     "− Contradiction",
    "remove_ambiguity":         "− Ambiguity",
    "remove_complexity":        "− Complexity",
}
baseline = ablation["full_model"]
ab_labels = [labels_map[k] for k in ablation]
ab_vals   = list(ablation.values())
ab_colors = ["#2196f3" if k == "full_model" else ("#f44336" if v < baseline else "#4caf50")
             for k, v in ablation.items()]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(ab_labels, ab_vals, color=ab_colors, alpha=0.85)
ax.axvline(baseline, color="#555", linestyle="--", linewidth=1.5, label=f"Full model baseline ({baseline:.3f})")
ax.set_xlabel("Action Accuracy"); ax.set_xlim(0, 0.6)
ax.set_title("Signal Ablation Study — Impact of Removing Each Signal")
ax.legend(); ax.grid(axis="x", alpha=0.2)
for bar, val in zip(bars, ab_vals):
    delta = val - baseline
    sign = "+" if delta >= 0 else ""
    ax.text(val + 0.005, bar.get_y()+bar.get_height()/2,
            f"{val:.3f} ({sign}{delta:.3f})", va="center", fontsize=8)
save("ablation_study")


# ── 10. RAGAS ────────────────────────────────────────────────────────────────
ragas_sys  = ["Adaptive\n(n=17)", "Standard RAG\n(n=56)"]
ragas_cols = ["#2196f3", "#4caf50"]
faith = [0.505, 0.769]
relev = [0.531, 0.501]

x = np.arange(len(ragas_sys))
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - 0.2, faith, 0.35, label="Faithfulness ↑", color=ragas_cols, alpha=0.9)
ax.bar(x + 0.2, relev, 0.35, label="Answer Relevancy ↑", color=ragas_cols, alpha=0.5, hatch="//")
ax.set_xticks(x); ax.set_xticklabels(ragas_sys)
ax.set_ylim(0, 1); ax.set_ylabel("Score")
ax.set_title("RAGAS — Faithfulness & Answer Relevancy\n(retrieve/verify rows only)")
ax.legend(); ax.grid(axis="y", alpha=0.2)
for vals, offset in [(faith, -0.2), (relev, 0.2)]:
    for i, v in enumerate(vals):
        ax.text(i + offset, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
save("ragas_results")


# ── 11. DeepEval GEval Correctness ───────────────────────────────────────────
geval = [0.243, 0.199, 0.210, 0.232, 0.240]  # from DeepEval run (unchanged)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(SYS, geval, color=SCOLS, alpha=0.85)
ax.set_ylim(0, 0.4); ax.set_ylabel("GEval Correctness Score")
ax.set_title("DeepEval — LLM-as-Judge GEval Correctness (higher = better)")
ax.set_xticklabels(SYS, rotation=15, ha="right", fontsize=9)
ax.grid(axis="y", alpha=0.2)
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
save("deepeval_geval")


print(f"\nAll graphs saved to {OUT}/")
