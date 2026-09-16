# src/eval/plot_threshold_sweep.py
import json
import matplotlib.pyplot as plt

with open("data/processed/threshold_sweep_results.json") as f:
    results = json.load(f)

thresholds = [r["threshold"] for r in results]
accuracies = [r["accuracy"] for r in results]
retrieval_rates = [r["retrieval_rate"] for r in results]

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(thresholds, accuracies, "o-", label="Accuracy", color="tab:blue")
ax1.set_xlabel("Uncertainty Threshold")
ax1.set_ylabel("Accuracy", color="tab:blue")

ax2 = ax1.twinx()
ax2.plot(thresholds, retrieval_rates, "s--", label="Retrieval Rate", color="tab:orange")
ax2.set_ylabel("Retrieval Rate", color="tab:orange")

plt.title("Fixed-Threshold Baseline: Accuracy vs. Retrieval Rate Across Thresholds")
fig.tight_layout()
plt.savefig("data/processed/threshold_sweep_plot.png", dpi=150)
print("Saved plot to data/processed/threshold_sweep_plot.png")