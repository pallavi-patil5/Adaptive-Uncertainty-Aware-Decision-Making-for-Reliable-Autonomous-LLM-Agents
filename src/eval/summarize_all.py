# src/eval/summarize_all.py
import json, os

print("=" * 70)
print("FULL EVALUATION SUMMARY")
print("=" * 70)

files = {
    "Threshold sweep":  "data/processed/threshold_sweep_results.json",
    "Lambda sweep":     "data/processed/lambda_sweep_results.json",
    "Ablation study":   "data/processed/ablation_results.json",
    "RAGAS results":    "data/processed/ragas_results.json",
    "DeepEval results": "data/processed/deepeval_results.json",
}

for label, path in files.items():
    print(f"\n--- {label} ---")
    if os.path.exists(path):
        with open(path) as f:
            print(json.dumps(json.load(f), indent=2))
    else:
        print(f"Not yet generated — run the corresponding script first.")

print("\nSklearn-based tiers: re-run compute_tier1.py, compute_tier2.py, compute_tier3.py, compute_tier4.py")
print("RAGAS:    python src/eval/compute_ragas.py")
print("DeepEval: python src/eval/compute_deepeval.py")
