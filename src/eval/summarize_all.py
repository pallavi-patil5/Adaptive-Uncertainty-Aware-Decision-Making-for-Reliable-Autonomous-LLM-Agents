# src/eval/summarize_all.py
import json, os

print("=" * 70)
print("WEEK 6 EVALUATION SUMMARY")
print("=" * 70)

files = {
    "Threshold sweep": "data/processed/threshold_sweep_results.json",
    "Lambda sweep": "data/processed/lambda_sweep_results.json",
    "Ablation study": "data/processed/ablation_results.json",
}

for label, path in files.items():
    print(f"\n--- {label} ---")
    if os.path.exists(path):
        with open(path) as f:
            print(json.dumps(json.load(f), indent=2))
    else:
        print("Not yet generated — run the corresponding script first.")

print("\nAlso re-run compute_tier1.py, compute_tier2.py, compute_tier3.py, compute_tier4.py")
print("and copy their printed tables into your report — they print directly, not saved to JSON here.")