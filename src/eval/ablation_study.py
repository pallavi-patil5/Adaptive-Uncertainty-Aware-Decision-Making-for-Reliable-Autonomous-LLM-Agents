# src/eval/ablation_study.py
import sys, os, json, copy
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from policy.decision_policy import decide_action

LOG_PATH = "data/processed/eval_results.jsonl"

ABLATIONS = {
    "full_model": None,  # no ablation — baseline for comparison
    "remove_uncertainty": {"uncertainty": 0.0},
    "remove_evidence_coverage": {"evidence_coverage": 0.0},
    "remove_contradiction": {"contradiction_prob": 0.0},
    "remove_ambiguity": {"ambiguity": 0.0},
    "remove_complexity": {"complexity": 0.0},
}


def run_ablation():
    rows = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["system"] == "adaptive" and "error" not in row["result"] and "_cached_features" in row["result"]:
                rows.append(row)

    print(f"Running ablation over {len(rows)} cached adaptive-agent runs.")

    results = {}
    for ablation_name, overrides in ABLATIONS.items():
        matches, total = 0, 0
        for row in rows:
            features = copy.deepcopy(row["result"]["_cached_features"])
            if overrides:
                features.update(overrides)
            decision = decide_action(features)
            total += 1
            if decision["action"] == row.get("expected_action"):
                matches += 1

        accuracy = matches / total if total else 0.0
        results[ablation_name] = accuracy
        print(f"{ablation_name:<28} action_accuracy={accuracy:.3f}")

    baseline_acc = results["full_model"]
    print(f"\n{'Signal removed':<28} {'Accuracy drop':<15}")
    print("-" * 45)
    for name, acc in results.items():
        if name == "full_model":
            continue
        drop = baseline_acc - acc
        print(f"{name:<28} {drop:+.3f}")

    with open("data/processed/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_ablation()