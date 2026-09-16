# src/eval/lambda_sweep.py
import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from policy.decision_policy import decide_action
from eval.correctness import is_correct

LOG_PATH = "data/processed/eval_results.jsonl"
LAMBDAS = [0.05, 0.10, 0.15, 0.20, 0.30]


def run_lambda_sweep():
    rows = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["system"] == "adaptive" and "error" not in row["result"] and "_cached_features" in row["result"]:
                rows.append(row)

    print(f"Re-evaluating {len(rows)} cached adaptive-agent runs across lambda values (no new LLM calls needed).")

    results = []
    for lam in LAMBDAS:
        action_matches, retrieve_count, verify_count, total = 0, 0, 0, 0
        for row in rows:
            features = row["result"]["_cached_features"]
            decision = decide_action(features, lam=lam)
            total += 1
            if decision["action"] == row.get("expected_action"):
                action_matches += 1
            if decision["action"] == "retrieve":
                retrieve_count += 1
            if decision["action"] == "verify":
                verify_count += 1

        action_accuracy = action_matches / total if total else 0.0
        results.append({
            "lambda": lam,
            "action_accuracy": action_accuracy,
            "retrieve_rate": retrieve_count / total if total else 0.0,
            "verify_rate": verify_count / total if total else 0.0,
        })
        print(f"lambda={lam}: action_accuracy={action_accuracy:.3f}, "
              f"retrieve_rate={retrieve_count/total:.3f}, verify_rate={verify_count/total:.3f}")

    with open("data/processed/lambda_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_lambda_sweep()