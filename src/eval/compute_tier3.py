# src/eval/compute_tier3.py
import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.metrics import classification_report

LOG_PATH = "data/processed/eval_results.jsonl"
ACTIONS = ["answer", "retrieve", "verify", "clarify", "abstain"]


def get_predicted_action(row):
    result = row["result"]
    if "error" in result:
        return None
    # adaptive agent logs both the top-level decision and the post-retrieve refinement
    return result.get("action")


def compute_tier3():
    rows = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    by_system = {}
    for row in rows:
        system = row["system"]
        expected = row.get("expected_action")
        predicted = get_predicted_action(row)
        if not expected or not predicted:
            continue
        by_system.setdefault(system, {"y_true": [], "y_pred": [], "unnecessary_retrieve": 0,
                                       "unnecessary_verify": 0, "total": 0})
        by_system[system]["y_true"].append(expected)
        by_system[system]["y_pred"].append(predicted)
        by_system[system]["total"] += 1
        if expected == "answer" and predicted == "retrieve":
            by_system[system]["unnecessary_retrieve"] += 1
        if expected == "answer" and predicted == "verify":
            by_system[system]["unnecessary_verify"] += 1

    for system, data in by_system.items():
        print(f"\n{'='*60}\n{system}\n{'='*60}")
        print(classification_report(data["y_true"], data["y_pred"], labels=ACTIONS, zero_division=0))
        n = data["total"]
        print(f"Unnecessary retrieval rate: {data['unnecessary_retrieve']/n:.3f}")
        print(f"Unnecessary verification rate: {data['unnecessary_verify']/n:.3f}")


if __name__ == "__main__":
    compute_tier3()