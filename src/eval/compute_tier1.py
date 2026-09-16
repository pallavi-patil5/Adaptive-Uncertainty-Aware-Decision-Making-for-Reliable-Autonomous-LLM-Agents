# src/eval/compute_tier1.py
import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.metrics import precision_score, recall_score, f1_score
from eval.correctness import is_correct

LOG_PATH = "data/processed/eval_results.jsonl"


def load_results():
    rows = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def compute_tier1(rows):
    by_system = {}
    for row in rows:
        system = row["system"]
        result = row["result"]
        if "error" in result:
            continue

        predicted = result.get("final_answer")
        correct = is_correct(predicted, row.get("acceptable_answers", []), row.get("ground_truth_answer"))
        if correct is None:
            continue  # no ground truth to judge against — skip for Tier 1, covered by Tier 3 instead

        by_system.setdefault(system, {"y_true": [], "y_pred": [], "hallucinations": 0, "total": 0})
        by_system[system]["y_true"].append(1)  # "should be correct" ground truth marker
        by_system[system]["y_pred"].append(1 if correct else 0)
        by_system[system]["total"] += 1
        if not correct and result.get("action") not in ("abstain", "clarify"):
            by_system[system]["hallucinations"] += 1

    print(f"{'system':<18} {'accuracy':<10} {'hallucination_rate':<20} {'n':<5}")
    print("-" * 60)
    summary = {}
    for system, data in by_system.items():
        accuracy = sum(data["y_pred"]) / data["total"] if data["total"] else 0.0
        halluc_rate = data["hallucinations"] / data["total"] if data["total"] else 0.0
        print(f"{system:<18} {accuracy:<10.3f} {halluc_rate:<20.3f} {data['total']:<5}")
        summary[system] = {"accuracy": accuracy, "hallucination_rate": halluc_rate, "n": data["total"]}

    return summary


if __name__ == "__main__":
    rows = load_results()
    compute_tier1(rows)