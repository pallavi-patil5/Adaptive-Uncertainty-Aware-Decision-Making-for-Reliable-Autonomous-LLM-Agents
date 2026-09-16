# src/eval/diagnose_evidence_signal.py
import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

LOG_PATH = "data/processed/eval_results.jsonl"
LAMBDA = 0.15
ACTION_COSTS = {"answer": 1, "verify": 2, "retrieve": 3, "clarify": 1, "abstain": 0}

def new_reliability(f):
    relevance = max(0.0, 1 - f.get("top1_distance", 1.0))  # continuous, not a diluted fraction
    return {
        "answer": (1 - f["uncertainty"]) * (1 - f["contradiction_prob"]) * (1 - f["ambiguity"]) * (1 - 0.3 * f.get("complexity", 0.0)),
        "retrieve": f["uncertainty"] * relevance,
        "verify": f["contradiction_prob"] * relevance,
        "clarify": f["ambiguity"],
        "abstain": f["uncertainty"] * (1 - relevance),
    }

def decide(f):
    rel = new_reliability(f)
    scores = {a: rel[a] - LAMBDA * ACTION_COSTS[a] for a in rel}
    return max(scores, key=scores.get)

rows = [json.loads(l) for l in open(LOG_PATH, encoding="utf-8")
        if json.loads(l)["system"] == "adaptive" and "error" not in json.loads(l)["result"]
        and "_cached_features" in json.loads(l)["result"]]

matches, dist = 0, {}
for row in rows:
    pred = decide(row["result"]["_cached_features"])
    dist[pred] = dist.get(pred, 0) + 1
    matches += (pred == row.get("expected_action"))

print(f"New-formula action accuracy: {matches/len(rows):.3f}  (old was 0.304)")
print("Predicted action distribution:", dist)