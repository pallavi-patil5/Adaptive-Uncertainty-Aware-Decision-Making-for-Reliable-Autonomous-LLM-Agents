# src/eval/diagnose_scores.py
import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

LOG_PATH = "data/processed/eval_results.jsonl"
LAMBDA = 0.15
ACTION_COSTS = {"answer": 1, "verify": 2, "retrieve": 3, "clarify": 1, "abstain": 0}

def new_reliability(f):
    relevance = max(0.0, 1 - f.get("top1_distance", 1.0))
    return {
        "answer": (1 - f["uncertainty"]) * (1 - f["contradiction_prob"]) * (1 - f["ambiguity"]) * (1 - 0.3 * f.get("complexity", 0.0)),
        "retrieve": f["uncertainty"] * relevance,
        "verify": f["contradiction_prob"] * relevance,
        "clarify": f["ambiguity"],
        "abstain": f["uncertainty"] * (1 - relevance),
    }

rows = [json.loads(l) for l in open(LOG_PATH, encoding="utf-8")
        if json.loads(l)["system"] == "adaptive" and "error" not in json.loads(l)["result"]
        and "_cached_features" in json.loads(l)["result"]]

# Look specifically at questions where expected_action == "retrieve"
print("Rows where expected_action=retrieve — why did retrieve lose?\n")
for row in rows:
    if row.get("expected_action") != "retrieve":
        continue
    f = row["result"]["_cached_features"]
    rel = new_reliability(f)
    scores = {a: rel[a] - LAMBDA * ACTION_COSTS[a] for a in rel}
    winner = max(scores, key=scores.get)
    print(f"{row['record_id']}: uncertainty={f['uncertainty']:.2f} top1_dist={f.get('top1_distance', 1.0):.2f} "
          f"relevance={max(0,1-f.get('top1_distance',1.0)):.2f}")
    print(f"  reliability={ {k: round(v,3) for k,v in rel.items()} }")
    print(f"  scores={ {k: round(v,3) for k,v in scores.items()} } -> chose: {winner}\n")