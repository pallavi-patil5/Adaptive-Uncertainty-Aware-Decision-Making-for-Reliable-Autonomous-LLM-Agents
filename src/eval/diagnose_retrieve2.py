import json, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from signals.complexity import complexity_score

LAMBDA = 0.05
COSTS = {"answer": 1.0, "retrieve": 1.3, "verify": 1.6, "clarify": 1.0, "abstain": 0.0}

rows = [json.loads(l) for l in open("data/processed/eval_results.jsonl", encoding="utf-8")]
retrieve_rows = [r for r in rows if r["system"] == "adaptive" and r.get("expected_action") == "retrieve"]

print(f"retrieve-expected rows: {len(retrieve_rows)}\n")
print(f"{'question':<50} {'winner':<10} {'comp_old':>8} {'comp_new':>8} {'unc':>5} | {'ans':>6} {'ret':>6}")
print("-" * 105)

correct = 0
for r in retrieve_rows:
    f = r["result"].get("_cached_features", {})
    unc  = f.get("uncertainty", 0)
    ev   = f.get("evidence_coverage", 0)
    con  = f.get("contradiction_prob", 0)
    amb  = f.get("ambiguity", 0)
    comp_old = f.get("complexity", 0)
    comp_new = complexity_score(r["question"])["complexity"]

    cw = 0.7 if comp_new >= 0.4 else 0.5
    rel = {
        "answer":   (1-unc)*(1-con)*(1-amb)*(1-0.3*comp_new),
        "retrieve": (1-cw)*unc + cw*comp_new,
        "verify":   con * ev,
        "clarify":  amb,
        "abstain":  unc*(1-comp_new)*(1-ev),
    }
    scores = {a: rel[a] - LAMBDA*COSTS[a] for a in rel}
    winner = max(scores, key=scores.get)
    if winner == "retrieve":
        correct += 1
    print(f"{r['question'][:49]:<50} {winner:<10} {comp_old:>8.3f} {comp_new:>8.3f} {unc:>5.2f} | {scores['answer']:>6.3f} {scores['retrieve']:>6.3f}")

print(f"\nRetrieve recall: {correct}/12 = {correct/12:.2f}")
