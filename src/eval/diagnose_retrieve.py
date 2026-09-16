import json, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

LAMBDA = 0.05
COSTS = {"answer": 1.0, "retrieve": 1.3, "verify": 1.6, "clarify": 1.0, "abstain": 0.0}

rows = [json.loads(l) for l in open("data/processed/eval_results.jsonl", encoding="utf-8")]
retrieve_rows = [r for r in rows if r["system"] == "adaptive" and r.get("expected_action") == "retrieve"]

print(f"retrieve-expected rows: {len(retrieve_rows)}\n")
print(f"{'question':<48} {'old':<10} {'new_winner':<12} {'unc':>5} {'comp':>5} {'ev':>5} | {'ans':>6} {'ret':>6} {'abs':>6}")
print("-" * 115)

for r in retrieve_rows:
    f = r["result"].get("_cached_features", {})
    unc  = f.get("uncertainty", 0)
    comp = f.get("complexity", 0)
    ev   = f.get("evidence_coverage", 0)
    con  = f.get("contradiction_prob", 0)
    amb  = f.get("ambiguity", 0)

    complexity_weight = 0.7 if comp >= 0.4 else 0.5
    rel = {
        "answer":   (1-unc)*(1-con)*(1-amb)*(1-0.3*comp),
        "retrieve": (1-complexity_weight)*unc + complexity_weight*comp,
        "verify":   con * ev,
        "clarify":  amb,
        "abstain":  unc*(1-comp)*(1-ev),
    }
    scores = {a: rel[a] - LAMBDA*COSTS[a] for a in rel}
    winner = max(scores, key=scores.get)
    old_action = r["result"].get("action", "?")

    print(f"{r['question'][:47]:<48} {old_action:<10} {winner:<12} {unc:>5.2f} {comp:>5.2f} {ev:>5.2f} | {scores['answer']:>6.3f} {scores['retrieve']:>6.3f} {scores['abstain']:>6.3f}")
