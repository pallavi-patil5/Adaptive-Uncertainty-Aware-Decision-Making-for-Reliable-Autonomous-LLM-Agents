# app/backend/main.py
import sys, os, json, time
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from policy.agent_graph import build_graph

app = FastAPI(title="Adaptive Uncertainty-Aware Agent API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

graph = build_graph()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
EVAL_RESULTS_PATH = os.path.join(DATA_DIR, "eval_results.jsonl")
RAGAS_PATH        = os.path.join(DATA_DIR, "ragas_results.json")
DEEPEVAL_PATH     = os.path.join(DATA_DIR, "deepeval_results.json")


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query_agent(request: QueryRequest):
    import time as _time
    state = graph.invoke({
        "question": request.question,
        "features": None, "decision": None,
        "result": None, "evidence": None,
        "llm_calls": 0, "retrieval_calls": 0,
        "start_time": _time.monotonic(),
    })

    features = state["features"]
    decision = state["decision"]
    result   = state["result"]
    evidence = state.get("evidence") or {}

    unc = features["uncertainty"]
    timeline = [
        {"step": "Query Received",         "detail": request.question[:80]},
        {"step": "Feature Extraction",     "detail": f"Uncertainty={unc:.3f}  Ambiguity={features['ambiguity']:.3f}  Complexity={features.get('complexity',0):.3f}"},
        {"step": "Uncertainty Estimation", "detail": f"Self-consistency={features['self_consistency_uncertainty']:.3f}  Self-eval={features['self_eval_uncertainty']:.3f}  Combined={unc:.3f}"},
        {"step": "Action Selection",       "detail": f"Policy chose: {decision['action'].upper()}  (λ=0.05)"},
    ]
    if decision["action"] in ("retrieve", "verify"):
        docs = evidence.get("retrieved_docs", [])
        timeline.append({"step": "Evidence Retrieval", "detail": f"{len(docs)} document(s) retrieved  top1_distance={features.get('top1_distance', 1.0):.3f}"})
    if decision["action"] == "verify":
        timeline.append({"step": "Verification", "detail": f"Contradiction prob={features['contradiction_prob']:.3f}  Support score={features.get('support_score',0):.3f}"})
    timeline.append({"step": "Final Response", "detail": f"Action={result.get('action','?').upper()}  LLM calls={result.get('llm_calls',0)}  Latency={result.get('latency_s',0):.2f}s"})

    retrieved_docs = evidence.get("retrieved_docs", [])
    evidence_chunks = [
        {"text": d["text"][:300], "distance": round(d["distance"], 4), "relevant": d["distance"] < 0.6}
        for d in retrieved_docs
    ]

    return {
        "question":         request.question,
        "action":           decision["action"],
        "final_answer":     result.get("final_answer"),
        "signals": {
            "uncertainty":        round(features["uncertainty"], 3),
            "self_consistency":   round(features["self_consistency_uncertainty"], 3),
            "self_eval":          round(features["self_eval_uncertainty"], 3),
            "ambiguity":          round(features["ambiguity"], 3),
            "complexity":         round(features.get("complexity", 0), 3),
            "evidence_coverage":  round(features["evidence_coverage"], 3),
            "contradiction_prob": round(features["contradiction_prob"], 3),
            "support_score":      round(features.get("support_score", 0), 3),
            "top1_distance":      round(features.get("top1_distance", 1.0), 3),
        },
        "action_scores":     {k: round(v, 3) for k, v in decision["scores"].items()},
        "action_reliability":{k: round(v, 3) for k, v in decision["reliability"].items()},
        "timeline":          timeline,
        "evidence_chunks":   evidence_chunks,
        "efficiency": {
            "llm_calls":       result.get("llm_calls", 0),
            "retrieval_calls": result.get("retrieval_calls", 0),
            "latency_s":       result.get("latency_s", 0),
        },
        "trace":            result.get("trace", ""),
        "candidate_answer": features.get("candidate_answer", ""),
        "clarifying_question": result.get("clarifying_question"),
    }


@app.get("/eval_metrics")
def get_eval_metrics():
    if not os.path.exists(EVAL_RESULTS_PATH):
        raise Exception("eval_results.jsonl not found — run src/eval/run_evaluation.py first")

    rows    = [json.loads(l) for l in open(EVAL_RESULTS_PATH, encoding="utf-8")]
    systems = ["adaptive", "normal_llm", "standard_rag", "self_reflection", "fixed_threshold"]
    ACTIONS = ["answer", "retrieve", "verify", "clarify", "abstain"]

    # Load RAGAS + DeepEval pre-computed results if available
    ragas_data    = json.load(open(RAGAS_PATH))    if os.path.exists(RAGAS_PATH)    else {}
    deepeval_data = json.load(open(DEEPEVAL_PATH)) if os.path.exists(DEEPEVAL_PATH) else {}

    metrics = {}
    for system in systems:
        sys_rows = [r for r in rows if r["system"] == system and "error" not in r["result"]]
        if not sys_rows:
            continue

        # Tier 1 — answer quality
        correct, halluc, t1_total = 0, 0, 0
        for r in sys_rows:
            gt  = r.get("ground_truth_answer")
            acc = r.get("acceptable_answers", [])
            if not gt and not acc:
                continue
            pred       = r["result"].get("final_answer") or ""
            candidates = acc if acc else ([gt] if gt else [])
            hit        = any(c.lower() in pred.lower() for c in candidates if c)
            correct   += int(hit)
            t1_total  += 1
            if not hit and r["result"].get("action") not in ("abstain", "clarify"):
                halluc += 1

        # Tier 3 — decision quality
        action_correct, unnecessary_ret, t3_total = 0, 0, 0
        action_counts = {a: {"tp": 0, "fp": 0, "fn": 0} for a in ACTIONS}
        for r in sys_rows:
            expected  = r.get("expected_action")
            predicted = r["result"].get("action")
            if not expected or not predicted:
                continue
            t3_total += 1
            if expected == predicted:
                action_correct += 1
                action_counts[expected]["tp"] += 1
            else:
                action_counts[predicted]["fp"] += 1
                action_counts[expected]["fn"]  += 1
            if expected == "answer" and predicted == "retrieve":
                unnecessary_ret += 1

        # Tier 4 — efficiency
        llm_calls = [r["result"].get("llm_calls", 0)       for r in sys_rows]
        ret_calls = [r["result"].get("retrieval_calls", 0)  for r in sys_rows]
        latencies = [r["result"].get("latency_s", 0)        for r in sys_rows]

        per_action = {}
        for a in ACTIONS:
            tp = action_counts[a]["tp"]
            fp = action_counts[a]["fp"]
            fn = action_counts[a]["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_action[a] = {
                "precision": round(precision, 3),
                "recall":    round(recall, 3),
                "f1":        round(f1, 3),
            }

        # RAGAS scores (faithfulness + answer_relevancy) — retrieve/verify rows only
        ragas = ragas_data.get(system, {})
        # DeepEval scores (hallucination_score, answer_relevancy, geval_correctness)
        deepeval = deepeval_data.get(system, {})

        metrics[system] = {
            # Tier 1
            "accuracy":                    round(correct / t1_total, 3) if t1_total else 0,
            "hallucination_rate":          round(halluc / t1_total, 3)  if t1_total else 0,
            # Tier 3
            "action_accuracy":             round(action_correct / t3_total, 3) if t3_total else 0,
            "unnecessary_retrieval":       round(unnecessary_ret / t3_total, 3) if t3_total else 0,
            "per_action":                  per_action,
            # Tier 4
            "avg_llm_calls":               round(sum(llm_calls) / len(llm_calls), 2) if llm_calls else 0,
            "avg_retrieval_calls":         round(sum(ret_calls) / len(ret_calls), 2)  if ret_calls else 0,
            "avg_latency_s":               round(sum(latencies) / len(latencies), 3)  if latencies else 0,
            # RAGAS
            "ragas_faithfulness":          ragas.get("faithfulness"),
            "ragas_answer_relevancy":      ragas.get("answer_relevancy"),
            # DeepEval
            "deepeval_hallucination":      deepeval.get("hallucination_score"),
            "deepeval_answer_relevancy":   deepeval.get("answer_relevancy"),
            "deepeval_geval_correctness":  deepeval.get("answer_correctness_geval"),
            "n": len(sys_rows),
        }

    return metrics


@app.get("/health")
def health():
    return {"status": "ok"}
