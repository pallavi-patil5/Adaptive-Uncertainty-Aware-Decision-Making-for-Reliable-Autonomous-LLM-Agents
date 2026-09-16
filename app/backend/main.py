# app/backend/main.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from policy.agent_graph import build_graph

app = FastAPI(title="Adaptive Uncertainty-Aware Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev demo; tighten if ever deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    action: str
    final_answer: str | None
    reasoning_trace: dict


@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    state = graph.invoke({
        "question": request.question,
        "features": None,
        "decision": None,
        "result": None,
        "evidence": None,
    })

    features = state["features"]
    decision = state["decision"]
    result = state["result"]

    reasoning_trace = {
        "uncertainty": round(features["uncertainty"], 3),
        "ambiguity": round(features["ambiguity"], 3),
        "contradiction_prob": round(features["contradiction_prob"], 3),
        "evidence_coverage": round(features["evidence_coverage"], 3),
        "action_scores": {k: round(v, 3) for k, v in decision["scores"].items()},
        "trace_text": result.get("trace", ""),
    }

    return QueryResponse(
        question=request.question,
        action=decision["action"],
        final_answer=result.get("final_answer"),
        reasoning_trace=reasoning_trace,
    )


@app.get("/health")
def health():
    return {"status": "ok"}