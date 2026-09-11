# src/policy/agent_graph.py

import sys
import os
import time
from typing import TypedDict, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langgraph.graph import StateGraph, END

from policy.feature_extractor import extract_features
from policy.decision_policy import decide_action

from policy.actions import (
    action_answer,
    action_verify,
    action_clarify,
    action_abstain,
)

from signals.evidence_retrieval import retrieval_signals


class AgentState(TypedDict):
    question: str
    features: Optional[dict]
    decision: Optional[dict]
    result: Optional[dict]
    evidence: Optional[dict]
    llm_calls: int          # accumulated across all nodes — for Tier 4 efficiency metrics
    retrieval_calls: int
    start_time: float       # set at entry via time.monotonic(), used to compute end-to-end latency_s


def _elapsed(state: AgentState) -> float:
    """Monotonic elapsed seconds since start_time, floored at 0.0.
    start_time must be set in the initial state via time.monotonic().
    Falls back to 0.0 (not time.monotonic()) so a missing start_time
    produces an explicit 0.0 rather than a silent near-zero from a
    same-instant fallback evaluation.
    """
    start = state.get("start_time")
    if start is None:
        return 0.0
    return max(0.0, round(time.monotonic() - start, 3))


# ---------------------------------------------------------
# 1. Feature Extraction
# ---------------------------------------------------------

def node_extract_features(state: AgentState) -> AgentState:
    # Stamp start_time here as a safety net — covers callers that forget to set it.
    # If already set by the caller, preserve it so end-to-end latency includes
    # any pre-graph overhead the caller wants to measure.
    features = extract_features(state["question"])
    return {
        **state,
        "features": features,
        "start_time": state.get("start_time") or time.monotonic(),
        "llm_calls": state.get("llm_calls", 0) + 6,
        "retrieval_calls": state.get("retrieval_calls", 0) + 1,
    }


# ---------------------------------------------------------
# 2. Adaptive Decision
# ---------------------------------------------------------

def node_decide(state: AgentState) -> AgentState:
    decision = decide_action(state["features"])
    return {
        **state,
        "decision": decision,
    }


# ---------------------------------------------------------
# 3. Direct Answer
# ---------------------------------------------------------

def node_answer(state: AgentState) -> AgentState:
    result = action_answer(state["question"], state["features"]["candidate_answer"])
    return {
        **state,
        "result": {**result,
                   "llm_calls": state.get("llm_calls", 0) + result["llm_calls"],
                   "retrieval_calls": state.get("retrieval_calls", 0) + result["retrieval_calls"],
                   "latency_s": _elapsed(state)},
    }


# ---------------------------------------------------------
# 4. Retrieve Evidence
# ---------------------------------------------------------

def node_retrieve(state: AgentState) -> AgentState:
    evidence = retrieval_signals(state["question"], k=3)
    return {
        **state,
        "evidence": evidence,
        "retrieval_calls": state.get("retrieval_calls", 0) + 1,
    }


# ---------------------------------------------------------
# 5. Post-Retrieve Decision: verify only if contradiction is high
# ---------------------------------------------------------

def node_post_retrieve_decide(state: AgentState) -> AgentState:
    """After retrieval, re-evaluate whether verification is actually needed.
    Avoids unnecessary verify calls when retrieved evidence already clearly supports the answer."""
    evidence = state["evidence"]
    retrieved_docs = evidence.get("retrieved_docs", [])

    if not retrieved_docs:
        # No evidence retrieved — answer directly with candidate
        return {**state, "decision": {**state["decision"], "post_retrieve_action": "answer_direct"}}

    # Re-check contradiction signal against freshly retrieved evidence
    from signals.contradiction import contradiction_score
    top_doc = retrieved_docs[0]["text"]
    contra = contradiction_score(top_doc, state["features"]["candidate_answer"])
    contradiction_prob = contra["contradiction"]

    post_action = "verify" if contradiction_prob > 0.3 else "answer_with_context"
    return {**state, "decision": {**state["decision"], "post_retrieve_action": post_action,
                                   "post_retrieve_contradiction": contradiction_prob}}


def route_post_retrieve(state: AgentState) -> str:
    return state["decision"].get("post_retrieve_action", "verify")


def node_answer_with_context(state: AgentState) -> AgentState:
    """Answer using retrieved context without a full verify LLM call."""
    from llm_client import call_llm
    evidence = state.get("evidence", {})
    docs = [d["text"] for d in evidence.get("retrieved_docs", [])]
    context = "\n\n".join(docs) if docs else ""
    prompt = f"Context:\n{context}\n\nQuestion: {state['question']}\nAnswer using the context above:"
    answer = call_llm(prompt, temperature=0.0)
    return {**state, "result": {
        "action": "retrieve",
        "final_answer": answer,
        "trace": "Retrieved evidence supported candidate answer — answered with context.",
        "llm_calls": state.get("llm_calls", 0) + 1,
        "retrieval_calls": state.get("retrieval_calls", 0),
        "latency_s": _elapsed(state),
    }}


def node_verify(state: AgentState) -> AgentState:
    extra_retrieval = 0
    evidence = state.get("evidence")
    if not evidence:
        evidence = retrieval_signals(state["question"], k=3)
        extra_retrieval = 1

    retrieved_docs = evidence.get("retrieved_docs", [])
    evidence_text = "\n\n".join(doc["text"] for doc in retrieved_docs) if retrieved_docs else "No evidence available."

    result = action_verify(state["question"], state["features"]["candidate_answer"], evidence_text)
    return {
        **state,
        "evidence": evidence,
        "result": {**result,
                   "llm_calls": state.get("llm_calls", 0) + result["llm_calls"],
                   "retrieval_calls": state.get("retrieval_calls", 0) + extra_retrieval,
                   "latency_s": _elapsed(state)},
    }


# ---------------------------------------------------------
# 6. Clarification
# ---------------------------------------------------------

def node_clarify(state: AgentState) -> AgentState:
    result = action_clarify(state["question"])
    return {
        **state,
        "result": {**result,
                   "llm_calls": state.get("llm_calls", 0) + result["llm_calls"],
                   "retrieval_calls": state.get("retrieval_calls", 0),
                   "latency_s": _elapsed(state)},
    }


# ---------------------------------------------------------
# 7. Abstention
# ---------------------------------------------------------

def node_abstain(state: AgentState) -> AgentState:
    result = action_abstain(state["question"])
    return {
        **state,
        "result": {**result,
                   "llm_calls": state.get("llm_calls", 0),
                   "retrieval_calls": state.get("retrieval_calls", 0),
                   "latency_s": _elapsed(state)},
    }


# ---------------------------------------------------------
# 8. Route Based on Adaptive Policy
# ---------------------------------------------------------

def route_action(state: AgentState) -> str:
    return state["decision"]["action"]


# ---------------------------------------------------------
# 9. Build Agent Graph
# ---------------------------------------------------------

def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("extract_features", node_extract_features)
    graph.add_node("decide", node_decide)
    graph.add_node("answer", node_answer)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("post_retrieve_decide", node_post_retrieve_decide)
    graph.add_node("answer_with_context", node_answer_with_context)
    graph.add_node("verify", node_verify)
    graph.add_node("clarify", node_clarify)
    graph.add_node("abstain", node_abstain)

    graph.set_entry_point("extract_features")
    graph.add_edge("extract_features", "decide")

    graph.add_conditional_edges(
        "decide",
        route_action,
        {
            "answer": "answer",
            "retrieve": "retrieve",
            "verify": "verify",
            "clarify": "clarify",
            "abstain": "abstain",
        },
    )

    graph.add_edge("retrieve", "post_retrieve_decide")
    graph.add_conditional_edges(
        "post_retrieve_decide",
        route_post_retrieve,
        {
            "verify": "verify",
            "answer_with_context": "answer_with_context",
            "answer_direct": "answer",
        },
    )

    graph.add_edge("answer_with_context", END)
    graph.add_edge("answer", END)
    graph.add_edge("verify", END)
    graph.add_edge("clarify", END)
    graph.add_edge("abstain", END)

    return graph.compile()


# ---------------------------------------------------------
# 10. Local Test
# ---------------------------------------------------------

if __name__ == "__main__":

    app = build_graph()

    test_questions = [
        "What is 15 * 23?",
        "Is the statement 'the Earth is flat' scientifically supported?",
        "Who was the president of the country where the 2016 Summer Olympics were held?",
        "Which university did the scientist who discovered penicillin attend?",
    ]

    for question in test_questions:

        initial_state = {
            "question": question,
            "features": None,
            "decision": None,
            "result": None,
            "evidence": None,
            "llm_calls": 0,
            "retrieval_calls": 0,
            "start_time": time.monotonic(),
        }

        final_state = app.invoke(initial_state)

        print("\n" + "=" * 70)
        print(f"Question: {question}")
        print(f"Action: {final_state['decision']['action']}")
        print(f"Decision details: {final_state['decision']}")
        print(f"Result: {final_state['result']}")
