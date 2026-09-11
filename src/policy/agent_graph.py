# src/policy/agent_graph.py

import sys
import os
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


# ---------------------------------------------------------
# 1. Feature Extraction
# ---------------------------------------------------------

def node_extract_features(state: AgentState) -> AgentState:
    features = extract_features(state["question"])

    return {
        **state,
        "features": features,
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
    result = action_answer(
        state["question"],
        state["features"]["candidate_answer"],
    )

    return {
        **state,
        "result": result,
    }


# ---------------------------------------------------------
# 4. Retrieve Evidence
# ---------------------------------------------------------

def node_retrieve(state: AgentState) -> AgentState:
    evidence = retrieval_signals(
        state["question"],
        k=3,
    )

    return {
        **state,
        "evidence": evidence,
    }


# ---------------------------------------------------------
# 5. Verify Retrieved Evidence
# ---------------------------------------------------------

def node_verify(state: AgentState) -> AgentState:

    evidence = state.get("evidence")

    if not evidence:
        # Reached "verify" directly (e.g. triggered by high contradiction_prob)
        # without passing through the retrieve node first — fetch evidence now.
        # This keeps the Retrieve -> Verify path efficient (no duplicate query)
        # while making the direct-Verify path actually have evidence to check against.
        evidence = retrieval_signals(state["question"], k=3)

    retrieved_docs = evidence.get("retrieved_docs", [])

    if retrieved_docs:
        evidence_text = "\n\n".join(
            doc["text"] for doc in retrieved_docs
        )
    else:
        evidence_text = "No evidence available."

    result = action_verify(
        state["question"],
        state["features"]["candidate_answer"],
        evidence_text,
    )

    return {
        **state,
        "evidence": evidence,
        "result": result,
    }


# ---------------------------------------------------------
# 6. Clarification
# ---------------------------------------------------------

def node_clarify(state: AgentState) -> AgentState:
    result = action_clarify(
        state["question"]
    )

    return {
        **state,
        "result": result,
    }


# ---------------------------------------------------------
# 7. Abstention
# ---------------------------------------------------------

def node_abstain(state: AgentState) -> AgentState:
    result = action_abstain(
        state["question"]
    )

    return {
        **state,
        "result": result,
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

    graph.add_node(
        "extract_features",
        node_extract_features,
    )

    graph.add_node(
        "decide",
        node_decide,
    )

    graph.add_node(
        "answer",
        node_answer,
    )

    graph.add_node(
        "retrieve",
        node_retrieve,
    )

    graph.add_node(
        "verify",
        node_verify,
    )

    graph.add_node(
        "clarify",
        node_clarify,
    )

    graph.add_node(
        "abstain",
        node_abstain,
    )

    # Entry
    graph.set_entry_point(
        "extract_features"
    )

    # Feature extraction -> decision
    graph.add_edge(
        "extract_features",
        "decide",
    )

    # Adaptive action selection
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

    # Retrieved evidence can then be verified.
    graph.add_edge(
        "retrieve",
        "verify",
    )

    # Terminal actions
    graph.add_edge(
        "answer",
        END,
    )

    graph.add_edge(
        "verify",
        END,
    )

    graph.add_edge(
        "clarify",
        END,
    )

    graph.add_edge(
        "abstain",
        END,
    )

    return graph.compile()


# ---------------------------------------------------------
# 10. Local Test
# ---------------------------------------------------------

if __name__ == "__main__":

    app = build_graph()

    test_questions = [
        "What is the capital of France?",
        "Who was the first person to walk on the Moon?",
        "What does 'the meeting' refer to, and when is it?",
        "Is the claim that humans can survive indefinitely without water true?",
    ]

    for question in test_questions:

        initial_state = {
            "question": question,
            "features": None,
            "decision": None,
            "result": None,
            "evidence": None,
        }

        final_state = app.invoke(
            initial_state
        )

        print("\n" + "=" * 70)
        print(f"Question: {question}")
        print(
            f"Action: "
            f"{final_state['decision']['action']}"
        )
        print(
            f"Decision details: "
            f"{final_state['decision']}"
        )
        print(
            f"Result: "
            f"{final_state['result']}"
        )