import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from policy.feature_extractor import extract_features
from policy.agent_graph import (
    node_retrieve, node_post_retrieve_decide,
    node_verify, node_answer_with_context,
)

question = "Which university did the scientist who discovered penicillin attend?"
features = extract_features(question)

state = {
    "question": question,
    "features": features,
    "decision": {"action": "retrieve", "scores": {}, "reliability": {}},
    "result": None,
    "evidence": None,
    "llm_calls": 0,
    "retrieval_calls": 0,
    "start_time": time.monotonic(),  # required by _elapsed() in agent_graph nodes
}

state = node_retrieve(state)
print("After node_retrieve — evidence keys:", state["evidence"].keys())

state = node_post_retrieve_decide(state)
print("post_retrieve_action:", state["decision"]["post_retrieve_action"])
print("post_retrieve_contradiction:", state["decision"].get("post_retrieve_contradiction"))

if state["decision"]["post_retrieve_action"] == "verify":
    state = node_verify(state)
else:
    state = node_answer_with_context(state)

print("\nFinal result dict:")
print(state["result"])
print("\nResult keys present:", list(state["result"].keys()))
print("Has llm_calls:", "llm_calls" in state["result"])
print("Has retrieval_calls:", "retrieval_calls" in state["result"])
print("Has latency_s:", "latency_s" in state["result"])