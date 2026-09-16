# src/policy/decision_policy.py
# Lambda empirically tuned via lambda_sweep.py: best action_accuracy at 0.05
# Lower lambda = lower cost penalty = more willing to retrieve/verify
LAMBDA = 0.05

# Costs calibrated from Tier 4 empirical averages (avg_llm_calls per action):
#   answer  ~1.0 LLM call
#   clarify ~1.0 LLM call
#   retrieve ~1.3 LLM calls (1 generate + occasional rerank)
#   verify  ~1.6 LLM calls (generate + verification pass)
#   abstain  0 LLM calls
# Normalised relative to answer=1.0
ACTION_COSTS = {
    "answer": 1.0,
    "retrieve": 1.3,
    "verify": 1.6,
    "clarify": 1.0,
    "abstain": 0.0,
}


def estimate_reliability(features: dict) -> dict:
    """Per-action reliability estimates, per the documented formal objective.

    Key design decisions:
    - retrieve is driven by uncertainty + complexity alone, NOT gated by evidence_coverage.
      evidence_coverage is 0 before retrieval happens — gating on it creates a catch-22
      where retrieve can never win on questions with no pre-cached evidence.
    - verify IS gated by evidence_coverage: it only makes sense after evidence exists.
    - abstain requires BOTH high uncertainty AND low complexity (simple question the model
      genuinely doesn't know). High-complexity uncertain questions should retrieve, not abstain.
    """
    uncertainty = features["uncertainty"]
    contradiction = features["contradiction_prob"]
    ambiguity = features["ambiguity"]
    evidence_coverage = features["evidence_coverage"]
    complexity = features.get("complexity", 0.0)

    # retrieve signal: uncertain OR complex question — attempt retrieval regardless of
    # whether corpus already has evidence (retrieval is the action that *produces* evidence).
    # Complexity weight is 0.7 for high-complexity questions (multi-hop, specific facts)
    # because the model is frequently confidently wrong on these even at low uncertainty.
    complexity_weight = 0.7 if complexity >= 0.4 else 0.5
    retrieve_signal = (1 - complexity_weight) * uncertainty + complexity_weight * complexity

    return {
        "answer": (1 - uncertainty) * (1 - contradiction) * (1 - ambiguity) * (1 - 0.3 * complexity),
        # not gated by evidence_coverage — retrieve is the action that fetches evidence
        "retrieve": retrieve_signal,
        # triggered by high contradiction OR high uncertainty on a non-trivial question
        "verify": max(contradiction, 0.5 * uncertainty) * (1 - ambiguity) * (1 - 0.3 * (1 - complexity)),
        "clarify": ambiguity * (1 + ambiguity),  # quadratic boost so high ambiguity clearly wins
        # abstain only when uncertain AND simple (low complexity) AND no evidence available
        # high-complexity uncertain questions should retrieve, not give up
        "abstain": uncertainty * (1 - complexity) * (1 - evidence_coverage),
    }


def decide_action(features: dict, lam: float = LAMBDA) -> dict:
    """A* = argmax_A [ Reliability(A) - lambda * Cost(A) ]"""
    reliability = estimate_reliability(features)
    scores = {action: reliability[action] - lam * ACTION_COSTS[action] for action in reliability}
    best_action = max(scores, key=scores.get)
    return {"action": best_action, "scores": scores, "reliability": reliability}


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from feature_extractor import extract_features

    for q in [
        "What is the capital of France?",
        "What does 'the meeting' refer to, and when is it?",
        "What was the exact color of the shirt worn by a random person in Pune yesterday?",
    ]:
        features = extract_features(q)
        decision = decide_action(features)
        print(f"\nQ: {q}")
        print(f"  -> action: {decision['action']}")
        print(f"  -> scores: { {k: round(v, 3) for k, v in decision['scores'].items()} }")
