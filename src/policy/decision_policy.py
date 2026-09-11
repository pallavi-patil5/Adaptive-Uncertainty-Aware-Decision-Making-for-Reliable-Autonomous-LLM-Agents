# src/policy/decision_policy.py
LAMBDA = 0.15

# Fixed cost proxies (relative LLM/tool calls) — from Step 1's documented objective
ACTION_COSTS = {
    "answer": 1,
    "verify": 2,
    "retrieve": 3,
    "clarify": 1,
    "abstain": 0,
}


def estimate_reliability(features: dict) -> dict:
    """Per-action reliability estimates, per the documented formal objective (Step 1)."""
    uncertainty = features["uncertainty"]
    contradiction = features["contradiction_prob"]
    ambiguity = features["ambiguity"]
    evidence_coverage = features["evidence_coverage"]

    return {
        "answer": (1 - uncertainty) * (1 - contradiction) * (1 - ambiguity),
        "retrieve": (1 - evidence_coverage),
        "verify": contradiction,
        "clarify": ambiguity,
        "abstain": uncertainty * (1 - evidence_coverage),
    }


def decide_action(features: dict, lam: float = LAMBDA) -> dict:
    """
    A* = argmax_A [ Reliability(A) - lambda * Cost(A) ]
    Returns the chosen action plus the full score breakdown (for logging/debugging/ablation).
    """
    reliability = estimate_reliability(features)
    scores = {
        action: reliability[action] - lam * ACTION_COSTS[action]
        for action in reliability
    }
    best_action = max(scores, key=scores.get)

    return {
        "action": best_action,
        "scores": scores,
        "reliability": reliability,
    }


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