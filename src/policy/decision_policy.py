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
        # High confidence + no contradiction + unambiguous -> answer directly
        "answer": (1 - uncertainty) * (1 - contradiction) * (1 - ambiguity),
        # Uncertain AND evidence exists in store to fetch -> retrieve is worthwhile.
        # Edge-case: if evidence_coverage == 0 (empty corpus or no relevant docs),
        # retrieve score collapses to 0 and the policy falls through to abstain
        # (uncertainty * 1.0). This is intentional — retrieving from an empty store
        # wastes calls and returns nothing. If your corpus is populated but coverage
        # is still 0 for a query, check the relevance_distance_threshold in
        # evidence_retrieval.py (currently 0.55) — it may be too tight.
        "retrieve": uncertainty * evidence_coverage,
        # Evidence contradicts candidate answer -> verify.
        # Also gated by evidence_coverage so verify isn't triggered when no docs exist.
        "verify": contradiction * evidence_coverage,
        # Question is genuinely ambiguous -> clarify
        "clarify": ambiguity,
        # Uncertain AND no evidence available -> abstain
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

# src/policy/decision_policy.py — add this to estimate_reliability()
def estimate_reliability(features: dict) -> dict:
    uncertainty = features["uncertainty"]
    contradiction = features["contradiction_prob"]
    ambiguity = features["ambiguity"]
    evidence_coverage = features["evidence_coverage"]
    complexity = features.get("complexity", 0.0)  # NEW

    return {
        # High confidence + no contradiction + unambiguous + not too complex -> answer directly
        "answer": (1 - uncertainty) * (1 - contradiction) * (1 - ambiguity) * (1 - 0.3 * complexity),
        "retrieve": uncertainty * evidence_coverage,
        "verify": contradiction * evidence_coverage,
        "clarify": ambiguity,
        "abstain": uncertainty * (1 - evidence_coverage),
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