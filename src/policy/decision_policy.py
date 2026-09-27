# src/policy/decision_policy.py
# Lambda empirically tuned via lambda_sweep.py: best action_accuracy at 0.05
LAMBDA = 0.05

ACTION_COSTS = {
    "answer": 1.0,
    "retrieve": 1.3,
    "verify": 1.6,
    "clarify": 1.0,
    "abstain": 0.0,
}


def estimate_reliability(features: dict) -> dict:
    uncertainty = features["uncertainty"]
    contradiction = features["contradiction_prob"]
    ambiguity = features["ambiguity"]
    evidence_coverage = features["evidence_coverage"]
    complexity = features.get("complexity", 0.0)
    self_eval_unc = features.get("self_eval_uncertainty", uncertainty)
    is_claim_check = features.get("is_claim_check", False)

    complexity_weight = 0.7 if complexity >= 0.4 else 0.5
    retrieve_signal = (1 - complexity_weight) * uncertainty + complexity_weight * complexity

    # verify fires on:
    #   (a) high contradiction from NLI against retrieved evidence
    #   (b) is_claim_check=True — question explicitly asks to verify a claim/myth
    #       boosted to 0.9 so it beats answer even when model is confident
    claim_signal = 0.9 if is_claim_check else 0.0
    verify_base = max(contradiction, claim_signal)
    # no complexity dampening for claim-check — always worth verifying a stated claim
    complexity_damp = 0.0 if is_claim_check else (1 - 0.2 * (1 - complexity))
    verify_rel = verify_base * (1 - ambiguity) * (1 - complexity_damp) + 0.2 * contradiction if not is_claim_check else verify_base * (1 - ambiguity)

    return {
        "answer": (1 - uncertainty) * (1 - contradiction) * (1 - ambiguity) * (1 - 0.3 * complexity),
        "retrieve": retrieve_signal,
        "verify": verify_rel,
        "clarify": ambiguity * (1 + 0.4 * ambiguity),
        # abstain: uncertain AND simple AND unambiguous
        # NOT gated by evidence_coverage — irrelevant vector store hits must not suppress abstain
        "abstain": uncertainty * (1 - complexity) * (1 - ambiguity),
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
        "Is it true that Einstein failed math as a child?",
        "What does 'the meeting' refer to, and when is it?",
        "What was the exact color of the shirt worn by a random person in Pune yesterday?",
        "Who won the 2024 US Presidential election?",
    ]:
        features = extract_features(q)
        decision = decide_action(features)
        print(f"\nQ: {q}")
        print(f"  self_eval_unc={features.get('self_eval_uncertainty', 'N/A'):.3f}  "
              f"contradiction={features['contradiction_prob']:.3f}  "
              f"ambiguity={features['ambiguity']:.3f}")
        print(f"  -> action: {decision['action']}")
        print(f"  -> scores: { {k: round(v, 3) for k, v in decision['scores'].items()} }")
