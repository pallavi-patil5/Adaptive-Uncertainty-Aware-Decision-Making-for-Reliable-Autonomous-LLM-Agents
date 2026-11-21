# src/policy/decision_policy.py
# Lambda empirically tuned via lambda_sweep.py: best action_accuracy at 0.05
import os
import joblib
import numpy as np

LAMBDA = 0.05

# Weight given to XGBoost probabilities vs rule-based reliability.
# 0.0 = pure rule-based, 1.0 = pure XGBoost, 0.4 = recommended blend.
XGB_ALPHA = 0.4

MODEL_PATH   = "data/processed/policy_classifier.joblib"
ENCODER_PATH = "data/processed/policy_label_encoder.joblib"

ACTION_COSTS = {
    "answer": 1.0,
    "retrieve": 1.3,
    "verify": 1.6,
    "clarify": 1.0,
    "abstain": 0.0,
}

FEATURE_COLUMNS = [
    "uncertainty", "self_consistency_uncertainty", "self_eval_uncertainty",
    "evidence_coverage", "top1_distance", "support_score",
    "contradiction_prob", "entailment_prob",
    "ambiguity", "complexity", "is_claim_check",
]


class _XGBPolicy:
    """Lazy-loaded XGBoost model + label encoder. Returns None if files are missing."""
    _model   = None
    _encoder = None
    _loaded  = False

    @classmethod
    def load(cls):
        if cls._loaded:
            return
        cls._loaded = True
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            cls._model   = joblib.load(MODEL_PATH)
            cls._encoder = joblib.load(ENCODER_PATH)

    @classmethod
    def predict_proba(cls, features: dict) -> dict | None:
        """Returns {action: probability} dict, or None if model not available."""
        cls.load()
        if cls._model is None:
            return None
        x = np.array([[features.get(col, 0.0) for col in FEATURE_COLUMNS]], dtype=float)
        probs = cls._model.predict_proba(x)[0]
        return {cls._encoder.classes_[i]: float(probs[i]) for i in range(len(probs))}


def estimate_reliability(features: dict) -> dict:
    uncertainty = features["uncertainty"]
    ambiguity = features["ambiguity"]
    evidence_coverage = features.get("evidence_coverage", 0.0)
    top1_distance = features.get("top1_distance", 1.0)
    evidence_relevant = features.get(
        "evidence_relevant",
        evidence_coverage > 0 and top1_distance < 0.6,
    )
    evidence_availability = evidence_coverage if evidence_relevant else 0.0
    entailment = features.get("entailment_prob", 0.0) if evidence_relevant else 0.0
    contradiction = features.get("contradiction_prob", 0.0) if evidence_relevant else 0.0
    complexity = features.get("complexity", 0.0)
    is_claim_check = features.get("is_claim_check", False)

    complexity_weight = 0.7 if complexity >= 0.4 else 0.5
    retrieve_signal = max(
        ((1 - complexity_weight) * uncertainty + complexity_weight * complexity)
        * evidence_availability,
        entailment,
    )

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
        "answer": (
            (1 - uncertainty) * (1 - contradiction) * (1 - ambiguity)
            * (1 - 0.3 * complexity)
            * (1 - 0.5 * evidence_availability * (1 - entailment))
        ),
        "retrieve": retrieve_signal,
        "verify": verify_rel,
        "clarify": ambiguity * (1 + 0.4 * ambiguity),
        "abstain": uncertainty * (1 - complexity) * (1 - ambiguity) * (1 - entailment),
    }


def decide_action(features: dict, lam: float = LAMBDA, alpha: float = XGB_ALPHA) -> dict:
    """
    Hybrid policy: blends XGBoost class probabilities with rule-based reliability.

    final_score(A) = [ alpha * xgb_prob(A) + (1-alpha) * rule_reliability(A) ] - lam * cost(A)

    Falls back to pure rule-based if XGBoost model files are not present.
    """
    reliability = estimate_reliability(features)
    xgb_probs   = _XGBPolicy.predict_proba(features)  # None if model missing

    if xgb_probs is None:
        # Pure rule-based fallback
        blended = reliability
        mode = "rule_based"
    else:
        # Ensure all 5 actions are present (model may not have seen all classes)
        xgb_probs = {a: xgb_probs.get(a, 0.0) for a in reliability}
        blended = {
            a: alpha * xgb_probs[a] + (1 - alpha) * reliability[a]
            for a in reliability
        }
        mode = "hybrid"

    scores = {a: blended[a] - lam * ACTION_COSTS[a] for a in blended}
    best_action = max(scores, key=scores.get)
    return {
        "action":      best_action,
        "scores":      scores,
        "reliability": reliability,
        "blended":     blended,
        "xgb_probs":   xgb_probs,
        "mode":        mode,
    }


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
        print(f"  -> action: {decision['action']}  (mode={decision['mode']})")
        print(f"  -> scores:  { {k: round(v, 3) for k, v in decision['scores'].items()} }")
        print(f"  -> blended: { {k: round(v, 3) for k, v in decision['blended'].items()} }")
