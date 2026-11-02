# src/uncertainty/combine.py
import sys, os
sys.path.append(os.path.dirname(__file__))

from self_consistency import self_consistency_score
from self_eval import get_answer, self_eval_uncertainty


def uncertainty_score(question: str, n_samples: int = 2, weights: dict | None = None) -> dict:
    """
    Combines self-consistency + P(True)/P(IK) uncertainty.

    LLM call budget per question (down from 9 → 3):
      1. get_answer          — 1 call  (reused by both signals)
      2. self_consistency    — 1 extra sample (seed_answer counts as first)
      3. self_eval combined  — 1 call  (PTRUE + PIK in one prompt)
    """
    weights = weights or {"self_consistency": 0.5, "self_eval": 0.5}

    # Single get_answer call — reused by both signals
    answer = get_answer(question)

    sc_result = self_consistency_score(question, n=n_samples, seed_answer=answer)
    se_result = self_eval_uncertainty(question, answer=answer)

    combined = (
        weights["self_consistency"] * sc_result["uncertainty"]
        + weights["self_eval"] * se_result["uncertainty"]
    )

    return {
        "question": question,
        "combined_uncertainty": combined,
        "self_consistency_uncertainty": sc_result["uncertainty"],
        "self_eval_uncertainty": se_result["uncertainty"],
        "p_true": se_result["p_true"],
        "p_ik": se_result["p_ik"],
        "answer": answer,
        "samples": sc_result["samples"],
    }


if __name__ == "__main__":
    result = uncertainty_score("What is the capital of France?")
    print(f"Combined uncertainty: {result['combined_uncertainty']:.3f}")
    print(f"  self-consistency: {result['self_consistency_uncertainty']:.3f}")
    print(f"  self-eval:        {result['self_eval_uncertainty']:.3f}")
