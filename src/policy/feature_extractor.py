# src/policy/feature_extractor.py
import sys, os, re
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from uncertainty.combine import uncertainty_score
from signals.evidence_retrieval import retrieval_signals
from signals.evidence_support import evidence_support_score
from signals.contradiction import contradiction_score
from signals.ambiguity import ambiguity_score
from signals.complexity import complexity_score

_CLAIM_CHECK_PATTERN = re.compile(
    r"^(is it true|is it a fact|is the claim|did .{1,40} really|was .{1,40} really|"
    r"is .{1,40} really|are .{1,40} really|does .{1,40} really|is the myth|is the rumou?r|"
    r"fact or (fiction|myth)|true or false|myth or fact)",
    re.IGNORECASE
)


def extract_features(question: str, n_consistency_samples: int = 2) -> dict:
    """
    Full signal pipeline for one question.
    LLM calls per question: 3  (down from 9)
      1. get_answer (inside uncertainty_score, reused by both sub-signals)
      2. self_consistency extra sample
      3. self_eval combined PTRUE+PIK prompt
    """
    # --- Uncertainty (3 LLM calls total, answer reused) ---
    unc = uncertainty_score(question, n_samples=n_consistency_samples)
    answer = unc["answer"]  # reuse — no extra LLM call

    # --- Retrieval signals (embedding only, no LLM) ---
    RELEVANCE_THRESHOLD = 0.75
    NLI_THRESHOLD = 0.60

    ret = retrieval_signals(question, k=3, relevance_distance_threshold=RELEVANCE_THRESHOLD)
    top1_distance = ret["top1_distance"] if ret["top1_distance"] is not None else 1.0

    support_score = 0.0
    contradiction_probs = {"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0}
    if ret["retrieved_docs"] and top1_distance < NLI_THRESHOLD:
        top_doc = ret["retrieved_docs"][0]["text"]
        support_score = evidence_support_score(question, answer, top_doc)
        contradiction_probs = contradiction_score(top_doc, answer)

    # --- Heuristic signals (no LLM) ---
    amb  = ambiguity_score(question)
    comp = complexity_score(question)
    is_claim_check = bool(_CLAIM_CHECK_PATTERN.match(question))

    return {
        "question": question,
        "candidate_answer": answer,

        "uncertainty": unc["combined_uncertainty"],
        "self_consistency_uncertainty": unc["self_consistency_uncertainty"],
        "self_eval_uncertainty": unc["self_eval_uncertainty"],

        "evidence_coverage": ret["evidence_coverage"],
        "top1_distance": top1_distance,
        "evidence_relevant": ret["retrieved_docs"] and top1_distance < NLI_THRESHOLD,
        "support_score": support_score,

        "contradiction_prob": contradiction_probs["contradiction"],
        "entailment_prob": contradiction_probs["entailment"],

        "ambiguity": amb["ambiguity"],
        "complexity": comp["complexity"],
        "is_claim_check": is_claim_check,
    }


if __name__ == "__main__":
    features = extract_features("What is the capital of France?")
    for k, v in features.items():
        print(f"{k}: {v}")
