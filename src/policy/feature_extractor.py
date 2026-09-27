# src/policy/feature_extractor.py
import sys, os, re
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from uncertainty.combine import uncertainty_score
from uncertainty.self_eval import get_answer
from signals.evidence_retrieval import retrieval_signals
from signals.evidence_support import evidence_support_score
from signals.contradiction import contradiction_score
from signals.ambiguity import ambiguity_score
from signals.complexity import complexity_score

# Questions phrased as claims to verify — model should verify regardless of confidence
_CLAIM_CHECK_PATTERN = re.compile(
    r"^(is it true|is it a fact|is the claim|did .{1,40} really|was .{1,40} really|"
    r"is .{1,40} really|are .{1,40} really|does .{1,40} really|is the myth|is the rumou?r|"
    r"fact or (fiction|myth)|true or false|myth or fact)",
    re.IGNORECASE
)


def extract_features(question: str, n_consistency_samples: int = 5) -> dict:
    """
    Runs the full Week 2 + Week 3 signal pipeline for one question.
    Returns a flat dict of features ready for both the rule-based policy
    and the classifier's training table.
    """
    # Candidate answer — needed for evidence-support / contradiction scoring
    answer = get_answer(question)

    # Week 2 signal
    unc = uncertainty_score(question, n_samples=n_consistency_samples)

    # Week 3 signals
    

    RELEVANCE_DISTANCE_THRESHOLD = 0.75  # for evidence_coverage calculation
    NLI_DISTANCE_THRESHOLD = 0.60        # tighter gate for NLI — only run on genuinely close docs

    ret = retrieval_signals(question, k=3, relevance_distance_threshold=RELEVANCE_DISTANCE_THRESHOLD)
    evidence_coverage = ret["evidence_coverage"]
    top1_distance = ret["top1_distance"] if ret["top1_distance"] is not None else 1.0

    evidence_relevant = False
    support_score = None
    contradiction_probs = {"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0}
    if ret["retrieved_docs"] and top1_distance < NLI_DISTANCE_THRESHOLD:
        evidence_relevant = True
        top_doc_text = ret["retrieved_docs"][0]["text"]
        support_score = evidence_support_score(question, answer, top_doc_text)
        # Only run NLI if cross-encoder confirms doc is topically relevant (support > 0.5)
        if support_score > 0.5:
            contradiction_probs = contradiction_score(top_doc_text, answer)
    
    amb = ambiguity_score(question)
    comp = complexity_score(question)
    is_claim_check = bool(_CLAIM_CHECK_PATTERN.match(question))

    return {
        "question": question,
        "candidate_answer": answer,

        "uncertainty": unc["combined_uncertainty"],
        "self_consistency_uncertainty": unc["self_consistency_uncertainty"],
        "self_eval_uncertainty": unc["self_eval_uncertainty"],

        "evidence_coverage": evidence_coverage,
        "top1_distance": top1_distance,
        "evidence_relevant": evidence_relevant,
        "support_score": support_score if support_score is not None else 0.0,

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