# src/policy/feature_extractor.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from uncertainty.combine import uncertainty_score
from uncertainty.self_eval import get_answer
from signals.evidence_retrieval import retrieval_signals
from signals.evidence_support import evidence_support_score
from signals.contradiction import contradiction_score
from signals.ambiguity import ambiguity_score
from signals.complexity import complexity_score


def extract_features(question: str, n_consistency_samples: int = 3) -> dict:
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
    

    RELEVANCE_DISTANCE_THRESHOLD = 0.55  # calibrated in Week 4 Step 0 against real indexed evidence

    ret = retrieval_signals(question, k=3)
    evidence_coverage = ret["evidence_coverage"]
    top1_distance = ret["top1_distance"] if ret["top1_distance"] is not None else 1.0

    evidence_relevant = False
    support_score = None
    contradiction_probs = {"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0}
    if ret["retrieved_docs"] and top1_distance < RELEVANCE_DISTANCE_THRESHOLD:
        evidence_relevant = True
        top_doc_text = ret["retrieved_docs"][0]["text"]
        support_score = evidence_support_score(question, answer, top_doc_text)
        contradiction_probs = contradiction_score(top_doc_text, answer)
    
    amb = ambiguity_score(question)
    comp = complexity_score(question)

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
    }


if __name__ == "__main__":
    features = extract_features("What is the capital of France?")
    for k, v in features.items():
        print(f"{k}: {v}")