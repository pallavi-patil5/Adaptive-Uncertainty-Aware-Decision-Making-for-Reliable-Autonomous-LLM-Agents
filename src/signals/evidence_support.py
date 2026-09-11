# src/signals/evidence_support.py
import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sentence_transformers import CrossEncoder

_reranker = CrossEncoder("BAAI/bge-reranker-base")


def _sigmoid(x: float) -> float:
    """Normalize raw cross-encoder logit to [0,1]."""
    return float(1.0 / (1.0 + np.exp(-x)))


def evidence_support_score(question: str, answer: str, evidence_text: str) -> float:
    """
    Scores how well `evidence_text` supports `answer` as a response to `question`.
    Returns a sigmoid-normalized score in [0,1] (higher = stronger support).
    """
    pair_input = f"{question} {answer}"
    raw_score = _reranker.predict([(pair_input, evidence_text)])[0]
    return _sigmoid(raw_score)


def best_evidence_support(question: str, answer: str, retrieved_docs: list[dict]) -> dict:
    """Score answer support against each retrieved doc, return the best match."""
    if not retrieved_docs:
        return {"best_support_score": None, "best_evidence_text": None}

    scored = [
        (evidence_support_score(question, answer, doc["text"]), doc["text"])
        for doc in retrieved_docs
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_text = scored[0]
    return {"best_support_score": best_score, "best_evidence_text": best_text}


if __name__ == "__main__":
    score = evidence_support_score(
        question="What is the capital of France?",
        answer="Paris",
        evidence_text="Paris is the capital of France.",
    )
    print("Support score (should be high/positive):", score)

    score_bad = evidence_support_score(
        question="What is the capital of France?",
        answer="Paris",
        evidence_text="The Eiffel Tower was completed in 1889.",
    )
    print("Support score for irrelevant evidence (should be lower):", score_bad)