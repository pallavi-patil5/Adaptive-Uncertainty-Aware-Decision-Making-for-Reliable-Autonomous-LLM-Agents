# src/signals/contradiction.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sentence_transformers import CrossEncoder
import numpy as np

_nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

# IMPORTANT: verify this against the model's card/config the first time you run it —
# some nli-deberta checkpoints order labels [contradiction, entailment, neutral],
# others differ. Print _nli_model.config.id2label to confirm before trusting results.
LABEL_ORDER = ["contradiction", "entailment", "neutral"]


def contradiction_score(evidence_text: str, answer: str) -> dict:
    """
    Returns softmax probabilities over [contradiction, entailment, neutral]
    for (evidence, answer) as (premise, hypothesis).
    """
    logits = _nli_model.predict([(evidence_text, answer)])
    probs = _softmax(logits[0])
    return dict(zip(LABEL_ORDER, [float(p) for p in probs]))


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


if __name__ == "__main__":
    # sanity-check label order first
    print("Model id2label:", _nli_model.config.id2label)

    result_entail = contradiction_score(
        evidence_text="Paris is the capital of France.",
        answer="The capital of France is Paris.",
    )
    print("Should show high 'entailment':", result_entail)

    result_contra = contradiction_score(
        evidence_text="Paris is the capital of France.",
        answer="The capital of France is Berlin.",
    )
    print("Should show high 'contradiction':", result_contra)