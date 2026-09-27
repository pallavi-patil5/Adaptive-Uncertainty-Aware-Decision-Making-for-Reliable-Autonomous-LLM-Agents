# src/signals/contradiction.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sentence_transformers import CrossEncoder
import numpy as np

_nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

# Read label order directly from model config — never hardcode, different checkpoints differ.
# id2label maps int index → label string; sort by key to get logit-position order.
_id2label = _nli_model.config.id2label  # e.g. {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
LABEL_ORDER = [_id2label[i] for i in sorted(_id2label.keys())]


def contradiction_score(evidence_text: str, answer: str) -> dict:
    """
    Returns softmax probabilities keyed by label name for (evidence, answer).
    Label order is read from model config so it is always correct.
    """
    logits = _nli_model.predict([(evidence_text, answer)])
    probs = _softmax(logits[0])
    result = dict(zip(LABEL_ORDER, [float(p) for p in probs]))
    # Normalise keys so callers always get 'contradiction', 'entailment', 'neutral'
    # regardless of which labels the checkpoint uses (some use 'LABEL_0' etc.)
    return {
        "contradiction": result.get("contradiction", 0.0),
        "entailment": result.get("entailment", 0.0),
        "neutral": result.get("neutral", 1.0),
    }


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


if __name__ == "__main__":
    print("Model id2label:", _nli_model.config.id2label)
    print("LABEL_ORDER resolved to:", LABEL_ORDER)

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

    result_myth = contradiction_score(
        evidence_text="Einstein excelled at mathematics from a young age and never failed math.",
        answer="Einstein failed math as a child.",
    )
    print("Einstein myth (should show high contradiction):", result_myth)
