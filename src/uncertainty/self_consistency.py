# src/uncertainty/self_consistency.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from itertools import combinations
from sentence_transformers import SentenceTransformer
from llm_client import call_llm

_embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")


def sample_answers(question: str, n: int = 5, temperature: float = 0.8) -> list[str]:
    """Sample N answers at higher temperature."""
    prompt = f"Question: {question}\nAnswer concisely (a few words or one sentence):"
    return [call_llm(prompt, temperature=temperature) for _ in range(n)]


def self_consistency_score(question: str, n: int = 5, temperature: float = 0.8) -> dict:
    """
    Returns a dict with:
      - uncertainty: 0 (fully consistent) to 1 (fully inconsistent)
      - samples: the raw sampled answers (useful for debugging/logging)
    """
    samples = sample_answers(question, n=n, temperature=temperature)
    embeddings = _embedder.encode(samples, normalize_embeddings=True)

    pairs = list(combinations(range(len(samples)), 2))
    if not pairs:
        return {"uncertainty": 0.0, "samples": samples}

    sims = [float(np.dot(embeddings[i], embeddings[j])) for i, j in pairs]
    mean_sim = float(np.mean(sims))

    uncertainty = 1.0 - mean_sim  # low agreement -> high uncertainty
    uncertainty = max(0.0, min(1.0, uncertainty))  # clip to [0,1]

    return {"uncertainty": uncertainty, "mean_similarity": mean_sim, "samples": samples}


if __name__ == "__main__":
    # Expect LOW uncertainty — factual, well-known
    result_easy = self_consistency_score("What is the capital of France?")
    print("Easy question:", result_easy["uncertainty"], result_easy["samples"])

    # Expect HIGHER uncertainty — obscure/unanswerable
    result_hard = self_consistency_score("What will the exact stock price of a random startup be in 2030?")
    print("Hard question:", result_hard["uncertainty"], result_hard["samples"])