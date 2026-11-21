# src/uncertainty/self_consistency.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from itertools import combinations
from sentence_transformers import SentenceTransformer
from llm_client import call_llm

_embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")


def sample_answers(question: str, n: int = 2, temperature: float = 0.8, seed_answer: str = None) -> list[str]:
    """
    Sample N answers. If seed_answer is provided it counts as the first sample,
    so only (n - 1) extra LLM calls are made instead of n.
    """
    prompt = f"Question: {question}\nAnswer concisely (a few words or one sentence):"
    samples = [seed_answer] if seed_answer else []
    remaining = n - len(samples)
    samples += [call_llm(prompt, temperature=temperature) for _ in range(remaining)]
    return samples


def self_consistency_score(question: str, n: int = 2, temperature: float = 0.8, seed_answer: str = None) -> dict:
    """
    Returns:
      - uncertainty: 0 (fully consistent) to 1 (fully inconsistent)
      - samples: raw sampled answers
    With seed_answer + n=2 this costs only 1 LLM call instead of 5.
    """
    samples = sample_answers(question, n=n, temperature=temperature, seed_answer=seed_answer)
    embeddings = _embedder.encode(samples, normalize_embeddings=True)

    pairs = list(combinations(range(len(samples)), 2))
    if not pairs:
        return {"uncertainty": 0.0, "samples": samples}

    sims = [float(np.dot(embeddings[i], embeddings[j])) for i, j in pairs]
    mean_sim = float(np.mean(sims))
    uncertainty = max(0.0, min(1.0, 1.0 - mean_sim))

    return {"uncertainty": uncertainty, "mean_similarity": mean_sim, "samples": samples}


if __name__ == "__main__":
    result_easy = self_consistency_score("What is the capital of France?")
    print("Easy question:", result_easy["uncertainty"], result_easy["samples"])

    result_hard = self_consistency_score("What will the exact stock price of a random startup be in 2030?")
    print("Hard question:", result_hard["uncertainty"], result_hard["samples"])
