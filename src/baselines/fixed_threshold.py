# src/baselines/fixed_threshold.py
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from baselines.common import CallCounter, make_result
from vector_store import query as vector_query
from uncertainty.self_consistency import sample_answers
from uncertainty.self_eval import p_true_score, p_ik_score

import numpy as np
from itertools import combinations
from sentence_transformers import SentenceTransformer

_embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")


def run(question: str, threshold: float = 0.5, n_consistency_samples: int = 5) -> dict:
    """
    threshold: fixed cutoff in [0,1]. uncertainty > threshold -> retrieve, else answer directly.
    Parameterized so Week 6 can sweep this across {0.1, ..., 0.9}.
    """
    t0 = time.monotonic()
    counter = CallCounter()

    # --- Self-consistency uncertainty ---
    samples = [counter.call(f"Question: {question}\nAnswer concisely:", temperature=0.8)
               for _ in range(n_consistency_samples)]
    embeddings = _embedder.encode(samples, normalize_embeddings=True)
    pairs = list(combinations(range(len(samples)), 2))
    mean_sim = float(np.mean([np.dot(embeddings[i], embeddings[j]) for i, j in pairs])) if pairs else 1.0
    sc_uncertainty = max(0.0, min(1.0, 1.0 - mean_sim))

    # --- Self-eval uncertainty (P(True) + P(IK)) ---
    answer = counter.call(f"Question: {question}\nAnswer concisely:", temperature=0.0)
    p_true = p_true_score(question, answer)
    counter.count += 1
    p_ik = p_ik_score(question)
    counter.count += 1
    se_uncertainty = 1.0 - (p_true + p_ik) / 2.0

    combined_uncertainty = 0.5 * sc_uncertainty + 0.5 * se_uncertainty

    if combined_uncertainty > threshold:
        retrieved = vector_query(question, k=3)
        docs = retrieved["documents"][0] if retrieved["documents"] else []
        context = "\n\n".join(docs) if docs else "No evidence found."
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer using the context above:"
        final_answer = counter.call(prompt, temperature=0.0)

        return make_result(
            action="retrieve",
            final_answer=final_answer,
            trace=f"Fixed-threshold agent — uncertainty {combined_uncertainty:.3f} > threshold {threshold} -> retrieved.",
            llm_calls=counter.count,
            retrieval_calls=1,
            latency_s=max(0.0, round(time.monotonic() - t0, 3)),
            extra={"uncertainty": combined_uncertainty, "threshold": threshold, "evidence_used": docs},
        )
    else:
        return make_result(
            action="answer",
            final_answer=answer,
            trace=f"Fixed-threshold agent — uncertainty {combined_uncertainty:.3f} <= threshold {threshold} -> answered directly.",
            llm_calls=counter.count,
            latency_s=max(0.0, round(time.monotonic() - t0, 3)),
            extra={"uncertainty": combined_uncertainty, "threshold": threshold},
        )


if __name__ == "__main__":
    for t in [0.3, 0.7]:
        result = run("What is the capital of France?", threshold=t)
        print(f"threshold={t}: action={result['action']}, uncertainty={result['extra']['uncertainty']:.3f}, llm_calls={result['llm_calls']}, latency={result['latency_s']}s")
