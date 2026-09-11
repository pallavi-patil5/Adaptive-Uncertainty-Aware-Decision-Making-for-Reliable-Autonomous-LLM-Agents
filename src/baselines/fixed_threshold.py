# src/baselines/fixed_threshold.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from baselines.common import CallCounter, make_result
from vector_store import query as vector_query
from llm_client import call_llm

# Import the raw uncertainty pipeline pieces so we can count their LLM calls explicitly
from uncertainty.self_consistency import self_consistency_score
from uncertainty.self_eval import self_eval_uncertainty


def run(question: str, threshold: float = 0.5, n_consistency_samples: int = 5) -> dict:
    """
    threshold: fixed cutoff in [0,1]. uncertainty > threshold -> retrieve, else answer directly.
    Parameterized so Week 6 can sweep this across {0.1, ..., 0.9}.
    """
    counter = CallCounter()

    # Uncertainty estimation — count these calls too (self-consistency = n samples, self-eval = 1 answer + 2 scoring calls)
    sc_result = self_consistency_score(question, n=n_consistency_samples)
    counter.count += n_consistency_samples

    se_result = self_eval_uncertainty(question)
    counter.count += 3  # get_answer + p_true + p_ik, per self_eval_uncertainty's internals

    combined_uncertainty = 0.5 * sc_result["uncertainty"] + 0.5 * se_result["uncertainty"]

    if combined_uncertainty > threshold:
        # Retrieve path
        retrieved = vector_query(question, k=3)
        docs = retrieved["documents"][0] if retrieved["documents"] else []
        context = "\n\n".join(docs) if docs else "No evidence found."
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer using the context above:"
        answer = counter.call(prompt, temperature=0.0)

        return make_result(
            action="retrieve",
            final_answer=answer,
            trace=f"Fixed-threshold agent — uncertainty {combined_uncertainty:.3f} > threshold {threshold} -> retrieved.",
            llm_calls=counter.count,
            retrieval_calls=1,
            extra={"uncertainty": combined_uncertainty, "threshold": threshold, "evidence_used": docs},
        )
    else:
        return make_result(
            action="answer",
            final_answer=se_result["answer"],
            trace=f"Fixed-threshold agent — uncertainty {combined_uncertainty:.3f} <= threshold {threshold} -> answered directly.",
            llm_calls=counter.count,
            extra={"uncertainty": combined_uncertainty, "threshold": threshold},
        )


if __name__ == "__main__":
    for t in [0.3, 0.7]:
        result = run("What is the capital of France?", threshold=t)
        print(f"threshold={t}: action={result['action']}, uncertainty={result['uncertainty']:.3f}, llm_calls={result['llm_calls']}")