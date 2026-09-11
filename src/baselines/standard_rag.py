# src/baselines/standard_rag.py
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from baselines.common import CallCounter, make_result
from vector_store import query as vector_query


def run(question: str, k: int = 3) -> dict:
    t0 = time.monotonic()
    counter = CallCounter()

    retrieved = vector_query(question, k=k)
    docs = retrieved["documents"][0] if retrieved["documents"] else []
    context = "\n\n".join(docs) if docs else "No evidence found."

    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer using the context above:"
    answer = counter.call(prompt, temperature=0.0)

    return make_result(
        action="retrieve",
        final_answer=answer,
        trace=f"Standard RAG agent — always retrieves ({len(docs)} docs), then answers.",
        llm_calls=counter.count,
        retrieval_calls=1,
        latency_s=max(0.0, round(time.monotonic() - t0, 3)),
        extra={"evidence_used": docs},
    )


if __name__ == "__main__":
    result = run("What is the capital of France?")
    print(result)