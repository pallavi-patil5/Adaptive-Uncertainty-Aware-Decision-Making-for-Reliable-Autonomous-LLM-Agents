# src/baselines/normal_llm.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from baselines.common import CallCounter, make_result


def run(question: str) -> dict:
    counter = CallCounter()
    prompt = f"Question: {question}\nAnswer concisely:"
    answer = counter.call(prompt, temperature=0.0)

    return make_result(
        action="answer",
        final_answer=answer,
        trace="Normal LLM agent — answered directly, no retrieval or self-checking.",
        llm_calls=counter.count,
    )


if __name__ == "__main__":
    result = run("What is the capital of France?")
    print(result)