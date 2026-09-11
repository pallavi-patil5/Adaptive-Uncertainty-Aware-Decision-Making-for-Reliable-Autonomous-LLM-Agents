# src/baselines/self_reflection.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from baselines.common import CallCounter, make_result


def run(question: str) -> dict:
    counter = CallCounter()

    # Step 1: generate initial answer
    initial_prompt = f"Question: {question}\nAnswer concisely:"
    initial_answer = counter.call(initial_prompt, temperature=0.0)

    # Step 2: critique + revise in one call
    reflection_prompt = (
        f"Question: {question}\n"
        f"Proposed answer: {initial_answer}\n\n"
        "Critically review this answer for factual errors, overconfidence, or missing caveats. "
        "Then provide a final, revised answer. Respond in exactly this format:\n"
        "CRITIQUE: <your critique>\n"
        "FINAL_ANSWER: <revised answer>"
    )
    raw = counter.call(reflection_prompt, temperature=0.0)

    final_answer = raw
    if "FINAL_ANSWER:" in raw:
        final_answer = raw.split("FINAL_ANSWER:")[-1].strip()

    return make_result(
        action="answer",
        final_answer=final_answer,
        trace=f"Self-reflection agent — generated, critiqued, revised. Initial: '{initial_answer}'",
        llm_calls=counter.count,
        extra={"initial_answer": initial_answer, "raw_reflection": raw},
    )


if __name__ == "__main__":
    result = run("What is the capital of France?")
    print(result)