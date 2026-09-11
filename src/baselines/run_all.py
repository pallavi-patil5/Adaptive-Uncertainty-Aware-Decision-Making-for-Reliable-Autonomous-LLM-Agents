# src/baselines/run_all.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from baselines import normal_llm, standard_rag, self_reflection, fixed_threshold

BASELINES = {
    "normal_llm": normal_llm.run,
    "standard_rag": standard_rag.run,
    "self_reflection": self_reflection.run,
    "fixed_threshold": lambda q: fixed_threshold.run(q, threshold=0.5),
}


def run_all(question: str) -> dict:
    results = {}
    for name, fn in BASELINES.items():
        results[name] = fn(question)
    return results


if __name__ == "__main__":
    test_questions = [
        "What is the capital of France?",
        "Is the claim that humans can survive indefinitely without water true?",
    ]

    for q in test_questions:
        print("\n" + "=" * 70)
        print(f"Question: {q}")
        results = run_all(q)
        for name, r in results.items():
            print(f"\n[{name}] action={r['action']} llm_calls={r['llm_calls']} retrieval_calls={r['retrieval_calls']}")
            print(f"  answer: {r['final_answer']}")