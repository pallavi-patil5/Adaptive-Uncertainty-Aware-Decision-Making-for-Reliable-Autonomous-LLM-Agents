# src/uncertainty/self_eval.py
import sys, os, re
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm_client import call_llm


def get_answer(question: str) -> str:
    prompt = f"Question: {question}\nAnswer concisely:"
    return call_llm(prompt, temperature=0.0)


def self_eval_uncertainty(question: str, answer: str = None) -> dict:
    """
    Combines P(True) + P(IK) in a SINGLE LLM call instead of 3 separate calls.
    Accepts pre-computed answer to avoid duplicate get_answer call.
    """
    if answer is None:
        answer = get_answer(question)

    prompt = (
        f"Question: {question}\n"
        f"Proposed answer: {answer}\n\n"
        "Respond with exactly two lines:\n"
        "PTRUE: <integer 0-100>  (how likely this answer is factually correct)\n"
        "PIK: <integer 0-100>    (how likely you have reliable knowledge to answer this)"
    )
    raw = call_llm(prompt, temperature=0.0)
    ptrue_match = re.search(r"PTRUE:\s*(\d+)", raw, re.IGNORECASE)
    pik_match   = re.search(r"PIK:\s*(\d+)",   raw, re.IGNORECASE)

    p_true = max(0, min(100, int(ptrue_match.group(1)))) / 100.0 if ptrue_match else 0.5
    p_ik   = max(0, min(100, int(pik_match.group(1))))   / 100.0 if pik_match   else 0.5

    confidence  = (p_true + p_ik) / 2.0
    uncertainty = 1.0 - confidence
    return {
        "uncertainty": uncertainty,
        "p_true": p_true,
        "p_ik": p_ik,
        "answer": answer,
    }


if __name__ == "__main__":
    result_easy = self_eval_uncertainty("What is the capital of France?")
    print("Easy question:", result_easy)

    result_hard = self_eval_uncertainty("What will the exact stock price of a random startup be in 2030?")
    print("Hard question:", result_hard)
