# src/uncertainty/self_eval.py
import sys, os, re
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm_client import call_llm


def get_answer(question: str) -> str:
    prompt = f"Question: {question}\nAnswer concisely:"
    return call_llm(prompt, temperature=0.0)




# src/uncertainty/self_eval.py — updated p_true_score
def p_true_score(question: str, answer: str, knowledge: str = "") -> float:
    context_block = f"Reference knowledge: {knowledge}\n\n" if knowledge else ""
    prompt = (
        f"{context_block}"
        f"Question: {question}\n"
        f"Proposed answer: {answer}\n\n"
        "Be critical and skeptical. Based only on the reference knowledge (if given) "
        "or well-established facts, on a scale of 0 to 100, how confident are you that "
        "this answer is factually correct? Respond with ONLY a single integer."
    )
    raw = call_llm(prompt, temperature=0.0)
    match = re.search(r"\d+", raw)
    if not match:
        return 0.5
    pct = max(0, min(100, int(match.group())))
    return pct / 100.0


def p_ik_score(question: str) -> float:
    """
    Ask the model to self-assess whether it likely KNOWS the answer to this
    type of question at all (independent of any specific answer generated).
    """
    prompt = (
        f"Question: {question}\n\n"
        "Without answering the question, rate on a scale of 0 to 100 how likely it is "
        "that you have reliable, factual knowledge to answer this question correctly. "
        "Respond with ONLY a single integer, nothing else."
    )
    raw = call_llm(prompt, temperature=0.0)
    match = re.search(r"\d+", raw)
    if not match:
        return 0.5
    pct = max(0, min(100, int(match.group())))
    return pct / 100.0


def self_eval_uncertainty(question: str) -> dict:
    """Combine P(True) and P(IK) into one uncertainty value (0=confident, 1=uncertain)."""
    answer = get_answer(question)
    p_true = p_true_score(question, answer)
    p_ik = p_ik_score(question)
    confidence = (p_true + p_ik) / 2.0
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