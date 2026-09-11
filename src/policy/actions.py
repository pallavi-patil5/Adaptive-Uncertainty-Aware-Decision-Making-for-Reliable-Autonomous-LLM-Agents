# src/policy/actions.py
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm_client import call_llm
from vector_store import query as vector_query


def action_answer(question: str, candidate_answer: str) -> dict:
    return {
        "action": "answer",
        "final_answer": candidate_answer,
        "trace": "Answered directly.",
        "llm_calls": 0,       # candidate_answer already generated in feature_extractor
        "retrieval_calls": 0,
        "latency_s": 0.0,
    }


def action_retrieve(question: str, k: int = 3) -> dict:
    t0 = time.monotonic()
    result = vector_query(question, k=k)
    docs = result["documents"][0] if result["documents"] else []
    context = "\n\n".join(docs) if docs else "No evidence found."
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer using the context above:"
    answer = call_llm(prompt, temperature=0.0)
    return {
        "action": "retrieve",
        "final_answer": answer,
        "evidence_used": docs,
        "trace": f"Retrieved {len(docs)} evidence docs, then answered.",
        "llm_calls": 1,
        "retrieval_calls": 1,
        "latency_s": max(0.0, round(time.monotonic() - t0, 3)),
    }


def action_verify(question: str, candidate_answer: str, evidence_text: str) -> dict:
    t0 = time.monotonic()
    prompt = (
        f"Question: {question}\n"
        f"Proposed answer: {candidate_answer}\n"
        f"Evidence: {evidence_text}\n\n"
        "Check the proposed answer against the evidence. If it's correct, restate it. "
        "If it's wrong, give the corrected answer based on the evidence. "
        "Respond with only the final, verified answer."
    )
    verified_answer = call_llm(prompt, temperature=0.0)
    return {
        "action": "verify",
        "final_answer": verified_answer,
        "trace": f"Verified '{candidate_answer}' against evidence -> '{verified_answer}'.",
        "llm_calls": 1,
        "retrieval_calls": 0,
        "latency_s": max(0.0, round(time.monotonic() - t0, 3)),
    }


def action_clarify(question: str) -> dict:
    t0 = time.monotonic()
    prompt = (
        f"Question: {question}\n\n"
        "This question is ambiguous. Write one short, specific clarifying question "
        "to ask the user so their intent becomes clear. Respond with only the clarifying question."
    )
    clarifying_question = call_llm(prompt, temperature=0.0)
    return {
        "action": "clarify",
        "final_answer": None,
        "clarifying_question": clarifying_question,
        "trace": "Question was ambiguous — asked for clarification instead of answering.",
        "llm_calls": 1,
        "retrieval_calls": 0,
        "latency_s": max(0.0, round(time.monotonic() - t0, 3)),
    }


def action_abstain(question: str, reason_hint: str = "") -> dict:
    return {
        "action": "abstain",
        "final_answer": "I don't have reliable enough information to answer this confidently.",
        "trace": f"Abstained — insufficient confidence/evidence. {reason_hint}".strip(),
        "llm_calls": 0,
        "retrieval_calls": 0,
        "latency_s": 0.0,
    }


if __name__ == "__main__":
    print(action_answer("What is the capital of France?", "Paris"))
    print(action_clarify("What does 'the meeting' refer to, and when is it?"))
    print(action_abstain("What was the exact color of a random shirt in Pune yesterday?"))
