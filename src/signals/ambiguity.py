# src/signals/ambiguity.py
import sys, os, re
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm_client import call_llm



def ambiguity_score(question: str) -> dict:
    prompt = (
        f"Question: {question}\n\n"
        "A question is AMBIGUOUS only if clarifying with the user could actually resolve it, "
        "and either:\n"
        "(a) it has multiple distinct, equally valid interpretations leading to different "
        "correct answers (e.g. a name shared by multiple people/places), or\n"
        "(b) it contains an unresolved reference (e.g. 'the meeting', 'it', 'that report') "
        "whose specific referent the user could supply if asked.\n\n"
        "A question is NOT ambiguous if it is merely open-ended, subjective, or broad, and it "
        "is NOT ambiguous if no amount of clarification from the user could make it answerable "
        "— for example, asking for an unrecorded, unobservable fact about an unspecified or "
        "unnamed person or moment (e.g. 'the exact thought of a random stranger', 'the color "
        "of a random person's shirt yesterday') is UNANSWERABLE, not ambiguous, because the "
        "user clarifying their intent would not produce a fact that exists to be known.\n\n"
        "On a scale of 0 to 100, how ambiguous is this question in the specific senses above "
        "(not how unanswerable it is)?\n"
        "Respond in exactly this format:\n"
        "SCORE: <integer 0-100>\n"
        "REASON: <one sentence>"
    )
    raw = call_llm(prompt, temperature=0.0)
    score_match = re.search(r"SCORE:\s*(\d+)", raw)
    reason_match = re.search(r"REASON:\s*(.+)", raw)
    score = int(score_match.group(1)) if score_match else 50
    score = max(0, min(100, score))
    reason = reason_match.group(1).strip() if reason_match else raw.strip()
    return {"ambiguity": score / 100.0, "reasoning": reason}


if __name__ == "__main__":
    print(ambiguity_score("What is the capital of France?"))
    print(ambiguity_score("What is the population of Springfield?"))
    print(ambiguity_score("What does 'the meeting' refer to, and when is it?"))
    print(ambiguity_score("What was the exact color of the shirt worn by a random person in Pune yesterday?"))