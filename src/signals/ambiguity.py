# src/signals/ambiguity.py
import sys, os, re
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm_client import call_llm


def ambiguity_score(question: str) -> dict:
    """
    Returns:
      - ambiguity: float in [0,1], 0 = unambiguous, 1 = clearly ambiguous
      - reasoning: model's brief justification (useful for debugging/report examples)
    """
    
    prompt = (
        f"Question: {question}\n\n"
        "A question is AMBIGUOUS if EITHER of these is true:\n"
        "(a) It has multiple distinct, equally valid interpretations that would lead to "
        "genuinely different correct answers (e.g. a name shared by multiple people/places).\n"
        "(b) It contains an unresolved reference (a pronoun or phrase like 'the meeting', 'it', "
        "'that report', 'this person') with no antecedent given in the question, making it "
        "impossible to answer without first knowing what is being referred to.\n\n"
        "A question is NOT ambiguous just because it is open-ended, subjective, or broad.\n\n"
        "On a scale of 0 to 100, how ambiguous is this question in either of the senses above?\n"
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