# src/signals/ambiguity.py
import sys, os, re
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm_client import call_llm

# Unresolved pronouns / demonstratives that make a question definitively ambiguous
_AMBIGUOUS_REFS = re.compile(
    r"\b(he|she|they|it|this|that|these|those|him|her|them|his|hers|its|their)\b",
    re.IGNORECASE
)
_VAGUE_REFS = re.compile(
    r"\b(the meeting|the report|the event|the project|the document|the file|the case|the issue)\b",
    re.IGNORECASE
)


def ambiguity_score(question: str) -> dict:
    # Rule-based fast path: unresolved pronoun or vague definite reference → high ambiguity
    words = question.split()
    has_pronoun = bool(_AMBIGUOUS_REFS.search(question))
    has_vague_ref = bool(_VAGUE_REFS.search(question))
    # Only trigger if the question is short (no named entity to resolve the pronoun)
    named_entity_likely = any(w[0].isupper() for w in words if len(w) > 2 and w not in ("What", "Who", "When", "Where", "Why", "How", "Is", "Are", "Was", "Were", "Did", "Do", "Does"))
    if (has_pronoun or has_vague_ref) and not named_entity_likely:
        reason = "Contains unresolved pronoun or vague reference that the user could clarify."
        return {"ambiguity": 0.9, "reasoning": reason}

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