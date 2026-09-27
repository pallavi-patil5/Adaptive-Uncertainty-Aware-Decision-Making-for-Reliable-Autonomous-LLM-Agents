# src/signals/ambiguity.py
import sys, os, re
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm_client import call_llm

# Unresolved pronouns / demonstratives — only standalone tokens, not substrings
_AMBIGUOUS_REFS = re.compile(
    r"(?<![a-zA-Z])\b(he|she|they|it|this|that|these|those|him|her|them|his|hers|its|their)\b(?![a-zA-Z])",
    re.IGNORECASE
)
_VAGUE_REFS = re.compile(
    r"\b(the meeting|the report|the event|the project|the document|the file|the case|the issue)\b",
    re.IGNORECASE
)

# Unanswerable patterns: unobservable/unrecorded facts — these are NOT ambiguous.
# Clarifying with the user cannot produce a fact that doesn't exist to be known.
_UNANSWERABLE_PATTERNS = re.compile(
    r"\b(random (person|stranger|individual|man|woman|people|someone)|"
    r"exact (color|colour|shade|thought|feeling|number|amount)|"
    r"unspecified|unnamed person|unknown person|"
    r"(yesterday|last night|this morning|today).{0,30}(wearing|ate|said|did|thought|had on))\b",
    re.IGNORECASE
)


def ambiguity_score(question: str) -> dict:
    # Fast path 1: unanswerable pattern → near-zero ambiguity (should ABSTAIN, not CLARIFY)
    if _UNANSWERABLE_PATTERNS.search(question):
        return {"ambiguity": 0.05, "reasoning": "Question asks for an unobservable/unrecorded fact — unanswerable, not ambiguous."}

    words = question.split()
    has_pronoun = bool(_AMBIGUOUS_REFS.search(question))
    has_vague_ref = bool(_VAGUE_REFS.search(question))
    named_entity_likely = any(
        w[0].isupper() for w in words
        if len(w) > 2 and w not in ("What", "Who", "When", "Where", "Why", "How",
                                     "Is", "Are", "Was", "Were", "Did", "Do", "Does")
    )

    # Fast path 2: short question with unresolved pronoun/vague ref and no named entity → high ambiguity
    if (has_pronoun or has_vague_ref) and not named_entity_likely and len(words) <= 12:
        return {"ambiguity": 0.9, "reasoning": "Contains unresolved pronoun or vague reference that the user could clarify."}

    prompt = (
        f"Question: {question}\n\n"
        "A question is AMBIGUOUS only if clarifying with the user could actually resolve it, "
        "and either:\n"
        "(a) it has multiple distinct, equally valid interpretations leading to different "
        "correct answers (e.g. a name shared by multiple people/places), or\n"
        "(b) it contains an unresolved reference (e.g. 'the meeting', 'it', 'that report') "
        "whose specific referent the user could supply if asked.\n\n"
        "A question is NOT ambiguous if it is merely open-ended, subjective, or broad.\n"
        "A question is NOT ambiguous if no clarification from the user could make it answerable "
        "— e.g. 'the exact color of a random person's shirt yesterday' is UNANSWERABLE (score 0), "
        "not ambiguous, because the fact does not exist to be known.\n\n"
        "On a scale of 0 to 100, how ambiguous is this question?\n"
        "Respond in exactly this format:\n"
        "SCORE: <integer 0-100>\n"
        "REASON: <one sentence>"
    )
    raw = call_llm(prompt, temperature=0.0)
    score_match = re.search(r"SCORE:\s*(\d+)", raw)
    reason_match = re.search(r"REASON:\s*(.+)", raw)
    score = int(score_match.group(1)) if score_match else 20
    # Cap at 60 — prevents LLM from over-scoring non-ambiguous questions
    score = max(0, min(60, score))
    reason = reason_match.group(1).strip() if reason_match else raw.strip()
    return {"ambiguity": score / 100.0, "reasoning": reason}


if __name__ == "__main__":
    print(ambiguity_score("What is the capital of France?"))
    print(ambiguity_score("What is the population of Springfield?"))
    print(ambiguity_score("What does 'the meeting' refer to, and when is it?"))
    print(ambiguity_score("What was the exact color of the shirt worn by a random person in Pune yesterday?"))
    print(ambiguity_score("What is my neighbour thinking right now?"))
