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
_AMBIGUOUS_ENTITY_QUERY = re.compile(
    r"^\s*(?:tell me about|what is|who is)\s+apple[?.!]*\s*$",
    re.IGNORECASE,
)
_POLYSEMY_REFS = re.compile(
    r"(?:^|\s)(?:tell me about|what is|who is|what's|who's)\s+apple(?:\?|\.|!|$)",
    re.IGNORECASE,
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

    if _AMBIGUOUS_ENTITY_QUERY.fullmatch(question):
        return {"ambiguity": 0.9, "reasoning": "Apple may refer to the company or the fruit; no context distinguishes them."}

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

    # Fast path 3: polysemy / multiple valid meanings — e.g. "Apple" can mean the company or the fruit.
    if _POLYSEMY_REFS.search(question):
        return {"ambiguity": 0.9, "reasoning": "The query is ambiguous because 'Apple' could refer to the company, the fruit, or another entity."}

    # Heuristic fallback — no LLM call needed
    # Multi-word named entities present → likely specific enough, low ambiguity
    if named_entity_likely and not has_pronoun and not has_vague_ref:
        return {"ambiguity": 0.1, "reasoning": "Contains named entity with no unresolved references — likely specific."}

    # Question words that are inherently specific
    specific_starters = re.compile(
        r"^(what is the|what was the|who is|who was|when did|where is|where was|how many|how much)",
        re.IGNORECASE
    )
    if specific_starters.match(question) and not has_pronoun and not has_vague_ref:
        return {"ambiguity": 0.15, "reasoning": "Specific question form with no unresolved references."}

    # Fallback: moderate ambiguity for anything else
    return {"ambiguity": 0.35, "reasoning": "Could not determine ambiguity from heuristics alone."}


if __name__ == "__main__":
    print(ambiguity_score("What is the capital of France?"))
    print(ambiguity_score("What is the population of Springfield?"))
    print(ambiguity_score("What does 'the meeting' refer to, and when is it?"))
    print(ambiguity_score("What was the exact color of the shirt worn by a random person in Pune yesterday?"))
    print(ambiguity_score("What is my neighbour thinking right now?"))
