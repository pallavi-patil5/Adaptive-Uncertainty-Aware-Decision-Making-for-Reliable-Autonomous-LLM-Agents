# src/signals/complexity.py
import re

MULTI_HOP_CUES = [
    "and", "who was", "where was", "which", "compared to",
    "before", "after", "same", "both", "difference between",
]


def complexity_score(question: str) -> dict:
    """
    Heuristic complexity signal in [0,1]. Combines:
      - length (normalized word count)
      - multi-hop cue phrase presence
      - count of capitalized tokens (proxy for named-entity density)
    """
    words = question.split()
    word_count = len(words)

    length_component = min(word_count / 25.0, 1.0)  # cap at 25 words = max length contribution

    cue_hits = sum(1 for cue in MULTI_HOP_CUES if cue in question.lower())
    cue_component = min(cue_hits / 3.0, 1.0)

    capitalized = re.findall(r"\b[A-Z][a-z]+\b", question)
    entity_component = min(len(capitalized) / 4.0, 1.0)

    complexity = (0.3 * length_component) + (0.4 * cue_component) + (0.3 * entity_component)
    complexity = max(0.0, min(1.0, complexity))

    return {
        "complexity": complexity,
        "word_count": word_count,
        "cue_hits": cue_hits,
        "entity_mentions": len(capitalized),
    }


if __name__ == "__main__":
    print(complexity_score("What is the capital of France?"))
    print(complexity_score(
        "Who was the president of the country where the 2016 Summer Olympics were held?"
    ))