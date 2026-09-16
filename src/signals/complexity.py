# src/signals/complexity.py
import re

MULTI_HOP_CUES = [
    "who was", "where was", "which", "compared to", "before", "after",
    "same", "both", "difference between", "what was", "when did", "how did",
    "what caused", "why did", "in what", "at what",
]

# Time-sensitive questions need retrieval regardless of model confidence
TIME_SENSITIVE_CUES = [
    "current", "currently", "today", "now", "latest", "recent", "this year",
    "this month", "price", "rate", "stock", "weather", "live", "right now",
    "as of", "2024", "2025", "2026",
]


def complexity_score(question: str) -> dict:
    """
    Complexity signal in [0,1]. Combines:
      - length (normalized word count)
      - multi-hop cue phrase presence
      - named-entity density (capitalized tokens)
      - time-sensitivity (forces high complexity so policy retrieves)
    """
    words = question.split()
    word_count = len(words)
    q_lower = question.lower()

    length_component = min(word_count / 25.0, 1.0)

    cue_hits = sum(1 for cue in MULTI_HOP_CUES if cue in q_lower)
    cue_component = min(cue_hits / 3.0, 1.0)

    capitalized = re.findall(r"\b[A-Z][a-z]+\b", question)
    entity_component = min(len(capitalized) / 5.0, 1.0)  # raised cap from 4 to 5

    time_sensitive = any(cue in q_lower for cue in TIME_SENSITIVE_CUES)
    time_component = 1.0 if time_sensitive else 0.0

    if time_sensitive:
        # Time-sensitive questions always get high complexity so retrieve fires
        complexity = max(0.75, (0.2 * length_component) + (0.3 * cue_component) + (0.2 * entity_component) + (0.3 * time_component))
    else:
        complexity = (0.25 * length_component) + (0.45 * cue_component) + (0.30 * entity_component)

    complexity = max(0.0, min(1.0, complexity))

    return {
        "complexity": complexity,
        "word_count": word_count,
        "cue_hits": cue_hits,
        "entity_mentions": len(capitalized),
        "time_sensitive": time_sensitive,
    }


if __name__ == "__main__":
    tests = [
        "What is the capital of France?",
        "What is the current price of gold in India?",
        "Who was the president of the country where the 2016 Summer Olympics were held?",
        "Which comic series involves characters such as Nick Fury and Baron von Strucker?",
        "In what city did the Prince of tenors star in a film based on an opera by Giacomo Puccini?",
        "What was the capital of India when the Taj Mahal was commissioned?",
    ]
    for q in tests:
        r = complexity_score(q)
        print(f"{r['complexity']:.3f}  ts={r['time_sensitive']}  {q[:60]}")
