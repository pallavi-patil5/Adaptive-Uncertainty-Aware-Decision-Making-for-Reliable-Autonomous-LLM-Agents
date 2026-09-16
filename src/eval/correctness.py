# src/eval/correctness.py
from difflib import SequenceMatcher

def is_correct(predicted_answer: str, acceptable_answers: list[str], ground_truth: str | None, threshold: float = 0.5) -> bool | None:
    """
    Returns True/False if a ground truth exists to check against, None if not applicable.
    Uses substring match first (handles verbose LLM answers containing the short ground truth),
    then token overlap (handles paraphrases), then SequenceMatcher as fallback.
    Threshold lowered to 0.5 to handle verbose answers vs short ground truths.
    """
    if not predicted_answer:
        return False if (ground_truth or acceptable_answers) else None

    candidates = list(acceptable_answers) if acceptable_answers else ([ground_truth] if ground_truth else [])
    if not candidates:
        return None

    predicted_lower = predicted_answer.lower().strip()
    for candidate in candidates:
        if not candidate:
            continue
        candidate_lower = candidate.lower().strip()
        # 1. Exact substring — handles "...the answer is NP-complete..." vs "NP-complete"
        if candidate_lower in predicted_lower:
            return True
        # 2. Reverse substring — predicted is shorter and contained in candidate
        if predicted_lower in candidate_lower:
            return True
        # 3. Token overlap — handles paraphrases and word-order differences
        pred_tokens = set(predicted_lower.split())
        cand_tokens = set(candidate_lower.split())
        if cand_tokens and pred_tokens:
            overlap = len(pred_tokens & cand_tokens) / len(cand_tokens)
            if overlap >= 0.8:  # 80% of candidate tokens present in prediction
                return True
        # 4. SequenceMatcher fallback
        similarity = SequenceMatcher(None, predicted_lower, candidate_lower).ratio()
        if similarity >= threshold:
            return True
    return False


if __name__ == "__main__":
    print(is_correct("Paris.", ["Paris"], "Paris"))                       # True
    print(is_correct("The capital is Paris, France.", ["Paris"], None))    # True
    print(is_correct("Berlin", ["Paris"], "Paris"))                        # False
    print(is_correct(None, [], None))                                     # None