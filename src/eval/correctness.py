# src/eval/correctness.py
from difflib import SequenceMatcher

def is_correct(predicted_answer: str, acceptable_answers: list[str], ground_truth: str | None, threshold: float = 0.6) -> bool | None:
    """
    Returns True/False if a ground truth exists to check against, None if not applicable
    (e.g. question has no fixed answer — abstain/clarify/open-ended cases).
    """
    if not predicted_answer:
        return False if (ground_truth or acceptable_answers) else None

    candidates = list(acceptable_answers) if acceptable_answers else ([ground_truth] if ground_truth else [])
    if not candidates:
        return None

    predicted_lower = predicted_answer.lower()
    for candidate in candidates:
        if not candidate:
            continue
        candidate_lower = candidate.lower()
        if candidate_lower in predicted_lower:
            return True
        similarity = SequenceMatcher(None, predicted_lower, candidate_lower).ratio()
        if similarity >= threshold:
            return True
    return False


if __name__ == "__main__":
    print(is_correct("Paris.", ["Paris"], "Paris"))                       # True
    print(is_correct("The capital is Paris, France.", ["Paris"], None))    # True
    print(is_correct("Berlin", ["Paris"], "Paris"))                        # False
    print(is_correct(None, [], None))                                     # None