# src/data/builders/build_from_squad2.py
"""
Extracts labeled (query -> expected_action) examples from SQuAD 2.0 and
converts them into the project's unified decision-dataset schema (same
fields as src/custom_decision_dataset.py).

Labeling rule (dataset-metadata-derived, no LLM call needed):
    - answerable question (has gold answer + context)      -> expected_action = "retrieve"
      (the model needs the passage to answer; without it, it can't reliably answer)
    - unanswerable question (SQuAD 2.0 flags this)          -> expected_action = "abstain"

Usage:
    python src/data/builders/build_from_squad2.py --n_answerable 150 --n_unanswerable 200 --seed 42

Requires:
    pip install datasets
"""

import argparse
import json
import os
import random

from datasets import load_dataset


def make_record(index, question, answer, context, action, difficulty):
    """Build a record matching src/custom_decision_dataset.py's schema exactly."""
    is_unanswerable = action == "abstain"
    return {
        "id": f"squad2_{index:05d}",
        "source_dataset": "squad2",
        "question": question,
        "ground_truth_answer": None if is_unanswerable else answer,
        "acceptable_answers": [] if is_unanswerable else [answer],

        "expected_action": action,
        "category": "unanswerable" if is_unanswerable else "retrieval_required",
        "evaluation_type": "abstention" if is_unanswerable else "retrieval",

        "answerable": not is_unanswerable,
        "ambiguity": False,
        "evidence_available": not is_unanswerable,
        "evidence": [] if is_unanswerable else [{"source": context[:300]}],

        "requires_retrieval": not is_unanswerable,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "high" if is_unanswerable else "medium",
        "difficulty": difficulty,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_answerable", type=int, default=150,
                         help="How many answerable (expected_action=retrieve) examples to sample")
    parser.add_argument("--n_unanswerable", type=int, default=200,
                         help="How many unanswerable (expected_action=abstain) examples to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                         default="data/processed/squad2_decision_dataset.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading SQuAD 2.0 (validation split) from Hugging Face...")
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    print(f"Loaded {len(ds)} raw examples.")

    answerable_pool = []
    unanswerable_pool = []

    for row in ds:
        question = row["question"].strip()
        context = row["context"].strip()
        answers = row["answers"]["text"]  # list of gold answer strings

        if len(answers) == 0:
            unanswerable_pool.append((question, context))
        else:
            answerable_pool.append((question, context, answers[0]))

    print(f"Answerable pool: {len(answerable_pool)} | Unanswerable pool: {len(unanswerable_pool)}")

    if args.n_answerable > len(answerable_pool):
        raise ValueError(f"Requested {args.n_answerable} answerable examples but pool only has {len(answerable_pool)}")
    if args.n_unanswerable > len(unanswerable_pool):
        raise ValueError(f"Requested {args.n_unanswerable} unanswerable examples but pool only has {len(unanswerable_pool)}")

    sampled_answerable = random.sample(answerable_pool, args.n_answerable)
    sampled_unanswerable = random.sample(unanswerable_pool, args.n_unanswerable)

    records = []
    idx = 1

    for question, context, answer in sampled_answerable:
        # crude difficulty heuristic: longer context = harder to ground -> retrieve is more clearly needed
        difficulty = "medium" if len(context) < 500 else "hard"
        records.append(make_record(idx, question, answer, context, "retrieve", difficulty))
        idx += 1

    for question, context in sampled_unanswerable:
        records.append(make_record(idx, question, None, context, "abstain", "medium"))
        idx += 1

    random.shuffle(records)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("SQuAD 2.0 -> DECISION DATASET")
    print("=" * 60)
    print(f"Total examples written: {len(records)}")
    print(f"  retrieve (answerable): {args.n_answerable}")
    print(f"  abstain (unanswerable): {args.n_unanswerable}")
    print(f"Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()