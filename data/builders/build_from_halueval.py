# src/data/builders/build_from_halueval.py
"""
Extracts labeled examples from HaluEval (QA subset) into the unified
decision-dataset schema.

Labeling rule:
    HaluEval QA rows give: question, knowledge (evidence passage),
    right_answer, hallucinated_answer. Since a candidate answer could be
    either correct or hallucinated, and the only way to tell them apart is
    to check against the knowledge passage -> expected_action = "verify"
    for all of them. This trains the policy to recognize "answer needs
    checking against evidence" situations.

Usage:
    python src/data/builders/build_from_halueval.py --n 200 --seed 42

Requires:
    pip install datasets
"""

import argparse
import json
import os
import random

from datasets import load_dataset


def make_record(index, question, knowledge, right_answer):
    return {
        "id": f"halueval_{index:05d}",
        "source_dataset": "halueval",
        "question": question,
        "ground_truth_answer": right_answer,
        "acceptable_answers": [right_answer],

        "expected_action": "verify",
        "category": "factual_verification",
        "evaluation_type": "verification",

        "answerable": True,
        "ambiguity": False,
        "evidence_available": True,
        "evidence": [{"source": knowledge[:400]}],

        "requires_retrieval": True,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "high",
        "difficulty": "medium",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                         default="data/processed/halueval_decision_dataset.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading HaluEval (qa config) from Hugging Face...")
    # NOTE: verify this repo id still resolves; search huggingface.co/datasets
    # for "halueval" if this fails and swap the id below.
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    print(f"Loaded {len(ds)} raw examples.")

    rows = list(ds)
    if args.n > len(rows):
        raise ValueError(f"Requested {args.n} but pool only has {len(rows)}")
    sampled = random.sample(rows, args.n)

    records = []
    for idx, row in enumerate(sampled, 1):
        question = row["question"].strip()
        knowledge = row["knowledge"].strip()
        right_answer = row["right_answer"].strip()
        records.append(make_record(idx, question, knowledge, right_answer))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("HaluEval -> DECISION DATASET")
    print("=" * 60)
    print(f"Total examples written: {len(records)} (all expected_action=verify)")
    print(f"Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()