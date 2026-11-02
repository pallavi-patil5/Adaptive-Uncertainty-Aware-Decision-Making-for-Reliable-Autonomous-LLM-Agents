# src/data/builders/build_from_hotpotqa.py
"""
Extracts labeled examples from HotpotQA into the unified decision-dataset
schema.

Labeling rule:
    All HotpotQA questions require combining 2+ supporting facts from
    different documents -> expected_action = "retrieve", category =
    "multi_hop", requires_multi_hop = True. Difficulty comes directly from
    HotpotQA's own "level" field (easy/medium/hard).

Usage:
    python src/data/builders/build_from_hotpotqa.py --n 300 --seed 42

Requires:
    pip install datasets
"""

import argparse
import json
import os
import random

from datasets import load_dataset


def make_record(index, question, answer, context_titles, level):
    difficulty = level if level in {"easy", "medium", "hard"} else "medium"
    evidence = [{"source": title} for title in context_titles[:3]]
    return {
        "id": f"hotpotqa_{index:05d}",
        "source_dataset": "hotpotqa",
        "question": question,
        "ground_truth_answer": answer,
        "acceptable_answers": [answer],

        "expected_action": "retrieve",
        "category": "multi_hop",
        "evaluation_type": "multi_hop_retrieval",

        "answerable": True,
        "ambiguity": False,
        "evidence_available": True,
        "evidence": evidence,

        "requires_retrieval": True,
        "requires_multi_hop": True,

        "dataset_hallucination_prior": "medium",
        "difficulty": difficulty,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                         default="data/processed/hotpotqa_decision_dataset.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading HotpotQA (distractor config) from Hugging Face...")
    # NOTE: some environments need trust_remote_code=True for this dataset's
    # loading script. Remove the arg if your `datasets` version errors on it.
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation",
                       trust_remote_code=True)
    print(f"Loaded {len(ds)} raw examples.")

    rows = list(ds)
    if args.n > len(rows):
        raise ValueError(f"Requested {args.n} but pool only has {len(rows)}")
    sampled = random.sample(rows, args.n)

    records = []
    for idx, row in enumerate(sampled, 1):
        question = row["question"].strip()
        answer = row["answer"].strip()
        level = row.get("level", "medium")
        context_titles = row["context"]["title"] if "context" in row else []
        records.append(make_record(idx, question, answer, context_titles, level))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("HotpotQA -> DECISION DATASET")
    print("=" * 60)
    print(f"Total examples written: {len(records)} (all expected_action=retrieve, multi_hop)")
    print(f"Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()