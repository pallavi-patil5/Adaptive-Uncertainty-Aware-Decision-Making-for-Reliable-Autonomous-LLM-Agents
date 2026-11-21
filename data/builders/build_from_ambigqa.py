# src/data/builders/build_from_ambigqa.py
"""
Extracts labeled examples from AmbigQA into the unified decision-dataset
schema.

Labeling rule:
    AmbigQA annotations have type "multipleQAs" (question is genuinely
    ambiguous, multiple valid interpretations/answers) or "singleAnswer"
    (not ambiguous).
        multipleQAs -> expected_action = "clarify"
        singleAnswer -> expected_action = "retrieve"
            (these are open-domain NQ-style questions; still need evidence
            lookup, they're just not ambiguous)

Usage:
    python src/data/builders/build_from_ambigqa.py --n_ambiguous 200 --n_single 150 --seed 42

Requires:
    pip install datasets
"""

import argparse
import json
import os
import random

from datasets import load_dataset


def make_record(index, question, action, answer=None):
    is_clarify = action == "clarify"
    return {
        "id": f"ambigqa_{index:05d}",
        "source_dataset": "ambigqa",
        "question": question,
        "ground_truth_answer": None if is_clarify else answer,
        "acceptable_answers": [] if is_clarify else ([answer] if answer else []),

        "expected_action": action,
        "category": "ambiguous" if is_clarify else "retrieval_required",
        "evaluation_type": "ambiguity" if is_clarify else "retrieval",

        "answerable": not is_clarify,
        "ambiguity": is_clarify,
        "evidence_available": not is_clarify,
        "evidence": [],

        "requires_retrieval": not is_clarify,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "medium",
        "difficulty": "medium",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ambiguous", type=int, default=200)
    parser.add_argument("--n_single", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                         default="data/processed/ambigqa_decision_dataset.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading AmbigQA (light config) from Hugging Face...")
    # NOTE: verify this repo id/config still resolves; search
    # huggingface.co/datasets for "ambig_qa" if this fails.
    ds = load_dataset("sewon/ambig_qa", "light", split="train")
    print(f"Loaded {len(ds)} raw examples.")

    ambiguous_pool = []
    single_pool = []

    for row in ds:
        question = row["question"].strip()
        annotations = row["annotations"]
        ann_type = annotations["type"][0] if len(annotations["type"]) > 0 else None

        if ann_type == "multipleQAs":
            ambiguous_pool.append(question)
        elif ann_type == "singleAnswer":
            # answer text is nested; take first available answer string
            answers = annotations.get("answer", [[]])
            flat_answer = None
            if len(answers) > 0 and len(answers[0]) > 0:
                flat_answer = answers[0][0]
            single_pool.append((question, flat_answer))

    print(f"Ambiguous pool: {len(ambiguous_pool)} | Single-answer pool: {len(single_pool)}")

    if args.n_ambiguous > len(ambiguous_pool):
        raise ValueError(f"Requested {args.n_ambiguous} ambiguous but pool only has {len(ambiguous_pool)}")
    if args.n_single > len(single_pool):
        raise ValueError(f"Requested {args.n_single} single-answer but pool only has {len(single_pool)}")

    sampled_ambiguous = random.sample(ambiguous_pool, args.n_ambiguous)
    sampled_single = random.sample(single_pool, args.n_single)

    records = []
    idx = 1
    for question in sampled_ambiguous:
        records.append(make_record(idx, question, "clarify"))
        idx += 1
    for question, answer in sampled_single:
        records.append(make_record(idx, question, "retrieve", answer))
        idx += 1

    random.shuffle(records)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("AmbigQA -> DECISION DATASET")
    print("=" * 60)
    print(f"Total examples written: {len(records)}")
    print(f"  clarify: {args.n_ambiguous}")
    print(f"  retrieve: {args.n_single}")
    print(f"Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()