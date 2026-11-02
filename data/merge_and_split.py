# src/data/merge_and_split.py
"""
Merges custom_decision_dataset.jsonl + all data/processed/*_decision_dataset.jsonl
files into one unified dataset, validates schema, dedupes by normalized
question text, and writes a stratified (by expected_action) train/val/test
split.

Usage:
    python src/data/merge_and_split.py

Reads (any that exist):
    data/processed/custom_decision_dataset.jsonl
    data/processed/squad2_decision_dataset.jsonl
    data/processed/halueval_decision_dataset.jsonl
    data/processed/ambigqa_decision_dataset.jsonl
    data/processed/hotpotqa_decision_dataset.jsonl
    data/processed/simpleqa_decision_dataset.jsonl

Writes:
    data/processed/full_decision_dataset.jsonl
    data/processed/train.jsonl
    data/processed/val.jsonl
    data/processed/test.jsonl
"""

import glob
import json
import os
import random
from collections import defaultdict

REQUIRED_FIELDS = [
    "id", "source_dataset", "question", "ground_truth_answer",
    "acceptable_answers", "expected_action", "category", "evaluation_type",
    "answerable", "ambiguity", "evidence_available", "evidence",
    "requires_retrieval", "requires_multi_hop",
    "dataset_hallucination_prior", "difficulty",
]

VALID_ACTIONS = {"answer", "retrieve", "verify", "clarify", "abstain"}

DATA_DIR = "data/processed"
SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test


def load_all_records():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_decision_dataset.jsonl")))
    if not files:
        raise FileNotFoundError(
            f"No *_decision_dataset.jsonl files found in {DATA_DIR}/. "
            "Run the builder scripts first."
        )

    all_records = []
    for path in files:
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                all_records.append(json.loads(line))
                count += 1
        print(f"  loaded {count:5d} from {path}")
    return all_records


def validate(record):
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        return f"missing fields: {missing}"
    if record["expected_action"] not in VALID_ACTIONS:
        return f"invalid expected_action: {record['expected_action']}"
    if not record["question"] or not isinstance(record["question"], str):
        return "empty/invalid question"
    return None


def normalize_question(q):
    return " ".join(q.strip().lower().split())


def dedupe(records):
    seen = set()
    deduped = []
    dropped = 0
    for r in records:
        key = normalize_question(r["question"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(r)
    return deduped, dropped


def stratified_split(records, ratios, seed):
    random.seed(seed)
    by_action = defaultdict(list)
    for r in records:
        by_action[r["expected_action"]].append(r)

    train, val, test = [], [], []
    for action, group in by_action.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_distribution(records, label):
    action_counts = defaultdict(int)
    category_counts = defaultdict(int)
    source_counts = defaultdict(int)
    for r in records:
        action_counts[r["expected_action"]] += 1
        category_counts[r["category"]] += 1
        source_counts[r["source_dataset"]] += 1

    print(f"\n--- {label} ({len(records)} examples) ---")
    print("By action:")
    for k, v in sorted(action_counts.items()):
        print(f"  {k:10s}: {v}")
    print("By source_dataset:")
    for k, v in sorted(source_counts.items()):
        print(f"  {k:10s}: {v}")


def main():
    print("Loading all dataset files...")
    all_records = load_all_records()
    print(f"\nTotal raw records loaded: {len(all_records)}")

    print("\nValidating schema...")
    valid_records = []
    invalid_count = 0
    for r in all_records:
        error = validate(r)
        if error:
            invalid_count += 1
            print(f"  DROPPED {r.get('id', '?')}: {error}")
        else:
            valid_records.append(r)
    print(f"Valid: {len(valid_records)} | Dropped for schema errors: {invalid_count}")

    print("\nDeduping by normalized question text...")
    deduped_records, dropped_dupes = dedupe(valid_records)
    print(f"Deduped: {len(deduped_records)} | Dropped as duplicates: {dropped_dupes}")

    full_path = os.path.join(DATA_DIR, "full_decision_dataset.jsonl")
    write_jsonl(deduped_records, full_path)
    print(f"\nWrote merged dataset -> {full_path}")

    print("\nSplitting (stratified by expected_action, 70/15/15)...")
    train, val, test = stratified_split(deduped_records, SPLIT_RATIOS, SEED)

    write_jsonl(train, os.path.join(DATA_DIR, "train.jsonl"))
    write_jsonl(val, os.path.join(DATA_DIR, "val.jsonl"))
    write_jsonl(test, os.path.join(DATA_DIR, "test.jsonl"))

    print_distribution(deduped_records, "FULL DATASET")
    print_distribution(train, "TRAIN")
    print_distribution(val, "VAL")
    print_distribution(test, "TEST")

    print("\nDone. Files written to data/processed/:")
    print("  full_decision_dataset.jsonl, train.jsonl, val.jsonl, test.jsonl")


if __name__ == "__main__":
    main()