# src/policy/build_feature_table.py

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from policy.feature_extractor import extract_features


INPUT_PATH = "data/processed/train.jsonl"
OUTPUT_PATH = "data/processed/train_feature_table.csv"
CHECKPOINT_EVERY = 10


def _write_table(rows, out_path):
    df = pd.DataFrame(rows)
    metadata_columns = [
        column for column in
        ["record_id", "source_dataset", "category", "expected_action"]
        if column in df.columns
    ]
    feature_columns = [column for column in df.columns if column not in metadata_columns]
    df = df[feature_columns + metadata_columns]

    output_dir = os.path.dirname(out_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    temporary_path = f"{out_path}.tmp"
    df.to_csv(temporary_path, index=False)
    os.replace(temporary_path, out_path)
    return df


def build_table(
    path=INPUT_PATH,
    out_path=OUTPUT_PATH,
    limit=None,
    checkpoint_every=CHECKPOINT_EVERY,
    n_consistency_samples=2,
    sleep_between=2,
    cooldown=30,
):
    with open(path, encoding="utf-8") as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]

    if limit is not None:
        records = records[:limit]

    expected_ids = {str(record["id"]) for record in records}
    rows_by_id = {}
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        if "record_id" in existing.columns:
            for row in existing.to_dict(orient="records"):
                record_id = str(row["record_id"])
                if record_id in expected_ids:
                    rows_by_id[record_id] = row

    for index, record in enumerate(records, 1):
        record_id = str(record["id"])
        if record_id in rows_by_id:
            continue

        question = record["question"]
        print(f"[{index}/{len(records)}] Extracting features for: {question[:60]}...", flush=True)
        try:
            features = extract_features(question, n_consistency_samples=n_consistency_samples)
        except Exception as e:
            print(f"  SKIPPED {record_id}: {e}", flush=True)
            time.sleep(sleep_between)
            continue
        features["record_id"] = record_id
        features["source_dataset"] = record.get("source_dataset", "")
        features["expected_action"] = record["expected_action"]
        if "category" in record:
            features["category"] = record["category"]
        rows_by_id[record_id] = features

        time.sleep(sleep_between)  # 2s between every question — lets CPU breathe

        if len(rows_by_id) % checkpoint_every == 0:
            _write_table(list(rows_by_id.values()), out_path)
            print(f"Checkpoint saved: {len(rows_by_id)}/{len(records)} rows — cooling down {cooldown}s...", flush=True)
            time.sleep(cooldown)  # longer cool-down every 10 questions

    df = _write_table(list(rows_by_id.values()), out_path)
    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"Columns: {len(df.columns)}")
    print("Action counts:")
    print(df["expected_action"].value_counts().sort_index())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract model features for a JSONL action dataset.")
    parser.add_argument("--input", default=INPUT_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--consistency-samples", type=int, default=2)
    parser.add_argument("--sleep", type=int, default=2, help="Seconds to sleep between questions")
    parser.add_argument("--cooldown", type=int, default=30, help="Seconds to cool down every N questions")
    args = parser.parse_args()
    build_table(
        path=args.input,
        out_path=args.output,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
        n_consistency_samples=args.consistency_samples,
        sleep_between=args.sleep,
        cooldown=args.cooldown,
    )