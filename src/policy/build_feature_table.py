# src/policy/build_feature_table.py

import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from policy.feature_extractor import extract_features


INPUT_PATH = "data/processed/custom_decision_dataset.jsonl"
OUTPUT_PATH = "data/processed/feature_table.csv"


def build_table(path=INPUT_PATH, out_path=OUTPUT_PATH):
    rows = []

    with open(path, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    for i, line in enumerate(lines, 1):
            record = json.loads(line)

            question = record["question"]

            print(
                f"[{i}/{len(lines)}] Extracting features for: "
                f"{question[:60]}..."
            )

            # Extract only observable features.
            features = extract_features(question)

            # Preserve ground-truth information separately.
            features["record_id"] = record["id"]
            features["expected_action"] = record["expected_action"]

            # Preserve question category if available.
            if "category" in record:
                features["category"] = record["category"]

            rows.append(features)

    df = pd.DataFrame(rows)

    # Keep identifiers / labels at the end of the table.
    metadata_columns = [
        col
        for col in ["record_id", "category", "expected_action"]
        if col in df.columns
    ]

    feature_columns = [
        col for col in df.columns
        if col not in metadata_columns
    ]

    df = df[feature_columns + metadata_columns]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"Columns: {len(df.columns)}")

    preview_columns = [
        col
        for col in [
            "record_id",
            "category",
            "expected_action",
            "uncertainty",
            "ambiguity",
            "contradiction_prob",
        ]
        if col in df.columns
    ]

    print("\nFeature table preview:")
    print(df[preview_columns].head(10))


if __name__ == "__main__":
    build_table()