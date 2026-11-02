# src/data/spot_check_sample.py
"""
Samples rows from the full merged decision dataset, stratified by
source_dataset, and writes them to a CSV for quick manual review in Excel.

Usage:
    python src/data/spot_check_sample.py --n_per_source 6 --seed 42
"""

import argparse
import csv
import json
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                         default="data/processed/full_decision_dataset.jsonl")
    parser.add_argument("--output", type=str,
                         default="data/processed/spot_check_sample.csv")
    parser.add_argument("--n_per_source", type=int, default=6,
                         help="How many rows to sample per source_dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    by_source = defaultdict(list)
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_source[row["source_dataset"]].append(row)

    sample = []
    for source, rows in sorted(by_source.items()):
        k = min(args.n_per_source, len(rows))
        sample.extend(random.sample(rows, k))

    random.shuffle(sample)

    fields = ["id", "source_dataset", "question", "expected_action",
              "category", "ground_truth_answer", "evidence_available",
              "difficulty", "review_ok_yn", "review_notes"]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sample:
            evidence_str = "yes" if row.get("evidence_available") else "no"
            writer.writerow({
                "id": row["id"],
                "source_dataset": row["source_dataset"],
                "question": row["question"],
                "expected_action": row["expected_action"],
                "category": row["category"],
                "ground_truth_answer": row.get("ground_truth_answer") or "",
                "evidence_available": evidence_str,
                "difficulty": row.get("difficulty", ""),
                "review_ok_yn": "",     # fill in Y/N while reviewing
                "review_notes": "",     # optional notes on mislabels
            })

    print(f"Wrote {len(sample)} rows to {args.output}")
    print("Open it in Excel, fill in review_ok_yn (Y/N) for each row, "
          "and note anything mislabeled in review_notes.")


if __name__ == "__main__":
    main()