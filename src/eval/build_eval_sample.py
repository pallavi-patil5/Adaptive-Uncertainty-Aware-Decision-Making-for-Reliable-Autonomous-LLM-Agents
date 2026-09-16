# src/eval/build_eval_sample.py
import sys, os, json, random
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def build_sample(
    unified_path="data/processed/unified_dataset.jsonl",
    custom_path="data/processed/custom_decision_dataset.jsonl",
    out_path="data/processed/eval_sample.jsonl",
    per_source=8,
    seed=42,
):
    by_source = {}
    with open(unified_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_source.setdefault(row["source_dataset"], []).append(row)

    random.seed(seed)
    sample = []
    for source, rows in by_source.items():
        sample.extend(random.sample(rows, min(per_source, len(rows))))

    with open(custom_path, encoding="utf-8") as f:
        for line in f:
            sample.append(json.loads(line))  # include ALL custom examples

    random.shuffle(sample)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Pilot evaluation sample: {len(sample)} questions "
          f"({per_source} per unified source x {len(by_source)} sources + all custom)")
    print(f"Saved to {out_path}")

    counts = {}
    for row in sample:
        counts[row["source_dataset"]] = counts.get(row["source_dataset"], 0) + 1
    print("Breakdown:", counts)


if __name__ == "__main__":
    build_sample()