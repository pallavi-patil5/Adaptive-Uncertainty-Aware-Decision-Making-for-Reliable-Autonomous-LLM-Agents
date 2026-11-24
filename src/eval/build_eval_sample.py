# src/eval/build_eval_sample.py
import sys, os, json, random
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def build_sample(
    test_path="data/processed/test.jsonl",
    out_path="data/processed/eval_sample.jsonl",
    seed=42,
):
    sample = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            sample.append(json.loads(line))

    random.seed(seed)
    random.shuffle(sample)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {}
    for row in sample:
        counts[row["source_dataset"]] = counts.get(row["source_dataset"], 0) + 1
    print(f"Test-split evaluation sample: {len(sample)} questions")
    print(f"Breakdown: {counts}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    build_sample()