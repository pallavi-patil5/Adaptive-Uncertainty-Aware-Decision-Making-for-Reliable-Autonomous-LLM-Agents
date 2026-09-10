# src/signals/sanity_test_week3.py
import sys, os, json, random
sys.path.append(os.path.join(os.path.dirname(__file__)))

from ambiguity import ambiguity_score
from complexity import complexity_score
from evidence_retrieval import retrieval_signals


def load_samples(path="data/processed/unified_dataset.jsonl", per_dataset=5, seed=42):
    by_dataset = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_dataset.setdefault(row["source_dataset"], []).append(row)

    random.seed(seed)
    sampled = []
    for name, rows in by_dataset.items():
        sampled.extend(random.sample(rows, min(per_dataset, len(rows))))
    return sampled


def run():
    samples = load_samples()
    print(f"{'dataset':<15} {'expected_action':<10} {'ambiguity':<10} {'complexity':<11} {'top1_dist':<10}")
    print("-" * 65)

    for row in samples:
        q = row["question"]
        amb = ambiguity_score(q)["ambiguity"]
        comp = complexity_score(q)["complexity"]
        ret = retrieval_signals(q, k=3)
        top1 = ret["top1_distance"]
        top1_str = f"{top1:.3f}" if top1 is not None else "N/A"

        print(f"{row['source_dataset']:<15} {row['expected_action']:<10} {amb:<10.3f} {comp:<11.3f} {top1_str:<10}")


if __name__ == "__main__":
    run()