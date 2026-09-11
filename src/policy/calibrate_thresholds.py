# src/policy/calibrate_thresholds.py
import sys, os, json, random
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from signals.evidence_retrieval import retrieval_signals
from signals.evidence_support import evidence_support_score

def calibrate(path="data/processed/unified_dataset.jsonl", n=15, seed=42):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("evidence_available"):
                records.append(row)

    random.seed(seed)
    sample = random.sample(records, min(n, len(records)))

    print(f"{'question':<50} {'top1_dist':<12} {'support_score':<15}")
    print("-" * 80)
    for row in sample:
        q = row["question"][:47]
        ret = retrieval_signals(row["question"], k=1)
        top1 = ret["top1_distance"]
        answer = row.get("ground_truth_answer") or "unknown"
        support = None
        if ret["retrieved_docs"]:
            support = evidence_support_score(row["question"], answer, ret["retrieved_docs"][0]["text"])
        print(f"{q:<50} {str(top1):<12} {str(support):<15}")

if __name__ == "__main__":
    calibrate()