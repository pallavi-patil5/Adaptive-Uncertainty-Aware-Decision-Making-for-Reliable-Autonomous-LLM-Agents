# src/uncertainty/sanity_test_halueval.py
import sys, os, json, random
sys.path.append(os.path.dirname(__file__))

from sklearn.metrics import roc_auc_score
from self_eval import p_true_score  # cheaper than full self-consistency for a quick sanity pass


def load_halueval_sample(path="data/processed/unified_dataset.jsonl", n=20, seed=42):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["source_dataset"] == "halueval_qa" and row.get("hallucinated_answer") and row.get("ground_truth_answer"):
                records.append(row)
    random.seed(seed)
    return random.sample(records, min(n, len(records)))


def run_sanity_test(n: int = 20):
    sample = load_halueval_sample(n=n)
    if not sample:
        print("No usable HaluEval records found — check unified_dataset.jsonl has hallucinated_answer/ground_truth_answer populated.")
        return

    uncertainties = []
    labels = []  # 1 = hallucinated, 0 = correct

    

    for row in sample:
        question = row["question"]
        knowledge = row.get("knowledge", "") or (row.get("evidence", [{}])[0].get("knowledge", "") if row.get("evidence") else "")

        # Score the CORRECT answer -> expect LOW uncertainty (label 0)
        u_correct = 1.0 - p_true_score(question, row["ground_truth_answer"], knowledge)
        uncertainties.append(u_correct)
        labels.append(0)

        # Score the HALLUCINATED answer -> expect HIGH uncertainty (label 1)
        u_halluc = 1.0 - p_true_score(question, row["hallucinated_answer"], knowledge)
        uncertainties.append(u_halluc)
        labels.append(1)

    auroc = roc_auc_score(labels, uncertainties)

    mean_correct = sum(u for u, l in zip(uncertainties, labels) if l == 0) / labels.count(0)
    mean_halluc = sum(u for u, l in zip(uncertainties, labels) if l == 1) / labels.count(1)

    print(f"Sample size: {len(sample)} question pairs ({len(uncertainties)} total scored answers)")
    print(f"Mean uncertainty — correct answers:     {mean_correct:.3f}")
    print(f"Mean uncertainty — hallucinated answers: {mean_halluc:.3f}")
    print(f"AUROC (uncertainty separating hallucinated vs correct): {auroc:.3f}")
    print()
    if auroc > 0.6:
        print("PASS: uncertainty signal shows meaningful separation. Proceed to Week 3.")
    elif auroc > 0.5:
        print("WEAK SIGNAL: some separation, but check prompts in self_eval.py before proceeding.")
    else:
        print("FAIL: no better than random. Debug self_eval.py / self_consistency.py before Week 3.")


if __name__ == "__main__":
    run_sanity_test(n=20)