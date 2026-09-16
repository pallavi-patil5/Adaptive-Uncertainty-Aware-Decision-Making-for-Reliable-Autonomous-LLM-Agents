# src/eval/threshold_sweep.py
import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from uncertainty.combine import uncertainty_score
from vector_store import query as vector_query
from llm_client import call_llm
from eval.correctness import is_correct

EVAL_PATH = "data/processed/eval_sample.jsonl"
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def run_sweep():
    records = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # Compute uncertainty ONCE per question — the expensive part
    cached = []
    for record in records:
        unc = uncertainty_score(record["question"])
        cached.append({"record": record, "uncertainty": unc["combined_uncertainty"], "candidate_answer": unc["answer"]})
        print(f"Cached uncertainty for: {record['question'][:50]}... -> {unc['combined_uncertainty']:.3f}")

    results = []
    for threshold in THRESHOLDS:
        correct_count, retrieve_count, total = 0, 0, 0
        for item in cached:
            record, uncertainty, candidate = item["record"], item["uncertainty"], item["candidate_answer"]
            total += 1

            if uncertainty > threshold:
                retrieve_count += 1
                retrieved = vector_query(record["question"], k=3)
                docs = retrieved["documents"][0] if retrieved["documents"] else []
                context = "\n\n".join(docs) if docs else "No evidence found."
                prompt = f"Context:\n{context}\n\nQuestion: {record['question']}\nAnswer using the context above:"
                final_answer = call_llm(prompt, temperature=0.0)
            else:
                final_answer = candidate

            correct = is_correct(final_answer, record.get("acceptable_answers", []), record.get("ground_truth_answer"))
            if correct:
                correct_count += 1

        accuracy = correct_count / total
        retrieval_rate = retrieve_count / total
        results.append({"threshold": threshold, "accuracy": accuracy, "retrieval_rate": retrieval_rate})
        print(f"threshold={threshold}: accuracy={accuracy:.3f}, retrieval_rate={retrieval_rate:.3f}")

    with open("data/processed/threshold_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_sweep()