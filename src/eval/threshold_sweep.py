# src/eval/threshold_sweep.py
import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from uncertainty.combine import uncertainty_score
from vector_store import query as vector_query
from llm_client import call_llm
from eval.correctness import is_correct

EVAL_PATH    = "data/processed/eval_sample.jsonl"
CACHE_PATH   = "data/processed/sweep_uncertainty_cache.jsonl"
RESULTS_PATH = "data/processed/threshold_sweep_results.json"
THRESHOLDS   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _load_cache():
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                cache[row["id"]] = row
    return cache


def run_sweep():
    records = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # --- Resume-aware uncertainty caching ---
    cache = _load_cache()
    print(f"Uncertainty cache: {len(cache)}/{len(records)} already done.")

    with open(CACHE_PATH, "a", encoding="utf-8") as cf:
        for record in records:
            if record["id"] in cache:
                continue
            unc = uncertainty_score(record["question"])
            row = {"id": record["id"], "uncertainty": unc["combined_uncertainty"], "candidate_answer": unc["answer"]}
            cf.write(json.dumps(row) + "\n")
            cf.flush()
            cache[record["id"]] = row
            print(f"  cached [{len(cache)}/{len(records)}] {record['question'][:50]}... -> {unc['combined_uncertainty']:.3f}")

    # --- Threshold sweep (fast, no LLM calls) ---
    results = []
    for threshold in THRESHOLDS:
        correct_count, retrieve_count, total = 0, 0, 0
        for record in records:
            item = cache[record["id"]]
            uncertainty, candidate = item["uncertainty"], item["candidate_answer"]
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

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_sweep()
