# src/eval/compute_ragas.py
# RAGAS evaluation — Faithfulness + Answer Relevancy
# Only runs on rows where evidence was retrieved (retrieve / verify actions)
# Reads eval_results.jsonl, no new LLM calls beyond what RAGAS needs internally.

import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

LOG_PATH = "data/processed/eval_results.jsonl"
OUT_PATH = "data/processed/ragas_results.json"

RETRIEVE_ACTIONS = {"retrieve", "verify"}


def build_ragas_dataset(rows, system):
    samples = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for row in rows:
        if row["system"] != system:
            continue
        result = row["result"]
        if "error" in result:
            continue
        action = result.get("action", "")
        if action not in RETRIEVE_ACTIONS:
            continue  # RAGAS faithfulness only meaningful when evidence was used

        question = row["question"]
        answer = result.get("final_answer") or ""
        # evidence stored as list of dicts or list of strings depending on action path
        raw_evidence = result.get("evidence") or result.get("retrieved_docs") or []
        if isinstance(raw_evidence, list) and raw_evidence and isinstance(raw_evidence[0], dict):
            contexts = [d.get("text", "") for d in raw_evidence if d.get("text")]
        elif isinstance(raw_evidence, list):
            contexts = [str(d) for d in raw_evidence if d]
        else:
            contexts = []

        if not contexts or not answer:
            continue

        gt = row.get("ground_truth_answer") or ""
        samples["question"].append(question)
        samples["answer"].append(answer)
        samples["contexts"].append(contexts)
        samples["ground_truth"].append(gt)

    return Dataset.from_dict(samples) if samples["question"] else None


def run_ragas():
    rows = [json.loads(l) for l in open(LOG_PATH, encoding="utf-8")]

    systems = ["adaptive", "standard_rag", "self_reflection"]  # only systems that retrieve
    all_results = {}

    for system in systems:
        ds = build_ragas_dataset(rows, system)
        if ds is None or len(ds) == 0:
            print(f"{system}: no retrieve/verify rows found — skipping RAGAS")
            continue

        print(f"\n{system}: running RAGAS on {len(ds)} retrieve/verify rows...")
        try:
            result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
            scores = {
                "faithfulness":      round(float(result["faithfulness"]), 3),
                "answer_relevancy":  round(float(result["answer_relevancy"]), 3),
                "n":                 len(ds),
            }
            all_results[system] = scores
            print(f"  faithfulness:     {scores['faithfulness']}")
            print(f"  answer_relevancy: {scores['answer_relevancy']}")
            print(f"  n = {scores['n']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[system] = {"error": str(e)}

    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
    return all_results


if __name__ == "__main__":
    run_ragas()
