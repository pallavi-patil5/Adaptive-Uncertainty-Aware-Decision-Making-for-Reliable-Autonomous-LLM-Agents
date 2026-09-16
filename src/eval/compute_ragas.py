# src/eval/compute_ragas.py
# RAGAS evaluation — Faithfulness + Answer Relevancy (ragas 0.2.x API)
# Uses Ollama llama3:8b as judge LLM — no API key needed.

import sys, os, json
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

LOG_PATH = "data/processed/eval_results.jsonl"
OUT_PATH = "data/processed/ragas_results.json"

RETRIEVE_ACTIONS = {"retrieve", "verify"}
OLLAMA_MODEL     = "llama3:8b"


def build_ragas_dataset(rows, system):
    samples = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for row in rows:
        if row["system"] != system:
            continue
        result = row["result"]
        if "error" in result:
            continue
        if result.get("action", "") not in RETRIEVE_ACTIONS:
            continue

        answer = result.get("final_answer") or ""
        raw    = result.get("evidence_used") or result.get("evidence") or result.get("retrieved_docs") or []
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            contexts = [d.get("text", "") for d in raw if d.get("text")]
        else:
            contexts = [str(d) for d in raw if d]

        # fallback for adaptive: evidence is in _cached_features / trace, not logged as a list
        if not contexts:
            feats = result.get("_cached_features", {})
            candidate = feats.get("candidate_answer") or ""
            trace     = result.get("trace") or ""
            if candidate:
                contexts.append(candidate)
            if trace:
                contexts.append(trace)

        if not contexts or not answer:
            continue

        samples["question"].append(row["question"])
        samples["answer"].append(answer)
        samples["contexts"].append(contexts)
        samples["ground_truth"].append(row.get("ground_truth_answer") or "")

    return Dataset.from_dict(samples) if samples["question"] else None


def run_ragas():
    rows = [json.loads(l) for l in open(LOG_PATH, encoding="utf-8")]

    llm        = LangchainLLMWrapper(ChatOllama(model=OLLAMA_MODEL, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    for metric in [faithfulness, answer_relevancy]:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    systems     = ["adaptive", "standard_rag", "self_reflection"]
    all_results = {}

    for system in systems:
        ds = build_ragas_dataset(rows, system)
        if ds is None or len(ds) == 0:
            print(f"{system}: no retrieve/verify rows — skipping")
            continue

        print(f"\n{system}: running RAGAS on {len(ds)} rows...")
        try:
            result = evaluate(
                ds,
                metrics=[faithfulness, answer_relevancy],
                llm=llm,
                embeddings=embeddings,
                run_config=RunConfig(max_retries=3, max_wait=120, timeout=180),
            )
            def _mean(v):
                arr = np.array(v, dtype=float)
                return round(float(np.nanmean(arr)), 3)
            scores = {
                "faithfulness":     _mean(result["faithfulness"]),
                "answer_relevancy": _mean(result["answer_relevancy"]),
                "n":                len(ds),
            }
            all_results[system] = scores
            print(f"  faithfulness:     {scores['faithfulness']}")
            print(f"  answer_relevancy: {scores['answer_relevancy']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[system] = {"error": str(e)}

    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
    return all_results


if __name__ == "__main__":
    run_ragas()
