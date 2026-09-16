# src/eval/run_evaluation.py
import sys, os, json, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from policy.agent_graph import build_graph
from policy.feature_extractor import extract_features
from baselines import normal_llm, standard_rag, self_reflection, fixed_threshold

EVAL_PATH = "data/processed/eval_sample.jsonl"
LOG_PATH = "data/processed/eval_results.jsonl"

SYSTEMS = ["adaptive", "normal_llm", "standard_rag", "self_reflection", "fixed_threshold"]


def load_completed(log_path):
    """Returns set of (record_id, system) pairs already logged, for resume support."""
    done = set()
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done.add((row["record_id"], row["system"]))
    return done


def run_system(system: str, question: str, app=None) -> dict:
    if system == "adaptive":
        state = app.invoke({"question": question, "features": None, "decision": None,
                            "result": None, "evidence": None,
                            "llm_calls": 0, "retrieval_calls": 0, "start_time": time.monotonic()})
        result = state["result"]
        result["decision_action"] = state["decision"]["action"]
        result["_cached_features"] = state["features"]  # reuse in Steps 8-10, strip before final metrics
        return result
    elif system == "normal_llm":
        return normal_llm.run(question)
    elif system == "standard_rag":
        return standard_rag.run(question)
    elif system == "self_reflection":
        return self_reflection.run(question)
    elif system == "fixed_threshold":
        return fixed_threshold.run(question, threshold=0.5)  # default threshold; sweep happens in Step 8
    else:
        raise ValueError(f"Unknown system: {system}")


def run_evaluation():
    app = build_graph()
    done = load_completed(LOG_PATH)

    records = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    total_pairs = len(records) * len(SYSTEMS)
    completed_count = len(done)
    print(f"Total (question, system) pairs: {total_pairs}. Already completed: {completed_count}.")

    with open(LOG_PATH, "a", encoding="utf-8") as log_f:
        for record in records:
            for system in SYSTEMS:
                key = (record["id"], system)
                if key in done:
                    continue

                print(f"[{system}] {record['id']}: {record['question'][:60]}...")
                try:
                    result = run_system(system, record["question"], app=app)
                except Exception as e:
                    print(f"  ERROR: {e} — logging failure, continuing.")
                    result = {"error": str(e)}

                log_row = {
                    "record_id": record["id"],
                    "source_dataset": record["source_dataset"],
                    "system": system,
                    "question": record["question"],
                    "expected_action": record.get("expected_action"),
                    "ground_truth_answer": record.get("ground_truth_answer"),
                    "acceptable_answers": record.get("acceptable_answers", []),
                    "result": result,
                }
                log_f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
                log_f.flush()  # ensure it's on disk immediately, not just buffered

    print("Evaluation run complete.")


if __name__ == "__main__":
    run_evaluation()