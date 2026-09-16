# src/eval/compute_tier4.py
import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

LOG_PATH = "data/processed/eval_results.jsonl"


def compute_tier4():
    rows = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    by_system = {}
    for row in rows:
        system = row["system"]
        result = row["result"]
        if "error" in result:
            continue
        by_system.setdefault(system, {"llm_calls": [], "retrieval_calls": [], "latency_s": []})
        by_system[system]["llm_calls"].append(result.get("llm_calls", 0))
        by_system[system]["retrieval_calls"].append(result.get("retrieval_calls", 0))
        by_system[system]["latency_s"].append(result.get("latency_s", 0))

    print(f"{'system':<18} {'avg_llm_calls':<15} {'avg_retrieval_calls':<20} {'avg_latency_s':<15}")
    print("-" * 70)
    for system, data in by_system.items():
        avg_llm = sum(data["llm_calls"]) / len(data["llm_calls"])
        avg_ret = sum(data["retrieval_calls"]) / len(data["retrieval_calls"])
        avg_lat = sum(data["latency_s"]) / len(data["latency_s"])
        print(f"{system:<18} {avg_llm:<15.2f} {avg_ret:<20.2f} {avg_lat:<15.3f}  [latency caveat — see note above]")


if __name__ == "__main__":
    compute_tier4()