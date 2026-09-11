# src/baselines/common.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm_client import call_llm


class CallCounter:
    """Wraps call_llm to count LLM calls per baseline run — needed for Week 6 efficiency metrics."""
    def __init__(self):
        self.count = 0

    def call(self, prompt: str, **kwargs) -> str:
        self.count += 1
        return call_llm(prompt, **kwargs)


def make_result(action: str, final_answer, trace: str, llm_calls: int, retrieval_calls: int = 0, extra: dict | None = None) -> dict:
    """Standard result shape — every baseline AND the Week 4 adaptive agent should produce this shape
    by Week 6, so the evaluation script can treat all 5 systems identically."""
    result = {
        "action": action,
        "final_answer": final_answer,
        "trace": trace,
        "llm_calls": llm_calls,
        "retrieval_calls": retrieval_calls,
    }
    if extra:
        result.update(extra)
    return result