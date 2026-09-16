# src/eval/compute_deepeval.py
# DeepEval 4.x evaluation — Hallucination + GEval Correctness
# Uses Ollama llama3:8b as judge LLM — no API key needed.

import sys, os, json, re, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from deepeval.metrics import HallucinationMetric, GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
try:
    from deepeval.test_case import SingleTurnParams as EvalParams
except ImportError:
    from deepeval.test_case import LLMTestCaseParams as EvalParams
import ollama as _ollama

LOG_PATH = "data/processed/eval_results.jsonl"
OUT_PATH = "data/processed/deepeval_results.json"

RETRIEVE_ACTIONS = {"retrieve", "verify"}
OLLAMA_MODEL     = "llama3:8b"


class OllamaJudge(DeepEvalBaseLLM):
    """
    DeepEval calls generate(prompt, schema=SomePydanticClass).
    It then parses the returned STRING as JSON itself via trimAndLoadJson.
    So generate() must always return a JSON string whose keys match the schema fields.
    """
    def load_model(self): return OLLAMA_MODEL

    def generate(self, prompt: str, schema=None) -> str:
        raw = _ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )["message"]["content"]

        if schema is None:
            return raw

        # Get the field names deepeval expects in the JSON response
        fields = list(schema.model_fields.keys()) if hasattr(schema, "model_fields") else []

        # Try to extract a JSON object from the LLM output
        data = _extract_json(raw) or {}

        # Fill any missing fields with sensible defaults
        for k in fields:
            if k not in data:
                if k == "score":   data[k] = 0.0  # explicit 0 so fallback is visible in averages
                elif k == "steps": data[k] = ["Evaluate whether the answer is correct."]
                else:              data[k] = "Unable to determine."

        return json.dumps(data)

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return OLLAMA_MODEL


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m: text = m.group(1)
    else:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m: text = m.group(1)
    try:
        return json.loads(text)
    except Exception:
        return None


def build_test_cases(rows, system):
    halluc_cases, correctness_cases = [], []
    for row in rows:
        if row["system"] != system:
            continue
        result = row["result"]
        if "error" in result:
            continue
        question = row["question"]
        answer   = result.get("final_answer") or ""
        if not answer:
            continue
        gt     = row.get("ground_truth_answer") or ""
        action = result.get("action", "")
        raw    = result.get("evidence_used") or result.get("evidence") or result.get("retrieved_docs") or []
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            contexts = [d.get("text", "") for d in raw if d.get("text")]
        else:
            contexts = [str(d) for d in raw if d]

        if action in RETRIEVE_ACTIONS and contexts:
            halluc_cases.append(LLMTestCase(
                input=question, actual_output=answer, context=contexts,
            ))
        if gt:
            correctness_cases.append(LLMTestCase(
                input=question, actual_output=answer, expected_output=gt
            ))
    return halluc_cases, correctness_cases


def run_deepeval():
    rows  = [json.loads(l) for l in open(LOG_PATH, encoding="utf-8")]
    judge = OllamaJudge()

    halluc_metric = HallucinationMetric(threshold=0.5, model=judge, include_reason=False)
    correctness_metric = GEval(
        name="AnswerCorrectness",
        criteria="Does the actual output correctly answer the question given the expected output as reference?",
        evaluation_params=[EvalParams.INPUT, EvalParams.ACTUAL_OUTPUT, EvalParams.EXPECTED_OUTPUT],
        model=judge,
        strict_mode=False,
        async_mode=False,
    )

    systems     = ["adaptive", "normal_llm", "standard_rag", "self_reflection", "fixed_threshold"]
    all_results = {}

    for system in systems:
        halluc_cases, correctness_cases = build_test_cases(rows, system)
        print(f"\n{system}: halluc={len(halluc_cases)}  correctness={len(correctness_cases)}")
        scores = {}

        if halluc_cases:
            halluc_scores, failed = [], 0
            for tc in halluc_cases:
                try:
                    halluc_metric.measure(tc)
                    s = getattr(halluc_metric, "score", None)
                    if s is not None:
                        halluc_scores.append(float(s))
                except Exception:
                    failed += 1
            scores["hallucination_score"] = round(sum(halluc_scores)/len(halluc_scores), 3) if halluc_scores else None
            print(f"  hallucination_score: {scores['hallucination_score']}  (n={len(halluc_scores)} failed={failed})")

        if correctness_cases:
            corr_scores, failed = [], 0
            for i, tc in enumerate(correctness_cases):
                try:
                    correctness_metric.measure(tc)
                    s = getattr(correctness_metric, "score", None)
                    if s is not None:
                        corr_scores.append(float(s))
                        # Spot-check first 3 cases so you can eyeball judge quality
                        if i < 3:
                            print(f"  [spot-check {i}] q={tc.input[:60]!r}")
                            print(f"    answer={tc.actual_output[:80]!r}")
                            print(f"    expected={tc.expected_output[:80]!r}")
                            print(f"    score={s:.3f}  reason={getattr(correctness_metric,'reason','n/a')}")
                except Exception as e:
                    failed += 1
                    if i < 3: print(f"  [spot-check {i}] EXCEPTION: {e}")
            scores["answer_correctness_geval"] = round(sum(corr_scores)/len(corr_scores), 3) if corr_scores else None
            print(f"  geval_correctness:   {scores['answer_correctness_geval']}  (n={len(corr_scores)} failed={failed})")

        scores.update({"n_halluc": len(halluc_cases), "n_correctness": len(correctness_cases)})
        all_results[system] = scores

    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
    return all_results


if __name__ == "__main__":
    run_deepeval()
