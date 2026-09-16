# src/eval/compute_deepeval.py
# DeepEval evaluation — Hallucination + Answer Correctness (LLM-as-judge)
# Stronger than substring matching in correctness.py — uses semantic judgment.
# Reads eval_results.jsonl, uses Groq as the judge LLM via deepeval's custom LLM wrapper.

import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric, GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from groq import Groq

LOG_PATH = "data/processed/eval_results.jsonl"
OUT_PATH = "data/processed/deepeval_results.json"

RETRIEVE_ACTIONS = {"retrieve", "verify"}


# ── Groq wrapper so DeepEval uses our existing LLM, not OpenAI ────────────────
class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = "llama3-8b-8192"

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model


# ── Build DeepEval test cases ─────────────────────────────────────────────────
def build_test_cases(rows, system):
    halluc_cases, relevancy_cases, correctness_cases = [], [], []

    for row in rows:
        if row["system"] != system:
            continue
        result = row["result"]
        if "error" in result:
            continue

        question = row["question"]
        answer = result.get("final_answer") or ""
        if not answer:
            continue

        gt = row.get("ground_truth_answer") or ""
        action = result.get("action", "")

        # Evidence context for hallucination metric
        raw_evidence = result.get("evidence") or result.get("retrieved_docs") or []
        if isinstance(raw_evidence, list) and raw_evidence and isinstance(raw_evidence[0], dict):
            contexts = [d.get("text", "") for d in raw_evidence if d.get("text")]
        elif isinstance(raw_evidence, list):
            contexts = [str(d) for d in raw_evidence if d]
        else:
            contexts = []

        base = LLMTestCase(input=question, actual_output=answer,
                           expected_output=gt if gt else None)

        # Hallucination — only for rows that used evidence
        if action in RETRIEVE_ACTIONS and contexts:
            halluc_cases.append(LLMTestCase(
                input=question, actual_output=answer,
                context=contexts,
                expected_output=gt if gt else None,
            ))

        # Answer relevancy — all rows
        relevancy_cases.append(base)

        # Answer correctness (GEval) — only rows with ground truth
        if gt:
            correctness_cases.append(LLMTestCase(
                input=question, actual_output=answer, expected_output=gt
            ))

    return halluc_cases, relevancy_cases, correctness_cases


def run_deepeval():
    rows = [json.loads(l) for l in open(LOG_PATH, encoding="utf-8")]
    judge = GroqJudge()

    halluc_metric     = HallucinationMetric(threshold=0.5, model=judge, include_reason=False)
    relevancy_metric  = AnswerRelevancyMetric(threshold=0.5, model=judge, include_reason=False)
    correctness_metric = GEval(
        name="AnswerCorrectness",
        criteria="Does the actual output correctly answer the question given the expected output as reference?",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=judge,
    )

    systems = ["adaptive", "normal_llm", "standard_rag", "self_reflection", "fixed_threshold"]
    all_results = {}

    for system in systems:
        halluc_cases, relevancy_cases, correctness_cases = build_test_cases(rows, system)
        print(f"\n{system}: halluc={len(halluc_cases)}  relevancy={len(relevancy_cases)}  correctness={len(correctness_cases)}")

        scores = {}

        # Hallucination score (lower = less hallucination)
        if halluc_cases:
            try:
                for tc in halluc_cases:
                    halluc_metric.measure(tc)
                avg_halluc = sum(tc.metrics_metadata[0].score for tc in halluc_cases
                                 if tc.metrics_metadata) / len(halluc_cases)
                scores["hallucination_score"] = round(avg_halluc, 3)
                print(f"  hallucination_score (DeepEval): {scores['hallucination_score']}")
            except Exception as e:
                print(f"  hallucination ERROR: {e}")
                scores["hallucination_score"] = None

        # Answer relevancy
        if relevancy_cases:
            try:
                for tc in relevancy_cases:
                    relevancy_metric.measure(tc)
                avg_rel = sum(tc.metrics_metadata[0].score for tc in relevancy_cases
                              if tc.metrics_metadata) / len(relevancy_cases)
                scores["answer_relevancy"] = round(avg_rel, 3)
                print(f"  answer_relevancy (DeepEval):    {scores['answer_relevancy']}")
            except Exception as e:
                print(f"  relevancy ERROR: {e}")
                scores["answer_relevancy"] = None

        # Answer correctness via GEval
        if correctness_cases:
            try:
                for tc in correctness_cases:
                    correctness_metric.measure(tc)
                avg_corr = sum(tc.metrics_metadata[0].score for tc in correctness_cases
                               if tc.metrics_metadata) / len(correctness_cases)
                scores["answer_correctness_geval"] = round(avg_corr, 3)
                print(f"  answer_correctness GEval:       {scores['answer_correctness_geval']}")
            except Exception as e:
                print(f"  correctness ERROR: {e}")
                scores["answer_correctness_geval"] = None

        scores["n_halluc"]      = len(halluc_cases)
        scores["n_relevancy"]   = len(relevancy_cases)
        scores["n_correctness"] = len(correctness_cases)
        all_results[system] = scores

    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
    return all_results


if __name__ == "__main__":
    run_deepeval()
