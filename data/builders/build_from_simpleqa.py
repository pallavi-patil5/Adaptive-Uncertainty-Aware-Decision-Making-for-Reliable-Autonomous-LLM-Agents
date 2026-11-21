# src/data/builders/build_from_simpleqa.py
"""
Extracts labeled examples from SimpleQA into the unified decision-dataset
schema.

Labeling rule (HEURISTIC — see caveat below):
    SimpleQA questions are short, single-fact-answer trivia questions.
    Default heuristic label: expected_action = "answer" for all of them,
    since they're presented as directly answerable factual lookups.

    CAVEAT: SimpleQA was specifically designed by OpenAI to be *hard* —
    many of these questions actually cause models to hallucinate or get
    wrong. A pure "always answer" label is a simplification and will teach
    your policy that all short factual questions = safe to answer, which
    isn't strictly true. The correct-but-more-expensive fix is to run each
    question through your actual base LLM (Groq), check correctness against
    ground_truth, and only keep expected_action="answer" for ones it gets
    right — relabeling the rest as "abstain" or "verify". See the stub
    function `refine_labels_with_model()` at the bottom: fill in your Groq
    call there if/when you have time (optional Week 4 enhancement, not
    required to hit your 1,000-2,000 target).

Usage:
    python src/data/builders/build_from_simpleqa.py --n 250 --seed 42

Requires:
    pip install datasets
"""

import argparse
import json
import os
import random

from datasets import load_dataset


def make_record(index, question, answer, difficulty):
    return {
        "id": f"simpleqa_{index:05d}",
        "source_dataset": "simpleqa",
        "question": question,
        "ground_truth_answer": answer,
        "acceptable_answers": [answer],

        "expected_action": "answer",
        "category": "known_easy" if difficulty == "easy" else "known_but_difficult",
        "evaluation_type": "qa",

        "answerable": True,
        "ambiguity": False,
        "evidence_available": False,
        "evidence": [],

        "requires_retrieval": False,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "medium",
        "difficulty": difficulty,
    }


def refine_labels_with_model(records):
    """
    OPTIONAL enhancement — not required for initial dataset build.

    Fill this in later: for each record, call your base LLM (Groq) zero-shot,
    compare its answer to ground_truth_answer (string/semantic match), and
    relabel:
        - correct + high self-consistency -> keep expected_action = "answer"
        - wrong or inconsistent           -> relabel to "abstain" (or
          "verify" if you want it routed through a check instead)

    Returns the same list with expected_action possibly updated.
    """
    raise NotImplementedError(
        "Fill in your Groq API call here if you want model-verified labels "
        "instead of the default heuristic 'always answer' labels."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                         default="data/processed/simpleqa_decision_dataset.jsonl")
    parser.add_argument("--refine_with_model", action="store_true",
                         help="If set, calls refine_labels_with_model() (must be implemented first)")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading SimpleQA from Hugging Face...")
    # NOTE: verify this repo id still resolves; search
    # huggingface.co/datasets for "simpleqa" if this fails and swap it in.
    ds = load_dataset("basicv8vc/SimpleQA", split="test")
    print(f"Loaded {len(ds)} raw examples.")

    rows = list(ds)
    if args.n > len(rows):
        raise ValueError(f"Requested {args.n} but pool only has {len(rows)}")
    sampled = random.sample(rows, args.n)

    records = []
    for idx, row in enumerate(sampled, 1):
        question = row["problem"].strip()
        answer = row["answer"].strip()
        # crude difficulty heuristic from question length; refine later if needed
        difficulty = "easy" if len(question) < 80 else "hard"
        records.append(make_record(idx, question, answer, difficulty))

    if args.refine_with_model:
        records = refine_labels_with_model(records)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("SimpleQA -> DECISION DATASET")
    print("=" * 60)
    print(f"Total examples written: {len(records)} (all expected_action=answer, heuristic)")
    print(f"Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()