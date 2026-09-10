# src/build_unified_schema.py

import json
import os


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


records = []


# ============================================================
# 1. SimpleQA
# ============================================================

for i, row in enumerate(load_jsonl("data/raw/simpleqa.jsonl")):

    answer = row.get("answer", "")

    records.append({
        "id": f"simpleqa_{i}",
        "source_dataset": "simpleqa",

        "question": row.get("problem", ""),
        "ground_truth_answer": answer,
        "acceptable_answers": [answer] if answer else [],

        "expected_action": "answer",

        "category": "known",
        "evaluation_type": "qa",

        "answerable": True,
        "ambiguity": False,

        "evidence_available": False,
        "evidence": [],

        "requires_retrieval": False,
        "requires_multi_hop": False,

        "hallucination_risk": "low",
        "difficulty": "easy"
    })


# ============================================================
# 2. SQuAD 2.0
# ============================================================

for i, row in enumerate(load_jsonl("data/raw/squad2.jsonl")):

    answers = row.get("answers", {}).get("text", [])
    unanswerable = len(answers) == 0

    ground_truth = None if unanswerable else answers[0]

    records.append({
        "id": f"squad2_{i}",
        "source_dataset": "squad2",

        "question": row.get("question", ""),
        "ground_truth_answer": ground_truth,
        "acceptable_answers": answers,

        "expected_action": "abstain" if unanswerable else "answer",

        "category": "unknown" if unanswerable else "known",
        "evaluation_type": "abstention",

        "answerable": not unanswerable,
        "ambiguity": False,

        "evidence_available": True,
        "evidence": [{"context": row.get("context", "")}],

        "requires_retrieval": False,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "high" if unanswerable else "low",  # heuristic label, not measured — do not use as eval ground truth
        "difficulty": "medium"
    })


# ============================================================
# 3. AmbigQA
# ============================================================

for i, row in enumerate(load_jsonl("data/raw/ambigqa.jsonl")):

    annotations = row.get("annotations", {})

    qa_pairs = annotations.get("qaPairs", [])

    # AmbigQA structure can contain nested QA pairs.
    # Count distinct interpretations safely.
    interpretations = []

    if qa_pairs:
        for group in qa_pairs:
            if isinstance(group, list):
                interpretations.extend(group)
            elif isinstance(group, dict):
                interpretations.append(group)

    # Extract answers from interpretations when possible
    distinct_answers = set()

    for item in interpretations:

        if not isinstance(item, dict):
            continue

        answer = item.get("answer")

        if isinstance(answer, str) and answer.strip():
            distinct_answers.add(answer.strip().lower())

        elif isinstance(answer, list):
            for ans in answer:
                if isinstance(ans, str) and ans.strip():
                    distinct_answers.add(ans.strip().lower())

    is_ambiguous = len(distinct_answers) > 1

    records.append({
        "id": f"ambigqa_{i}",
        "source_dataset": "ambigqa",

        "question": row.get("question", ""),
        "ground_truth_answer": None,
        "acceptable_answers": list(distinct_answers),

        "expected_action": "clarify" if is_ambiguous else "answer",

        "category": "ambiguous" if is_ambiguous else "known",
        "evaluation_type": "ambiguity",

        "answerable": True,
        "ambiguity": is_ambiguous,

        "evidence_available": True,
        "evidence": [],

        "requires_retrieval": False,
        "requires_multi_hop": False,

        "hallucination_risk": "medium" if is_ambiguous else "low",
        "difficulty": "hard"
    })


# ============================================================
# 4. HotpotQA
# ============================================================

for i, row in enumerate(load_jsonl("data/raw/hotpotqa.jsonl")):

    answer = row.get("answer", "")

    # HotpotQA supporting facts
    supporting_facts = row.get("supporting_facts", {})

    evidence = []

    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title", [])
        sentences = supporting_facts.get("sent_id", [])

        for title, sent_id in zip(titles, sentences):
            evidence.append({
                "title": title,
                "sentence_id": sent_id
            })

    records.append({
        "id": f"hotpotqa_{i}",
        "source_dataset": "hotpotqa",

        "question": row.get("question", ""),
        "ground_truth_answer": answer,
        "acceptable_answers": [answer] if answer else [],

        "expected_action": "retrieve",

        "category": "retrieval_required",
        "evaluation_type": "multi_hop_retrieval",

        "answerable": True,
        "ambiguity": False,

        "evidence_available": True,
        "evidence": evidence,

        "requires_retrieval": True,
        "requires_multi_hop": True,

        "hallucination_risk": "medium",
        "difficulty": "hard"
    })


# ============================================================
# 5. HaluEval QA
# ============================================================

for i, row in enumerate(load_jsonl("data/raw/halueval_qa.jsonl")):

    question = row.get(
        "question",
        row.get("query", "")
    )

    right_answer = row.get(
        "right_answer",
        row.get("answer", "")
    )

    hallucinated_answer = row.get("hallucinated_answer", "")
    knowledge = row.get("knowledge", "")

    # HaluEval is primarily used for hallucination /
    # verification evaluation rather than forcing
    # every example to have "verify" as the final action.

    records.append({
        "id": f"halueval_{i}",
        "source_dataset": "halueval_qa",

        "question": question,
        "ground_truth_answer": right_answer,
        "acceptable_answers": [right_answer] if right_answer else [],

        "hallucinated_answer": hallucinated_answer,
        "knowledge": knowledge,

        "expected_action": "verify",

        "category": "hallucination_check",
        "evaluation_type": "hallucination",

        "answerable": True,
        "ambiguity": False,

        "evidence_available": True,
        "evidence": [{"knowledge": knowledge}],

        "requires_retrieval": False,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "high",  # heuristic label, not measured — do not use as eval ground truth
        "difficulty": "medium"
    })


# ============================================================
# Save Unified Dataset
# ============================================================

output_path = "data/processed/unified_dataset.jsonl"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:

    for record in records:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


# ============================================================
# Summary
# ============================================================

print("=" * 60)
print("Unified Dataset Created")
print("=" * 60)

print(f"Total records: {len(records)}")
print(f"Output: {output_path}")

print("\nRecords by dataset:")

dataset_counts = {}

for record in records:
    dataset = record["source_dataset"]
    dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

for dataset, count in dataset_counts.items():
    print(f"  {dataset}: {count}")

print("\nRecords by expected action:")

action_counts = {}

for record in records:
    action = record["expected_action"]
    action_counts[action] = action_counts.get(action, 0) + 1

for action, count in action_counts.items():
    print(f"  {action}: {count}")

print("=" * 60)