# src/custom_decision_dataset.py

import json
import os


custom_examples = [

    # ============================================================
    # 1. Known / Direct Answer
    # ============================================================

    {
        "id": "custom_0001",
        "source_dataset": "custom",
        "question": "What is 15 * 23?",
        "ground_truth_answer": "345",
        "acceptable_answers": ["345"],

        "expected_action": "answer",
        "category": "known_easy",
        "evaluation_type": "qa",

        "answerable": True,
        "ambiguity": False,

        "evidence_available": False,
        "evidence": [],

        "requires_retrieval": False,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "low",  # heuristic label, not measured — do not use as eval ground truth
        "difficulty": "easy"
    },


    # ============================================================
    # 2. Time-Sensitive / Retrieval Required
    # ============================================================

    {
        "id": "custom_0002",
        "source_dataset": "custom",
        "question": "Who won the Nobel Prize in Physics in 2025?",
        "ground_truth_answer": None,
        "acceptable_answers": [],

        "expected_action": "retrieve",
        "category": "time_sensitive",
        "evaluation_type": "retrieval",

        "answerable": True,
        "ambiguity": False,

        "evidence_available": True,
        "evidence": [{"source": "News report on 2025 Nobel Physics Prize announcement"}],

        "requires_retrieval": True,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "high",  # heuristic label, not measured — do not use as eval ground truth
        "difficulty": "medium"
    },


    # ============================================================
    # 3. Ambiguous / Clarification Required
    # ============================================================

    {
        "id": "custom_0003",
        "source_dataset": "custom",
        "question": "What does 'the meeting' refer to, and when is it?",
        "ground_truth_answer": None,
        "acceptable_answers": [],

        "expected_action": "clarify",
        "category": "ambiguous",
        "evaluation_type": "ambiguity",

        "answerable": False,
        "ambiguity": True,

        "evidence_available": False,
        "evidence": [],

        "requires_retrieval": False,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "medium",  # heuristic label, not measured — do not use as eval ground truth
        "difficulty": "medium"
    },


    # ============================================================
    # 4. Unanswerable / Abstention Required
    # ============================================================

    {
        "id": "custom_0004",
        "source_dataset": "custom",
        "question": "What was the exact color of the shirt worn by a random person in Pune yesterday?",
        "ground_truth_answer": None,
        "acceptable_answers": [],

        "expected_action": "abstain",
        "category": "unanswerable",
        "evaluation_type": "abstention",

        "answerable": False,
        "ambiguity": False,

        "evidence_available": False,
        "evidence": [],

        "requires_retrieval": False,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "high",  # heuristic label, not measured — do not use as eval ground truth
        "difficulty": "medium"
    },


    # ============================================================
    # 5. Verification Required
    # ============================================================

    {
        "id": "custom_0005",
        "source_dataset": "custom",
        "question": "Is the claim that humans can survive indefinitely without water true?",
        "ground_truth_answer": "No",
        "acceptable_answers": ["No", "False"],

        "expected_action": "verify",
        "category": "factual_verification",
        "evaluation_type": "verification",

        "answerable": True,
        "ambiguity": False,

        "evidence_available": True,
        "evidence": [{"source": "Medical/physiological fact: humans typically survive 3-4 days without water"}],

        "requires_retrieval": True,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "high",  # heuristic label, not measured — do not use as eval ground truth
        "difficulty": "medium"
    },


    # ============================================================
    # 6. Multi-Hop Retrieval
    # ============================================================

    {
        "id": "custom_0006",
        "source_dataset": "custom",
        "question": "Who was the president of the country where the 2016 Summer Olympics were held?",
        "ground_truth_answer": "Michel Temer",
        "acceptable_answers": ["Michel Temer"],

        "expected_action": "retrieve",
        "category": "multi_hop",
        "evaluation_type": "multi_hop_retrieval",

        "answerable": True,
        "ambiguity": False,

        "evidence_available": True,
        "evidence": [{"source": "2016 Summer Olympics held in Rio de Janeiro, Brazil; Michel Temer was president at the time"}],

        "requires_retrieval": True,
        "requires_multi_hop": True,

        "dataset_hallucination_prior": "medium",  # heuristic label, not measured — do not use as eval ground truth
        "difficulty": "hard"
    }

    # ============================================================
    # NOTE — category coverage gap (flagged, not yet filled):
    # The 8-category table from the project plan also includes:
    #   - "known_but_difficult"  -> expected_action "answer" (with verify follow-up)
    #   - "conflicting_evidence" -> expected_action "verify" (two contradicting sources,
    #     distinct from single-source factual_verification above)
    # Add seed examples for these two categories before scaling up authoring to
    # 1,000-2,000 examples, and keep category proportions roughly even so Week 4
    # policy training and Week 6 per-action precision/recall aren't skewed.
    # ============================================================
]


# ============================================================
# Save Dataset
# ============================================================

output_path = "data/processed/custom_decision_dataset.jsonl"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:

    for record in custom_examples:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


print("=" * 60)
print("Custom Decision Dataset")
print("=" * 60)
print(f"Seed examples: {len(custom_examples)}")
print(f"Output: {output_path}")

print("\nExpected actions:")

action_counts = {}

for record in custom_examples:
    action = record["expected_action"]
    action_counts[action] = action_counts.get(action, 0) + 1

for action, count in action_counts.items():
    print(f"  {action}: {count}")

print("\nTarget: 1,000-2,000 validated examples")
print("=" * 60)