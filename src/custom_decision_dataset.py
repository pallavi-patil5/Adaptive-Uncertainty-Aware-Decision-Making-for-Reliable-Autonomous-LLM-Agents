# src/custom_decision_dataset.py

import json
import os


custom_examples = [

    # ============================================================
    # 1. KNOWN - EASY
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

        "dataset_hallucination_prior": "low",
        "difficulty": "easy"
    },

    {
        "id": "custom_0002",
        "source_dataset": "custom",
        "question": "What is the capital of France?",
        "ground_truth_answer": "Paris",
        "acceptable_answers": ["Paris"],

        "expected_action": "answer",
        "category": "known_easy",
        "evaluation_type": "qa",

        "answerable": True,
        "ambiguity": False,
        "evidence_available": False,
        "evidence": [],

        "requires_retrieval": False,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "low",
        "difficulty": "easy"
    },


    # ============================================================
    # 2. KNOWN BUT DIFFICULT
    # ============================================================

    {
        "id": "custom_0003",
        "source_dataset": "custom",
        "question": "Explain why the Chandrasekhar limit is approximately 1.4 solar masses.",
        "ground_truth_answer": "It is the maximum mass at which electron degeneracy pressure can support a white dwarf against gravitational collapse.",
        "acceptable_answers": [
            "electron degeneracy pressure limits the mass of a white dwarf",
            "maximum mass supported by electron degeneracy pressure"
        ],

        "expected_action": "answer",
        "category": "known_but_difficult",
        "evaluation_type": "qa",

        "answerable": True,
        "ambiguity": False,
        "evidence_available": False,
        "evidence": [],

        "requires_retrieval": False,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "medium",
        "difficulty": "hard"
    },

    {
        "id": "custom_0004",
        "source_dataset": "custom",
        "question": "Why does increasing model depth sometimes lead to vanishing gradients?",
        "ground_truth_answer": "Repeated multiplication by small derivatives during backpropagation can make gradients shrink toward zero.",
        "acceptable_answers": [
            "repeated multiplication of small derivatives causes gradients to vanish"
        ],

        "expected_action": "answer",
        "category": "known_but_difficult",
        "evaluation_type": "qa",

        "answerable": True,
        "ambiguity": False,
        "evidence_available": False,
        "evidence": [],

        "requires_retrieval": False,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "medium",
        "difficulty": "hard"
    },


    # ============================================================
    # 3. RETRIEVAL REQUIRED
    # ============================================================

    {
        "id": "custom_0005",
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
        "evidence": [],

        "requires_retrieval": True,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "high",
        "difficulty": "medium"
    },

    {
        "id": "custom_0006",
        "source_dataset": "custom",
        "question": "What is the current price of gold in India?",
        "ground_truth_answer": None,
        "acceptable_answers": [],

        "expected_action": "retrieve",
        "category": "time_sensitive",
        "evaluation_type": "retrieval",

        "answerable": True,
        "ambiguity": False,
        "evidence_available": True,
        "evidence": [],

        "requires_retrieval": True,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "high",
        "difficulty": "medium"
    },


    # ============================================================
    # 4. CLARIFICATION REQUIRED
    # ============================================================

    {
        "id": "custom_0007",
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

        "dataset_hallucination_prior": "medium",
        "difficulty": "medium"
    },

    {
        "id": "custom_0008",
        "source_dataset": "custom",
        "question": "Tell me about Apple.",
        "ground_truth_answer": None,
        "acceptable_answers": [],

        "expected_action": "clarify",
        "category": "ambiguous",
        "evaluation_type": "ambiguity",

        "answerable": False,
        "ambiguity": True,
        "evidence_available": True,
        "evidence": [],

        "requires_retrieval": False,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "medium",
        "difficulty": "easy"
    },


    # ============================================================
    # 5. ABSTENTION REQUIRED
    # ============================================================

    {
        "id": "custom_0009",
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

        "dataset_hallucination_prior": "high",
        "difficulty": "medium"
    },

    {
        "id": "custom_0010",
        "source_dataset": "custom",
        "question": "What was the exact thought of Albert Einstein immediately before his death?",
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

        "dataset_hallucination_prior": "high",
        "difficulty": "hard"
    },


    # ============================================================
    # 6. VERIFICATION REQUIRED
    # ============================================================

    {
        "id": "custom_0011",
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
        "evidence": [],

        "requires_retrieval": True,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "high",
        "difficulty": "medium"
    },

    {
        "id": "custom_0012",
        "source_dataset": "custom",
        "question": "Is the statement 'the Earth is flat' scientifically supported?",
        "ground_truth_answer": "No",
        "acceptable_answers": ["No", "False"],

        "expected_action": "verify",
        "category": "factual_verification",
        "evaluation_type": "verification",

        "answerable": True,
        "ambiguity": False,
        "evidence_available": True,
        "evidence": [],

        "requires_retrieval": True,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "high",
        "difficulty": "easy"
    },


    # ============================================================
    # 7. CONFLICTING EVIDENCE
    # ============================================================

    {
        "id": "custom_0013",
        "source_dataset": "custom",
        "question": "Two sources give different population figures for the same city and year. Which figure is correct?",
        "ground_truth_answer": None,
        "acceptable_answers": [],

        "expected_action": "verify",
        "category": "conflicting_evidence",
        "evaluation_type": "conflict_resolution",

        "answerable": True,
        "ambiguity": False,
        "evidence_available": True,
        "evidence": [
            {
                "source": "source_A",
                "claim": "Population is 1.2 million"
            },
            {
                "source": "source_B",
                "claim": "Population is 1.5 million"
            }
        ],

        "requires_retrieval": True,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "high",
        "difficulty": "hard"
    },

    {
        "id": "custom_0014",
        "source_dataset": "custom",
        "question": "One document says the company's revenue was $10 million, while another says $12 million for the same financial year. What is the correct figure?",
        "ground_truth_answer": None,
        "acceptable_answers": [],

        "expected_action": "verify",
        "category": "conflicting_evidence",
        "evaluation_type": "conflict_resolution",

        "answerable": True,
        "ambiguity": False,
        "evidence_available": True,
        "evidence": [
            {
                "source": "document_A",
                "claim": "Revenue was $10 million"
            },
            {
                "source": "document_B",
                "claim": "Revenue was $12 million"
            }
        ],

        "requires_retrieval": True,
        "requires_multi_hop": False,

        "dataset_hallucination_prior": "high",
        "difficulty": "hard"
    },


    # ============================================================
    # 8. MULTI-HOP RETRIEVAL
    # ============================================================

    {
        "id": "custom_0015",
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
        "evidence": [],

        "requires_retrieval": True,
        "requires_multi_hop": True,

        "dataset_hallucination_prior": "medium",
        "difficulty": "hard"
    },

    {
        "id": "custom_0016",
        "source_dataset": "custom",
        "question": "Which university did the scientist who discovered penicillin attend?",
        "ground_truth_answer": "University of London",
        "acceptable_answers": ["University of London"],

        "expected_action": "retrieve",
        "category": "multi_hop",
        "evaluation_type": "multi_hop_retrieval",

        "answerable": True,
        "ambiguity": False,
        "evidence_available": True,
        "evidence": [],

        "requires_retrieval": True,
        "requires_multi_hop": True,

        "dataset_hallucination_prior": "medium",
        "difficulty": "hard"
    }
]


# ============================================================
# Save Dataset
# ============================================================

output_path = "data/processed/custom_decision_dataset.jsonl"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for record in custom_examples:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


# ============================================================
# Summary
# ============================================================

print("=" * 60)
print("CUSTOM DECISION DATASET")
print("=" * 60)

print(f"Total seed examples: {len(custom_examples)}")
print(f"Output: {output_path}")

print("\nBy category:")

category_counts = {}

for record in custom_examples:
    category = record["category"]
    category_counts[category] = category_counts.get(category, 0) + 1

for category, count in category_counts.items():
    print(f"  {category}: {count}")

print("\nBy expected action:")

action_counts = {}

for record in custom_examples:
    action = record["expected_action"]
    action_counts[action] = action_counts.get(action, 0) + 1

for action, count in action_counts.items():
    print(f"  {action}: {count}")

print("\nTarget: 1,000-2,000 validated examples")
print("Expand this dataset in parallel with Week 4.")
print("=" * 60)