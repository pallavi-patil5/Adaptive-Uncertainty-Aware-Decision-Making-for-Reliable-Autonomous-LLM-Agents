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
        "evidence": [{"source": "News report on 2025 Nobel Physics Prize announcement"}],

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
        "evidence": [{"source": "Live commodity price feed for gold rates in India"}],

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
        "evidence": [{"source": "Medical/physiological fact: humans typically survive only 3-4 days without water"}],

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
        "evidence": [{"source": "Scientific consensus: Earth's spherical shape is confirmed by satellite imagery, gravity measurements, and centuries of physical evidence"}],

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
        "evidence": [{"source": "2016 Summer Olympics held in Rio de Janeiro, Brazil; Michel Temer was president at the time"}],

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
        "evidence": [{"source": "Alexander Fleming discovered penicillin; he studied medicine at St Mary's Hospital Medical School, part of the University of London"}],

        "requires_retrieval": True,
        "requires_multi_hop": True,

        "dataset_hallucination_prior": "medium",
        "difficulty": "hard"
    }
]


def make_example(index, question, answer, action, category, evaluation_type, difficulty):
    """Build a consistently labelled custom record from a curated example."""
    answerable = action not in {"clarify", "abstain"}
    requires_retrieval = action in {"retrieve", "verify"}
    evidence = []
    if requires_retrieval:
        evidence = [{"source": "External or retrieved evidence required for this example"}]

    return {
        "id": f"custom_{index:04d}",
        "source_dataset": "custom",
        "question": question,
        "ground_truth_answer": answer,
        "acceptable_answers": [answer] if answer is not None else [],
        "expected_action": action,
        "category": category,
        "evaluation_type": evaluation_type,
        "answerable": answerable,
        "ambiguity": action == "clarify",
        "evidence_available": requires_retrieval,
        "evidence": evidence,
        "requires_retrieval": requires_retrieval,
        "requires_multi_hop": category == "multi_hop",
        "dataset_hallucination_prior": (
            "high" if action in {"retrieve", "verify", "abstain"}
            else "medium" if difficulty == "hard" else "low"
        ),
        "difficulty": difficulty,
    }


additional_examples = [
    # Direct questions with stable, independently checkable answers.
    ("What is the chemical formula for water?", "H2O", "known_easy", "qa", "easy", "answer"),
    ("What is the square root of 144?", "12", "known_easy", "qa", "easy", "answer"),
    ("Which planet is known as the Red Planet?", "Mars", "known_easy", "qa", "easy", "answer"),
    ("Who wrote Pride and Prejudice?", "Jane Austen", "known_easy", "qa", "easy", "answer"),
    ("What is the SI unit of force?", "newton", "known_easy", "qa", "easy", "answer"),
    ("What is 3/4 + 1/8?", "7/8", "known_easy", "qa", "easy", "answer"),
    ("What is the largest planet in the Solar System?", "Jupiter", "known_easy", "qa", "easy", "answer"),
    ("What is the decimal value of the binary number 1010?", "10", "known_easy", "qa", "easy", "answer"),
    ("At standard atmospheric pressure, what is the boiling point of water in Celsius?", "100 degrees Celsius", "known_easy", "qa", "easy", "answer"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "known_easy", "qa", "easy", "answer"),
    ("What gas do plants primarily absorb from the atmosphere during photosynthesis?", "Carbon dioxide", "known_easy", "qa", "easy", "answer"),
    ("Approximately how fast does light travel in a vacuum?", "299,792,458 meters per second", "known_but_difficult", "qa", "medium", "answer"),

    # Current or multi-hop facts need an external source or evidence lookup.
    ("What is the current USD-to-INR exchange rate?", None, "time_sensitive", "retrieval", "medium", "retrieve"),
    ("What is the weather in Mumbai right now?", None, "time_sensitive", "retrieval", "easy", "retrieve"),
    ("What is the latest released version of Python?", None, "time_sensitive", "retrieval", "medium", "retrieve"),
    ("What is the current market price of Nvidia stock?", None, "time_sensitive", "retrieval", "medium", "retrieve"),
    ("What is the most recent official estimate of India's population?", None, "time_sensitive", "retrieval", "medium", "retrieve"),
    ("Who currently holds the office of Prime Minister of the United Kingdom?", None, "time_sensitive", "retrieval", "medium", "retrieve"),
    ("What is today's air quality index in Delhi?", None, "time_sensitive", "retrieval", "easy", "retrieve"),
    ("What is the latest official unemployment rate reported for the United States?", None, "time_sensitive", "retrieval", "medium", "retrieve"),
    ("Which country hosted the 2014 Winter Olympics, and what is that country's capital?", "Russia; Moscow", "multi_hop", "multi_hop_retrieval", "hard", "retrieve"),
    ("What is the capital of the country where the 2010 FIFA World Cup final was played?", "Pretoria (executive capital of South Africa)", "multi_hop", "multi_hop_retrieval", "hard", "retrieve"),
    ("Which element is named after the scientist who created the first widely recognized periodic table?", "Mendelevium", "multi_hop", "multi_hop_retrieval", "hard", "retrieve"),
    ("What is the capital of the country that contains the city of Kyoto?", "Tokyo", "multi_hop", "multi_hop_retrieval", "medium", "retrieve"),

    # Explicit factual claims should be checked against reliable evidence.
    ("Is it true that Earth's natural satellite is the Moon?", "True", "factual_verification", "verification", "easy", "verify"),
    ("Is it true that Venus is closer to the Sun than Earth is?", "True", "factual_verification", "verification", "easy", "verify"),
    ("Is it true that sound can travel through a vacuum?", "False", "factual_verification", "verification", "easy", "verify"),
    ("Is it true that water molecules contain hydrogen and oxygen atoms?", "True", "factual_verification", "verification", "easy", "verify"),
    ("Is it true that the Great Wall of China is readily visible to the unaided eye from the Moon?", "False", "factual_verification", "verification", "medium", "verify"),
    ("Is it true that bats are blind?", "False", "factual_verification", "verification", "easy", "verify"),
    ("Is it true that lightning can strike the same place more than once?", "True", "factual_verification", "verification", "easy", "verify"),
    ("Is it true that Earth's seasons are caused primarily by its changing distance from the Sun?", "False", "factual_verification", "verification", "medium", "verify"),
    ("Is it true that antibiotics treat infections caused by viruses?", "False", "factual_verification", "verification", "easy", "verify"),
    ("Is it true that every metal is strongly attracted to a magnet?", "False", "factual_verification", "verification", "medium", "verify"),
    ("Is it true that whales are mammals rather than fish?", "True", "factual_verification", "verification", "easy", "verify"),
    ("Is it true that the Earth completes one orbit around the Sun in about one year?", "True", "factual_verification", "verification", "easy", "verify"),

    # Missing referents or criteria make clarification the useful next action.
    ("Can you make it shorter?", None, "ambiguous", "ambiguity", "easy", "clarify"),
    ("How much does it cost?", None, "ambiguous", "ambiguity", "easy", "clarify"),
    ("When does it start?", None, "ambiguous", "ambiguity", "easy", "clarify"),
    ("Is it safe to use?", None, "ambiguous", "ambiguity", "easy", "clarify"),
    ("Which one is better?", None, "ambiguous", "ambiguity", "easy", "clarify"),
    ("Can you send it to them?", None, "ambiguous", "ambiguity", "easy", "clarify"),
    ("What happened there?", None, "ambiguous", "ambiguity", "easy", "clarify"),
    ("Should I choose the first option?", None, "ambiguous", "ambiguity", "easy", "clarify"),
    ("How far away is the place you mentioned?", None, "ambiguous", "ambiguity", "easy", "clarify"),
    ("What is the best programming language?", None, "ambiguous", "ambiguity", "medium", "clarify"),
    ("Can you compare these two?", None, "ambiguous", "ambiguity", "easy", "clarify"),
    ("What does that mean?", None, "ambiguous", "ambiguity", "easy", "clarify"),

    # These ask for private, unrecorded, or inherently unknowable specifics.
    ("What is my neighbor's email password?", None, "unanswerable", "abstention", "medium", "abstain"),
    ("What exact words did my colleague say in a private conversation I did not record?", None, "unanswerable", "abstention", "medium", "abstain"),
    ("What number will win the next national lottery draw?", None, "unanswerable", "abstention", "medium", "abstain"),
    ("What is the exact temperature inside an uninstrumented room in a private home right now?", None, "unanswerable", "abstention", "medium", "abstain"),
    ("What is the precise location of a random person who is not sharing their location?", None, "unanswerable", "abstention", "medium", "abstain"),
    ("What did a stranger I passed yesterday have for breakfast?", None, "unanswerable", "abstention", "medium", "abstain"),
    ("How many grains of sand are currently inside my closed shoe?", None, "unanswerable", "abstention", "medium", "abstain"),
    ("What is the exact birthday of a person I have not identified?", None, "unanswerable", "abstention", "medium", "abstain"),
    ("What is the private PIN for someone else's bank card?", None, "unanswerable", "abstention", "medium", "abstain"),
    ("What is the exact thought a specific person is having right now?", None, "unanswerable", "abstention", "medium", "abstain"),
    ("Why did an unidentified stranger stop replying to a message?", None, "unanswerable", "abstention", "medium", "abstain"),
    ("How many people attended an unrecorded private gathering last night?", None, "unanswerable", "abstention", "medium", "abstain"),
]

for index, (question, answer, category, evaluation_type, difficulty, action) in enumerate(additional_examples, 17):
    custom_examples.append(
        make_example(index, question, answer, action, category, evaluation_type, difficulty)
    )


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