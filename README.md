# Adaptive Uncertainty-Aware Decision Making for Reliable Autonomous LLM Agents

A research implementation of an adaptive action-selection policy for autonomous LLM agents. The agent estimates uncertainty and evidence quality at inference time to decide the optimal action — **Answer, Retrieve, Verify, Clarify, or Abstain** — minimising hallucination and unnecessary computational cost.

---

## Research Question

> Can an adaptive uncertainty-aware policy improve the reliability of autonomous LLM agents while minimising unnecessary retrieval, verification, and computational cost?

**Formal decision objective:**

$$A^* = \arg\max_{A} \left[ \text{Expected Reliability}(A) - \lambda \cdot \text{Cost}(A) \right]$$

$$A \in \{\text{Answer},\ \text{Retrieve},\ \text{Verify},\ \text{Clarify},\ \text{Abstain}\}$$

**Contribution:** This work focuses on the *decision layer* — integrating multiple uncertainty and evidence signals into a unified adaptive policy — rather than the underlying detection primitives (which are well-studied).

---

## Architecture

```
USER QUERY
    ↓
Query Analyzer
    ↓
Uncertainty Engine ──┬── Model Confidence  (self-consistency, P(True)/P(IK))
                     ├── Evidence Quality  (embedding similarity + cross-encoder support)
                     ├── Contradiction     (answer–evidence NLI/cross-encoder)
                     └── Ambiguity / Complexity
    ↓
Adaptive Policy  (rule-based → XGBoost refinement)
    ↓
ANSWER / RETRIEVE / VERIFY / CLARIFY / ABSTAIN
    ↓
(if Retrieve) → VERIFY → FINAL ANSWER
    ↓
EVALUATION  (Accuracy · Hallucination · Calibration · Decision Quality · Cost)
```

Agent orchestration is handled by **LangGraph** for explicit state/action graph with conditional branching and loopable Retrieve → Verify → Answer cycles.

---

## Project Structure

```
├── src/
│   ├── signals/          # Uncertainty & evidence signals
│   ├── uncertainty/      # Self-consistency, P(True)/P(IK), entropy
│   ├── policy/           # Decision policy, feature extractor, LangGraph agent
│   ├── baselines/        # 4 baseline agents for comparison
│   ├── eval/             # Evaluation scripts (Tier 1–4, RAGAS, DeepEval, ablation)
│   ├── llm_client.py     # Groq (primary) / Ollama (fallback) wrapper
│   └── vector_store.py   # FAISS / ChromaDB retrieval
├── app/
│   ├── backend/          # FastAPI endpoint
│   ├── frontend/         # Streamlit demo
│   └── cli_demo.py
├── data/
│   ├── raw/              # Source datasets (gitignored)
│   └── processed/        # Eval outputs, trained policy (gitignored)
├── docs/
│   └── methodology_notes.md
├── notebooks/
├── requirements.txt
└── .env.example
```

---

## Datasets

| Dataset | Purpose |
|---|---|
| SimpleQA | Factual QA — accuracy, calibration, Answer/Abstain |
| SQuAD 2.0 | Unanswerable subset — Abstain behaviour |
| AmbigQA | Ambiguous questions — Clarify behaviour |
| HotpotQA | Multi-hop — Retrieve + Verify behaviour |
| HaluEval | Pre-labelled hallucinations — hallucination rate, AUROC |
| Custom Decision Dataset (1,000–2,000 Qs) | Core research contribution — labelled `(query → expected_action)` pairs |

---

## Baselines

| # | Agent | Description |
|---|---|---|
| 1 | Normal LLM | No uncertainty awareness |
| 2 | Standard RAG | Always retrieves |
| 3 | Self-reflection | Generate → Critique → Revise |
| 4 | Fixed-threshold | Single uncertainty cutoff |

All baselines share the same LLM, datasets, and interface for fair comparison.

---

## Evaluation Metrics

- **Tier 1 — Answer quality:** Accuracy, Precision, Recall, F1, hallucination rate, evidence-groundedness
- **Tier 2 — Uncertainty quality:** ECE, Brier score, AUROC (hallucination detection)
- **Tier 3 — Decision quality:** Action accuracy, per-action precision/recall, unnecessary retrieval/verification rate
- **Tier 4 — Efficiency:** Token usage, LLM calls, tool calls, latency

Key experiments: threshold sweep (0.1–0.9) producing a reliability–cost curve, and a signal ablation study (remove one signal at a time).

---

## Setup

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) with `llama3:8b` pulled, Groq API key (optional, for primary LLM).

```bash
git clone https://github.com/<your-username>/EDAI7.git
cd EDAI7
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```
GROQ_API_KEY=<your_groq_api_key>
```

Pull the local model:

```bash
ollama pull llama3:8b
```

---

## Usage

```bash
# Run the adaptive agent over the evaluation set
python src/baselines/run_all.py

# Compute all evaluation tiers
python src/eval/run_evaluation.py

# RAGAS faithfulness + answer relevancy (uses local Ollama)
python src/eval/compute_ragas.py

# Ablation study
python src/eval/ablation_study.py

# Launch the demo
python app/cli_demo.py
# or
uvicorn app.backend.main:app --reload
streamlit run app/frontend/streamlit_app.py
```

---

## Tech Stack

| Component | Library |
|---|---|
| Agent orchestration | LangGraph |
| LLM (primary) | Groq API (`llama3-8b-8192`) |
| LLM (local fallback) | Ollama (`llama3:8b`) |
| Vector store | FAISS / ChromaDB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Policy classifier | XGBoost / scikit-learn |
| RAG evaluation | RAGAS 0.2.x |
| LLM evaluation | DeepEval |
| Backend | FastAPI |
| Frontend | Streamlit |

---

## License

MIT
