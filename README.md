# Adaptive Uncertainty-Aware Decision Making for Reliable Autonomous LLM Agents
### Final Consolidated Project Plan (2-Month Timeline)

---

## 1. Feasibility

Fully buildable at zero monetary cost. Free-tier constraints (API rate limits, capped GPU hours) affect speed/scale, not feasibility — mitigated by using **Groq (free tier) as primary LLM, Ollama as local backup** for whenever you hit rate limits.

---

## 2. Core Research Framing

**Research question:**
Can an adaptive uncertainty-aware policy improve the reliability of autonomous LLM agents while minimizing unnecessary retrieval, verification, and computational cost?

**Formal decision objective** (state this explicitly in your methodology section — even if the actual implementation is rule-based/XGBoost underneath, this framing gives the policy theoretical grounding):

$$ A^* = \arg\max_{A} \left[ \text{Expected Reliability}(A) - \lambda \cdot \text{Cost}(A) \right] $$
$$ A \in \{\text{Answer, Retrieve, Verify, Clarify, Abstain}\} $$

**What NOT to claim as your research gap** (use near-verbatim in related-work framing to avoid overclaiming):
- ❌ "LLMs cannot estimate uncertainty" — already studied (Kadavath et al.)
- ❌ "No one has worked on hallucination detection" — extensive research exists (SelfCheckGPT, semantic entropy)
- ❌ "No one has studied abstention" — already studied (InDU paper)

**Your actual contribution**: integrating multiple uncertainty/evidence signals into an **adaptive action-selection policy** — the decision layer, not the underlying detection primitives.

---

## 3. Requirements / Prerequisites

**Knowledge:** Python (intermediate+), LLM prompting/API basics, RAG & embeddings, agent framework concepts (LangGraph), probability/calibration/entropy, precision/recall/F1/ECE, FastAPI + Streamlit basics, familiarity with the 6 core papers (P(True)/P(IK), SelfCheckGPT, semantic entropy, InDU abstention, SAUP, SELAUR).

**Software:** Python 3.10+, Git, Ollama + Groq API key, LangGraph, FAISS/ChromaDB, RAGAS, DeepEval, scikit-learn, XGBoost, Sentence-Transformers/BGE, a cross-encoder model (e.g. BGE-reranker), FastAPI, Streamlit.

**Hardware:** 8GB+ RAM laptop sufficient (API-first approach via Groq); GPU optional/backup via Colab/Kaggle for Ollama fallback.

**Datasets (final — all 6 kept):**

| Dataset | Purpose |
|---|---|
| SimpleQA | Main factual QA benchmark — accuracy, calibration, Answer/Abstain decisions |
| SQuAD 2.0 | Unanswerable subset — trains/tests Abstain behavior |
| AmbigQA | Ambiguous questions — tests Clarify behavior |
| HotpotQA | Multi-hop — tests Retrieve + Verify behavior |
| HaluEval | Pre-labeled hallucinated/non-hallucinated examples — hallucination-rate metric, AUROC eval |
| Custom Decision Dataset (1,000–2,000 Qs) | Your core research contribution — labeled `(query → expected_action)` pairs across known/retrieval-required/unknown/ambiguous/conflicting-evidence/time-sensitive categories |

---

## 4. Architecture

```
USER QUERY
    ↓
Query Analyzer
    ↓
Uncertainty Engine ──┬── Model Confidence (self-consistency, P(True)/P(IK))
                      ├── Evidence Quality (embedding similarity + cross-encoder support score)
                      ├── Answer–Evidence Contradiction (NLI/cross-encoder)
                      └── Question Ambiguity / Complexity
    ↓
Adaptive Policy (rule-based → XGBoost/Random Forest refinement)
    ↓
ANSWER / RETRIEVE / VERIFY / CLARIFY / ABSTAIN
    ↓
(if Retrieve) → VERIFY → FINAL ANSWER
    ↓
EVALUATION (Accuracy / Hallucination / Calibration / Decision Quality / Cost)
```

**Agent orchestration:** LangGraph (explicit state/action graph — better fit than LangChain chaining for this branching, loopable decision structure). LangChain/LlamaIndex utilities can still be used underneath for retrievers/loaders.

**Baselines (4, same LLM/dataset/interface as proposed agent for fair comparison):**
1. Normal LLM agent (no uncertainty awareness)
2. Standard RAG agent (always retrieves)
3. Self-reflection agent (generate → critique → revise)
4. Fixed-threshold uncertainty agent (single cutoff → retrieve or answer)

---

## 5. Signals — Full List (including new additions)

- **Question:** embedding, length, complexity, ambiguity score
- **Model:** confidence, response consistency, multi-sample agreement, self-evaluation (P(True)/P(IK))
- **Retrieval:** top-1 similarity, avg top-K similarity, evidence coverage
- **Semantic (expanded):**
  - question–answer similarity
  - **answer–evidence support score via cross-encoder** *(new — stronger than plain cosine similarity for triggering Verify)*
  - **explicit answer–evidence contradiction score (NLI/cross-encoder)** *(new — distinct signal from "low similarity"; cleaner Verify trigger)*

---

## 6. Evaluation — Full Metric Taxonomy (new structure adopted)

**1. Final answer quality:** Accuracy, **Precision, Recall**, F1, hallucination rate (with **precision/recall of hallucination flags** — of flagged answers, how many really were hallucinated; of actual hallucinations, how many were caught), evidence-groundedness

**2. Uncertainty quality:** Expected Calibration Error (ECE), **Brier score** *(new)*, **AUROC for hallucination detection** *(new)*

**3. Agent decision quality** *(new explicit category — this is what actually proves your contribution over the baselines)*:
- Action accuracy
- **Per-action precision/recall** — e.g. of all Abstain decisions, what fraction were correct (precision); of all questions that should have triggered Abstain, how many did (recall). Same treatment for Retrieve/Verify/Clarify.
- Correct abstention / retrieval / verification / clarification rate
- **Unnecessary retrieval rate, unnecessary verification rate**

**4. Efficiency:** Token usage, # LLM calls, # tool calls, retrieval calls, latency, API cost

**Key experiments:**
- **Threshold-sweep** for the fixed-threshold baseline across 0.1–0.9 → produces a reliability–cost curve to plot the adaptive agent against (your main proof figure) *(new, made explicit)*
- **Ablation study**: remove one signal at a time (semantic uncertainty → retrieval confidence → self-consistency → question complexity) and measure the drop — shows each component's contribution *(new, made explicit as its own task)*

---

## 7. Week-by-Week Gantt Timeline (8 Weeks)

### Week 1 — Foundation & Setup
**Revise first:** venv/conda, pip, Git basics, REST API concepts, structure of all 6 datasets (SQuAD 2.0 answerable/unanswerable format, HotpotQA multi-hop, AmbigQA ambiguity annotations, SimpleQA format, HaluEval categories). Re-skim the 6 literature papers.

Steps:
1. Create Git repo (`/data`, `/src`, `/notebooks`, `/eval`, `/app`), venv, Python 3.10+
2. `requirements.txt` + README with project scope
3. Set up Groq API key (primary) + install Ollama (backup)
4. Set up FAISS/ChromaDB
5. Load and inspect all 6 datasets via Hugging Face `datasets` library (budget extra time for AmbigQA — fiddlier to load)
6. **Design and lock the unified schema** — this is the most important deliverable of the week; every baseline and your proposed agent get scored against this file later. Include fields for `expected_action`, `category`, `evidence_available`, `difficulty`.
7. One trivial end-to-end call (Query → LLM → Answer) to confirm plumbing works — no uncertainty logic yet
8. Commit + write README notes

### Week 2 — Uncertainty Estimation Module
**Revise first:** calibration/confidence concepts, entropy as disagreement measure, temperature sampling, self-consistency decoding, SelfCheckGPT approach, P(True)/P(IK) prompting (Kadavath et al.). Be comfortable writing prompts that elicit a confidence score and computing agreement across sampled outputs (string/embedding similarity).

1. Implement self-consistency/entropy signal (multi-sample at high temp, measure agreement)
2. Implement P(True)/P(IK) self-assessment prompting
3. Combine into a single uncertainty score per query
4. Sanity-test against HaluEval's labeled hallucinated/non-hallucinated examples

### Week 3 — Evidence Quality, Ambiguity, Complexity Signals
**Revise first:** cosine similarity, retrieval relevance scoring, prompt-based zero-shot classification, what makes a question genuinely ambiguous vs. underspecified, LangGraph retriever integration, **basics of cross-encoders / NLI models for entailment vs. contradiction** *(new prerequisite)*.

1. Retrieval evidence-scoring step (FAISS/ChromaDB + web search retrieval)
2. **Add cross-encoder evidence-support scoring** (does retrieved evidence actually support the candidate answer) *(new)*
3. **Add contradiction detection signal** (answer vs. evidence, via NLI/cross-encoder) *(new)*
4. Ambiguity detector (LLM classification prompt or entropy over paraphrased-question answers)
5. Task-complexity estimator (heuristic or LLM classification)

### Week 4 — Adaptive Decision Policy
**Revise first:** rule-based vs. learned classifiers, logistic regression/decision trees/XGBoost basics, combining normalized scores into one decision, the formal objective (Reliability − λ·Cost). Re-review SAUP and SELAUR for multi-signal decision structuring.

1. Formalize the decision objective for your write-up
2. Design policy logic: start rule-based/threshold combination over (uncertainty, evidence quality, contradiction, ambiguity, complexity)
3. Build labeled feature set from Custom Decision Dataset
4. Train lightweight classifier (XGBoost/Random Forest) as refinement over rule-based policy
5. Implement each action as real agent behavior: Answer / Retrieve (RAG+web) / Verify (second LLM pass against evidence) / Clarify (follow-up question) / Abstain ("I don't know" + reasoning)
6. Wire policy into LangGraph as explicit nodes/edges with conditional branching (including Retrieve → Verify → Answer loop)

### Week 5 — Baseline Agents
**Revise first:** standard retrieve-then-generate RAG architecture, self-reflection patterns (generate → critique → revise), fixed-threshold agent logic (single cutoff). Ensure all baselines and proposed agent share identical LLM, dataset, and interface for fair comparison.

1. Implement Baseline 1 — Normal LLM agent
2. Implement Baseline 2 — Standard RAG agent
3. Implement Baseline 3 — Self-reflection agent
4. Implement Baseline 4 — Fixed-threshold uncertainty agent (parameterize the threshold for the sweep in Week 6)

### Week 6 — Evaluation & Metrics
**Revise first:** precision/recall/F1, Expected Calibration Error, **Brier score, AUROC** *(new)*, operational definitions of hallucination rate and abstention rate, how RAGAS/DeepEval compute their metrics, basic statistical comparison (is a 3% F1 difference meaningful?).

1. Run all 5 agents (proposed + 4 baselines) over the labeled dataset (all 6 sources)
2. Compute Tier 1 metrics: accuracy, F1, hallucination rate, evidence-groundedness
3. Compute Tier 2 metrics: ECE, **Brier score, AUROC for hallucination detection** *(new)*
4. Compute Tier 3 — **agent decision quality metrics**: action accuracy, correct/unnecessary retrieval & verification rates, correct abstention/clarification rate *(new explicit tier)*
5. Compute Tier 4 — efficiency: tokens, LLM calls, tool calls, latency, API cost
6. **Run the threshold-sweep experiment** (0.1–0.9) for the fixed-threshold baseline; plot reliability–cost curve against your adaptive agent *(new)*
7. **Run the ablation study**: remove semantic uncertainty → retrieval confidence → self-consistency → complexity, one at a time, and re-measure *(new)*
8. Tabulate and visualize all comparative results

### Week 7 — Backend + Frontend
**Revise first:** FastAPI routes/Pydantic models, wrapping a pipeline as an API endpoint, Streamlit widgets/session state/calling a backend from frontend. Plan the reasoning-trace JSON shape (uncertainty scores → action taken → evidence used → contradiction flag).

1. FastAPI endpoint: query in → {action, answer, reasoning trace} out
2. Streamlit frontend: live demo showing decision path (e.g. "Uncertainty: high → Evidence: weak → Action: Retrieve → Verify → Answer")
3. If time-constrained: a working CLI/notebook demo is an acceptable substitute for full Streamlit UI — protect Weeks 2–4 instead

### Week 8 — Write-up, Testing, Buffer
**Revise first:** technical report structure (intro, related work, methodology, results, discussion, conclusion), presenting comparative tables/charts, citation formatting. Intentionally light on new concepts — buffer week.

1. Document methodology + architecture diagrams (both the core and full agentic versions)
2. Results tables/charts: baseline comparison, reliability–cost curve, ablation results
3. Related work section — explicitly state what you are **not** claiming as novel (Section 2 list above), and frame your contribution as the adaptive action-selection layer
4. Final report/paper + presentation, tying back to the 5 identified research gaps
5. Proofread, buffer for anything that slipped from Weeks 2–4

---

## 8. Timeline Risk Notes

- **Weeks 2–4** (uncertainty + policy core) are the intellectually hardest and most likely to overrun. If tight on time, compress **Week 1** (setup) or **Week 7** (interface) instead — protect Weeks 2–4.
- **Week 1, Step 6** (unified schema) is the highest-leverage task of the whole project — every later evaluation depends on getting it right now.
- **Week 6** now carries more weight than originally scoped (threshold sweep + ablation added) — if Week 1 or 7 gets compressed, do **not** compress Week 6; the sweep and ablation are your strongest evidence for the paper.
- Six datasets means more schema-unification work in Week 1 — budgeted for, but don't let AmbigQA loading eat into Week 2's start.
