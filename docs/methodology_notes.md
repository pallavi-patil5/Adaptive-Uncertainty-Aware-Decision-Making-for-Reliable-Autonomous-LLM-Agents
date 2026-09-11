# Decision Objective — Research Version

## 1. Core Research Objective

The agent must select the action that provides the best expected reliability–cost trade-off for the current query:

\[
A^*=\arg\max_{A\in\mathcal A}
\left[
\mathbb{E}[\text{Reliability}\mid A,x]
-\lambda\,\mathbb{E}[\text{Cost}\mid A,x]
\right]
\]

where:

\[
\mathcal A=
\{\text{Answer, Retrieve, Verify, Clarify, Abstain}\}
\]

and \(x\) represents the observed uncertainty, semantic, question, and evidence features.

---

## 2. Action Utility

Instead of assuming that each action's heuristic score is its true reliability, define:

\[
U(A|x)=R(A|x)-\lambda C(A|x)
\]

where:

- \(R(A|x)\) = expected reliability after taking action \(A\)
- \(C(A|x)\) = expected computational cost
- \(\lambda\) = cost sensitivity parameter

The agent selects:

\[
A^*=\arg\max_A U(A|x)
\]

This makes the objective experimentally testable.

---

## 3. Initial Reliability Proxies

Before sufficient experimental data exists, use the following proxy estimates:

| Action | Initial Reliability Proxy |
|---|---|
| **Answer** | \((1-U)(1-C)(1-Amb)\) |
| **Retrieve** | \(1-E_C\) |
| **Verify** | \(C\) |
| **Clarify** | \(Amb\) |
| **Abstain** | \(U(1-E_C)\) |

Where:

- \(U\) = uncertainty
- \(C\) = contradiction probability
- \(Amb\) = ambiguity
- \(E_C\) = evidence coverage

**Important:** These are initial heuristic/proxy estimates, not ground-truth reliability values.

For Retrieve specifically:

\[
G_{retrieve}=1-E_C
\]

should be interpreted as **potential reliability gain from obtaining additional evidence**, rather than reliability itself.

---

## 4. Cost Model

Use normalized empirical cost rather than claiming that the numbers represent actual monetary cost:

\[
C(A)=
\begin{cases}
1 & \text{Answer}\\
3 & \text{Retrieve}\\
2 & \text{Verify}\\
1 & \text{Clarify}\\
0 & \text{Abstain}
\end{cases}
\]

Initially these represent **relative computational effort**.

Later, measure actual:

- LLM calls
- Retrieval calls
- Tokens
- Latency
- API cost

and construct an empirical cost model.

---

## 5. Final Research-Based Policy

After collecting evaluation data, estimate:

\[
\hat R(A|x)
\]

using observed outcomes from the benchmark and action logs.

Then:

\[
A^*=
\arg\max_A
[\hat R(A|x)-\lambda C(A|x)]
\]

This is the actual research version of the policy.

The ML model can learn:

\[
P(A|x)
\]

for the five actions, while the policy uses the predicted action utility to make the final decision.

---

## 6. Lambda Sensitivity Experiment

Do **not** permanently fix \(\lambda=0.15\).

Use:

\[
\lambda\in
\{0.05,0.10,0.15,0.20,0.30\}
\]

and evaluate:

- Accuracy
- Hallucination rate
- Abstention quality
- Retrieval frequency
- Verification frequency
- Token consumption
- Latency
- Number of LLM/tool calls
- Overall reliability–cost score

Select the value using the **validation set**, then evaluate the selected lambda once on the held-out test set.

---

## 7. Research Hypothesis

> **An adaptive uncertainty-aware decision policy can improve reliability while reducing unnecessary retrieval and verification compared with always-answer, always-RAG, self-reflection, and fixed-threshold strategies.**

The key comparison is:

\[
\text{Reliability} \quad \leftrightarrow \quad \text{Computational Cost}
\]

rather than simply:

> "Our model predicts the correct action."

Action accuracy alone does not establish that the policy is useful.

---

## 8. Final Experimental Policy

The complete research pipeline is:

**Query**

↓

**Feature / Uncertainty Extraction**

↓

**ML Decision Model**

\[
x\rightarrow P(A|x)
\]

↓

**Action Utility Estimation**

\[
U(A|x)=\hat R(A|x)-\lambda C(A|x)
\]

↓

**Adaptive Action Selection**

**ANSWER / RETRIEVE / VERIFY / CLARIFY / ABSTAIN**

↓

**Final Response**

↓

**Evaluation**

- Reliability
- Hallucination
- Calibration
- Decision quality
- Unnecessary actions
- Token cost
- Latency
- LLM/tool calls

---

## 9. Research Contribution

The research contribution is **not the LLM itself**.

The contribution is the **adaptive uncertainty-aware reliability–cost action-selection policy** that integrates multiple signals to determine whether an autonomous LLM agent should:

- Answer directly
- Retrieve additional evidence
- Verify existing evidence or an answer
- Ask for clarification
- Abstain

The heuristic equations provide the initial policy formulation. Empirical experiments, learned reliability estimates, lambda sensitivity analysis, baselines, and ablation studies determine whether the proposed adaptive policy actually improves the reliability–cost trade-off.
