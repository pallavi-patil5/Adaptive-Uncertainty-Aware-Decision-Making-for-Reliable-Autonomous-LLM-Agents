# app/frontend/streamlit_app.py
import streamlit as st
import requests
import json, os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Adaptive Uncertainty-Aware Agent", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:#0f1117;color:#e0e0e0}
.action-badge{display:inline-block;padding:6px 20px;border-radius:20px;font-size:1.1rem;font-weight:700;letter-spacing:1px;margin:4px 0}
.action-ANSWER{background:#1a3a1a;color:#4caf50;border:1px solid #4caf50}
.action-RETRIEVE{background:#1a2a3a;color:#2196f3;border:1px solid #2196f3}
.action-VERIFY{background:#3a2a00;color:#ff9800;border:1px solid #ff9800}
.action-CLARIFY{background:#2a1a3a;color:#9c27b0;border:1px solid #9c27b0}
.action-ABSTAIN{background:#3a1a1a;color:#f44336;border:1px solid #f44336}
.sig-wrap{margin:4px 0}
.sig-lbl{font-size:0.78rem;color:#aaa;margin-bottom:2px}
.sig-bg{background:#1e1e2e;border-radius:4px;height:10px;width:100%}
.sig-fill{height:10px;border-radius:4px}
.tl-step{border-left:2px solid #2196f3;padding:6px 12px;margin:6px 0;background:#111827;border-radius:0 6px 6px 0}
.tl-title{font-size:0.82rem;color:#2196f3;font-weight:600}
.tl-detail{font-size:0.78rem;color:#ccc;margin-top:2px}
.ev-card{background:#111827;border:1px solid #1e293b;border-radius:8px;padding:10px 14px;margin:8px 0}
.ev-rel{border-left:3px solid #4caf50}
.ev-irr{border-left:3px solid #555}
.mc{background:#111827;border:1px solid #1e293b;border-radius:8px;padding:12px;text-align:center}
.mv{font-size:1.5rem;font-weight:700}
.ml{font-size:0.75rem;color:#888;margin-top:2px}
.ans-box{background:#111827;border:1px solid #1e293b;border-radius:10px;padding:16px 20px;margin:12px 0;font-size:0.95rem;line-height:1.6}
.sec{font-size:0.72rem;font-weight:700;letter-spacing:2px;color:#555;text-transform:uppercase;margin:16px 0 8px 0}
</style>
""", unsafe_allow_html=True)

DEMOS = [
    {"label": "🟢 Certain — ANSWER",        "q": "What is the capital of France?"},
    {"label": "🔵 Knowledge gap — RETRIEVE", "q": "What is the current price of gold in India?"},
    {"label": "🟠 Conflicting — VERIFY",     "q": "Is the statement 'the Earth is flat' scientifically supported?"},
    {"label": "🟣 Ambiguous — CLARIFY",      "q": "Tell me about Apple."},
    {"label": "🔴 Unknowable — ABSTAIN",     "q": "What was the exact thought of Albert Einstein immediately before his death?"},
]

COLORS = {"answer": "#4caf50", "retrieve": "#2196f3", "verify": "#ff9800", "clarify": "#9c27b0", "abstain": "#f44336"}

EXPLAIN = {
    "answer":   "Model confidence is high and the question is unambiguous — answering directly is optimal.",
    "retrieve": "Uncertainty or complexity is high — fetching external evidence before answering.",
    "verify":   "Evidence contradicts the candidate answer — a verification pass is required.",
    "clarify":  "Question has multiple valid interpretations — clarification needed from the user.",
    "abstain":  "Uncertainty is high and no reliable evidence is available — abstaining is safer than hallucinating.",
}


def sbar(label, value, color):
    pct = int(value * 100)
    st.markdown(f'<div class="sig-wrap"><div class="sig-lbl">{label} — {value:.3f}</div>'
                f'<div class="sig-bg"><div class="sig-fill" style="width:{pct}%;background:{color}"></div></div></div>',
                unsafe_allow_html=True)


def badge(action):
    a = action.upper()
    st.markdown(f'<span class="action-badge action-{a}">{a}</span>', unsafe_allow_html=True)


def call_api(q):
    try:
        r = requests.post(f"{API_URL}/query", json={"question": q}, timeout=180)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def load_metrics():
    try:
        r = requests.get(f"{API_URL}/eval_metrics", timeout=10)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Adaptive Agent")
    st.markdown("*Uncertainty-Aware Decision Making*")
    st.markdown("---")
    page = st.radio("Navigate", ["① Agent Demo", "② Decision Trace", "③ Research Metrics"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Research Question**")
    st.caption("Can an adaptive uncertainty-aware policy improve reliability while minimising unnecessary retrieval and verification cost?")
    st.markdown("---")
    st.markdown("**Formal Objective**")
    st.latex(r"A^* = \arg\max_A \left[ R(A) - \lambda \cdot C(A) \right]")
    st.markdown("---")
    for a, c in COLORS.items():
        st.markdown(f'<span style="color:{c}">●</span> `{a.upper()}`', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — AGENT DEMO
# ══════════════════════════════════════════════════════════════════════════════
if page == "① Agent Demo":
    st.markdown("## Agent Playground")
    st.caption("Observe how the agent selects an action based on uncertainty — not just answers the question.")

    st.markdown('<div class="sec">Demo Scenarios</div>', unsafe_allow_html=True)
    dcols = st.columns(len(DEMOS))
    for i, s in enumerate(DEMOS):
        if dcols[i].button(s["label"], use_container_width=True):
            st.session_state["prefill"] = s["q"]

    prefill = st.session_state.get("prefill", "")
    question = st.text_input("Ask a question:", value=prefill, placeholder="Type any question…")
    if st.button("▶  Run Agent", type="primary", use_container_width=True) and question.strip():
        st.session_state.pop("prefill", None)
        with st.spinner("Agent is reasoning…"):
            data, err = call_api(question)
        if err:
            st.error(f"API error: {err}")
        else:
            st.session_state["last_result"] = data

    data = st.session_state.get("last_result")
    if not data:
        st.info("Run a query above or click a demo scenario.")
        st.stop()

    action = data["action"]
    signals = data["signals"]
    color = COLORS.get(action, "#888")
    conf = round(1 - signals["uncertainty"], 3)
    conf_color = "#4caf50" if conf > 0.7 else "#ff9800" if conf > 0.4 else "#f44336"

    st.markdown("---")
    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="sec">Selected Action</div>', unsafe_allow_html=True)
        badge(action)
        st.markdown(f'<div style="font-size:0.82rem;color:#aaa;margin-top:8px">{EXPLAIN[action]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec" style="margin-top:16px">Confidence</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mc"><div class="mv" style="color:{conf_color}">{conf:.0%}</div><div class="ml">Model Confidence</div></div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="sec">Final Answer</div>', unsafe_allow_html=True)
        ans = data.get("final_answer") or "*(No direct answer)*"
        source_map = {
            "answer":   ("🧠", "From model knowledge",       "#4caf50"),
            "retrieve": ("📚", "From retrieved evidence",    "#2196f3"),
            "verify":   ("✓",  "Verified against evidence",  "#ff9800"),
            "clarify":  ("❓", "Clarification requested",    "#9c27b0"),
            "abstain":  ("🚫", "Abstained — too uncertain",  "#f44336"),
        }
        icon, src_label, src_color = source_map.get(action, ("🧠", "Model", "#888"))
        st.markdown(f'<div style="font-size:0.78rem;color:{src_color};margin-bottom:6px">{icon} {src_label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ans-box">{ans}</div>', unsafe_allow_html=True)
        eff = data["efficiency"]
        c1, c2, c3 = st.columns(3)
        c1.metric("LLM Calls", eff["llm_calls"])
        c2.metric("Retrieval Calls", eff["retrieval_calls"])
        c3.metric("Latency", f"{eff['latency_s']:.2f}s")

    st.markdown("---")
    st.markdown('<div class="sec">Uncertainty & Evidence Signals</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        sbar("Uncertainty (combined)", signals["uncertainty"], "#f44336")
        sbar("Self-Consistency Uncertainty", signals["self_consistency"], "#ff7043")
        sbar("Self-Eval Uncertainty", signals["self_eval"], "#ff8a65")
        sbar("Ambiguity", signals["ambiguity"], "#9c27b0")
    with s2:
        sbar("Complexity", signals["complexity"], "#2196f3")
        sbar("Evidence Coverage", signals["evidence_coverage"], "#4caf50")
        sbar("Contradiction Probability", signals["contradiction_prob"], "#ff9800")
        sbar("Support Score", signals["support_score"], "#00bcd4")

    st.markdown("---")
    st.markdown('<div class="sec">Action Scores — A* = argmax [ Reliability(A) − λ·Cost(A) ]</div>', unsafe_allow_html=True)
    scores = data["action_scores"]
    reliability = data["action_reliability"]
    acols = st.columns(len(scores))
    for i, (a, s) in enumerate(scores.items()):
        c = COLORS.get(a, "#888")
        rel = reliability.get(a, 0)
        border = f"border:2px solid {c}" if a == action else "border:1px solid #1e293b"
        acols[i].markdown(
            f'<div class="mc" style="{border}">'
            f'<div class="mv" style="color:{c}">{s:+.3f}</div>'
            f'<div class="ml">{a.upper()}</div>'
            f'<div style="font-size:0.7rem;color:#555;margin-top:4px">rel={rel:.3f}</div>'
            f'</div>', unsafe_allow_html=True)

    # Action scores bar chart
    import pandas as pd, altair as alt
    df_scores = pd.DataFrame([
        {"action": a.upper(), "score": s, "color": COLORS.get(a, "#888")}
        for a, s in scores.items()
    ])
    chart = alt.Chart(df_scores).mark_bar().encode(
        x=alt.X("score:Q", title="Score"),
        y=alt.Y("action:N", sort="-x", title=None),
        color=alt.Color("color:N", scale=None),
        tooltip=["action", "score"]
    ).properties(height=160, title="Action Score Comparison")
    st.altair_chart(chart, use_container_width=True)

    # Radar-style signal chart
    st.markdown('<div class="sec">Signal Radar</div>', unsafe_allow_html=True)
    sig_items = [
        ("Uncertainty",       signals["uncertainty"]),
        ("Self-Consistency",  signals["self_consistency"]),
        ("Self-Eval",         signals["self_eval"]),
        ("Ambiguity",         signals["ambiguity"]),
        ("Complexity",        signals["complexity"]),
        ("Evidence Coverage", signals["evidence_coverage"]),
        ("Contradiction",     signals["contradiction_prob"]),
        ("Support Score",     signals["support_score"]),
    ]
    df_sig = pd.DataFrame(sig_items, columns=["signal", "value"])
    radar = alt.Chart(df_sig).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
        y=alt.Y("signal:N", sort="-x", title=None),
        color=alt.condition(
            alt.datum.value > 0.6,
            alt.value("#f44336"),
            alt.condition(alt.datum.value > 0.3, alt.value("#ff9800"), alt.value("#4caf50"))
        ),
        tooltip=["signal", "value"]
    ).properties(height=220, title="Uncertainty & Evidence Signals (0–1)")
    st.altair_chart(radar, use_container_width=True)

    chunks = data.get("evidence_chunks", [])
    if chunks:
        st.markdown("---")
        st.markdown('<div class="sec">Retrieved Evidence Chunks</div>', unsafe_allow_html=True)
        for i, chunk in enumerate(chunks):
            cls = "ev-rel" if chunk["relevant"] else "ev-irr"
            lbl = "✓ Relevant" if chunk["relevant"] else "✗ Below threshold"
            lc = "#4caf50" if chunk["relevant"] else "#555"
            st.markdown(
                f'<div class="ev-card {cls}">'
                f'<div style="font-size:0.72rem;color:{lc};margin-bottom:4px">Chunk {i+1} — distance={chunk["distance"]}  {lbl}</div>'
                f'<div style="font-size:0.82rem;color:#ccc">{chunk["text"]}</div>'
                f'</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DECISION TRACE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "② Decision Trace":
    st.markdown("## Decision Trace")
    st.caption("Step-by-step view of how the agent processed the last query.")

    data = st.session_state.get("last_result")
    if not data:
        st.info("Run a query on the Agent Demo page first.")
        st.stop()

    action = data["action"]
    signals = data["signals"]
    color = COLORS.get(action, "#888")

    # Header
    st.markdown(f'**Question:** {data["question"]}')
    badge(action)
    st.markdown(f'<span style="color:#aaa;font-size:0.85rem">{EXPLAIN[action]}</span>', unsafe_allow_html=True)
    st.markdown("---")

    # Timeline
    st.markdown('<div class="sec">Agent Decision Pipeline</div>', unsafe_allow_html=True)
    for step in data.get("timeline", []):
        st.markdown(
            f'<div class="tl-step">'
            f'<div class="tl-title">▶ {step["step"]}</div>'
            f'<div class="tl-detail">{step["detail"]}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="sec">Signal Values</div>', unsafe_allow_html=True)
        rows = [
            ("Uncertainty",          signals["uncertainty"],        "#f44336"),
            ("Self-Consistency",     signals["self_consistency"],   "#ff7043"),
            ("Self-Eval",            signals["self_eval"],          "#ff8a65"),
            ("Ambiguity",            signals["ambiguity"],          "#9c27b0"),
            ("Complexity",           signals["complexity"],         "#2196f3"),
            ("Evidence Coverage",    signals["evidence_coverage"],  "#4caf50"),
            ("Contradiction Prob",   signals["contradiction_prob"], "#ff9800"),
            ("Support Score",        signals["support_score"],      "#00bcd4"),
        ]
        for lbl, val, col in rows:
            sbar(lbl, val, col)

    with right:
        st.markdown('<div class="sec">Action Score Breakdown</div>', unsafe_allow_html=True)
        scores = data["action_scores"]
        reliability = data["action_reliability"]
        for a, s in sorted(scores.items(), key=lambda x: -x[1]):
            c = COLORS.get(a, "#888")
            rel = reliability.get(a, 0)
            is_w = a == action
            border = f"border:2px solid {c}" if is_w else "border:1px solid #1e293b"
            winner_tag = " ← selected" if is_w else ""
            st.markdown(
                f'<div class="mc" style="{border};margin-bottom:8px;text-align:left;padding:10px 14px">'
                f'<span style="color:{c};font-weight:700">{a.upper()}</span>'
                f'<span style="color:#555;font-size:0.75rem">{winner_tag}</span>'
                f'<div style="font-size:0.85rem;margin-top:4px">Score: <b style="color:{c}">{s:+.3f}</b> &nbsp; Reliability: {rel:.3f}</div>'
                f'</div>', unsafe_allow_html=True)

        # Signal vs threshold chart for this query
        import pandas as pd, altair as alt
        sig_df = pd.DataFrame([
            {"signal": k, "value": round(v, 3)}
            for k, v in signals.items()
        ])
        sig_chart = alt.Chart(sig_df).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title="Value"),
            y=alt.Y("signal:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.value > 0.6,
                alt.value("#f44336"),
                alt.condition(alt.datum.value > 0.3, alt.value("#ff9800"), alt.value("#4caf50"))
            ),
            tooltip=["signal", "value"]
        ).properties(height=220, title="Signal Values for This Query")
        st.altair_chart(sig_chart, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec">Candidate Answer (before action)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ans-box" style="color:#888">{data.get("candidate_answer","—")}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Final Answer (after action)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ans-box">{data.get("final_answer") or "*(abstained / clarification requested)*"}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Raw Trace</div>', unsafe_allow_html=True)
    st.code(data.get("trace", ""), language=None)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RESEARCH METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "③ Research Metrics":
    st.markdown("## Research Evaluation Metrics")
    st.caption("Comparative results across all 5 systems — Tier 1 answer quality, Tier 3 decision quality, Tier 4 efficiency.")

    metrics, err = load_metrics()
    if err or not metrics or "error" in (metrics or {}):
        st.error(f"Could not load metrics: {err or metrics.get('error','unknown')}. Run `python src/eval/run_evaluation.py` first.")
        st.stop()

    SYSTEMS = ["adaptive", "normal_llm", "standard_rag", "self_reflection", "fixed_threshold"]
    LABELS  = {"adaptive": "Adaptive Agent", "normal_llm": "Normal LLM",
                "standard_rag": "Standard RAG", "self_reflection": "Self-Reflection",
                "fixed_threshold": "Fixed Threshold"}
    SYS_COLORS = {"adaptive": "#2196f3", "normal_llm": "#9e9e9e",
                   "standard_rag": "#4caf50", "self_reflection": "#ff9800",
                   "fixed_threshold": "#9c27b0"}

    present = [s for s in SYSTEMS if s in metrics]

    import pandas as pd, altair as alt

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Tier 1 — Answer Quality")
    t1cols = st.columns(len(present))
    for i, sys in enumerate(present):
        m = metrics[sys]
        c = SYS_COLORS[sys]
        is_best_acc  = m["accuracy"]          == max(metrics[s]["accuracy"]          for s in present)
        is_best_hall = m["hallucination_rate"] == min(metrics[s]["hallucination_rate"] for s in present)
        acc_color  = "#4caf50" if is_best_acc  else "#e0e0e0"
        hall_color = "#4caf50" if is_best_hall else "#e0e0e0"
        t1cols[i].markdown(
            f'<div class="mc" style="border-top:3px solid {c}">'
            f'<div style="font-size:0.78rem;color:{c};font-weight:700;margin-bottom:8px">{LABELS[sys]}</div>'
            f'<div class="mv" style="color:{acc_color}">{m["accuracy"]:.1%}</div>'
            f'<div class="ml">Accuracy</div>'
            f'<div style="margin-top:10px;font-size:0.85rem;color:{hall_color}">{m["hallucination_rate"]:.1%}</div>'
            f'<div class="ml">Hallucination Rate ↓</div>'
            f'<div style="margin-top:6px;font-size:0.72rem;color:#555">n={m["n"]}</div>'
            f'</div>', unsafe_allow_html=True)

    # Grouped bar: Accuracy vs Hallucination per system
    t1_rows = []
    for s in present:
        t1_rows += [
            {"System": LABELS[s], "Metric": "Accuracy ↑",        "Value": metrics[s]["accuracy"],          "color": SYS_COLORS[s]},
            {"System": LABELS[s], "Metric": "Hallucination ↓",   "Value": metrics[s]["hallucination_rate"], "color": SYS_COLORS[s]},
        ]
    df_t1 = pd.DataFrame(t1_rows)
    t1_chart = alt.Chart(df_t1).mark_bar().encode(
        x=alt.X("System:N", title=None),
        y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
        color=alt.Color("System:N", scale=alt.Scale(domain=list(LABELS.values()), range=[SYS_COLORS[s] for s in present])),
        column=alt.Column("Metric:N", title=None),
        tooltip=["System", "Metric", alt.Tooltip("Value:Q", format=".1%")]
    ).properties(width=220, height=220)
    st.altair_chart(t1_chart)

    # ── Tier 3 ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Tier 3 — Decision Quality")
    st.caption("Per-action precision and recall for the adaptive agent vs baselines. This is the core research contribution metric.")

    t3cols = st.columns(len(present))
    for i, sys in enumerate(present):
        m = metrics[sys]
        c = SYS_COLORS[sys]
        is_best = m["action_accuracy"] == max(metrics[s]["action_accuracy"] for s in present)
        aa_color = "#4caf50" if is_best else "#e0e0e0"
        t3cols[i].markdown(
            f'<div class="mc" style="border-top:3px solid {c}">'
            f'<div style="font-size:0.78rem;color:{c};font-weight:700;margin-bottom:8px">{LABELS[sys]}</div>'
            f'<div class="mv" style="color:{aa_color}">{m["action_accuracy"]:.1%}</div>'
            f'<div class="ml">Action Accuracy</div>'
            f'<div style="margin-top:10px;font-size:0.85rem;color:#ff9800">{m["unnecessary_retrieval"]:.1%}</div>'
            f'<div class="ml">Unnecessary Retrieval ↓</div>'
            f'</div>', unsafe_allow_html=True)

    # Per-action precision/recall for adaptive only
    st.markdown("---")
    st.markdown("#### Adaptive Agent — Per-Action Precision / Recall")
    if "adaptive" in metrics:
        pa = metrics["adaptive"]["per_action"]
        actions = ["answer", "retrieve", "verify", "clarify", "abstain"]
        pacols = st.columns(len(actions))
        for i, a in enumerate(actions):
            c = COLORS.get(a, "#888")
            p = pa.get(a, {}).get("precision", 0)
            r = pa.get(a, {}).get("recall", 0)
            f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
            pacols[i].markdown(
                f'<div class="mc" style="border-top:3px solid {c}">'
                f'<div style="color:{c};font-weight:700;font-size:0.82rem;margin-bottom:6px">{a.upper()}</div>'
                f'<div style="font-size:0.85rem">P: <b>{p:.2f}</b></div>'
                f'<div style="font-size:0.85rem">R: <b>{r:.2f}</b></div>'
                f'<div style="font-size:0.85rem">F1: <b>{f1:.2f}</b></div>'
                f'</div>', unsafe_allow_html=True)

    # ── Tier 4 ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Tier 4 — Efficiency")
    st.caption("Lower LLM calls and latency = lower cost. Adaptive agent should use fewer calls than fixed_threshold while being more accurate than normal_llm.")

    t4cols = st.columns(len(present))
    for i, sys in enumerate(present):
        m = metrics[sys]
        c = SYS_COLORS[sys]
        t4cols[i].markdown(
            f'<div class="mc" style="border-top:3px solid {c}">'
            f'<div style="font-size:0.78rem;color:{c};font-weight:700;margin-bottom:8px">{LABELS[sys]}</div>'
            f'<div class="mv">{m["avg_llm_calls"]:.1f}</div>'
            f'<div class="ml">Avg LLM Calls</div>'
            f'<div style="margin-top:10px;font-size:0.85rem">{m["avg_retrieval_calls"]:.2f}</div>'
            f'<div class="ml">Avg Retrieval Calls</div>'
            f'<div style="margin-top:10px;font-size:0.85rem">{m["avg_latency_s"]:.2f}s</div>'
            f'<div class="ml">Avg Latency</div>'
            f'</div>', unsafe_allow_html=True)

    # Scatter: Accuracy vs Avg LLM Calls — efficiency frontier
    df_scatter = pd.DataFrame([
        {"System": LABELS[s], "Accuracy": metrics[s]["accuracy"],
         "Avg LLM Calls": metrics[s]["avg_llm_calls"], "color": SYS_COLORS[s]}
        for s in present
    ])
    scatter = alt.Chart(df_scatter).mark_circle(size=120).encode(
        x=alt.X("Avg LLM Calls:Q", title="Avg LLM Calls (lower = cheaper)"),
        y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="Accuracy (higher = better)"),
        color=alt.Color("color:N", scale=None),
        tooltip=["System", alt.Tooltip("Accuracy:Q", format=".1%"), "Avg LLM Calls"]
    ).mark_circle(size=120) + alt.Chart(df_scatter).mark_text(dy=-12, fontSize=11).encode(
        x="Avg LLM Calls:Q", y="Accuracy:Q", text="System:N",
        color=alt.Color("color:N", scale=None)
    )
    st.altair_chart(scatter.properties(height=280, title="Efficiency Frontier — Accuracy vs Cost"), use_container_width=True)

    # ── RAGAS ─────────────────────────────────────────────────────────────────
    ragas_present = [s for s in present if metrics[s].get("ragas_faithfulness") is not None]
    if ragas_present:
        st.markdown("---")
        st.markdown("### RAGAS — Faithfulness & Answer Relevancy")
        st.caption("Computed on retrieve/verify rows only. Faithfulness: does the answer stay grounded in evidence? Answer Relevancy: is the answer on-topic?")
        rcols = st.columns(len(ragas_present))
        for i, sys in enumerate(ragas_present):
            m = metrics[sys]
            c = SYS_COLORS[sys]
            faith = m["ragas_faithfulness"]
            relev = m["ragas_answer_relevancy"]
            best_f = faith == max(metrics[s]["ragas_faithfulness"] for s in ragas_present if metrics[s].get("ragas_faithfulness") is not None)
            best_r = relev == max(metrics[s]["ragas_answer_relevancy"] for s in ragas_present if metrics[s].get("ragas_answer_relevancy") is not None)
            rcols[i].markdown(
                f'<div class="mc" style="border-top:3px solid {c}">'
                f'<div style="font-size:0.78rem;color:{c};font-weight:700;margin-bottom:8px">{LABELS[sys]}</div>'
                f'<div class="mv" style="color:{"#4caf50" if best_f else "#e0e0e0"}">{faith:.3f}</div>'
                f'<div class="ml">Faithfulness ↑</div>'
                f'<div style="margin-top:10px;font-size:0.85rem;color:{"#4caf50" if best_r else "#e0e0e0"}">{relev:.3f}</div>'
                f'<div class="ml">Answer Relevancy ↑</div>'
                f'</div>', unsafe_allow_html=True)
    else:
        st.info("RAGAS scores not yet computed — run `python src/eval/compute_ragas.py` first.")

    # ── DeepEval ──────────────────────────────────────────────────────────────
    de_present = [s for s in present if metrics[s].get("deepeval_geval_correctness") is not None]
    if de_present:
        st.markdown("---")
        st.markdown("### DeepEval — LLM-as-Judge Metrics")
        st.caption("Hallucination score (lower = less hallucination). GEval Correctness: semantic correctness judged by LLM, stronger than substring match.")
        dcols = st.columns(len(de_present))
        for i, sys in enumerate(de_present):
            m = metrics[sys]
            c = SYS_COLORS[sys]
            halluc = m.get("deepeval_hallucination")
            geval  = m.get("deepeval_geval_correctness")
            relev  = m.get("deepeval_answer_relevancy")
            best_g = geval is not None and geval == max((metrics[s].get("deepeval_geval_correctness") or 0) for s in de_present)
            best_h = halluc is not None and halluc == min((metrics[s].get("deepeval_hallucination") or 1) for s in de_present)
            dcols[i].markdown(
                f'<div class="mc" style="border-top:3px solid {c}">'
                f'<div style="font-size:0.78rem;color:{c};font-weight:700;margin-bottom:8px">{LABELS[sys]}</div>'
                f'<div class="mv" style="color:{"#4caf50" if best_g else "#e0e0e0"}">{geval:.3f if geval is not None else "n/a"}</div>'
                f'<div class="ml">GEval Correctness ↑</div>'
                f'<div style="margin-top:10px;font-size:0.85rem;color:{"#4caf50" if best_h else "#e0e0e0"}">{halluc:.3f if halluc is not None else "n/a"}</div>'
                f'<div class="ml">Hallucination Score ↓</div>'
                f'<div style="margin-top:6px;font-size:0.82rem;color:#aaa">{relev:.3f if relev is not None else "n/a"}</div>'
                f'<div class="ml">Answer Relevancy ↑</div>'
                f'</div>', unsafe_allow_html=True)
    else:
        st.info("DeepEval scores not yet computed — run `python src/eval/compute_deepeval.py` first.")

    # ── Threshold Sweep ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Threshold Sweep — Reliability vs Cost")
    st.caption("As the uncertainty threshold increases, retrieval rate drops but so does accuracy. The optimal threshold balances both.")
    sweep_path = os.path.join(DATA_DIR, "threshold_sweep_results.json")
    if os.path.exists(sweep_path):
        import pandas as pd
        sweep = json.load(open(sweep_path))
        df_sweep = pd.DataFrame(sweep)
        import altair as alt
        base = alt.Chart(df_sweep).encode(x=alt.X("threshold:Q", title="Uncertainty Threshold"))
        acc_line = base.mark_line(color="#2196f3", strokeWidth=2).encode(
            y=alt.Y("accuracy:Q", title="Value", scale=alt.Scale(domain=[0, 1])),
            tooltip=["threshold", "accuracy"]
        )
        ret_line = base.mark_line(color="#ff9800", strokeWidth=2, strokeDash=[4, 2]).encode(
            y="retrieval_rate:Q",
            tooltip=["threshold", "retrieval_rate"]
        )
        st.altair_chart((acc_line + ret_line).properties(height=280), use_container_width=True)
        st.markdown('<span style="color:#2196f3">— Accuracy</span> &nbsp;&nbsp; <span style="color:#ff9800">- - Retrieval Rate</span>', unsafe_allow_html=True)
    else:
        st.info("Run `python src/eval/threshold_sweep.py` first.")

    # ── Ablation Study ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Signal Ablation Study")
    st.caption("Action accuracy when each signal is removed. A larger drop = that signal contributes more to the policy.")
    ablation_path = os.path.join(DATA_DIR, "ablation_results.json")
    if os.path.exists(ablation_path):
        ablation = json.load(open(ablation_path))
        baseline = ablation.get("full_model", 0)
        signal_labels = {
            "full_model":               "Full Model",
            "remove_uncertainty":       "− Uncertainty",
            "remove_evidence_coverage": "− Evidence Coverage",
            "remove_contradiction":     "− Contradiction",
            "remove_ambiguity":         "− Ambiguity",
            "remove_complexity":        "− Complexity",
        }
        df_ab = pd.DataFrame([
            {"Condition": signal_labels.get(k, k), "Accuracy": v, "Delta": v - baseline, "is_base": k == "full_model"}
            for k, v in ablation.items()
        ])
        ab_chart = alt.Chart(df_ab).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="Action Accuracy"),
            y=alt.Y("Condition:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.is_base,
                alt.value("#2196f3"),
                alt.condition(alt.datum.Delta >= 0, alt.value("#4caf50"), alt.value("#f44336"))
            ),
            tooltip=["Condition", alt.Tooltip("Accuracy:Q", format=".1%"), alt.Tooltip("Delta:Q", format="+.1%")]
        ).properties(height=220, title="Ablation: Action Accuracy per Signal Removed")
        rule = alt.Chart(pd.DataFrame([{"x": baseline}])).mark_rule(strokeDash=[4, 2], color="#555").encode(x="x:Q")
        st.altair_chart((ab_chart + rule), use_container_width=True)
    else:
        st.info("Run `python src/eval/ablation_study.py` first.")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Full Comparison Table")
    import pandas as pd
    rows = []
    for sys in present:
        m = metrics[sys]
        rows.append({
            "System":               LABELS[sys],
            "Accuracy ↑":           f'{m["accuracy"]:.1%}',
            "Hallucination ↓":      f'{m["hallucination_rate"]:.1%}',
            "Action Accuracy ↑":    f'{m["action_accuracy"]:.1%}',
            "Unnecessary Ret ↓":    f'{m["unnecessary_retrieval"]:.1%}',
            "GEval Correctness ↑":  str(m.get("deepeval_geval_correctness") or "n/a"),
            "RAGAS Faithfulness ↑": str(m.get("ragas_faithfulness") or "n/a"),
            "Avg LLM Calls ↓":      f'{m["avg_llm_calls"]:.1f}',
            "Avg Latency ↓":        f'{m["avg_latency_s"]:.2f}s',
        })
    st.dataframe(pd.DataFrame(rows).set_index("System"), use_container_width=True)

    st.markdown("---")
    st.markdown("### Research Interpretation")
    if "adaptive" in metrics and "normal_llm" in metrics and "standard_rag" in metrics:
        adp = metrics["adaptive"]
        nrm = metrics["normal_llm"]
        rag = metrics["standard_rag"]
        acc_vs_nrm  = adp["accuracy"] - nrm["accuracy"]
        acc_vs_rag  = adp["accuracy"] - rag["accuracy"]
        ret_vs_rag  = adp["avg_retrieval_calls"] - rag["avg_retrieval_calls"]
        hall_vs_nrm = nrm["hallucination_rate"] - adp["hallucination_rate"]
        sign = lambda x: ("+" if x >= 0 else "") + f"{x:.1%}"
        st.markdown(f"""
| Comparison | Result |
|---|---|
| Adaptive vs Normal LLM accuracy | `{sign(acc_vs_nrm)}` |
| Adaptive vs Standard RAG accuracy | `{sign(acc_vs_rag)}` |
| Adaptive retrieval calls vs Standard RAG | `{ret_vs_rag:+.2f}` avg calls (negative = fewer) |
| Hallucination reduction vs Normal LLM | `{sign(hall_vs_nrm)}` fewer hallucinations |
| Adaptive action accuracy | `{adp["action_accuracy"]:.1%}` — policy correctly identifies what to do |
""")
