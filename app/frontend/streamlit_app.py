# app/frontend/streamlit_app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="Adaptive Uncertainty-Aware Agent", layout="centered")
st.title("Adaptive Uncertainty-Aware Agent — Live Demo")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask a question:")

if st.button("Submit") and question.strip():
    with st.spinner("Agent is thinking..."):
        try:
            response = requests.post(API_URL, json={"question": question}, timeout=120)
            response.raise_for_status()
            data = response.json()
            st.session_state.history.insert(0, data)
        except Exception as e:
            st.error(f"Request failed: {e}")

for entry in st.session_state.history:
    st.markdown("---")
    st.markdown(f"**Q:** {entry['question']}")

    action = entry["action"]
    action_colors = {
        "answer": "🟢", "retrieve": "🔵", "verify": "🟡",
        "clarify": "🟠", "abstain": "🔴",
    }
    st.markdown(f"**Action taken:** {action_colors.get(action, '⚪')} `{action.upper()}`")

    if entry["final_answer"]:
        st.markdown(f"**Answer:** {entry['final_answer']}")
    else:
        st.markdown("*(No direct answer — clarification requested or abstained)*")

    with st.expander("Reasoning trace"):
        trace = entry["reasoning_trace"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Uncertainty", trace["uncertainty"])
            st.metric("Ambiguity", trace["ambiguity"])
        with col2:
            st.metric("Contradiction", trace["contradiction_prob"])
            st.metric("Evidence Coverage", trace["evidence_coverage"])

        st.markdown("**Action scores:**")
        st.json(trace["action_scores"])

        st.markdown("**Trace:**")
        st.text(trace["trace_text"])