"""TPM Dashboard - Streamlit-based UI for V10."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src.*' imports resolve
# regardless of how streamlit is invoked (directly vs via CLI).
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    st = None

from src.config import DATA_DIR, LOGS_DIR


def run_dashboard():
    """Run the Streamlit dashboard."""
    if st is None:
        print("Streamlit is required: pip install streamlit")
        return

    st.set_page_config(
        page_title="AI-TPM Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )

    st.title("AI-TPM Benchmarking Dashboard")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Benchmarks", "Scorecard", "GPU Readiness", "Tickets", "RAG KB", "AI Agent"]
    )

    # --- Benchmarks Tab ---
    with tab1:
        st.header("Benchmark Results")

        csv_path = LOGS_DIR / "inference_results.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            st.dataframe(df, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Latency", f"{df['latency_sec'].mean():.2f}s")
            col2.metric("Avg TTFT", f"{df['ttft_sec'].mean():.2f}s")
            col3.metric("Avg TPS", f"{df['tokens_per_sec'].mean():.1f}")

            st.subheader("Latency Distribution")
            st.line_chart(df[["latency_sec", "ttft_sec"]])
        else:
            st.info("No benchmark results yet. Run a benchmark first.")

    # --- Scorecard Tab ---
    with tab2:
        st.header("TPM Scorecard")

        sc_path = LOGS_DIR / "scorecard.json"
        if sc_path.exists():
            with open(sc_path) as f:
                data = json.load(f)

            scorecard = data.get("scorecard", {})
            summary = data.get("summary", {})

            st.json(summary)

            overall = scorecard.get("overall", "N/A")
            color = {"PASS": "green", "WARN": "orange", "FAIL": "red"}.get(overall, "gray")
            st.markdown(f"### Overall Status: :{color}[{overall}]")

            for key, val in scorecard.items():
                if key == "overall":
                    continue
                status = val.get("status", "N/A")
                icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(status, "")
                st.write(f"{icon} **{key}**: {val.get('value', 'N/A')} (target: {val.get('target', 'N/A')})")
        else:
            st.info("No scorecard yet. Run a benchmark first.")

    # --- GPU Readiness Tab ---
    with tab3:
        st.header("GPU Model Readiness")

        ticket_file = DATA_DIR / "synthetic_tickets.json"
        model = st.selectbox(
            "Select Model",
            ["", "Llama-3-8B", "Mistral-7B", "Gemma-2-9B", "Phi-3-medium", "Qwen2-7B"],
        )

        if st.button("Analyze Readiness"):
            from src.jira_analyzer import load_and_analyze
            result = load_and_analyze(model=model)

            level = result.readiness_level
            color = {"Production Ready": "green", "Near Ready": "orange", "Needs Work": "red", "Not Ready": "red"}.get(level, "gray")
            st.markdown(f"### {level} :{color}[{result.overall_readiness_pct}%]")

            for dim, detail in result.dimension_details.items():
                st.progress(detail["score"] / 100, text=f"{dim}: {detail['score']}%")

            st.subheader("Suggestions")
            for i, s in enumerate(result.suggestions, 1):
                st.info(f"**{i}.** {s}")

    # --- Tickets Tab ---
    with tab4:
        st.header("Jira Tickets (Synthetic)")

        if st.button("Generate Synthetic Data"):
            from src.synthetic import save_synthetic_data
            save_synthetic_data()
            st.success("Synthetic data generated!")

        if ticket_file.exists():
            with open(ticket_file) as f:
                tickets = json.load(f)

            for ticket in tickets:
                with st.expander(f"{ticket['key']}: {ticket['summary'][:60]}..."):
                    st.json(ticket)

                    if st.button(f"Triage {ticket['key']}"):
                        from src.agents import BugTriageAgent, FixRecommendationAgent
                        agent = BugTriageAgent()
                        fix_agent = FixRecommendationAgent()
                        triage = agent.triage(ticket)
                        fix = fix_agent.recommend(triage, ticket)

                        st.write(f"**Severity:** {triage.severity}")
                        st.write(f"**Category:** {triage.category}")
                        st.write(f"**Root Cause:** {triage.root_cause_hypothesis}")
                        st.write("**Fix Options:**")
                        for opt in fix.fix_options:
                            st.write(f"  - {opt['action']} (impact: {opt['impact']}, effort: {opt['effort']})")
        else:
            st.info("No tickets. Generate synthetic data first.")


# --- RAG KB Tab ---
    with tab5:
        st.header("RAG Knowledge Base")
        st.caption("V3: Ask questions grounded in the project KB (benchmark metrics, GPU readiness, Intel guide, vLLM tuning).")

        question = st.text_input("Question", value="why is ttft important")
        gen_model = st.text_input("Generative model (optional, blank = retrieval only)", value="")
        if st.button("Ask KB"):
            from src.rag.pipeline import rag_query
            answer = rag_query(question, model=gen_model or None)
            st.markdown("**Answer:**")
            st.write(answer.answer)
            st.subheader("Evidence")
            for ev in answer.retrieved_context:
                st.info(f"[{ev['rank']}] **{ev['title']}** (score {ev['score']}) — {ev['snippet']}")

    # --- AI Agent Tab ---
    with tab6:
        st.header("Agentic Tool-Calling")
        st.caption("V5: Natural-language requests routed to platform tools (benchmark, readiness, tickets, triage, HF search, scorecard, RAG).")

        query = st.text_input("Ask the agent", value="what is the readiness of llama-3-8b")
        if st.button("Run Agent"):
            from src.agents import ToolCallingAgent
            response = ToolCallingAgent().ask(query)
            st.markdown(f"**Intent:** `{response.intent}` (confidence {response.confidence:.2f})")
            st.success(response.summary)
            for call in response.calls:
                with st.expander(f"Tool: {call.tool}  args={call.args}"):
                    st.json(call.result)


if __name__ == "__main__":
    run_dashboard()
