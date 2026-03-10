"""Traces page — view Application Insights traces for the selected agent."""
import streamlit as st
from lib.trace_query import query_summary, query_recent_spans, query_agent_nodes, query_errors


def render(agent: dict, workspace_id: str):
    st.title(f"📊 Traces — {agent['name']}")

    col1, col2 = st.columns([1, 3])
    with col1:
        minutes = st.number_input("Look-back (min)", min_value=5, max_value=1440, value=30, step=5)
    with col2:
        st.write("")  # spacer
        if st.button("🔄 Refresh"):
            st.cache_data.clear()

    role = agent["cloud_role_name"]

    # --- Summary ---
    st.subheader("Summary")
    try:
        df_summary = query_summary(workspace_id, role, minutes)
        if df_summary.empty:
            st.info("No spans found in the selected time window.")
        else:
            cols = st.columns(4)
            row = df_summary.iloc[0]
            cols[0].metric("Total Spans", int(row.get("spans", 0)))
            cols[1].metric("GenAI Spans", int(row.get("genai_spans", 0)))
            cols[2].metric("Input Tokens", f"{int(row.get('input_tok', 0)):,}")
            cols[3].metric("Output Tokens", f"{int(row.get('output_tok', 0)):,}")
    except Exception as e:
        st.error(f"Failed to query summary: {e}")

    # --- Agent Node Breakdown ---
    st.subheader("Agent Node Breakdown")
    try:
        df_nodes = query_agent_nodes(workspace_id, role, minutes)
        if df_nodes.empty:
            st.info("No agent node data.")
        else:
            st.dataframe(df_nodes, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Failed to query agent nodes: {e}")

    # --- Recent Spans ---
    st.subheader("Recent GenAI Spans")
    try:
        df_spans = query_recent_spans(workspace_id, role, minutes)
        if df_spans.empty:
            st.info("No recent GenAI spans.")
        else:
            st.dataframe(df_spans, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Failed to query spans: {e}")

    # --- Errors ---
    st.subheader("Errors")
    try:
        df_errors = query_errors(workspace_id, role, minutes)
        if df_errors.empty:
            st.success("No errors in the selected time window. ✅")
        else:
            st.dataframe(df_errors, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Failed to query errors: {e}")
