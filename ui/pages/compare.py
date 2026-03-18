"""Compare page — side-by-side traces from both agents with inline evaluation."""
import json

import pandas as pd
import streamlit as st

from lib.trace_query import query_conversations, query_conversation_detail, _AGENT_ICONS
from lib.trace_evaluator import evaluate_trace

_SCORE_LABELS = {
    "routing_appropriateness": "Routing",
    "escalation_judgment": "Escalation",
    "specialist_alignment": "Specialist",
}
_REASON_KEYS = {
    "routing_appropriateness": "routing_reason",
    "escalation_judgment": "escalation_reason",
    "specialist_alignment": "specialist_reason",
}


def _score_badge(score) -> str:
    """Return a coloured emoji badge for a 1-5 score."""
    if score is None:
        return "—"
    color = "🟢" if score >= 4 else ("🟡" if score >= 3 else "🔴")
    return f"{color} {score}"


def _render_scores(result: dict):
    """Render three score columns with reason text."""
    cols = st.columns(3)
    for col, (score_key, label) in zip(cols, _SCORE_LABELS.items()):
        score = result.get(score_key)
        reason = result.get(_REASON_KEYS[score_key], "")
        with col:
            st.metric(label, _score_badge(score))
            if reason and not reason.startswith("Error:"):
                st.caption(reason)
            elif reason.startswith("Error:"):
                st.error(reason)


def render(workspace_id: str):
    st.title("🔍 Agent Comparison")
    st.caption("Traces from both agents side-by-side. Select any trace to run an AI quality evaluation.")

    # ── Controls ───────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 4])
    with ctrl1:
        hours = st.number_input("Look-back (hours)", min_value=1, max_value=168, value=24, step=1)
    with ctrl2:
        agent_filter = st.selectbox("Agent", ["Both", "🟠 AWS", "🔵 GCP"], index=0)
    with ctrl3:
        st.write("")
        if st.button("🔄 Refresh", use_container_width=False):
            st.cache_data.clear()
            st.session_state.pop("compare_df", None)

    # ── Load traces ────────────────────────────────────────────────────────────
    if "compare_df" not in st.session_state or st.session_state.get("compare_hours") != hours:
        with st.spinner("Querying App Insights…"):
            try:
                df = query_conversations(workspace_id, hours)
                st.session_state["compare_df"] = df
                st.session_state["compare_hours"] = hours
            except Exception as e:
                st.error(f"Failed to query traces: {e}")
                return
    else:
        df = st.session_state["compare_df"]

    if df.empty:
        st.info(f"No conversations found in the last {hours}h.")
        return

    # Apply agent filter
    if agent_filter == "🟠 AWS":
        df = df[df["agent"] == "aws"]
    elif agent_filter == "🔵 GCP":
        df = df[df["agent"] == "gcp"]

    # ── Summary metrics ────────────────────────────────────────────────────────
    aws_count = int((df["agent"] == "aws").sum())
    gcp_count = int((df["agent"] == "gcp").sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🟠 AWS Traces", aws_count)
    m2.metric("🔵 GCP Traces", gcp_count)
    m3.metric("Total", len(df))
    avg_ms = df["duration_ms"].dropna()
    m4.metric("Avg Duration", f"{avg_ms.mean():.0f} ms" if not avg_ms.empty else "—")

    # ── Trace table ────────────────────────────────────────────────────────────
    st.subheader("Conversations")

    # Build display dataframe
    display = df.copy()
    display["Agent"] = display["agent"].map(lambda a: _AGENT_ICONS.get(a, a))
    display["Time"] = pd.to_datetime(display["timestamp"]).dt.strftime("%b %d, %H:%M")
    display["Query"] = display["query"].fillna("").str.slice(0, 80)
    display["Type"] = display["query_type"].fillna("—")
    display["Handled By"] = display["handled_by"].fillna("—")
    display["Duration (ms)"] = display["duration_ms"].apply(
        lambda v: f"{int(v):,}" if pd.notna(v) else "—"
    )

    st.dataframe(
        display[["Time", "Agent", "Query", "Type", "Handled By", "Duration (ms)"]],
        use_container_width=True,
        hide_index=True,
    )

    # ── Evaluate a trace ───────────────────────────────────────────────────────
    st.subheader("Evaluate a Trace")
    st.caption("Pick a conversation and run an AI quality evaluation on its full trace.")

    if "eval_results" not in st.session_state:
        st.session_state["eval_results"] = {}

    # Build selectbox options: label → operation_id mapping
    def _label(row) -> str:
        ts = pd.to_datetime(row["timestamp"]).strftime("%b %d, %H:%M")
        icon = _AGENT_ICONS.get(row["agent"], row["agent"])
        q = (row["query"] or "")[:60]
        return f"{ts} | {icon} | {q}"

    options = {_label(row): row["operation_id"] for _, row in df.iterrows()}
    if not options:
        st.info("No traces to evaluate.")
        return

    selected_label = st.selectbox("Select trace", list(options.keys()), label_visibility="collapsed")
    selected_op_id = options[selected_label]

    # Show selected trace details
    sel = df[df["operation_id"] == selected_op_id].iloc[0]
    with st.expander("Trace details", expanded=False):
        c1, c2 = st.columns(2)
        c1.markdown(f"**Agent:** {_AGENT_ICONS.get(sel['agent'], sel['agent'])}")
        c1.markdown(f"**Query type:** `{sel.get('query_type') or '—'}`")
        c1.markdown(f"**Handled by:** {sel.get('handled_by') or '—'}")
        c2.markdown(f"**Duration:** {int(sel['duration_ms']):,} ms" if pd.notna(sel.get('duration_ms')) else "**Duration:** —")
        c2.markdown(f"**Operation ID:** `{selected_op_id[:24]}…`")
        st.markdown("**Query:**")
        st.markdown(f"> {sel.get('query', '—')}")
        st.markdown("**Response:**")
        st.markdown(f"> {sel.get('response', '—')}")

    # Evaluate button
    already_evaluated = selected_op_id in st.session_state["eval_results"]
    btn_label = "✅ Re-evaluate" if already_evaluated else "🔍 Evaluate this trace"

    if st.button(btn_label, type="primary"):
        with st.spinner("Fetching LLM call spans and running evaluation…"):
            # Build full trace dict for the evaluator
            llm_calls = query_conversation_detail(workspace_id, selected_op_id)
            trace_dict = {
                "query": sel.get("query", ""),
                "response": sel.get("response", ""),
                "query_type": sel.get("query_type", "unknown"),
                "handled_by": sel.get("handled_by", ""),
                "needs_escalation": str(sel.get("needs_escalation", "")).lower() == "true",
                "agent": sel.get("agent", ""),
                "duration_ms": sel.get("duration_ms"),
                "llm_calls": llm_calls,
            }
            try:
                result = evaluate_trace(trace_dict)
                st.session_state["eval_results"][selected_op_id] = result
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

    # Show results if available
    if selected_op_id in st.session_state["eval_results"]:
        result = st.session_state["eval_results"][selected_op_id]
        st.markdown("#### Evaluation Results")
        _render_scores(result)

    # ── Comparison summary (if both agents have evaluated traces) ──────────────
    eval_results = st.session_state["eval_results"]
    aws_scores = [v for op, v in eval_results.items()
                  if df[df["operation_id"] == op]["agent"].values[:1] == ["aws"]]
    gcp_scores = [v for op, v in eval_results.items()
                  if df[df["operation_id"] == op]["agent"].values[:1] == ["gcp"]]

    if aws_scores and gcp_scores:
        st.subheader("Evaluated Traces Summary")
        _render_comparison_table(aws_scores, gcp_scores)


def _avg_score(scores_list: list[dict], key: str) -> float | None:
    vals = [s[key] for s in scores_list if isinstance(s.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None


def _render_comparison_table(aws_scores: list[dict], gcp_scores: list[dict]):
    rows = []
    for score_key, label in _SCORE_LABELS.items():
        aws_avg = _avg_score(aws_scores, score_key)
        gcp_avg = _avg_score(gcp_scores, score_key)
        delta = (gcp_avg - aws_avg) if (aws_avg is not None and gcp_avg is not None) else None
        rows.append({
            "Dimension": label,
            "🟠 AWS (avg)": f"{aws_avg:.2f}" if aws_avg is not None else "—",
            "🔵 GCP (avg)": f"{gcp_avg:.2f}" if gcp_avg is not None else "—",
            "Δ (GCP − AWS)": (f"{delta:+.2f}" if delta is not None else "—"),
        })

    st.caption(f"Averaged over {len(aws_scores)} AWS and {len(gcp_scores)} GCP evaluated trace(s).")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
