"""Query Azure Application Insights for agent traces via the Azure Monitor SDK."""
from datetime import timedelta

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient, LogsQueryStatus


_credential = DefaultAzureCredential()
_client = LogsQueryClient(_credential)

_AGENT_ROLES = {
    "aws": "aws-langgraph-customer-support",
    "gcp": "gcp-langgraph-customer-support",
}
_ALL_ROLES = list(_AGENT_ROLES.values())
_AGENT_ICONS = {"aws": "🟠 AWS", "gcp": "🔵 GCP"}


def _run_kql(workspace_id: str, kql: str, minutes: int = 30) -> pd.DataFrame:
    """Execute a KQL query and return results as a DataFrame."""
    response = _client.query_workspace(
        workspace_id=workspace_id,
        query=kql,
        timespan=timedelta(minutes=minutes),
    )
    if response.status == LogsQueryStatus.SUCCESS and response.tables:
        table = response.tables[0]
        return pd.DataFrame(data=table.rows, columns=[c.name for c in table.columns])
    return pd.DataFrame()


def _run_kql_hours(workspace_id: str, kql: str, hours: int) -> pd.DataFrame:
    """Execute a KQL query with an hours-based lookback window."""
    response = _client.query_workspace(
        workspace_id=workspace_id,
        query=kql,
        timespan=timedelta(hours=hours),
    )
    if response.status == LogsQueryStatus.SUCCESS and response.tables:
        table = response.tables[0]
        return pd.DataFrame(data=table.rows, columns=[c.name for c in table.columns])
    return pd.DataFrame()


def _role_filter(cloud_role_name: str) -> str:
    return f'| where cloud_RoleName == "{cloud_role_name}"'


def query_summary(workspace_id: str, cloud_role_name: str, minutes: int = 30) -> pd.DataFrame:
    """Span counts and total token usage."""
    kql = f"""
dependencies
{_role_filter(cloud_role_name)}
| summarize
    spans      = count(),
    genai_spans = countif(type startswith "GenAI"),
    input_tok  = sum(toint(customDimensions["gen_ai.usage.input_tokens"])),
    output_tok = sum(toint(customDimensions["gen_ai.usage.output_tokens"]))
  by cloud_RoleName
| order by spans desc
"""
    return _run_kql(workspace_id, kql, minutes)


def query_recent_spans(workspace_id: str, cloud_role_name: str, minutes: int = 30) -> pd.DataFrame:
    """Recent GenAI spans with agent name, model, duration, tokens."""
    kql = f"""
dependencies
{_role_filter(cloud_role_name)}
| where type startswith "GenAI"
| extend
    agent   = tostring(customDimensions["gen_ai.agent.name"]),
    model   = tostring(customDimensions["gen_ai.request.model"]),
    in_tok  = toint(customDimensions["gen_ai.usage.input_tokens"]),
    out_tok = toint(customDimensions["gen_ai.usage.output_tokens"])
| project timestamp, name, agent, model, duration, in_tok, out_tok
| order by timestamp desc
| take 50
"""
    return _run_kql(workspace_id, kql, minutes)


def query_agent_nodes(workspace_id: str, cloud_role_name: str, minutes: int = 30) -> pd.DataFrame:
    """Per-agent-node breakdown: call count, avg duration, total tokens."""
    kql = f"""
dependencies
{_role_filter(cloud_role_name)}
| where type startswith "GenAI"
| extend
    agent_node = tostring(customDimensions["gen_ai.agent.name"]),
    in_tok     = toint(customDimensions["gen_ai.usage.input_tokens"]),
    out_tok    = toint(customDimensions["gen_ai.usage.output_tokens"])
| summarize
    calls       = count(),
    avg_ms      = round(avg(duration)),
    total_in    = sum(in_tok),
    total_out   = sum(out_tok)
  by agent_node
| order by calls desc
"""
    return _run_kql(workspace_id, kql, minutes)


def query_errors(workspace_id: str, cloud_role_name: str, minutes: int = 30) -> pd.DataFrame:
    """Failed spans for the agent."""
    kql = f"""
dependencies
{_role_filter(cloud_role_name)}
| where success == false
| extend agent_node = tostring(customDimensions["gen_ai.agent.name"])
| project timestamp, name, agent_node, resultCode, duration
| order by timestamp desc
| take 20
"""
    return _run_kql(workspace_id, kql, minutes)


# ── Cross-agent conversation queries ─────────────────────────────────────────


def query_conversations(workspace_id: str, hours: int = 24) -> pd.DataFrame:
    """One row per end-to-end conversation for both agents.

    Anchors on ``invoke_agent Customer Support Workflow`` spans and left-joins
    routing metadata from the companion ``POST /support`` span.

    Returns columns: timestamp, agent, operation_id, query, response,
    query_type, handled_by, needs_escalation, duration_ms.
    """
    roles = ", ".join(f'"{r}"' for r in _ALL_ROLES)
    kql = f"""
let wf =
    union dependencies, requests
    | where cloud_RoleName in ({roles})
    | where name == "invoke_agent Customer Support Workflow"
    | project
        timestamp,
        operation_id  = operation_Id,
        agent         = iff(cloud_RoleName has "aws", "aws", "gcp"),
        query         = tostring(parse_json(tostring(customDimensions["gen_ai.input.messages"]))[0].parts[0].content),
        response      = tostring(parse_json(tostring(customDimensions["gen_ai.output.messages"]))[0].parts[0].content),
        duration_ms   = duration / 10000;
let routing =
    union dependencies, requests
    | where cloud_RoleName in ({roles})
    | where customDimensions has "agent.query_type"
    | project
        operation_Id,
        query_type       = tostring(customDimensions["agent.query_type"]),
        handled_by       = tostring(customDimensions["agent.handled_by"]),
        needs_escalation = tostring(customDimensions["agent.needs_escalation"]);
wf
| join kind=leftouter routing on $left.operation_id == $right.operation_Id
| project-away operation_Id
| order by timestamp desc
"""
    return _run_kql_hours(workspace_id, kql, hours)


def query_conversation_detail(workspace_id: str, operation_id: str) -> list[dict]:
    """Return the list of LLM call spans for a single conversation.

    Used to build the full trace dict passed to TraceQualityEvaluator.
    Looks back 7 days so any recent conversation can be retrieved by operation_id.
    """
    kql = f"""
union dependencies, requests
| where operation_Id == "{operation_id}"
| where customDimensions has "gen_ai.system"
| where isnotempty(tostring(customDimensions["gen_ai.output.messages"]))
| project
    span_name           = name,
    duration_ms         = duration / 10000,
    gen_ai_system       = tostring(customDimensions["gen_ai.system"]),
    gen_ai_operation    = tostring(customDimensions["gen_ai.operation.name"]),
    gen_ai_model        = tostring(customDimensions["gen_ai.provider.name"]),
    system_instructions = tostring(customDimensions["gen_ai.system_instructions"]),
    input_messages      = tostring(customDimensions["gen_ai.input.messages"]),
    output_message      = tostring(customDimensions["gen_ai.output.messages"]),
    input_tokens        = toint(customDimensions["gen_ai.usage.input_tokens"]),
    output_tokens       = toint(customDimensions["gen_ai.usage.output_tokens"])
"""
    df = _run_kql_hours(workspace_id, kql, hours=168)  # 7-day window
    if df.empty:
        return []
    return df.to_dict(orient="records")
