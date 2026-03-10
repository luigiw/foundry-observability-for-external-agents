"""Query Azure Application Insights for agent traces via the Azure Monitor SDK."""
from datetime import timedelta

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient, LogsQueryStatus


_credential = DefaultAzureCredential()
_client = LogsQueryClient(_credential)


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
