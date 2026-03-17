"""Query Application Insights for customer support agent traces.

Reconstructs evaluation-ready trace dicts from the existing OTEL span structure
emitted by both agents. Two span types are used as anchors:

  1. ``invoke_agent Customer Support Workflow`` — carries the user query
     (gen_ai.input.messages) and final response (gen_ai.output.messages).

  2. Any span with ``agent.query_type`` in customDimensions (e.g. ``POST /support``)
     — carries routing metadata: agent.query_type, agent.handled_by.

LLM call details (chat / invoke_agent spans with gen_ai content) are collected as
a ``llm_calls`` list and included in the trace dict for the TraceQualityEvaluator.

Authentication (tried in order):
  1. APPLICATIONINSIGHTS_QUERY_API_KEY env var  →  x-api-key header
  2. Azure CLI token  →  `az account get-access-token` bearer token

Requires env var:
    APPLICATIONINSIGHTS_APP_ID   — App Insights Application ID
                                   (Azure Portal → App Insights → API Access)

Usage (standalone):
    python query_app_insights.py --agent aws --hours 168
    python query_app_insights.py --agent gcp --hours 48 --output data/gcp_traces.jsonl
    python query_app_insights.py --agent both --hours 168 --output data/both_traces.jsonl
"""

import os
import json
import argparse
import subprocess
from pathlib import Path

import requests
from dotenv import load_dotenv

# cloud_RoleName values set by OTEL_SERVICE_NAME in each deployment
SERVICE_NAMES = {
    "aws": "aws-langgraph-customer-support",
    "gcp": "gcp-langgraph-customer-support",
}

# Simple query: reconstruct evaluation fields from workflow + routing spans only.
# Uses invoke_agent Customer Support Workflow as the anchor (has query + response),
# joined with any span carrying agent.query_type (routing metadata).
_KQL_SIMPLE = """
let workflow_spans =
    union dependencies, requests
    | where cloud_RoleName == "{service_name}"
    | where name == "invoke_agent Customer Support Workflow"
    | where timestamp > ago({hours}h)
    | project
        eval_timestamp   = timestamp,
        operation_Id,
        cloud_RoleName,
        session_id       = tostring(customDimensions["gen_ai.conversation.id"]),
        query            = tostring(parse_json(tostring(customDimensions["gen_ai.input.messages"]))[0].parts[0].content),
        response         = tostring(parse_json(tostring(customDimensions["gen_ai.output.messages"]))[0].parts[0].content),
        duration_ms      = duration / 10000;
let routing_spans =
    union dependencies, requests
    | where cloud_RoleName == "{service_name}"
    | where customDimensions has "agent.query_type"
    | where timestamp > ago({hours}h)
    | project
        operation_Id,
        query_type       = tostring(customDimensions["agent.query_type"]),
        handled_by       = tostring(customDimensions["agent.handled_by"]),
        needs_escalation = tostring(customDimensions["agent.needs_escalation"]);
workflow_spans
| join kind=leftouter routing_spans on operation_Id
| project-away operation_Id1
| order by eval_timestamp desc
""".strip()

# Enriched query: additionally collects all LLM call spans (chat / invoke_agent spans
# that have gen_ai content) and aggregates them as a llm_calls list per operation.
# Field names reflect what the Azure Monitor OTEL exporter actually writes:
#   gen_ai.input.messages / gen_ai.output.messages (not gen_ai.prompt / gen_ai.completion)
#   gen_ai.usage.input_tokens / gen_ai.usage.output_tokens
_KQL_WITH_LLM_CALLS = """
let workflow_spans =
    union dependencies, requests
    | where cloud_RoleName == "{service_name}"
    | where name == "invoke_agent Customer Support Workflow"
    | where timestamp > ago({hours}h)
    | project
        eval_timestamp   = timestamp,
        operation_Id,
        cloud_RoleName,
        session_id       = tostring(customDimensions["gen_ai.conversation.id"]),
        query            = tostring(parse_json(tostring(customDimensions["gen_ai.input.messages"]))[0].parts[0].content),
        response         = tostring(parse_json(tostring(customDimensions["gen_ai.output.messages"]))[0].parts[0].content),
        duration_ms      = duration / 10000;
let routing_spans =
    union dependencies, requests
    | where cloud_RoleName == "{service_name}"
    | where customDimensions has "agent.query_type"
    | where timestamp > ago({hours}h)
    | project
        operation_Id,
        query_type       = tostring(customDimensions["agent.query_type"]),
        handled_by       = tostring(customDimensions["agent.handled_by"]),
        needs_escalation = tostring(customDimensions["agent.needs_escalation"]);
let llm_spans =
    union dependencies, requests
    | where cloud_RoleName == "{service_name}"
    | where customDimensions has "gen_ai.system"
    | where isnotempty(tostring(customDimensions["gen_ai.output.messages"]))
    | where timestamp > ago({hours}h)
    | project
        operation_Id,
        llm_call = pack(
            "span_name",           name,
            "duration_ms",         duration / 10000,
            "gen_ai_system",       tostring(customDimensions["gen_ai.system"]),
            "gen_ai_operation",    tostring(customDimensions["gen_ai.operation.name"]),
            "gen_ai_model",        tostring(customDimensions["gen_ai.provider.name"]),
            "system_instructions", tostring(customDimensions["gen_ai.system_instructions"]),
            "input_messages",      tostring(customDimensions["gen_ai.input.messages"]),
            "output_message",      tostring(customDimensions["gen_ai.output.messages"]),
            "input_tokens",        toint(customDimensions["gen_ai.usage.input_tokens"]),
            "output_tokens",       toint(customDimensions["gen_ai.usage.output_tokens"])
        )
    | summarize llm_calls = make_list(llm_call) by operation_Id;
workflow_spans
| join kind=leftouter routing_spans on operation_Id
| join kind=leftouter llm_spans on operation_Id
| project-away operation_Id1, operation_Id2
| order by eval_timestamp desc
""".strip()


# ── Authentication ────────────────────────────────────────────────────────────


def _get_auth_headers() -> dict:
    """Return auth headers for the App Insights REST API.

    Tries APPLICATIONINSIGHTS_QUERY_API_KEY first; falls back to Azure CLI token.
    """
    api_key = os.environ.get("APPLICATIONINSIGHTS_QUERY_API_KEY", "").strip()
    if api_key:
        return {"x-api-key": api_key}

    # Fall back to Azure CLI bearer token
    result = subprocess.run(
        [
            "az", "account", "get-access-token",
            "--resource", "https://api.applicationinsights.io",
            "--query", "accessToken", "-o", "tsv",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return {"Authorization": f"Bearer {result.stdout.strip()}"}

    raise EnvironmentError(
        "No App Insights auth available.\n"
        "Set APPLICATIONINSIGHTS_QUERY_API_KEY in eval/.env, or run 'az login'."
    )


def _run_kql(app_id: str, kql: str) -> list[dict]:
    """Execute a KQL query against the App Insights REST API and return parsed rows."""
    url = f"https://api.applicationinsights.io/v1/apps/{app_id}/query"
    headers = {**_get_auth_headers(), "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"query": kql}, timeout=30)
    resp.raise_for_status()
    tables = resp.json().get("tables", [])
    if not tables:
        return []
    table = tables[0]
    columns = [col["name"] for col in table["columns"]]
    return [dict(zip(columns, row)) for row in table["rows"]]


# ── Public API ────────────────────────────────────────────────────────────────


def query_traces(agent: str, hours: int = 24, include_llm_calls: bool = True) -> list[dict]:
    """Query App Insights for customer support traces emitted by *agent*.

    Reconstructs trace dicts from the existing OTEL span structure
    (``invoke_agent Customer Support Workflow`` + routing spans + LLM call spans).

    Args:
        agent: ``"aws"``, ``"gcp"``, or ``"both"``
        hours: Lookback window in hours (default 24)
        include_llm_calls: When True (default), also collects sibling LLM call spans
            and includes them as ``llm_calls`` in each trace dict.

    Returns:
        List of trace dicts with keys: query, response, query_type, handled_by,
        needs_escalation, session_id, timestamp, duration_ms, operation_id, agent,
        and optionally llm_calls.
    """
    app_id = os.environ.get("APPLICATIONINSIGHTS_APP_ID", "").strip()
    if not app_id:
        raise EnvironmentError(
            "Missing APPLICATIONINSIGHTS_APP_ID.\n"
            "Set it in eval/.env — see .env.example for instructions."
        )

    kql_template = _KQL_WITH_LLM_CALLS if include_llm_calls else _KQL_SIMPLE
    agents = ["aws", "gcp"] if agent == "both" else [agent]
    traces: list[dict] = []

    for ag in agents:
        kql = kql_template.format(service_name=SERVICE_NAMES[ag], hours=hours)
        rows = _run_kql(app_id, kql)
        for row in rows:
            # needs_escalation: use the span attribute if present, else derive from
            # handled_by (Escalation Handler implies escalation was triggered).
            raw_esc = (row.get("needs_escalation") or "").lower()
            handled_by = row.get("handled_by", "") or ""
            needs_escalation = (
                raw_esc == "true" or handled_by == "Escalation Handler"
            )
            trace: dict = {
                "query": row.get("query", ""),
                "response": row.get("response", ""),
                "query_type": row.get("query_type", "unknown"),
                "handled_by": handled_by,
                "needs_escalation": needs_escalation,
                "session_id": row.get("session_id", ""),
                "timestamp": row.get("eval_timestamp", ""),
                "duration_ms": row.get("duration_ms"),
                "operation_id": row.get("operation_Id", ""),
                "agent": ag,
            }
            if include_llm_calls:
                raw_calls = row.get("llm_calls")
                trace["llm_calls"] = raw_calls if isinstance(raw_calls, list) else []
            traces.append(trace)

    return traces


def main():
    eval_dir = Path(__file__).parent
    load_dotenv(eval_dir / ".env")

    parser = argparse.ArgumentParser(
        description="Query App Insights for customer support agent traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--agent", choices=["aws", "gcp", "both"], required=True,
                        help="Which agent's traces to fetch.")
    parser.add_argument("--hours", type=int, default=24,
                        help="Lookback window in hours (default: 24).")
    parser.add_argument("--no-llm-calls", action="store_true",
                        help="Fetch evaluation span attributes only, skip LLM call spans.")
    parser.add_argument("--output", default=None,
                        help="Write traces to this JSONL file (default: print to stdout).")
    args = parser.parse_args()

    traces = query_traces(args.agent, args.hours, include_llm_calls=not args.no_llm_calls)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")
        print(f"Wrote {len(traces)} trace(s) to {args.output}")
    else:
        for t in traces:
            print(json.dumps(t, indent=2))
        print(f"\n{len(traces)} trace(s) found ({args.agent}, last {args.hours}h)",
              flush=True)


if __name__ == "__main__":
    main()
