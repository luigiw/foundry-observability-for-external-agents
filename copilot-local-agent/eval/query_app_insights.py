"""Query Application Insights for Copilot SDK local agent traces.

Reconstructs evaluation-ready dicts from ``copilot.session.turn`` spans
emitted by src/agent.py (requires OTEL_CAPTURE_CONTENT=true) and joins
child ``tool.*`` spans to build the ``tools_used`` field.

Authentication (tried in order):
  1. APPLICATIONINSIGHTS_QUERY_API_KEY env var  →  x-api-key header
  2. Azure CLI token  →  ``az account get-access-token`` bearer token

Required env vars:
    APPLICATIONINSIGHTS_APP_ID   — App Insights Application ID
                                   (Azure Portal → App Insights → API Access)

Usage (standalone):
    python query_app_insights.py --hours 24
    python query_app_insights.py --hours 48 --output data/traces.jsonl
"""

import os
import json
import argparse
import subprocess
from pathlib import Path

import requests
from dotenv import load_dotenv

SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "copilot-local-agent")

_KQL = """
let turns =
    union dependencies, requests
    | where cloud_RoleName == "{service_name}"
    | where name == "copilot.session.turn"
    | where timestamp > ago({hours}h)
    | project
        operation_Id,
        timestamp,
        duration_ms      = duration,
        user_input       = tostring(customDimensions["user.input"]),
        response         = tostring(customDimensions["assistant.response"]),
        model            = tostring(customDimensions["gen_ai.request.model"]),
        agent_name       = tostring(customDimensions["gen_ai.agent.name"]);
let tool_spans =
    union dependencies, requests
    | where cloud_RoleName == "{service_name}"
    | where name startswith "tool."
    | where timestamp > ago({hours}h)
    | project operation_Id, tool_name = name;
let tools_per_turn =
    tool_spans
    | summarize tools_used = make_list(tool_name) by operation_Id;
turns
| join kind=leftouter tools_per_turn on operation_Id
| extend tools_used = iff(isempty(tools_used), dynamic([]), tools_used)
| project-away operation_Id1
| where isnotempty(user_input)
| order by timestamp desc
| limit {limit}
""".strip()


def _get_auth_headers() -> dict:
    api_key = os.environ.get("APPLICATIONINSIGHTS_QUERY_API_KEY", "").strip()
    if api_key:
        return {"x-api-key": api_key}

    result = subprocess.run(
        ["az", "account", "get-access-token",
         "--resource", "https://api.applicationinsights.io",
         "--query", "accessToken", "-o", "tsv"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return {"Authorization": f"Bearer {result.stdout.strip()}"}

    raise EnvironmentError(
        "No App Insights auth available.\n"
        "Set APPLICATIONINSIGHTS_QUERY_API_KEY in eval/.env, or run 'az login'."
    )


def _run_kql(app_id: str, kql: str) -> list[dict]:
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


def query_traces(hours: int = 24, limit: int = 100) -> list[dict]:
    """Query App Insights for ``copilot.session.turn`` spans.

    Returns evaluation-ready dicts with keys: user_input, response,
    tools_used (comma-separated string), model, duration_ms, timestamp,
    operation_id.

    Requires OTEL_CAPTURE_CONTENT=true to have been set when traces were
    generated; rows with empty user_input are filtered out.
    """
    app_id = os.environ.get("APPLICATIONINSIGHTS_APP_ID", "").strip()
    if not app_id:
        raise EnvironmentError(
            "Missing APPLICATIONINSIGHTS_APP_ID.\n"
            "Set it in eval/.env — see .env.example for instructions."
        )

    kql = _KQL.format(service_name=SERVICE_NAME, hours=hours, limit=limit)
    rows = _run_kql(app_id, kql)

    traces = []
    for row in rows:
        raw_tools = row.get("tools_used") or []
        # App Insights may return tools_used as a list or a JSON string
        if isinstance(raw_tools, str):
            try:
                raw_tools = json.loads(raw_tools)
            except json.JSONDecodeError:
                raw_tools = [t.strip() for t in raw_tools.split(",") if t.strip()]
        # Strip the "tool." prefix for readability (e.g. "tool.list_files" → "list_files")
        tools_clean = [t.replace("tool.", "") for t in raw_tools]

        traces.append({
            "user_input": row.get("user_input", ""),
            "response": row.get("response", ""),
            "tools_used": ",".join(tools_clean),
            "model": row.get("model", ""),
            "duration_ms": row.get("duration_ms"),
            "timestamp": row.get("timestamp", ""),
            "operation_id": row.get("operation_Id", ""),
        })

    return traces


def main():
    eval_dir = Path(__file__).parent
    load_dotenv(eval_dir.parent / ".env")
    load_dotenv(eval_dir / ".env", override=True)

    parser = argparse.ArgumentParser(
        description="Query App Insights for Copilot SDK agent traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--hours", type=int, default=24,
                        help="Lookback window in hours (default: 24).")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max traces to return (default: 100).")
    parser.add_argument("--output", default=None,
                        help="Write traces to this JSONL file (default: print to stdout).")
    args = parser.parse_args()

    traces = query_traces(args.hours, args.limit)

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
        print(f"\n{len(traces)} trace(s) found (last {args.hours}h)")


if __name__ == "__main__":
    main()
