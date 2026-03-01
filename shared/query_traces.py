#!/usr/bin/env python3
"""
Query Azure Application Insights for traces/spans from the customer support agents.

Agent names (cloud_RoleName in App Insights):
  gcp  -> gcp-langgraph-customer-support   (GCP Cloud Run)
  aws  -> aws-langgraph-customer-support   (AWS Lambda)
  all  -> no filter, shows both agents side-by-side

Usage:
  python query_traces.py                   # summary for all agents
  python query_traces.py --agent gcp       # GCP agent only
  python query_traces.py --agent aws       # AWS agent only
  python query_traces.py --agent gcp --query spans    # recent GenAI spans
  python query_traces.py --agent all --query compare  # side-by-side comparison
  python query_traces.py --minutes 60      # look back 60 minutes (default: 30)

Requires: az cli logged in  (az login)
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

# ── Agent name constants ────────────────────────────────────────────────────
AGENT_NAMES = {
    "gcp": "gcp-langgraph-customer-support",
    "aws": "aws-langgraph-customer-support",
}

# App Insights application ID (shared by both agents)
APP_ID = "e93e55ce-5468-4d9c-a532-8887871161ed"


# ── Helpers ─────────────────────────────────────────────────────────────────

def run_query(kql: str, minutes: int = 30) -> dict | None:
    """Execute a KQL query against Application Insights via az CLI."""
    cmd = [
        "az", "monitor", "app-insights", "query",
        "--app", APP_ID,
        "--analytics-query", kql,
        "--offset", f"PT{minutes}M",
        "--output", "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] az CLI failed:\n{result.stderr}", file=sys.stderr)
        return None
    return json.loads(result.stdout)


def print_table(data: dict | None, max_col_width: int = 60) -> None:
    """Print KQL result as an aligned table."""
    if not data or "tables" not in data:
        print("  (no data)")
        return
    table = data["tables"][0]
    columns = [col["name"] for col in table["columns"]]
    rows = table["rows"]
    if not rows:
        print("  (no rows)")
        return

    widths = []
    for i, col in enumerate(columns):
        w = len(col)
        for row in rows:
            w = max(w, min(len(str(row[i] or "")), max_col_width))
        widths.append(w)

    sep = "-+-".join("-" * w for w in widths)
    header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
    print(header)
    print(sep)
    for row in rows:
        cells = []
        for i, val in enumerate(row):
            s = str(val) if val is not None else ""
            if len(s) > max_col_width:
                s = s[:max_col_width - 3] + "..."
            cells.append(s.ljust(widths[i]))
        print(" | ".join(cells))


def role_filter(agent: str) -> str:
    """Return a KQL where-clause fragment for the given agent key."""
    if agent == "all":
        names = " or ".join(f'cloud_RoleName == "{n}"' for n in AGENT_NAMES.values())
        return f"| where {names}"
    name = AGENT_NAMES[agent]
    return f'| where cloud_RoleName == "{name}"'


# ── Queries ──────────────────────────────────────────────────────────────────

def query_summary(agent: str, minutes: int) -> None:
    """Overall activity: span counts and total tokens per agent."""
    print(f"\n{'='*60}")
    print("  SUMMARY — span counts & token usage per agent")
    print(f"{'='*60}\n")
    kql = f"""
dependencies
{role_filter(agent)}
| summarize
    spans      = count(),
    genai_spans = countif(type startswith "GenAI"),
    input_tok  = sum(toint(customDimensions["gen_ai.usage.input_tokens"])),
    output_tok = sum(toint(customDimensions["gen_ai.usage.output_tokens"]))
  by cloud_RoleName
| order by spans desc
"""
    print_table(run_query(kql, minutes))


def query_recent_spans(agent: str, minutes: int) -> None:
    """Recent GenAI spans with agent name, model, duration, tokens."""
    print(f"\n{'='*60}")
    print("  RECENT GenAI SPANS")
    print(f"{'='*60}\n")
    kql = f"""
dependencies
{role_filter(agent)}
| where type startswith "GenAI"
| extend
    agent   = tostring(customDimensions["gen_ai.agent.name"]),
    model   = tostring(customDimensions["gen_ai.request.model"]),
    in_tok  = toint(customDimensions["gen_ai.usage.input_tokens"]),
    out_tok = toint(customDimensions["gen_ai.usage.output_tokens"])
| project timestamp, cloud_RoleName, name, agent, model, duration, in_tok, out_tok
| order by timestamp desc
| take 25
"""
    print_table(run_query(kql, minutes))


def query_agent_nodes(agent: str, minutes: int) -> None:
    """Per-agent-node breakdown: call count, avg duration, total tokens."""
    print(f"\n{'='*60}")
    print("  AGENT NODE BREAKDOWN")
    print(f"{'='*60}\n")
    kql = f"""
dependencies
{role_filter(agent)}
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
  by cloud_RoleName, agent_node
| order by cloud_RoleName asc, calls desc
"""
    print_table(run_query(kql, minutes))


def query_compare(minutes: int) -> None:
    """Side-by-side comparison of AWS vs GCP for key metrics."""
    print(f"\n{'='*60}")
    print("  AWS vs GCP COMPARISON")
    print(f"{'='*60}\n")
    aws_name = AGENT_NAMES["aws"]
    gcp_name = AGENT_NAMES["gcp"]
    kql = f"""
dependencies
| where cloud_RoleName in ("{aws_name}", "{gcp_name}")
| where type startswith "GenAI"
| extend
    in_tok  = toint(customDimensions["gen_ai.usage.input_tokens"]),
    out_tok = toint(customDimensions["gen_ai.usage.output_tokens"])
| summarize
    requests   = count(),
    avg_ms     = round(avg(duration)),
    p95_ms     = round(percentile(duration, 95)),
    total_in   = sum(in_tok),
    total_out  = sum(out_tok)
  by cloud_RoleName
| order by cloud_RoleName asc
"""
    print_table(run_query(kql, minutes))


def query_errors(agent: str, minutes: int) -> None:
    """Failed spans per agent."""
    print(f"\n{'='*60}")
    print("  ERRORS")
    print(f"{'='*60}\n")
    kql = f"""
dependencies
{role_filter(agent)}
| where success == false
| extend agent_node = tostring(customDimensions["gen_ai.agent.name"])
| project timestamp, cloud_RoleName, name, agent_node, resultCode, duration
| order by timestamp desc
| take 20
"""
    print_table(run_query(kql, minutes))


# ── CLI ───────────────────────────────────────────────────────────────────────

QUERIES = {
    "summary":  query_summary,
    "spans":    query_recent_spans,
    "nodes":    query_agent_nodes,
    "errors":   query_errors,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query App Insights traces for customer support agents."
    )
    parser.add_argument(
        "--agent", choices=["gcp", "aws", "all"], default="all",
        help="Filter by agent (default: all)",
    )
    parser.add_argument(
        "--query", choices=list(QUERIES.keys()) + ["compare", "all"], default="all",
        help="Which query to run (default: all)",
    )
    parser.add_argument(
        "--minutes", type=int, default=30,
        help="Look-back window in minutes (default: 30)",
    )
    args = parser.parse_args()

    print(f"\nApp Insights query — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Agent filter : {args.agent}  |  Look-back : {args.minutes} min")
    print(f"Agent names  : {', '.join(f'{k}={v}' for k, v in AGENT_NAMES.items())}")

    if args.query == "compare":
        query_compare(args.minutes)
    elif args.query == "all":
        query_summary(args.agent, args.minutes)
        query_recent_spans(args.agent, args.minutes)
        query_agent_nodes(args.agent, args.minutes)
        query_errors(args.agent, args.minutes)
        if args.agent == "all":
            query_compare(args.minutes)
    else:
        fn = QUERIES[args.query]
        # query_compare doesn't take an agent arg
        fn(args.agent, args.minutes)


if __name__ == "__main__":
    main()
