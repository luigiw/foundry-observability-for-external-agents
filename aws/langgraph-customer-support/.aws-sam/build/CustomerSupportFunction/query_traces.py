#!/usr/bin/env python3
"""Query Azure Application Insights traces to verify agent instrumentation."""
import subprocess
import json
import sys
from datetime import datetime

# Application Insights App ID (from az monitor app-insights component show)
APP_ID = "e93e55ce-5468-4d9c-a532-8887871161ed"

def run_query(kql: str, timespan: str = "PT30M") -> dict:
    """Run a KQL query against Application Insights using Azure CLI."""
    cmd = [
        "az", "monitor", "app-insights", "query",
        "--app", APP_ID,
        "--analytics-query", kql,
        "--offset", timespan,
        "--output", "json"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return json.loads(result.stdout)


def print_table(data: dict, max_col_width: int = 50):
    """Print query results as a formatted table."""
    if not data or "tables" not in data:
        print("No data returned")
        return
    
    table = data["tables"][0]
    columns = [col["name"] for col in table["columns"]]
    rows = table["rows"]
    
    if not rows:
        print("No rows returned")
        return
    
    # Calculate column widths
    widths = []
    for i, col in enumerate(columns):
        max_width = len(col)
        for row in rows:
            val = str(row[i]) if row[i] is not None else ""
            max_width = max(max_width, min(len(val), max_col_width))
        widths.append(max_width)
    
    # Print header
    header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
    print(header)
    print("-" * len(header))
    
    # Print rows
    for row in rows:
        values = []
        for i, val in enumerate(row):
            s = str(val) if val is not None else ""
            if len(s) > max_col_width:
                s = s[:max_col_width-3] + "..."
            values.append(s.ljust(widths[i]))
        print(" | ".join(values))


def query_all_tables():
    """Check what data exists in all tables."""
    print("\n=== Data in All Tables (last 30 min) ===\n")
    kql = """
    union traces, dependencies, requests, customEvents, customMetrics
    | where timestamp > ago(30m)
    | summarize count() by itemType
    | order by count_ desc
    """
    data = run_query(kql)
    print_table(data)


def query_dependencies():
    """Query dependencies table for Gen AI traces."""
    print("\n=== Dependencies with Gen AI Attributes ===\n")
    kql = """
    dependencies
    | where timestamp > ago(30m)
    | extend 
        agent_name = tostring(customDimensions["gen_ai.agent.name"]),
        model = tostring(customDimensions["gen_ai.request.model"]),
        run_name = tostring(customDimensions["run_name"])
    | project timestamp, name, agent_name, model, run_name, duration
    | order by timestamp desc
    | take 20
    """
    data = run_query(kql)
    print_table(data)


def query_traces():
    """Query traces table."""
    print("\n=== Traces Table ===\n")
    kql = """
    traces
    | where timestamp > ago(30m)
    | extend 
        agent_name = tostring(customDimensions["gen_ai.agent.name"]),
        model = tostring(customDimensions["gen_ai.request.model"])
    | project timestamp, message, agent_name, model
    | order by timestamp desc
    | take 20
    """
    data = run_query(kql)
    print_table(data)


def query_custom_events():
    """Query custom events."""
    print("\n=== Custom Events ===\n")
    kql = """
    customEvents
    | where timestamp > ago(30m)
    | extend 
        agent_name = tostring(customDimensions["gen_ai.agent.name"]),
        model = tostring(customDimensions["gen_ai.request.model"])
    | project timestamp, name, agent_name, model
    | order by timestamp desc
    | take 20
    """
    data = run_query(kql)
    print_table(data)


def query_all_custom_dimensions():
    """Show all custom dimensions to see what's being captured."""
    print("\n=== All Custom Dimension Keys (from dependencies) ===\n")
    kql = """
    dependencies
    | where timestamp > ago(30m)
    | mv-expand customDimensions
    | extend key = tostring(bag_keys(customDimensions)[0])
    | summarize count() by key
    | order by count_ desc
    | take 30
    """
    data = run_query(kql)
    print_table(data)


def query_raw_dependencies():
    """Show raw dependency data."""
    print("\n=== Raw Dependencies (last 5) ===\n")
    kql = """
    dependencies
    | where timestamp > ago(30m)
    | project timestamp, name, type, target, data, duration, success
    | order by timestamp desc
    | take 5
    """
    data = run_query(kql)
    print_table(data)


def query_gen_ai_spans():
    """Query specifically for Gen AI semantic convention spans."""
    print("\n=== Gen AI Semantic Convention Attributes ===\n")
    kql = """
    dependencies
    | where timestamp > ago(30m)
    | where isnotempty(customDimensions["gen_ai.request.model"]) 
         or isnotempty(customDimensions["gen_ai.agent.name"])
         or name contains "ChatBedrock" 
         or name contains "LLM"
    | extend 
        agent_name = tostring(customDimensions["gen_ai.agent.name"]),
        agent_id = tostring(customDimensions["gen_ai.agent.id"]),
        model = tostring(customDimensions["gen_ai.request.model"]),
        provider = tostring(customDimensions["gen_ai.provider.name"]),
        conversation_id = tostring(customDimensions["gen_ai.conversation.id"]),
        input_tokens = toint(customDimensions["gen_ai.usage.input_tokens"]),
        output_tokens = toint(customDimensions["gen_ai.usage.output_tokens"])
    | project timestamp, name, agent_name, model, provider, duration
    | order by timestamp desc
    | take 20
    """
    data = run_query(kql)
    print_table(data)


def main():
    print("=" * 60)
    print("Azure Application Insights Trace Query")
    print(f"App ID: {APP_ID}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "all":
            query_all_tables()
        elif cmd == "deps":
            query_dependencies()
        elif cmd == "traces":
            query_traces()
        elif cmd == "events":
            query_custom_events()
        elif cmd == "keys":
            query_all_custom_dimensions()
        elif cmd == "raw":
            query_raw_dependencies()
        elif cmd == "genai":
            query_gen_ai_spans()
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python query_traces.py [all|deps|traces|events|keys|raw|genai]")
    else:
        # Run all queries
        query_all_tables()
        query_raw_dependencies()
        query_gen_ai_spans()


if __name__ == "__main__":
    main()
