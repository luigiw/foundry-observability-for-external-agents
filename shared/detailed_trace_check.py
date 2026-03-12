"""Detailed check of recent GCP traces."""
import os
import requests

APP_ID = os.environ.get("APPINSIGHTS_APP_ID", "e93e55ce-5468-4d9c-a532-8887871161ed")
API_KEY = os.environ.get("APPINSIGHTS_API_KEY", "")

query = """
dependencies
| where timestamp > ago(10m)
| where name contains "invoke_agent"
| summarize arg_min(timestamp, *) by id
| extend props = todynamic(customDimensions)
| project timestamp, name, type, target, duration, success, operation_Id, id, props
| order by timestamp desc
| take 5
"""

response = requests.get(
    f"https://api.applicationinsights.io/v1/apps/{APP_ID}/query",
    headers={"x-api-key": API_KEY},
    params={"query": query}
)

print(f"Status: {response.status_code}\n")

if response.status_code == 200:
    result = response.json()
    tables = result.get("tables", [])
    if tables:
        rows = tables[0].get("rows", [])
        cols = [c["name"] for c in tables[0].get("columns", [])]

        print("=== Recent invoke_agent Spans ===\n")
        for i, row in enumerate(rows, 1):
            data = dict(zip(cols, row))
            print(f"--- Span {i} ---")
            print(f"Timestamp: {data.get('timestamp')}")
            print(f"Name: {data.get('name')}")
            print(f"Type: {data.get('type')}")
            print(f"Duration: {data.get('duration')}ms")
            print(f"Operation ID: {data.get('operation_Id')}")
            print(f"Span ID: {data.get('id')}")
            print(f"Custom Dimensions/Properties:")

            props = data.get('props', {})
            if props:
                for key in sorted(props.keys()):
                    val = props[key]
                    if 'gen_ai' in key.lower() or 'agent' in key.lower() or 'model' in key.lower():
                        print(f"  {key}: {val}")
            else:
                print("  (none)")
            print()
else:
    print(f"Error: {response.text}")
