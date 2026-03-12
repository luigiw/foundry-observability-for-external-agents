"""Check GCP agent traces in Azure Application Insights."""
import os
import requests

APP_ID = os.environ.get("APPINSIGHTS_APP_ID", "e93e55ce-5468-4d9c-a532-8887871161ed")
API_KEY = os.environ.get("APPINSIGHTS_API_KEY", "")
BASE_URL = f"https://api.applicationinsights.io/v1/apps/{APP_ID}"

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

query = """
dependencies
| where timestamp > ago(10m)
| where name contains "invoke_agent" or name contains "chat" or name contains "POST"
| summarize arg_min(timestamp, *) by id
| project timestamp, name, type, target, duration, success, customDimensions
| order by timestamp desc
| take 10
"""

response = requests.post(
    f"{BASE_URL}/query",
    headers=headers,
    json={"query": query}
)

if response.status_code == 200:
    result = response.json()
    rows = result.get("tables", [{}])[0].get("rows", [])
    columns = result.get("tables", [{}])[0].get("columns", [])

    print("=== Recent Dependencies (last 10 min) ===\n")
    for row in rows:
        data = dict(zip([c["name"] for c in columns], row))
        print(f"Timestamp: {data.get('timestamp')}")
        print(f"Name: {data.get('name')}")
        print(f"Type: {data.get('type')}")
        print(f"Target: {data.get('target')}")
        print(f"Duration: {data.get('duration')}ms")
        print(f"Custom Dimensions: {data.get('customDimensions')}")
        print("-" * 80)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
