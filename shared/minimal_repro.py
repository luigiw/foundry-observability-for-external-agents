"""Minimal reproduction case for duplicate span issue."""
import os
from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

# Set your connection string
os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"] = "InstrumentationKey=320c6a3f-989f-4303-8bbe-f7682ddec3bc;IngestionEndpoint=https://eastus2-3.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus2.livediagnostics.monitor.azure.com/;ApplicationId=e93e55ce-5468-4d9c-a532-8887871161ed"

print("Initializing AzureAIOpenTelemetryTracer...")
tracer = AzureAIOpenTelemetryTracer(
    connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"],
    enable_content_recording=True,
    provider_name="aws.bedrock",
)

print("Creating LLM...")
llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    temperature=0.2,
    region_name="us-east-1",
)

print("Invoking LLM with tracer callback...")
config = {
    "callbacks": [tracer],
    "run_name": "Test Duplicate Spans",
    "metadata": {
        "test": "duplicate_spans",
    }
}

response = llm.invoke([HumanMessage(content="Say hello")], config=config)
print(f"Response: {response.content}")

print("\nDone! Wait 2-3 minutes, then check Azure App Insights:")
print("Query: dependencies | where timestamp > ago(5m) | where name contains 'chat'")
print("Expected: 1 span")
print("Actual: 2 identical spans (same ID, timestamp, duration)")
