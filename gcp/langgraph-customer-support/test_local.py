"""Local test script for the customer support agent with Azure AI tracing."""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

# Set the Application Insights connection string
os.environ.setdefault(
    "APPLICATIONINSIGHTS_CONNECTION_STRING",
    "InstrumentationKey=320c6a3f-989f-4303-8bbe-f7682ddec3bc;IngestionEndpoint=https://eastus2-3.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus2.livediagnostics.monitor.azure.com/;ApplicationId=e93e55ce-5468-4d9c-a532-8887871161ed"
)

# Enable content recording for Gen AI traces
os.environ.setdefault("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")

# Initialize tracer before importing graph
from src.tracing import get_azure_tracer, flush_traces
get_azure_tracer()

from src.graph import invoke_support


def test_billing():
    print("=" * 50)
    print("Testing BILLING query...")
    result = invoke_support("I was charged twice for my subscription last month. Can you help me get a refund?")
    print(f"Query Type: {result['query_type']}")
    print(f"Handled By: {result['handled_by']}")
    print(f"Response: {result['response'][:200]}...")
    print()


def test_technical():
    print("=" * 50)
    print("Testing TECHNICAL query...")
    result = invoke_support("The app keeps crashing when I try to upload files larger than 10MB. Error code: ERR_FILE_TOO_LARGE")
    print(f"Query Type: {result['query_type']}")
    print(f"Handled By: {result['handled_by']}")
    print(f"Response: {result['response'][:200]}...")
    print()


def test_general():
    print("=" * 50)
    print("Testing GENERAL query...")
    result = invoke_support("Hi! What are your business hours?")
    print(f"Query Type: {result['query_type']}")
    print(f"Handled By: {result['handled_by']}")
    print(f"Response: {result['response'][:200]}...")
    print()


def test_escalation():
    print("=" * 50)
    print("Testing ESCALATION query...")
    result = invoke_support("This is unacceptable! I've been waiting for 3 weeks and nobody has helped me. I want to speak to a manager immediately or I'm calling my lawyer!")
    print(f"Query Type: {result['query_type']}")
    print(f"Handled By: {result['handled_by']}")
    print(f"Needs Escalation: {result['needs_escalation']}")
    print(f"Response: {result['response'][:200]}...")
    print()


if __name__ == "__main__":
    print("LangGraph Customer Support Agent - Local Test with Azure AI Tracing")
    print("=" * 50)
    print("Using Anthropic API with langchain-azure-ai Gen AI semantic conventions")
    print()

    test_general()
    test_billing()

    print("\n" + "=" * 50)
    print("Tests complete! Flushing traces to Azure Application Insights...")
    flush_traces()
    print("Done. Check Application Insights for traces (cloud_RoleName=gcp-langgraph-customer-support).")
