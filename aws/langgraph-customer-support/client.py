"""Client to invoke the LangGraph Customer Support Agent API with OTel context propagation."""
import os
import json
import requests
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.propagate import inject

API_ENDPOINT = "https://6n9k7anskk.execute-api.us-east-1.amazonaws.com/prod/support"
API_KEY = os.environ.get("CUSTOMER_SUPPORT_API_KEY", "")

APPINSIGHTS_CONNECTION_STRING = os.environ.get(
    "APPLICATIONINSIGHTS_CONNECTION_STRING",
    "InstrumentationKey=320c6a3f-989f-4303-8bbe-f7682ddec3bc;IngestionEndpoint=https://eastus2-3.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus2.livediagnostics.monitor.azure.com/;ApplicationId=e93e55ce-5468-4d9c-a532-8887871161ed",
)

CLIENT_SERVICE_NAME = "customer-support-client"
_client_tracing_configured = False


def _setup_client_tracing():
    """Set up client-side TracerProvider with Azure Monitor exporter."""
    global _client_tracing_configured
    if _client_tracing_configured:
        return
    try:
        from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

        resource = Resource.create({"service.name": CLIENT_SERVICE_NAME})
        provider = TracerProvider(
            resource=resource,
            sampler=ParentBased(TraceIdRatioBased(1.0)),
        )
        exporter = AzureMonitorTraceExporter(
            connection_string=APPINSIGHTS_CONNECTION_STRING,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _client_tracing_configured = True
    except Exception as e:
        print(f"Warning: Failed to set up client tracing: {e}")


def _flush_client_traces(timeout_millis: int = 10000):
    """Flush pending client traces to Azure Monitor."""
    try:
        provider = trace.get_tracer_provider()
        if not hasattr(provider, "_active_span_processor"):
            return
        for sp in getattr(provider._active_span_processor, "_span_processors", []):
            if isinstance(sp, BatchSpanProcessor):
                sp.force_flush(timeout_millis)
    except Exception:
        pass


def invoke_agent(message: str, customer_id: str | None = None) -> dict:
    """Send a message to the customer support agent with OTel context propagation.

    Creates a CLIENT span and injects traceparent/tracestate/baggage headers
    so the Lambda-side spans appear as children of this invocation.
    """
    _setup_client_tracing()
    tracer = trace.get_tracer(CLIENT_SERVICE_NAME)

    with tracer.start_as_current_span(
        "invoke_agent",
        kind=trace.SpanKind.CLIENT,
        attributes={
            "peer.service": "aws-langgraph-customer-support",
            "http.method": "POST",
            "http.url": API_ENDPOINT,
        },
    ) as span:
        payload = {"message": message}
        if customer_id:
            payload["customer_id"] = customer_id

        headers = {"Content-Type": "application/json", "x-api-key": API_KEY}
        inject(headers)

        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=120,
        )

        span.set_attribute("http.status_code", response.status_code)
        response.raise_for_status()
        return response.json()


def chat():
    """Interactive chat session with the agent."""
    print("=" * 60)
    print("LangGraph Customer Support Agent")
    print("Type 'quit' or 'exit' to end the conversation")
    print("=" * 60)

    try:
        while True:
            try:
                message = input("\nYou: ").strip()
                if not message:
                    continue
                if message.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break

                print("Agent is thinking...")
                result = invoke_agent(message)

                print(f"\nAgent ({result['metadata']['handled_by']}): {result['response']}")
                print(f"  [Query type: {result['metadata']['query_type']}, "
                      f"Escalation needed: {result['metadata']['needs_escalation']}]")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except requests.exceptions.RequestException as e:
                print(f"Error: {e}")
    finally:
        _flush_client_traces()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        result = invoke_agent(message)
        print(json.dumps(result, indent=2))
        _flush_client_traces()
    else:
        chat()
