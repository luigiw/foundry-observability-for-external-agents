"""Azure Application Insights tracing using OpenTelemetry Gen AI semantic conventions."""
import os
import logging
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

PROVIDER_NAME = "aws.bedrock"
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "aws-langgraph-customer-support")

_monitor_configured = False
_azure_tracer = None


def get_connection_string() -> str:
    """Get Application Insights connection string from env or default."""
    return os.environ.get(
        "APPLICATIONINSIGHTS_CONNECTION_STRING",
        "InstrumentationKey=320c6a3f-989f-4303-8bbe-f7682ddec3bc;IngestionEndpoint=https://eastus2-3.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus2.livediagnostics.monitor.azure.com/;ApplicationId=e93e55ce-5468-4d9c-a532-8887871161ed",
    )


def _configure_azure_monitor():
    """Configure Azure Monitor OpenTelemetry exporter."""
    global _monitor_configured
    if _monitor_configured:
        return

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor(
            connection_string=get_connection_string(),
            resource_attributes={"service.name": SERVICE_NAME},
        )

        # AzureAIOpenTelemetryTracer sets gen_ai.provider.name but the Azure Monitor
        # exporter reads gen_ai.system (standard OTel semconv) to set the dependency
        # Type column. This processor bridges the gap for all Gen AI spans.
        from opentelemetry import trace as _trace
        from opentelemetry.sdk.trace import SpanProcessor

        class _ProviderNameToSystemProcessor(SpanProcessor):
            def on_start(self, span, parent_context=None):
                attrs = span.attributes or {}
                if "gen_ai.provider.name" in attrs and "gen_ai.system" not in attrs:
                    span.set_attribute("gen_ai.system", attrs["gen_ai.provider.name"])
            def on_end(self, span):
                pass

        provider = _trace.get_tracer_provider()
        if hasattr(provider, "add_span_processor"):
            provider.add_span_processor(_ProviderNameToSystemProcessor())

        logger.info(f"Azure Monitor configured (service.name={SERVICE_NAME!r})")
        _monitor_configured = True
    except Exception as e:
        logger.warning(f"Failed to configure Azure Monitor: {e}")


def get_azure_tracer():
    """Get the AzureAIOpenTelemetryTracer for LangChain/LangGraph callbacks."""
    global _azure_tracer

    if _azure_tracer is not None:
        return _azure_tracer

    _configure_azure_monitor()

    try:
        from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer
        _azure_tracer = AzureAIOpenTelemetryTracer(
            name=SERVICE_NAME,
            provider_name=PROVIDER_NAME,
        )
        logger.info("AzureAIOpenTelemetryTracer initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize AzureAIOpenTelemetryTracer: {e}")

    return _azure_tracer


# Keep get_otel_tracer as an alias so test_local.py doesn't need changing
get_otel_tracer = get_azure_tracer


@contextmanager
def agent_span(agent_name: str, agent_description: str = None, session_id: str = None):
    """
    Context manager creating an invoke_agent span per OTel Gen AI semantic conventions.
    Wraps each agent node so it appears as a named parent span in App Insights.
    """
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, StatusCode

    tracer = trace.get_tracer(SERVICE_NAME)
    attributes: dict[str, Any] = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.provider.name": PROVIDER_NAME,
        "gen_ai.system": PROVIDER_NAME,  # read by Azure Monitor exporter for Type column
        "gen_ai.agent.name": agent_name,
        "gen_ai.agent.description": agent_description or agent_name,
    }
    if session_id:
        attributes["gen_ai.conversation.id"] = session_id
        attributes["gen_ai.agent.id"] = f"{agent_name.lower().replace(' ', '_')}_{session_id}"

    with tracer.start_as_current_span(
        f"invoke_agent {agent_name}", kind=SpanKind.INTERNAL, attributes=attributes
    ) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.set_attribute("error.type", type(e).__name__)
            span.record_exception(e)
            raise


def flush_traces():
    """Flush any pending traces to Azure Monitor."""
    try:
        from opentelemetry import trace
        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()
            logger.info("Traces flushed successfully")
    except Exception as e:
        logger.warning(f"Error flushing traces: {e}")
