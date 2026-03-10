"""Azure Application Insights tracing using OpenTelemetry Gen AI semantic conventions."""
import os
import logging
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

PROVIDER_NAME = "aws.bedrock"
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "aws-langgraph-customer-support")

# Ensure OTEL_SERVICE_NAME is in the environment so that
# AzureAIOpenTelemetryTracer's internal configure_azure_monitor() picks it up
# for the service.name resource attribute (cloud_RoleName in App Insights).
if "OTEL_SERVICE_NAME" not in os.environ:
    os.environ["OTEL_SERVICE_NAME"] = SERVICE_NAME

_azure_tracer = None


def get_connection_string() -> str:
    """Get Application Insights connection string from env or default."""
    return os.environ.get(
        "APPLICATIONINSIGHTS_CONNECTION_STRING",
        "InstrumentationKey=320c6a3f-989f-4303-8bbe-f7682ddec3bc;IngestionEndpoint=https://eastus2-3.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus2.livediagnostics.monitor.azure.com/;ApplicationId=e93e55ce-5468-4d9c-a532-8887871161ed",
    )


def _add_genai_system_processor():
    """Add a span processor that sets gen_ai.system on GenAI spans.

    AzureAIOpenTelemetryTracer sets gen_ai.provider.name via attributes= at span
    creation, but azure-monitor-opentelemetry ignores attributes= and only honours
    set_attribute(). The Azure Monitor exporter reads gen_ai.system to populate the
    dependency Type column. This processor bridges the gap.
    """
    try:
        from opentelemetry import trace as _trace
        from opentelemetry.sdk.trace import SpanProcessor

        _GENAI_SPAN_NAMES = {"chat", "text_completion", "embeddings"}
        _GENAI_SPAN_PREFIXES = ("invoke_agent ", "execute_tool ", "chat ")

        class _GenAiSystemProcessor(SpanProcessor):
            def on_start(self, span, parent_context=None):
                name = getattr(span, "name", "") or ""
                is_genai = name in _GENAI_SPAN_NAMES or any(
                    name.startswith(p) for p in _GENAI_SPAN_PREFIXES
                )
                if is_genai:
                    span.set_attribute("gen_ai.system", PROVIDER_NAME)
            def on_end(self, span):
                pass

        provider = _trace.get_tracer_provider()
        if hasattr(provider, "add_span_processor"):
            provider.add_span_processor(_GenAiSystemProcessor())
    except Exception as e:
        logger.warning(f"Failed to add GenAI system processor: {e}")


def get_azure_tracer():
    """Get the AzureAIOpenTelemetryTracer for LangChain/LangGraph callbacks."""
    global _azure_tracer

    if _azure_tracer is not None:
        return _azure_tracer

    try:
        from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer
        _azure_tracer = AzureAIOpenTelemetryTracer(
            connection_string=get_connection_string(),
            name=SERVICE_NAME,
            provider_name=PROVIDER_NAME,
        )
        # Add processor after the tracer has internally configured Azure Monitor
        _add_genai_system_processor()
        logger.info("AzureAIOpenTelemetryTracer initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize AzureAIOpenTelemetryTracer: {e}")

    return _azure_tracer


# Keep get_otel_tracer as an alias so test_local.py doesn't need changing
get_otel_tracer = get_azure_tracer


@contextmanager
def agent_span(
    agent_name: str,
    agent_description: str = None,
    session_id: str = None,
    input_text: str = None,
):
    """
    Context manager creating an invoke_agent span per OTel Gen AI semantic conventions.
    Wraps each agent node so it appears as a named parent span in App Insights.

    Emits gen_ai.user.message event for input_text if provided.
    Callers should emit gen_ai.assistant.message on the yielded span for output.
    """
    import json
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

    # Note: attributes= at span creation is ignored by azure-monitor-opentelemetry;
    # use set_attribute() inside the context instead.
    with tracer.start_as_current_span(
        f"invoke_agent {agent_name}", kind=SpanKind.INTERNAL
    ) as span:
        for k, v in attributes.items():
            span.set_attribute(k, v)
        if input_text:
            span.set_attribute(
                "gen_ai.input.messages",
                json.dumps([{"role": "user", "parts": [{"type": "text", "content": input_text}]}]),
            )
        try:
            yield span
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.set_attribute("error.type", type(e).__name__)
            span.record_exception(e)
            raise


def flush_traces(timeout_millis: int = 30000):
    """Flush any pending traces to Azure Monitor.

    We call force_flush directly on the BatchSpanProcessor rather than
    on the TracerProvider because azure-monitor-opentelemetry's
    _QuickpulseSpanProcessor.force_flush() returns None (falsy), which
    causes the SynchronousMultiSpanProcessor to short-circuit before
    reaching the BatchSpanProcessor.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = trace.get_tracer_provider()
        if not hasattr(provider, "_active_span_processor"):
            return

        flushed = False
        for sp in getattr(
            provider._active_span_processor, "_span_processors", []
        ):
            if isinstance(sp, BatchSpanProcessor):
                sp.force_flush(timeout_millis)
                flushed = True

        if flushed:
            logger.info("Traces flushed successfully")
    except Exception as e:
        logger.warning(f"Error flushing traces: {e}")
