"""Azure Application Insights tracing using langchain-azure-ai with Gen AI semantic conventions."""
import os
import logging

logger = logging.getLogger(__name__)

_azure_tracer = None
_provider_configured = False

# OTel Gen AI semantic convention constants
PROVIDER_NAME = "anthropic"
# Read from OTEL_SERVICE_NAME so this agent is distinguishable in shared App Insights
# (sets cloud_RoleName in Application Insights)
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "gcp-langgraph-customer-support")

if "OTEL_SERVICE_NAME" not in os.environ:
    os.environ["OTEL_SERVICE_NAME"] = SERVICE_NAME


def get_connection_string() -> str | None:
    """Get Application Insights connection string from env."""
    return os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")


def _get_server_address() -> tuple[str, int] | None:
    """Extract server address and port from the Foundry resource env var."""
    resource = os.environ.get("AZURE_FOUNDRY_RESOURCE", "")
    if resource:
        return (f"{resource}.services.ai.azure.com", 443)
    return None


def _setup_tracer_provider():
    """Set up TracerProvider with AzureMonitorTraceExporter directly.

    This avoids configure_azure_monitor() which adds auto-instrumentation
    for urllib3/requests that causes duplicate spans in App Insights.
    """
    global _provider_configured
    if _provider_configured:
        return

    connection_string = get_connection_string()
    if not connection_string:
        logger.warning("No connection string; TracerProvider not configured")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
        from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

        resource = Resource.create({"service.name": SERVICE_NAME})
        sampler = ParentBased(TraceIdRatioBased(1.0))
        provider = TracerProvider(resource=resource, sampler=sampler)

        exporter = AzureMonitorTraceExporter(
            connection_string=connection_string,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))

        _GENAI_SPAN_NAMES = {"chat", "text_completion", "embeddings"}
        _GENAI_SPAN_PREFIXES = ("invoke_agent ", "execute_tool ", "chat ")

        class _GenAiSystemBridge(SpanProcessor):
            def on_start(self, span, parent_context=None):
                name = getattr(span, "name", "") or ""
                if name in _GENAI_SPAN_NAMES or any(
                    name.startswith(p) for p in _GENAI_SPAN_PREFIXES
                ):
                    span.set_attribute("gen_ai.system", PROVIDER_NAME)

            def on_end(self, span):
                pass

        provider.add_span_processor(_GenAiSystemBridge())
        trace.set_tracer_provider(provider)

        _provider_configured = True
        logger.info("TracerProvider configured with AzureMonitorTraceExporter")
    except Exception as e:
        logger.warning(f"Failed to configure TracerProvider: {e}")


def get_azure_tracer():
    """
    Get the Azure AI OpenTelemetry tracer for LangChain/LangGraph.

    Returns:
        AzureAIOpenTelemetryTracer instance to use as a callback
    """
    global _azure_tracer

    if _azure_tracer is not None:
        return _azure_tracer

    _setup_tracer_provider()

    try:
        from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer

        # Prevent the tracer from calling configure_azure_monitor() internally,
        # which would replace our TracerProvider with one that auto-instruments
        # urllib3/requests (causing duplicate spans in App Insights).
        AzureAIOpenTelemetryTracer._azure_monitor_configured = True
        _azure_tracer = AzureAIOpenTelemetryTracer(
            connection_string=get_connection_string(),
            name=SERVICE_NAME,
            provider_name=PROVIDER_NAME,
        )

        enable_content = os.environ.get("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true").lower() == "true"
        logger.info(f"Azure AI OpenTelemetry tracer initialized (content recording: {enable_content})")
    except Exception as e:
        logger.warning(f"Failed to initialize Azure AI tracer: {e}")

    return _azure_tracer


def flush_traces(timeout_millis: int = 30000):
    """Flush any pending traces to Azure Monitor.

    Calls force_flush directly on the BatchSpanProcessor to avoid
    issues with other processors short-circuiting the flush chain.
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
