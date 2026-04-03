"""OpenTelemetry tracing for the Copilot SDK local agent.

Two modes, selected by environment:
  - OTLP (sidecar, default): sends all spans to the local OTel Collector
    (docker-compose.yml) which forwards to Azure Application Insights.
    Set OTEL_EXPORTER_OTLP_ENDPOINT (default: http://localhost:4318).
  - Direct Azure Monitor (fallback): set OTEL_EXPORTER=azure_monitor and
    APPLICATIONINSIGHTS_CONNECTION_STRING to bypass the collector.
"""
import logging
import os

logger = logging.getLogger(__name__)

SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "copilot-local-agent")
if "OTEL_SERVICE_NAME" not in os.environ:
    os.environ["OTEL_SERVICE_NAME"] = SERVICE_NAME

OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
EXPORTER_MODE = os.environ.get("OTEL_EXPORTER", "otlp").lower()  # "otlp" or "azure_monitor"

_provider_configured = False


def get_connection_string() -> str:
    conn = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
    if not conn:
        raise ValueError(
            "APPLICATIONINSIGHTS_CONNECTION_STRING is not set. "
            "Copy .env.example to .env and fill in the value."
        )
    return conn


def _build_otlp_exporter():
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    endpoint = OTLP_ENDPOINT.rstrip("/") + "/v1/traces"
    logger.info("OTLP exporter → %s", endpoint)
    return OTLPSpanExporter(endpoint=endpoint)


def _build_azure_monitor_exporter():
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
    logger.info("Azure Monitor exporter → App Insights (direct)")
    return AzureMonitorTraceExporter(connection_string=get_connection_string())


def setup_tracer_provider() -> None:
    """Configure the global TracerProvider. Safe to call multiple times.

    Uses OTLP exporter by default (requires the OTel Collector sidecar).
    Set OTEL_EXPORTER=azure_monitor to send directly to App Insights instead.
    """
    global _provider_configured
    if _provider_configured:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

        resource = Resource.create({"service.name": SERVICE_NAME})
        sampler = ParentBased(TraceIdRatioBased(1.0))
        provider = TracerProvider(resource=resource, sampler=sampler)

        if EXPORTER_MODE == "azure_monitor":
            exporter = _build_azure_monitor_exporter()
        else:
            exporter = _build_otlp_exporter()

        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _provider_configured = True
        logger.info(
            "TracerProvider configured (mode=%s, service=%s)",
            EXPORTER_MODE, SERVICE_NAME,
        )
    except Exception as exc:
        logger.warning("Failed to configure TracerProvider: %s", exc)


def get_tracer(name: str = __name__):
    """Return an OTel tracer. Call setup_tracer_provider() first."""
    from opentelemetry import trace
    return trace.get_tracer(name)


def flush_traces(timeout_millis: int = 30_000) -> None:
    """Flush pending spans.

    Calls force_flush() directly on each BatchSpanProcessor to avoid the
    known issue where QuickPulseSpanProcessor.force_flush() returns None,
    causing SynchronousMultiSpanProcessor to short-circuit.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = trace.get_tracer_provider()
        if not hasattr(provider, "_active_span_processor"):
            return

        flushed = False
        for sp in getattr(provider._active_span_processor, "_span_processors", []):
            if isinstance(sp, BatchSpanProcessor):
                sp.force_flush(timeout_millis)
                flushed = True

        if flushed:
            logger.info("Traces flushed successfully")
    except Exception as exc:
        logger.warning("Error flushing traces: %s", exc)
