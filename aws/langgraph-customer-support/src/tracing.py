"""Azure Application Insights tracing using OpenTelemetry Gen AI semantic conventions."""
import os
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Default connection string - can be overridden via environment variable
DEFAULT_CONNECTION_STRING = "InstrumentationKey=320c6a3f-989f-4303-8bbe-f7682ddec3bc;IngestionEndpoint=https://eastus2-3.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus2.livediagnostics.monitor.azure.com/;ApplicationId=e93e55ce-5468-4d9c-a532-8887871161ed"

_otel_configured = False
_otel_tracer = None


def get_connection_string() -> str:
    """Get Application Insights connection string from env or default."""
    return os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", DEFAULT_CONNECTION_STRING)


def _ensure_otel_configured():
    """Ensure OpenTelemetry is configured with Azure Monitor exporter."""
    global _otel_configured
    if _otel_configured:
        return
    
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    
    # Check if a TracerProvider is already set
    current_provider = trace.get_tracer_provider()
    if isinstance(current_provider, TracerProvider):
        # Already configured, just mark as done
        _otel_configured = True
        return
    
    # Create a new TracerProvider with service name
    resource = Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME", "customer-support-agents"),
        "service.version": "1.0.0",
    })
    provider = TracerProvider(resource=resource)
    
    # Add Azure Monitor exporter
    connection_string = get_connection_string()
    exporter = AzureMonitorTraceExporter(connection_string=connection_string)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    
    # Set as global provider
    trace.set_tracer_provider(provider)
    
    _otel_configured = True
    logger.info("OpenTelemetry configured with Azure Monitor exporter")


def get_otel_tracer():
    """Get OpenTelemetry tracer for creating custom spans."""
    global _otel_tracer
    
    _ensure_otel_configured()
    
    if _otel_tracer is None:
        from opentelemetry import trace
        _otel_tracer = trace.get_tracer("customer-support-agents", "1.0.0")
    return _otel_tracer


@contextmanager
def agent_span(agent_name: str, agent_description: str = None, session_id: str = None):
    """
    Context manager to create a named agent span with Gen AI semantic conventions.
    
    Follows OpenTelemetry Gen AI semantic conventions for invoke_agent spans:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
    
    Usage:
        with agent_span("Router Agent", "Routes queries to specialists", session_id):
            # Agent logic here
            response = llm.invoke(messages)
    """
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind
    
    tracer = get_otel_tracer()
    description = agent_description or agent_name
    
    # Span name follows convention: "invoke_agent {gen_ai.agent.name}"
    span_name = f"invoke_agent {agent_name}"
    
    # Create span with INTERNAL kind for in-process agents
    with tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
        # Required attributes per OTel Gen AI semantic conventions
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.provider.name", "aws.bedrock")
        # gen_ai.system is the standard OTel semconv attribute that Azure Monitor
        # exporter reads to set the dependency Type ("GenAI | aws.bedrock")
        span.set_attribute("gen_ai.system", "aws.bedrock")
        
        # Agent identification attributes
        span.set_attribute("gen_ai.agent.name", agent_name)
        span.set_attribute("gen_ai.agent.description", description)
        
        if session_id:
            span.set_attribute("gen_ai.conversation.id", session_id)
            span.set_attribute("gen_ai.agent.id", f"{agent_name.lower().replace(' ', '_')}_{session_id}")
        
        try:
            yield span
        except Exception as e:
            span.set_attribute("error.type", type(e).__name__)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def flush_traces():
    """Flush any pending traces."""
    try:
        from opentelemetry import trace
        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()
            logger.info("Traces flushed successfully")
    except Exception as e:
        logger.warning(f"Error flushing traces: {e}")
