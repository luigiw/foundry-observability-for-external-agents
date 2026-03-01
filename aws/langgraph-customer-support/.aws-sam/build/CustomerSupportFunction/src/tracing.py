"""Azure Application Insights tracing using langchain-azure-ai with Gen AI semantic conventions."""
import os
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Default connection string - can be overridden via environment variable
DEFAULT_CONNECTION_STRING = "InstrumentationKey=320c6a3f-989f-4303-8bbe-f7682ddec3bc;IngestionEndpoint=https://eastus2-3.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus2.livediagnostics.monitor.azure.com/;ApplicationId=e93e55ce-5468-4d9c-a532-8887871161ed"

_azure_tracer = None
_otel_tracer = None


def get_connection_string() -> str:
    """Get Application Insights connection string from env or default."""
    return os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", DEFAULT_CONNECTION_STRING)


def get_azure_tracer():
    """
    Get the Azure AI OpenTelemetry tracer for LangChain/LangGraph.
    
    This tracer implements the OpenTelemetry Gen AI semantic conventions
    and automatically traces LLM calls, tool invocations, and agent steps.
    
    Returns:
        AzureAIOpenTelemetryTracer instance to use as a callback
    """
    global _azure_tracer
    
    if _azure_tracer is not None:
        return _azure_tracer
    
    from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer
    
    connection_string = get_connection_string()
    enable_content = os.environ.get("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true").lower() == "true"
    
    # Create tracer with connection string - this was working at 8:46
    _azure_tracer = AzureAIOpenTelemetryTracer(
        connection_string=connection_string,
        enable_content_recording=enable_content,
    )
    
    logger.info(f"Azure AI OpenTelemetry tracer initialized (content recording: {enable_content})")
    
    return _azure_tracer


def get_otel_tracer():
    """Get OpenTelemetry tracer for creating custom spans."""
    global _otel_tracer
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
        span.set_attribute("gen_ai.provider.name", "aws.bedrock")  # Use standard value
        
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


def get_tracer_callbacks() -> list:
    """Get the list of tracer callbacks to pass to LangChain/LangGraph."""
    tracer = get_azure_tracer()
    return [tracer] if tracer else []


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
