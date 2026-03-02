"""Azure Application Insights tracing using langchain-azure-ai with Gen AI semantic conventions."""
import os
import json
import logging
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

_azure_tracer = None
_monitor_configured = False

# OTel Gen AI semantic convention constants
PROVIDER_NAME = "anthropic"
# Read from OTEL_SERVICE_NAME so this agent is distinguishable in shared App Insights
# (sets cloud_RoleName in Application Insights)
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "gcp-langgraph-customer-support")


def get_connection_string() -> str | None:
    """Get Application Insights connection string from env."""
    return os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")


def _get_server_address() -> tuple[str, int] | None:
    """Extract server address and port from the Foundry resource env var."""
    resource = os.environ.get("AZURE_FOUNDRY_RESOURCE", "")
    if resource:
        return (f"{resource}.services.ai.azure.com", 443)
    return None


def _configure_azure_monitor():
    """Configure Azure Monitor OpenTelemetry exporter."""
    global _monitor_configured
    if _monitor_configured:
        return

    try:
        connection_string = get_connection_string()
        if not connection_string:
            logger.warning("APPLICATIONINSIGHTS_CONNECTION_STRING not set, tracing disabled")
            return

        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor(
            connection_string=connection_string,
            resource_attributes={"service.name": SERVICE_NAME},
        )

        # AzureAIOpenTelemetryTracer sets gen_ai.provider.name but the Azure Monitor
        # exporter looks for gen_ai.system (standard OTel semconv) to set the
        # dependency Type column. This processor bridges the gap so all Gen AI spans
        # show "GenAI | anthropic" instead of "N/A".
        from opentelemetry import trace as _trace
        from opentelemetry.sdk.trace import SpanProcessor

        # azure-monitor-opentelemetry ignores `attributes=` at span creation time;
        # only set_attribute() calls work. AzureAIOpenTelemetryTracer sets
        # gen_ai.provider.name via attributes= (so it's dropped), and the Azure Monitor
        # exporter reads gen_ai.system to populate the Type column.
        # Fix: identify GenAI spans by name at on_start and set gen_ai.system directly.
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

        logger.info(f"Azure Monitor configured successfully (service.name={SERVICE_NAME!r})")
        _monitor_configured = True
    except Exception as e:
        logger.warning(f"Failed to configure Azure Monitor: {e}")


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

    # First configure Azure Monitor
    _configure_azure_monitor()

    try:
        from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer

        _azure_tracer = AzureAIOpenTelemetryTracer(
            name=SERVICE_NAME,
            provider_name=PROVIDER_NAME,
        )

        enable_content = os.environ.get("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true").lower() == "true"
        logger.info(f"Azure AI OpenTelemetry tracer initialized (content recording: {enable_content})")
    except Exception as e:
        logger.warning(f"Failed to initialize Azure AI tracer: {e}")

    return _azure_tracer


def get_tracer_callbacks() -> list:
    """Get the list of tracer callbacks to pass to LangChain/LangGraph."""
    tracer = get_azure_tracer()
    return [tracer] if tracer else []


@contextmanager
def invoke_agent_span(
    agent_name: str,
    *,
    agent_id: str | None = None,
    agent_description: str | None = None,
    conversation_id: str | None = None,
    input_text: str | None = None,
    request_model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_instructions: str | None = None,
):
    """
    Create an OTel span following Gen AI Agent semantic conventions.

    Spec: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/

    The span follows the invoke_agent convention:
    - span name: "invoke_agent {gen_ai.agent.name}"
    - span kind: INTERNAL (in-process LangGraph agent)
    - Required: gen_ai.operation.name, gen_ai.provider.name
    - Conditionally Required: gen_ai.agent.*, gen_ai.conversation.id, gen_ai.request.model

    Yields a dict that callers can update with response attributes
    (e.g., token usage) before the span closes.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.trace import StatusCode, SpanKind
    except ImportError:
        # OTel not available — yield a no-op dict
        result: dict[str, Any] = {}
        yield result
        return

    tracer = trace.get_tracer(SERVICE_NAME)

    # Required attributes (set at span creation for sampling)
    attributes: dict[str, Any] = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.provider.name": PROVIDER_NAME,
        # gen_ai.system is the standard OTel semconv attribute that Azure Monitor
        # exporter reads to set the dependency Type (e.g. "GenAI | anthropic")
        "gen_ai.system": PROVIDER_NAME,
    }

    # Conditionally Required
    if agent_name:
        attributes["gen_ai.agent.name"] = agent_name
    if agent_id:
        attributes["gen_ai.agent.id"] = agent_id
    if agent_description:
        attributes["gen_ai.agent.description"] = agent_description
    if conversation_id:
        attributes["gen_ai.conversation.id"] = conversation_id
    if request_model:
        attributes["gen_ai.request.model"] = request_model

    # Recommended
    if temperature is not None:
        attributes["gen_ai.request.temperature"] = temperature
    if max_tokens is not None:
        attributes["gen_ai.request.max_tokens"] = max_tokens

    # Server address
    server = _get_server_address()
    if server:
        attributes["server.address"] = server[0]
        attributes["server.port"] = server[1]

    # Opt-In: system instructions
    if system_instructions:
        attributes["gen_ai.system_instructions"] = json.dumps(
            [{"type": "text", "content": system_instructions}]
        )

    span_name = f"invoke_agent {agent_name}" if agent_name else "invoke_agent"
    result: dict[str, Any] = {}

    # Note: attributes= at span creation is ignored by azure-monitor-opentelemetry;
    # use set_attribute() inside the context instead.
    with tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
        for k, v in attributes.items():
            span.set_attribute(k, v)
        if input_text:
            span.add_event(
                "gen_ai.user.message",
                {"gen_ai.event.content": json.dumps({"role": "user", "content": input_text})},
            )
        try:
            yield result
            # Set response attributes after the agent runs
            if result.get("input_tokens"):
                span.set_attribute("gen_ai.usage.input_tokens", result["input_tokens"])
            if result.get("output_tokens"):
                span.set_attribute("gen_ai.usage.output_tokens", result["output_tokens"])
            if result.get("response_model"):
                span.set_attribute("gen_ai.response.model", result["response_model"])
            if result.get("finish_reasons"):
                span.set_attribute("gen_ai.response.finish_reasons", result["finish_reasons"])
            if result.get("output_text"):
                span.add_event(
                    "gen_ai.assistant.message",
                    {"gen_ai.event.content": json.dumps({"role": "assistant", "content": result["output_text"]})},
                )
            span.set_status(StatusCode.OK)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.set_attribute("error.type", type(e).__name__)
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
