"""Agent nodes for the customer support system with OTel Gen AI semantic conventions.

Uses Anthropic Claude models via Microsoft Foundry (Azure AI Foundry).
Set AZURE_FOUNDRY_RESOURCE and AZURE_FOUNDRY_API_KEY environment variables.

Spans follow: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
"""
import json
import os
from typing import Any
from uuid import uuid4
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .tracing import get_azure_tracer, invoke_agent_span, PROVIDER_NAME

# Session ID for the current request (set per invocation)
_current_session_id: str = str(uuid4())

# Microsoft Foundry endpoint
_FOUNDRY_RESOURCE = os.environ.get("AZURE_FOUNDRY_RESOURCE", "")
_FOUNDRY_BASE_URL = (
    f"https://{_FOUNDRY_RESOURCE}.services.ai.azure.com/anthropic/"
    if _FOUNDRY_RESOURCE
    else None
)


def set_session_id(session_id: str) -> None:
    """Set the session ID for the current request."""
    global _current_session_id
    _current_session_id = session_id


def get_session_id() -> str:
    """Get the current session ID."""
    return _current_session_id


def _agent_metadata(
    agent_name: str,
    *,
    agent_description: str | None = None,
    temperature: float = 0.1,
    model_id: str = "claude-sonnet-4-5",
) -> dict[str, Any]:
    """Build LangChain invoke metadata aligned with OTel Gen AI semantic conventions."""
    session_id = get_session_id()
    description = agent_description or agent_name

    metadata: dict[str, Any] = {
        # Gen AI semantic conventions (Required + Conditionally Required)
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": PROVIDER_NAME,
        "gen_ai.agent.name": agent_name,
        "gen_ai.agent.id": f"{agent_name.lower().replace(' ', '_')}_{session_id}",
        "gen_ai.agent.description": description,
        "gen_ai.request.model": model_id,
        "gen_ai.conversation.id": session_id,
        # Recommended
        "gen_ai.request.temperature": temperature,
        "gen_ai.request.max_tokens": 1024,
        "gen_ai.output.type": "text",
    }
    return metadata


def _extract_token_usage(response) -> dict[str, Any]:
    """Extract token usage from an LLM response for OTel span attributes."""
    usage: dict[str, Any] = {}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        if hasattr(um, "input_tokens") and um.input_tokens:
            usage["input_tokens"] = um.input_tokens
        if hasattr(um, "output_tokens") and um.output_tokens:
            usage["output_tokens"] = um.output_tokens
    if hasattr(response, "response_metadata") and response.response_metadata:
        rm = response.response_metadata
        if rm.get("model"):
            usage["response_model"] = rm["model"]
        if rm.get("stop_reason"):
            usage["finish_reasons"] = [rm["stop_reason"]]
    return usage


def get_llm(
    model_id: str = "claude-sonnet-4-5",
    name: str = "LLM",
    temperature: float = 0.1,
):
    """Get an Anthropic LLM instance via Microsoft Foundry."""
    kwargs: dict[str, Any] = {
        "model": model_id,
        "temperature": temperature,
        "max_tokens": 1024,
        "tags": [f"agent:{name}", "customer-support"],
        "metadata": {
            "agent_name": name,
            "session_id": get_session_id(),
            "ls_model_name": model_id,
            "ls_temperature": temperature,
        },
    }

    # Point to Microsoft Foundry endpoint if configured
    if _FOUNDRY_BASE_URL:
        kwargs["base_url"] = _FOUNDRY_BASE_URL
        # Use Foundry API key (falls back to ANTHROPIC_API_KEY if not set)
        foundry_key = os.environ.get("AZURE_FOUNDRY_API_KEY")
        if foundry_key:
            kwargs["api_key"] = foundry_key

    return ChatAnthropic(**kwargs)


def router_agent(state: dict) -> dict:
    """Route incoming queries to the appropriate specialist."""
    agent_name = "Router Agent"
    model_id = "claude-haiku-4-5"
    llm = get_llm(model_id, name=agent_name, temperature=0.2)

    system_prompt = """You are a customer support router. Analyze the customer's message and classify it.

Respond with ONLY a JSON object (no markdown, no explanation):
{"query_type": "billing|technical|general|escalation", "confidence": 0.0-1.0, "reasoning": "brief reason"}

Classification guide:
- billing: payments, invoices, refunds, subscription, pricing, charges
- technical: product issues, bugs, how-to, troubleshooting, errors
- general: FAQs, company info, general inquiries, greetings
- escalation: angry customer, legal threats, requests human, complex multi-issue"""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    input_text = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")

    # Build invoke config with agent metadata and tracer
    metadata = _agent_metadata(
        agent_name,
        agent_description="Routes customer queries to appropriate specialist agents",
        temperature=0.2,
        model_id=model_id,
    )
    invoke_config: dict[str, Any] = {
        "metadata": metadata,
        "run_name": agent_name,
        "tags": [f"agent:{agent_name}", "router"],
    }
    tracer = get_azure_tracer()
    if tracer:
        invoke_config["callbacks"] = [tracer]

    with invoke_agent_span(
        agent_name,
        agent_description="Routes customer queries to appropriate specialist agents",
        conversation_id=get_session_id(),
        input_text=input_text,
        request_model=model_id,
        temperature=0.2,
    ) as span_result:
        response = llm.invoke(messages, config=invoke_config)
        span_result.update({**_extract_token_usage(response), "output_text": response.content})

    try:
        result = json.loads(response.content)
        state["query_type"] = result.get("query_type", "general")
        state["confidence"] = result.get("confidence", 0.5)

        # Auto-escalate if confidence is too low
        if state["confidence"] < 0.4:
            state["needs_escalation"] = True
    except json.JSONDecodeError:
        state["query_type"] = "general"
        state["confidence"] = 0.5

    return state


def billing_specialist(state: dict) -> dict:
    """Handle billing-related queries."""
    agent_name = "Billing Specialist"
    model_id = "claude-sonnet-4-5"
    llm = get_llm(model_id, name=agent_name, temperature=0.3)

    system_prompt = """You are a billing specialist for customer support. You help with:
- Payment questions and issues
- Invoice inquiries
- Refund requests
- Subscription management
- Pricing questions

Be helpful, empathetic, and concise. If you cannot resolve the issue, indicate that escalation to a human agent is needed.
Always maintain a professional and friendly tone."""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    input_text = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")

    metadata = _agent_metadata(
        agent_name,
        agent_description="Handles billing, payments, refunds, and subscription queries",
        temperature=0.3,
        model_id=model_id,
    )
    invoke_config: dict[str, Any] = {
        "metadata": metadata,
        "run_name": agent_name,
        "tags": [f"agent:{agent_name}", "specialist", "billing"],
    }
    tracer = get_azure_tracer()
    if tracer:
        invoke_config["callbacks"] = [tracer]

    with invoke_agent_span(
        agent_name,
        agent_description="Handles billing, payments, refunds, and subscription queries",
        conversation_id=get_session_id(),
        input_text=input_text,
        request_model=model_id,
        temperature=0.3,
    ) as span_result:
        response = llm.invoke(messages, config=invoke_config)
        span_result.update({**_extract_token_usage(response), "output_text": response.content})

    state["messages"] = state["messages"] + [AIMessage(content=response.content)]
    state["handled_by"] = agent_name
    state["final_response"] = response.content

    if any(phrase in response.content.lower() for phrase in ["escalate", "human agent", "supervisor"]):
        state["needs_escalation"] = True

    return state


def technical_specialist(state: dict) -> dict:
    """Handle technical support queries."""
    agent_name = "Technical Specialist"
    model_id = "claude-sonnet-4-5"
    llm = get_llm(model_id, name=agent_name, temperature=0.3)

    system_prompt = """You are a technical support specialist. You help with:
- Product troubleshooting
- Bug reports
- How-to questions
- Error resolution
- Technical configuration

Provide clear, step-by-step solutions when possible. If the issue requires investigation, acknowledge this and set expectations.
Be patient and thorough in your explanations."""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    input_text = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")

    metadata = _agent_metadata(
        agent_name,
        agent_description="Handles technical issues, troubleshooting, and product support",
        temperature=0.3,
        model_id=model_id,
    )
    invoke_config: dict[str, Any] = {
        "metadata": metadata,
        "run_name": agent_name,
        "tags": [f"agent:{agent_name}", "specialist", "technical"],
    }
    tracer = get_azure_tracer()
    if tracer:
        invoke_config["callbacks"] = [tracer]

    with invoke_agent_span(
        agent_name,
        agent_description="Handles technical issues, troubleshooting, and product support",
        conversation_id=get_session_id(),
        input_text=input_text,
        request_model=model_id,
        temperature=0.3,
    ) as span_result:
        response = llm.invoke(messages, config=invoke_config)
        span_result.update({**_extract_token_usage(response), "output_text": response.content})

    state["messages"] = state["messages"] + [AIMessage(content=response.content)]
    state["handled_by"] = agent_name
    state["final_response"] = response.content

    if any(phrase in response.content.lower() for phrase in ["escalate", "engineering team", "investigate further"]):
        state["needs_escalation"] = True

    return state


def general_specialist(state: dict) -> dict:
    """Handle general inquiries."""
    agent_name = "General Specialist"
    model_id = "claude-sonnet-4-5"
    llm = get_llm(model_id, name=agent_name, temperature=0.4)

    system_prompt = """You are a general customer support agent. You help with:
- General inquiries about the company
- FAQs
- Directing customers to the right resources
- Friendly greetings and conversation

Be warm, helpful, and informative. Guide customers to the right department if needed."""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    input_text = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")

    metadata = _agent_metadata(
        agent_name,
        agent_description="Handles general inquiries, FAQs, and company information",
        temperature=0.4,
        model_id=model_id,
    )
    invoke_config: dict[str, Any] = {
        "metadata": metadata,
        "run_name": agent_name,
        "tags": [f"agent:{agent_name}", "specialist", "general"],
    }
    tracer = get_azure_tracer()
    if tracer:
        invoke_config["callbacks"] = [tracer]

    with invoke_agent_span(
        agent_name,
        agent_description="Handles general inquiries, FAQs, and company information",
        conversation_id=get_session_id(),
        input_text=input_text,
        request_model=model_id,
        temperature=0.4,
    ) as span_result:
        response = llm.invoke(messages, config=invoke_config)
        span_result.update({**_extract_token_usage(response), "output_text": response.content})

    state["messages"] = state["messages"] + [AIMessage(content=response.content)]
    state["handled_by"] = agent_name
    state["final_response"] = response.content

    return state


def escalation_handler(state: dict) -> dict:
    """Handle cases that need human escalation."""
    agent_name = "Escalation Handler"
    model_id = "claude-sonnet-4-5"
    llm = get_llm(model_id, name=agent_name, temperature=0.2)

    system_prompt = """You are handling a case that needs human attention. Your job is to:
1. Acknowledge the customer's concerns empathetically
2. Explain that you're connecting them with a human specialist
3. Summarize the issue so the human agent has context
4. Provide an estimated wait time or next steps

Be calm, professional, and reassuring."""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    input_text = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")

    metadata = _agent_metadata(
        agent_name,
        agent_description="Handles escalation cases requiring human intervention",
        temperature=0.2,
        model_id=model_id,
    )
    invoke_config: dict[str, Any] = {
        "metadata": metadata,
        "run_name": agent_name,
        "tags": [f"agent:{agent_name}", "escalation"],
    }
    tracer = get_azure_tracer()
    if tracer:
        invoke_config["callbacks"] = [tracer]

    with invoke_agent_span(
        agent_name,
        agent_description="Handles escalation cases requiring human intervention",
        conversation_id=get_session_id(),
        input_text=input_text,
        request_model=model_id,
        temperature=0.2,
    ) as span_result:
        response = llm.invoke(messages, config=invoke_config)
        span_result.update({**_extract_token_usage(response), "output_text": response.content})

    state["messages"] = state["messages"] + [AIMessage(content=response.content)]
    state["handled_by"] = agent_name
    state["needs_escalation"] = True
    state["final_response"] = response.content

    return state
