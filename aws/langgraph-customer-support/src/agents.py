"""Agent nodes for the customer support system with Azure AI tracing."""
import json
from uuid import uuid4
from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from .tracing import get_azure_tracer, agent_span

# Session ID for the current request (set per invocation)
_current_session_id: str = str(uuid4())


def set_session_id(session_id: str) -> None:
    """Set the session ID for the current request."""
    global _current_session_id
    _current_session_id = session_id


def get_session_id() -> str:
    """Get the current session ID."""
    return _current_session_id


def get_llm(
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    name: str = "LLM",
    temperature: float = 0.1,
):
    """Get a Bedrock LLM instance."""
    llm = ChatBedrock(
        model_id=model_id,
        model_kwargs={"temperature": temperature, "max_tokens": 1024},
        region_name="us-east-1",
    )
    return llm


def router_agent(state: dict) -> dict:
    """Route incoming queries to the appropriate specialist."""
    agent_name = "Router Agent"
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    
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
    tracer = get_azure_tracer()
    invoke_config = {"callbacks": [tracer]} if tracer else {}
    with agent_span(agent_name, "Routes customer queries to appropriate specialist agents", get_session_id(), input_text=input_text) as span:
        response = llm.invoke(messages, config=invoke_config)
        span.set_attribute("gen_ai.output.messages", json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": response.content}], "finish_reason": "stop"}]))

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
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
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
    tracer = get_azure_tracer()
    invoke_config = {"callbacks": [tracer]} if tracer else {}
    with agent_span(agent_name, "Handles billing, payments, refunds, and subscription queries", get_session_id(), input_text=input_text) as span:
        response = llm.invoke(messages, config=invoke_config)
        span.set_attribute("gen_ai.output.messages", json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": response.content}], "finish_reason": "stop"}]))

    state["messages"] = state["messages"] + [AIMessage(content=response.content)]
    state["handled_by"] = agent_name
    state["final_response"] = response.content

    if any(phrase in response.content.lower() for phrase in ["escalate", "human agent", "supervisor"]):
        state["needs_escalation"] = True
    
    return state


def technical_specialist(state: dict) -> dict:
    """Handle technical support queries."""
    agent_name = "Technical Specialist"
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
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
    tracer = get_azure_tracer()
    invoke_config = {"callbacks": [tracer]} if tracer else {}
    with agent_span(agent_name, "Handles technical issues, troubleshooting, and product support", get_session_id(), input_text=input_text) as span:
        response = llm.invoke(messages, config=invoke_config)
        span.set_attribute("gen_ai.output.messages", json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": response.content}], "finish_reason": "stop"}]))

    state["messages"] = state["messages"] + [AIMessage(content=response.content)]
    state["handled_by"] = agent_name
    state["final_response"] = response.content

    if any(phrase in response.content.lower() for phrase in ["escalate", "engineering team", "investigate further"]):
        state["needs_escalation"] = True
    
    return state


def general_specialist(state: dict) -> dict:
    """Handle general inquiries."""
    agent_name = "General Specialist"
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    llm = get_llm(model_id, name=agent_name, temperature=0.4)
    
    system_prompt = """You are a general customer support agent. You help with:
- General inquiries about the company
- FAQs
- Directing customers to the right resources
- Friendly greetings and conversation

Be warm, helpful, and informative. Guide customers to the right department if needed."""
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    input_text = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    tracer = get_azure_tracer()
    invoke_config = {"callbacks": [tracer]} if tracer else {}
    with agent_span(agent_name, "Handles general inquiries, FAQs, and company information", get_session_id(), input_text=input_text) as span:
        response = llm.invoke(messages, config=invoke_config)
        span.set_attribute("gen_ai.output.messages", json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": response.content}], "finish_reason": "stop"}]))

    state["messages"] = state["messages"] + [AIMessage(content=response.content)]
    state["handled_by"] = agent_name
    state["final_response"] = response.content

    return state


def escalation_handler(state: dict) -> dict:
    """Handle cases that need human escalation."""
    agent_name = "Escalation Handler"
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    llm = get_llm(model_id, name=agent_name, temperature=0.2)
    
    system_prompt = """You are handling a case that needs human attention. Your job is to:
1. Acknowledge the customer's concerns empathetically
2. Explain that you're connecting them with a human specialist
3. Summarize the issue so the human agent has context
4. Provide an estimated wait time or next steps

Be calm, professional, and reassuring."""
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    input_text = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    tracer = get_azure_tracer()
    invoke_config = {"callbacks": [tracer]} if tracer else {}
    with agent_span(agent_name, "Handles escalation cases requiring human intervention", get_session_id(), input_text=input_text) as span:
        response = llm.invoke(messages, config=invoke_config)
        span.set_attribute("gen_ai.output.messages", json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": response.content}], "finish_reason": "stop"}]))

    state["messages"] = state["messages"] + [AIMessage(content=response.content)]
    state["handled_by"] = agent_name
    state["needs_escalation"] = True
    state["final_response"] = response.content
    
    return state
