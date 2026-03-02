"""LangGraph workflow for customer support multi-agent system."""
from typing import TypedDict, Annotated, Any
from uuid import uuid4
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from .tracing import get_tracer_callbacks, get_azure_tracer, SERVICE_NAME
from .agents import (
    router_agent,
    billing_specialist,
    technical_specialist,
    general_specialist,
    escalation_handler,
    set_session_id,
)


class AgentState(TypedDict):
    """State that flows through the customer support graph."""
    messages: Annotated[list, add_messages]
    query_type: str
    confidence: float
    needs_escalation: bool
    customer_id: str | None
    handled_by: str | None
    final_response: str | None


def route_to_specialist(state: dict) -> str:
    """Conditional edge: route to appropriate specialist based on query type."""
    if state.get("needs_escalation"):
        return "escalation"

    routing_map = {
        "billing": "billing",
        "technical": "technical",
        "general": "general",
        "escalation": "escalation",
        "unknown": "general",
    }
    return routing_map.get(state.get("query_type", "unknown"), "general")


def build_graph() -> StateGraph:
    """Build and compile the customer support graph."""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("Router Agent", router_agent)
    workflow.add_node("Billing Specialist", billing_specialist)
    workflow.add_node("Technical Specialist", technical_specialist)
    workflow.add_node("General Specialist", general_specialist)
    workflow.add_node("Escalation Handler", escalation_handler)

    # Set entry point
    workflow.set_entry_point("Router Agent")

    # Conditional routing from router to specialists
    workflow.add_conditional_edges(
        "Router Agent",
        route_to_specialist,
        {
            "billing": "Billing Specialist",
            "technical": "Technical Specialist",
            "general": "General Specialist",
            "escalation": "Escalation Handler",
        }
    )

    # All specialists end after responding
    workflow.add_edge("Billing Specialist", END)
    workflow.add_edge("Technical Specialist", END)
    workflow.add_edge("General Specialist", END)
    workflow.add_edge("Escalation Handler", END)

    return workflow.compile()


# Create the compiled graph
customer_support_graph = build_graph()


def invoke_support(message: str, customer_id: str | None = None) -> dict:
    """Invoke the customer support graph with a message, with Azure AI tracing."""
    from langchain_core.messages import HumanMessage

    session_id = str(uuid4())
    set_session_id(session_id)

    initial_state = {
        "messages": [HumanMessage(content=message)],
        "customer_id": customer_id,
        "query_type": "unknown",
        "confidence": 0.0,
        "needs_escalation": False,
        "handled_by": None,
        "final_response": None,
    }

    workflow_metadata: dict[str, Any] = {
        "workflow_type": "customer_support",
        "customer_id": customer_id,
        "session_id": session_id,
        "thread_id": session_id,
        "gen_ai.agent.name": "Customer Support Workflow",
        "gen_ai.agent.id": f"customer_support_workflow_{session_id}",
        "gen_ai.agent.description": "Multi-agent customer support system with routing and specialist agents",
        "gen_ai.conversation.id": session_id,
        "otel_agent_span": True,
        "service.name": SERVICE_NAME,
    }

    config: RunnableConfig = {
        "run_name": "Customer Support Workflow",
        "tags": ["customer-support", "multi-agent", "langgraph"],
        "metadata": workflow_metadata,
        "configurable": {"thread_id": session_id},
    }

    # Ensure tracing is initialised; agents.py adds callbacks per llm.invoke call
    get_azure_tracer()

    result = customer_support_graph.invoke(initial_state, config=config)

    return {
        "response": result["final_response"],
        "handled_by": result["handled_by"],
        "query_type": result["query_type"],
        "needs_escalation": result["needs_escalation"],
    }
