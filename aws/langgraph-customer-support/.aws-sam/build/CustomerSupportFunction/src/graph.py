"""LangGraph workflow for customer support multi-agent system."""
from typing import TypedDict, Annotated, Any
from uuid import uuid4
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from .tracing import agent_span
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
        "unknown": "general",  # Default to general for unknown
    }
    return routing_map.get(state.get("query_type", "unknown"), "general")


def build_graph() -> StateGraph:
    """Build and compile the customer support graph."""
    
    # Create the graph with our state schema
    workflow = StateGraph(AgentState)
    
    # Add nodes with descriptive names
    workflow.add_node("Router Agent", router_agent)
    workflow.add_node("Billing Specialist", billing_specialist)
    workflow.add_node("Technical Specialist", technical_specialist)
    workflow.add_node("General Specialist", general_specialist)
    workflow.add_node("Escalation Handler", escalation_handler)
    
    # Set entry point
    workflow.set_entry_point("Router Agent")
    
    # Add conditional edges from router to specialists
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
    
    # Compile with a name for the graph
    return workflow.compile()


# Create the compiled graph
customer_support_graph = build_graph()


def invoke_support(message: str, customer_id: str | None = None) -> dict:
    """Invoke the customer support graph with a message, with Azure AI tracing."""
    from langchain_core.messages import HumanMessage
    from .tracing import agent_span
    
    # Generate a unique session ID for this conversation
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
    
    # Wrap the entire workflow in an OTel span with proper Gen AI conventions
    # We use our own agent_span instead of LangChain tracer to avoid duplicates
    with agent_span(
        "Customer Support Workflow",
        "Multi-agent customer support system with routing and specialist agents",
        session_id
    ):
        # Don't pass tracer callback - we're using OTel spans directly
        config: RunnableConfig = {
            "run_name": "Customer Support Workflow",
            "tags": ["customer-support", "multi-agent", "langgraph"],
            "metadata": {
                "workflow_type": "customer_support",
                "customer_id": customer_id,
                "session_id": session_id,
            },
            "configurable": {"thread_id": session_id},
        }
        
        result = customer_support_graph.invoke(initial_state, config=config)
    
    return {
        "response": result["final_response"],
        "handled_by": result["handled_by"],
        "query_type": result["query_type"],
        "needs_escalation": result["needs_escalation"],
    }
