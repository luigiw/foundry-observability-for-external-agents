"""LangGraph workflow for customer support multi-agent system."""
from langgraph.graph import StateGraph, END
from ..models import AgentState
from ..agents import CustomerSupportAgents


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


def build_support_graph(agents: CustomerSupportAgents) -> StateGraph:
    """
    Build the customer support graph with the given agents.
    
    Args:
        agents: CustomerSupportAgents instance with LLM provider configured
        
    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("Router Agent", agents.router_agent)
    workflow.add_node("Billing Specialist", agents.billing_specialist)
    workflow.add_node("Technical Specialist", agents.technical_specialist)
    workflow.add_node("General Specialist", agents.general_specialist)
    workflow.add_node("Escalation Handler", agents.escalation_handler)
    
    # Set entry point
    workflow.set_entry_point("Router Agent")
    
    # Add conditional routing
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
