"""AWS Lambda-specific graph wrapper with tracing."""
import sys
from pathlib import Path

# Add shared package to path
shared_path = Path(__file__).parent.parent.parent.parent / "shared"
sys.path.insert(0, str(shared_path))

from uuid import uuid4
from langchain_core.messages import HumanMessage
from customer_support_agents.graph import build_support_graph
from .agents_aws import get_agents
from .tracing import agent_span


_support_graph = None


def get_support_graph():
    """Get or create the customer support graph."""
    global _support_graph
    if _support_graph is None:
        agents = get_agents()
        _support_graph = build_support_graph(agents)
    return _support_graph


def invoke_support(message: str, customer_id: str | None = None) -> dict:
    """
    Invoke the customer support graph with tracing.
    
    Args:
        message: Customer's message
        customer_id: Optional customer ID
        
    Returns:
        dict with response, handled_by, query_type, needs_escalation
    """
    session_id = str(uuid4())
    
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "customer_id": customer_id,
        "query_type": "unknown",
        "confidence": 0.0,
        "needs_escalation": False,
        "handled_by": None,
        "final_response": None,
    }
    
    # Wrap in OTel span for observability
    with agent_span(
        "Customer Support Workflow",
        "Multi-agent customer support system with routing and specialist agents",
        session_id
    ):
        graph = get_support_graph()
        result = graph.invoke(initial_state)
    
    return {
        "response": result["final_response"],
        "handled_by": result["handled_by"],
        "query_type": result["query_type"],
        "needs_escalation": result["needs_escalation"],
    }
