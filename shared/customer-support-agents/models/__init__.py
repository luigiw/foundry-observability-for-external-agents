"""Data models for customer support agent state."""
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State that flows through the customer support graph."""
    messages: Annotated[list, add_messages]
    query_type: str
    confidence: float
    needs_escalation: bool
    customer_id: Optional[str]
    handled_by: Optional[str]
    final_response: Optional[str]


class QueryClassification(TypedDict):
    """Result of query classification."""
    query_type: str  # "billing", "technical", "general", "escalation"
    confidence: float
    reasoning: str
