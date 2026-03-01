"""State schema for the customer support multi-agent system."""
from typing import Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class AgentState(BaseModel):
    """State that flows through the customer support graph."""

    # Conversation messages with reducer for appending
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    # Current classification of the customer query
    query_type: Literal["billing", "technical", "general", "escalation", "unknown"] = "unknown"

    # Confidence score from router (0-1)
    confidence: float = 0.0

    # Whether the conversation should be escalated to human
    needs_escalation: bool = False

    # Customer context (could be enriched from CRM)
    customer_id: str | None = None

    # Track which specialist handled the query
    handled_by: str | None = None

    # Final response to send back
    final_response: str | None = None

    class Config:
        arbitrary_types_allowed = True
