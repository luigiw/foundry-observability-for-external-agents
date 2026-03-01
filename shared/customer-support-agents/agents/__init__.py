"""Agent node implementations for customer support system.

This module contains the business logic for each agent, decoupled from
cloud-specific implementations.
"""
from abc import ABC, abstractmethod
import json
from typing import Dict, Any, Callable
from langchain_core.messages import SystemMessage, AIMessage


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def invoke(self, messages: list, **kwargs) -> Any:
        """Invoke the LLM with messages."""
        pass


class CustomerSupportAgents:
    """Customer support agent implementations."""
    
    def __init__(self, llm_factory: Callable[[str, float], BaseLLMProvider]):
        """
        Initialize agents with an LLM factory function.
        
        Args:
            llm_factory: Function that takes (model_id, temperature) and returns an LLM instance
        """
        self.llm_factory = llm_factory
    
    def router_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Route incoming queries to the appropriate specialist."""
        llm = self.llm_factory("haiku", 0.2)
        
        system_prompt = """You are a customer support router. Analyze the customer's message and classify it.

Respond with ONLY a JSON object (no markdown, no explanation):
{"query_type": "billing|technical|general|escalation", "confidence": 0.0-1.0, "reasoning": "brief reason"}

Classification guide:
- billing: payments, invoices, refunds, subscription, pricing, charges
- technical: product issues, bugs, how-to, troubleshooting, errors
- general: FAQs, company info, general inquiries, greetings
- escalation: angry customer, legal threats, requests human, complex multi-issue"""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        
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
    
    def billing_specialist(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle billing-related queries."""
        llm = self.llm_factory("sonnet", 0.3)
        
        system_prompt = """You are a billing specialist for customer support. You help with:
- Payment questions and issues
- Invoice inquiries
- Refund requests
- Subscription management
- Pricing questions

Be helpful, empathetic, and concise. If you cannot resolve the issue, indicate that escalation to a human agent is needed.
Always maintain a professional and friendly tone."""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        
        state["messages"] = state["messages"] + [AIMessage(content=response.content)]
        state["handled_by"] = "Billing Specialist"
        state["final_response"] = response.content
        
        if any(phrase in response.content.lower() for phrase in ["escalate", "human agent", "supervisor"]):
            state["needs_escalation"] = True
        
        return state
    
    def technical_specialist(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle technical support queries."""
        llm = self.llm_factory("sonnet", 0.3)
        
        system_prompt = """You are a technical support specialist. You help with:
- Product troubleshooting
- Bug reports
- How-to questions
- Error resolution
- Technical configuration

Provide clear, step-by-step solutions when possible. If the issue requires investigation, acknowledge this and set expectations.
Be patient and thorough in your explanations."""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        
        state["messages"] = state["messages"] + [AIMessage(content=response.content)]
        state["handled_by"] = "Technical Specialist"
        state["final_response"] = response.content
        
        if any(phrase in response.content.lower() for phrase in ["escalate", "engineering team", "investigate further"]):
            state["needs_escalation"] = True
        
        return state
    
    def general_specialist(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general inquiries."""
        llm = self.llm_factory("sonnet", 0.4)
        
        system_prompt = """You are a general customer support agent. You help with:
- General inquiries about the company
- FAQs
- Directing customers to the right resources
- Friendly greetings and conversation

Be warm, helpful, and informative. Guide customers to the right department if needed."""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        
        state["messages"] = state["messages"] + [AIMessage(content=response.content)]
        state["handled_by"] = "General Specialist"
        state["final_response"] = response.content
        
        return state
    
    def escalation_handler(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cases that need human escalation."""
        llm = self.llm_factory("sonnet", 0.2)
        
        system_prompt = """You are handling a case that needs human attention. Your job is to:
1. Acknowledge the customer's concerns empathetically
2. Explain that you're connecting them with a human specialist
3. Summarize the issue so the human agent has context
4. Provide an estimated wait time or next steps

Be calm, professional, and reassuring."""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        
        state["messages"] = state["messages"] + [AIMessage(content=response.content)]
        state["handled_by"] = "Escalation Handler"
        state["needs_escalation"] = True
        state["final_response"] = response.content
        
        return state
