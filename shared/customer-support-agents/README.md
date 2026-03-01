# Customer Support Agents - Shared Package

Cloud-agnostic customer support multi-agent system built with LangGraph.

## Overview

This package contains the core business logic for a multi-agent customer support system that can be deployed on any cloud platform. The agents are designed to be platform-agnostic, with cloud-specific implementations providing the LLM integration.

## Architecture

```
┌─────────────────┐
│  Router Agent   │ ← Classifies incoming queries
└────────┬────────┘
         │
    ┌────┴────┬─────────────┬──────────────┐
    │         │             │              │
┌───▼───┐ ┌──▼──┐ ┌────────▼────┐ ┌───────▼────────┐
│Billing│ │Tech │ │   General   │ │  Escalation    │
│Spec.  │ │Spec.│ │   Specialist│ │  Handler       │
└───────┘ └─────┘ └─────────────┘ └────────────────┘
```

## Agents

- **Router Agent**: Analyzes customer messages and routes to the appropriate specialist
- **Billing Specialist**: Handles payments, invoices, refunds, subscriptions
- **Technical Specialist**: Handles product issues, bugs, troubleshooting
- **General Specialist**: Handles FAQs, general inquiries
- **Escalation Handler**: Handles cases requiring human intervention

## Usage

### 1. Create an LLM Factory

The LLM factory is a function that creates LLM instances for each agent:

```python
def my_llm_factory(model_name: str, temperature: float):
    """
    Create an LLM instance.
    
    Args:
        model_name: "haiku" or "sonnet" (cloud implementations map to actual models)
        temperature: Temperature setting
        
    Returns:
        LLM instance with invoke() method
    """
    # Cloud-specific LLM initialization
    if model_name == "haiku":
        return ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", ...)
    else:  # sonnet
        return ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", ...)
```

### 2. Create Agents and Build Graph

```python
from customer_support_agents.agents import CustomerSupportAgents
from customer_support_agents.graph import build_support_graph

# Create agents with your LLM factory
agents = CustomerSupportAgents(llm_factory=my_llm_factory)

# Build the graph
graph = build_support_graph(agents)

# Invoke the graph
from langchain_core.messages import HumanMessage

initial_state = {
    "messages": [HumanMessage(content="I need help with my bill")],
    "query_type": "unknown",
    "confidence": 0.0,
    "needs_escalation": False,
    "customer_id": None,
    "handled_by": None,
    "final_response": None,
}

result = graph.invoke(initial_state)
print(result["final_response"])
```

## Cloud Implementations

See platform-specific implementations in:
- `/aws/langgraph-customer-support` - AWS Lambda + Bedrock
- `/gcp/langgraph-customer-support` - GCP Cloud Functions + Vertex AI

Each implementation provides:
1. Cloud-specific LLM factory
2. Tracing/observability integration
3. Deployment configuration
4. API Gateway/HTTP handler
