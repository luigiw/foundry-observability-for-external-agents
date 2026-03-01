"""Client to invoke the LangGraph Customer Support Agent API."""
import os
import requests
import json

API_ENDPOINT = "https://6n9k7anskk.execute-api.us-east-1.amazonaws.com/prod/support"

# Azure Application Insights connection string (optional for local tracing)
APPINSIGHTS_CONNECTION_STRING = os.environ.get(
    "APPLICATIONINSIGHTS_CONNECTION_STRING",
    "InstrumentationKey=320c6a3f-989f-4303-8bbe-f7682ddec3bc;IngestionEndpoint=https://eastus2-3.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus2.livediagnostics.monitor.azure.com/;ApplicationId=e93e55ce-5468-4d9c-a532-8887871161ed"
)


def invoke_agent(message: str, customer_id: str | None = None) -> dict:
    """
    Send a message to the customer support agent.
    
    Args:
        message: The customer's message/query
        customer_id: Optional customer identifier
        
    Returns:
        Dict with response and metadata
    """
    payload = {"message": message}
    if customer_id:
        payload["customer_id"] = customer_id
    
    response = requests.post(
        API_ENDPOINT,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120  # Agent may take time to respond
    )
    response.raise_for_status()
    return response.json()


def chat():
    """Interactive chat session with the agent."""
    print("=" * 60)
    print("LangGraph Customer Support Agent")
    print("Type 'quit' or 'exit' to end the conversation")
    print("=" * 60)
    
    while True:
        try:
            message = input("\nYou: ").strip()
            if not message:
                continue
            if message.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            
            print("Agent is thinking...")
            result = invoke_agent(message)
            
            print(f"\nAgent ({result['metadata']['handled_by']}): {result['response']}")
            print(f"  [Query type: {result['metadata']['query_type']}, "
                  f"Escalation needed: {result['metadata']['needs_escalation']}]")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single message mode
        message = " ".join(sys.argv[1:])
        result = invoke_agent(message)
        print(json.dumps(result, indent=2))
    else:
        # Interactive chat mode
        chat()
