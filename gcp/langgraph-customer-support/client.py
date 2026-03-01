"""Client to invoke the LangGraph Customer Support Agent API."""
import os
import json
import httpx

# Default to local; override with SUPPORT_API_URL env var for Cloud Run
API_ENDPOINT = os.environ.get("SUPPORT_API_URL", "http://localhost:8080/support")


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

    with httpx.Client(timeout=120) as client:
        response = client.post(
            API_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        return response.json()


def chat():
    """Interactive chat session with the agent."""
    print("=" * 60)
    print("LangGraph Customer Support Agent (GCP)")
    print(f"Endpoint: {API_ENDPOINT}")
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
        except httpx.HTTPError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        result = invoke_agent(message)
        print(json.dumps(result, indent=2))
    else:
        chat()
