"""HTTP client to invoke customer support agents."""
import requests


def invoke_agent(url: str, message: str, customer_id: str | None = None) -> dict:
    """Send a message to a customer support agent endpoint.

    Returns dict with 'response' (str) and 'metadata' (dict with
    handled_by, query_type, needs_escalation).
    """
    payload = {"message": message}
    if customer_id:
        payload["customer_id"] = customer_id

    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()
