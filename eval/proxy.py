"""OpenAI-compatible proxy server with two modes:

1. Anthropic mode (default):
   Forwards to Azure AI Foundry Anthropic endpoint, translating OpenAI ↔ Anthropic format.
   Request flow: azure-ai-evaluation → (OpenAI) → proxy → (Anthropic) → Azure AI Foundry

2. GPT-5 pass-through mode:
   Forwards to Azure AI Foundry /openai/v1/ endpoint, replacing max_tokens →
   max_completion_tokens (required by GPT-5 which does not accept max_tokens).
   Request flow: azure-ai-evaluation → (OpenAI + max_tokens) → proxy → (OpenAI + max_completion_tokens) → GPT-5

The azure-ai-evaluation evaluators use the openai Python SDK internally and therefore
require an OpenAI-compatible /v1/chat/completions endpoint.

Usage (standalone):
    python proxy.py

Environment variables (read from eval/.env or shell):
    AZURE_FOUNDRY_RESOURCE   – Foundry resource name (e.g. my-resource)
    AZURE_FOUNDRY_API_KEY    – Foundry API key
    PROXY_MODEL              – Default model when none specified (default: claude-haiku-4-5)
    PROXY_PORT               – Local port (default: 4000)
    AZURE_GPT5_ENDPOINT      – GPT-5 endpoint base URL (for gpt5 mode)
    AZURE_GPT5_API_KEY       – GPT-5 API key (for gpt5 mode)
    AZURE_GPT5_DEPLOYMENT    – GPT-5 deployment/model name (default: gpt-5)
"""
import json
import logging
import os
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logger = logging.getLogger(__name__)

_FOUNDRY_RESOURCE = os.environ.get("AZURE_FOUNDRY_RESOURCE", "")
_FOUNDRY_BASE_URL = (
    f"https://{_FOUNDRY_RESOURCE}.services.ai.azure.com/anthropic"
    if _FOUNDRY_RESOURCE
    else ""
)
_FOUNDRY_API_KEY = os.environ.get("AZURE_FOUNDRY_API_KEY", "")
_DEFAULT_MODEL = os.environ.get("PROXY_MODEL", "claude-haiku-4-5")
PORT = int(os.environ.get("PROXY_PORT", "4000"))

_GPT5_ENDPOINT = os.environ.get("AZURE_GPT5_ENDPOINT", "")
_GPT5_API_KEY = os.environ.get("AZURE_GPT5_API_KEY", "")
_GPT5_DEPLOYMENT = os.environ.get("AZURE_GPT5_DEPLOYMENT", "gpt-5")


def _strip_json_fences(text: str) -> str:
    """Remove markdown JSON code fences that Claude adds even when not asked."""
    text = text.strip()
    if text.startswith("```"):
        # Drop opening fence (```json or ```)
        text = text[text.index("\n") + 1 :] if "\n" in text else text[3:]
        # Drop closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text


def _openai_to_anthropic(openai_req: dict) -> dict:
    """Convert an OpenAI chat completion request body to Anthropic messages format."""
    messages = list(openai_req.get("messages", []))

    # Anthropic requires system prompt as a top-level field, not a message role
    system = None
    if messages and messages[0].get("role") == "system":
        system = messages[0]["content"]
        messages = messages[1:]

    # When the caller requests JSON output, instruct Claude to return raw JSON
    # (no markdown fences). This mirrors the OpenAI response_format=json_object
    # constraint which Claude's API does not natively support.
    wants_json = (
        isinstance(openai_req.get("response_format"), dict)
        and openai_req["response_format"].get("type") == "json_object"
    )

    req: dict = {
        "model": openai_req.get("model", _DEFAULT_MODEL),
        "messages": messages,
        "max_tokens": openai_req.get("max_tokens", 1024),
    }
    if wants_json:
        json_instruction = "Return ONLY a raw JSON object. Do not use markdown code fences, do not include ```json, do not include any text before or after the JSON object."
        if system:
            req["system"] = json_instruction + "\n\n" + system
        else:
            req["system"] = json_instruction
    elif system:
        req["system"] = system
    if "temperature" in openai_req:
        req["temperature"] = openai_req["temperature"]

    return req


def _anthropic_to_openai(anthropic_resp: dict, model: str) -> dict:
    """Convert an Anthropic messages response to OpenAI chat completion format."""
    content = ""
    for block in anthropic_resp.get("content", []):
        if block.get("type") == "text":
            content = _strip_json_fences(block["text"])
            break

    usage = anthropic_resp.get("usage", {})
    return {
        "id": anthropic_resp.get("id", "proxy-1"),
        "object": "chat.completion",
        "model": anthropic_resp.get("model", model),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


class _ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Accept both /v1/chat/completions and /chat/completions
        if not (
            self.path.startswith("/v1/chat/completions")
            or self.path.startswith("/chat/completions")
        ):
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            body = b'{"error": {"message": "Not found", "type": "not_found"}}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        length = int(self.headers.get("Content-Length", 0))
        openai_req = json.loads(self.rfile.read(length))
        anthropic_req = _openai_to_anthropic(openai_req)

        try:
            resp = httpx.post(
                f"{_FOUNDRY_BASE_URL}/v1/messages",
                headers={
                    "x-api-key": _FOUNDRY_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=anthropic_req,
                timeout=120.0,
            )
            resp.raise_for_status()
            openai_resp = _anthropic_to_openai(resp.json(), anthropic_req["model"])
            status = 200
        except httpx.HTTPStatusError as exc:
            openai_resp = {"error": {"message": str(exc), "type": "proxy_error"}}
            status = exc.response.status_code
        except Exception as exc:  # noqa: BLE001
            openai_resp = {"error": {"message": str(exc), "type": "proxy_error"}}
            status = 500

        body = json.dumps(openai_resp).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # suppress default access log noise
        pass


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Thread-per-request server — needed because evaluate() calls the LLM in parallel."""

    daemon_threads = True


def start(port: int = PORT) -> _ThreadedHTTPServer:
    """Start the proxy server and return the server object (non-blocking)."""
    if not _FOUNDRY_RESOURCE or not _FOUNDRY_API_KEY:
        raise RuntimeError(
            "AZURE_FOUNDRY_RESOURCE and AZURE_FOUNDRY_API_KEY must be set to use the proxy."
        )
    server = _ThreadedHTTPServer(("127.0.0.1", port), _ProxyHandler)
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Proxy started on http://127.0.0.1:%d", port)
    return server


def stop(server: _ThreadedHTTPServer) -> None:
    server.shutdown()


class _GPT5ProxyHandler(BaseHTTPRequestHandler):
    """Pass-through proxy for GPT-5.

    Translates max_tokens → max_completion_tokens and forwards to the
    Azure AI Foundry /openai/v1/ endpoint.  GPT-5 rejects max_tokens with
    a 400 error; the azure-ai-evaluation SDK always sends max_tokens.
    """

    gpt5_endpoint: str = ""
    gpt5_api_key: str = ""

    def do_POST(self):
        if not (
            self.path.startswith("/v1/chat/completions")
            or self.path.startswith("/chat/completions")
        ):
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            body = b'{"error": {"message": "Not found", "type": "not_found"}}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))

        # GPT-5 uses max_completion_tokens; the SDK sends max_tokens.
        if "max_tokens" in req:
            req["max_completion_tokens"] = req.pop("max_tokens")

        # GPT-5 only supports the default temperature (1); remove explicit 0.0.
        req.pop("temperature", None)

        # Ensure model is set to the GPT-5 deployment
        req.setdefault("model", _GPT5_DEPLOYMENT)

        # Forward to the GPT-5 endpoint
        target_url = self.__class__.gpt5_endpoint.rstrip("/") + "/chat/completions"
        try:
            resp = httpx.post(
                target_url,
                headers={
                    "api-key": self.__class__.gpt5_api_key,
                    "content-type": "application/json",
                },
                json=req,
                timeout=120.0,
            )
            body = resp.content
            status = resp.status_code
        except Exception as exc:  # noqa: BLE001
            body = json.dumps({"error": {"message": str(exc), "type": "proxy_error"}}).encode()
            status = 500

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def start_gpt5(port: int, endpoint: str, api_key: str) -> "_ThreadedHTTPServer":
    """Start a GPT-5 pass-through proxy and return the server object (non-blocking)."""
    if not endpoint or not api_key:
        raise RuntimeError(
            "AZURE_GPT5_ENDPOINT and AZURE_GPT5_API_KEY must be set to use the GPT-5 proxy."
        )

    # Create a handler subclass with the endpoint/key baked in
    handler_cls = type(
        "_GPT5Handler",
        (_GPT5ProxyHandler,),
        {"gpt5_endpoint": endpoint, "gpt5_api_key": api_key},
    )
    server = _ThreadedHTTPServer(("127.0.0.1", port), handler_cls)
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("GPT-5 proxy started on http://127.0.0.1:%d → %s", port, endpoint)
    return server



    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not _FOUNDRY_RESOURCE or not _FOUNDRY_API_KEY:
        print(
            "ERROR: Set AZURE_FOUNDRY_RESOURCE and AZURE_FOUNDRY_API_KEY "
            "(in eval/.env or shell) before running the proxy."
        )
        raise SystemExit(1)

    print(f"Proxying → {_FOUNDRY_BASE_URL}")
    print(f"Listening on http://127.0.0.1:{PORT}/v1/chat/completions")
    print("Press Ctrl+C to stop.\n")
    server = _ThreadedHTTPServer(("127.0.0.1", PORT), _ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nProxy stopped.")
