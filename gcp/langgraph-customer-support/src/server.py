"""FastAPI server for the customer support agent — Cloud Run entrypoint.

Uses Anthropic Claude models via Microsoft Foundry (Azure AI Foundry).

Trace-context note:
  Cloud Run's HTTP proxy rewrites the ``traceparent`` header, replacing
  the parent span-id with its own infrastructure span that is exported
  to Google Cloud Trace — **not** Azure Application Insights.  This
  breaks the parent→child link in App Insights.

  Workaround (automatic): when the request arrives via Azure API
  Management, APIM sends a ``request-id`` header in Application Insights
  legacy format (``|<traceId>.<spanId>.``) containing its *dependency*
  span-id.  Cloud Run does not modify this header, so we parse it and
  reconstruct a ``traceparent`` to restore the parent→child link.

  Fallback: callers may also set a custom ``x-ms-traceparent`` header
  (Cloud Run leaves it untouched).
"""
import logging
import os
from pathlib import Path

# Load .env before any other imports so env vars are available at module level
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import secrets
from opentelemetry import context as otel_context, trace
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from .tracing import get_azure_tracer, setup_tracer_provider, extract_context_from_headers, flush_traces
from .graph import invoke_support

SUPPORT_API_KEY = os.environ.get("SUPPORT_API_KEY")

# Custom header that bypasses Cloud Run's traceparent rewriting.
_CUSTOM_TRACEPARENT_HEADER = "x-ms-traceparent"


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid x-api-key header.

    Skipped when SUPPORT_API_KEY is not set (local dev) or for health checks.
    """
    EXEMPT_PATHS = {"/health", "/docs", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        if not SUPPORT_API_KEY or request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        key = request.headers.get("x-api-key", "")
        if not secrets.compare_digest(key, SUPPORT_API_KEY):
            return JSONResponse(status_code=403, content={"error": "Forbidden"})

        return await call_next(request)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize tracing on startup
_tracing_initialized = False


def _init_tracing():
    global _tracing_initialized
    if not _tracing_initialized:
        os.environ.setdefault("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
        get_azure_tracer()
        _tracing_initialized = True


def _build_propagation_headers(request: Request) -> dict:
    """Return a header dict suitable for OTel ``extract()``.

    Cloud Run rewrites the ``traceparent`` header, replacing the span-id
    with its own infrastructure span (sent to Google Cloud Trace, **not**
    App Insights).  This breaks the parent→child link in App Insights.

    To restore the link we look for alternative headers that Cloud Run
    does **not** modify, in priority order:

    1. **``request-id``** (Application Insights legacy format sent by
       Azure API Management): ``|<traceId>.<spanId>.``  — contains the
       APIM *dependency* span-id, which is the ideal parent.
    2. **``x-ms-traceparent``** — custom W3C traceparent set by APIM
       inbound policy (carries the *caller's* traceparent, useful when
       the caller has its own trace context).
    3. **``traceparent``** — standard W3C header (may be rewritten by
       Cloud Run).
    """
    headers = dict(request.headers)

    # --- Priority 1: APIM request-id (legacy AI hierarchical format) ---
    request_id = headers.get("request-id", "")
    if request_id.startswith("|"):
        parts = request_id[1:].split(".")
        if len(parts) >= 2:
            trace_id, span_id = parts[0], parts[1]
            if len(trace_id) == 32 and len(span_id) == 16:
                constructed = f"00-{trace_id}-{span_id}-01"
                headers["traceparent"] = constructed
                logger.info(
                    "Constructed traceparent from request-id header: %s",
                    constructed,
                )
                return headers

    # --- Priority 2: custom x-ms-traceparent (Cloud Run leaves it alone) ---
    custom_tp = headers.get(_CUSTOM_TRACEPARENT_HEADER)
    if custom_tp:
        headers["traceparent"] = custom_tp
        logger.info("Using %s header: %s", _CUSTOM_TRACEPARENT_HEADER, custom_tp)
        return headers

    # --- Priority 3: standard traceparent (may be Cloud-Run-rewritten) ---
    logger.info(
        "Falling back to traceparent: %s",
        headers.get("traceparent", "(none)"),
    )
    return headers


app = FastAPI(title="LangGraph Customer Support Agent", version="1.0.0")

app.add_middleware(ApiKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SupportRequest(BaseModel):
    message: str
    customer_id: str | None = None


@app.on_event("startup")
async def startup():
    _init_tracing()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/support")
async def support(body: SupportRequest, request: Request):
    """Handle support requests with W3C Trace Context propagation."""
    setup_tracer_provider()
    prop_headers = _build_propagation_headers(request)
    parent_ctx = extract_context_from_headers(prop_headers)
    token = otel_context.attach(parent_ctx)

    tracer = trace.get_tracer(__name__)

    try:
        with tracer.start_as_current_span(
            "handle_request",
            kind=trace.SpanKind.SERVER,
            attributes={"http.method": "POST", "http.route": "/support"},
        ) as span:
            result = invoke_support(body.message, body.customer_id)

            logger.info(f"Response generated by: {result['handled_by']}")
            span.set_attribute("agent.handled_by", result["handled_by"])
            span.set_attribute("agent.query_type", result["query_type"])
            span.set_attribute("http.status_code", 200)
            flush_traces()

            return {
                "response": result["response"],
                "metadata": {
                    "handled_by": result["handled_by"],
                    "query_type": result["query_type"],
                    "needs_escalation": result["needs_escalation"],
                },
            }
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
    finally:
        otel_context.detach(token)
        flush_traces()
