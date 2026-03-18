"""Thin wrapper around TraceQualityEvaluator for use in the Streamlit UI.

Reads the shared prompty template from eval/evaluators/trace_quality.prompty and
calls an Azure OpenAI deployment to score a trace on three dimensions:
  - routing_appropriateness  (1-5)
  - escalation_judgment      (1-5)
  - specialist_alignment     (1-5)

Required environment variables:
    AZURE_OPENAI_ENDPOINT    — Azure OpenAI resource endpoint
    AZURE_OPENAI_API_KEY     — API key
    AZURE_OPENAI_DEPLOYMENT  — Deployment name (e.g. gpt-4o)
    AZURE_OPENAI_API_VERSION — Optional, defaults to "2024-02-01"
"""

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Resolve the shared prompty template (lives in eval/, sibling of ui/)
_PROMPTY_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "eval" / "evaluators" / "trace_quality.prompty"
)

_SCORE_KEYS = ("routing_appropriateness", "escalation_judgment", "specialist_alignment")
_REASON_KEYS = {
    "routing_appropriateness": "routing_reason",
    "escalation_judgment": "escalation_reason",
    "specialist_alignment": "specialist_reason",
}


def _parse_prompty(path: Path) -> tuple[str, str]:
    """Extract system and user prompt strings from a .prompty file."""
    text = path.read_text()
    text = re.sub(r"^---.*?---\s*", "", text, count=1, flags=re.DOTALL)
    system_m = re.search(r"^system:\s*\n(.*?)(?=^user:|\Z)", text, re.MULTILINE | re.DOTALL)
    user_m = re.search(r"^user:\s*\n(.*)", text, re.MULTILINE | re.DOTALL)
    return (
        system_m.group(1).strip() if system_m else "",
        user_m.group(1).strip() if user_m else "",
    )


def _build_client():
    """Return (openai.AzureOpenAI client, deployment_name) from env vars."""
    import openai  # noqa: PLC0415

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

    if not (endpoint and api_key and deployment):
        raise EnvironmentError(
            "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and "
            "AZURE_OPENAI_DEPLOYMENT in ui/.env to enable trace evaluation."
        )

    return openai.AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    ), deployment


def evaluate_trace(trace: dict) -> dict:
    """Evaluate a single trace dict and return a scores dict.

    Args:
        trace: A trace dict as returned by ``query_app_insights.query_traces``
            or assembled from ``trace_query.query_conversations`` +
            ``query_conversation_detail``. Must contain at least:
            query, response, query_type, handled_by, needs_escalation.

    Returns:
        Dict with keys: routing_appropriateness, routing_reason,
        escalation_judgment, escalation_reason, specialist_alignment,
        specialist_reason.  Score values are int 1-5 or None on error.
    """
    if not _PROMPTY_PATH.exists():
        return _error_result(f"Prompty not found at {_PROMPTY_PATH}")

    system, user_template = _parse_prompty(_PROMPTY_PATH)
    trace_json = json.dumps(trace, separators=(",", ":"), default=str)
    user_prompt = user_template.replace("{{trace}}", trace_json)

    try:
        client, model = _build_client()
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=1000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content
    except Exception as exc:
        logger.warning("evaluate_trace LLM call failed: %s", exc)
        return _error_result(str(exc))

    return _parse_scores(raw)


def _parse_scores(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        return _error_result(f"JSON parse error: {exc}")

    result = {}
    for score_key in _SCORE_KEYS:
        try:
            result[score_key] = int(data.get(score_key))
        except (TypeError, ValueError):
            result[score_key] = None
        result[_REASON_KEYS[score_key]] = data.get(_REASON_KEYS[score_key], "")
    return result


def _error_result(message: str) -> dict:
    result = {}
    for score_key in _SCORE_KEYS:
        result[score_key] = None
        result[_REASON_KEYS[score_key]] = f"Error: {message}"
    return result
