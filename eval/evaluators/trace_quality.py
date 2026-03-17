"""Trace-aware evaluator for the multi-agent customer support pipeline.

Takes an entire span tree (as returned by query_app_insights.query_traces) and
evaluates three dimensions that are only assessable from the full trace context:

  routing_appropriateness  — was the query routed to the right specialist?
  escalation_judgment      — was the escalation decision appropriate?
  specialist_alignment     — does the response reflect genuine domain expertise?

The trace dict is serialised to a compact JSON string and substituted into the
trace_quality.prompty template, which is then sent directly to the OpenAI-compatible
judge LLM. No promptflow dependency required.
"""

import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTY_PATH = Path(__file__).parent / "trace_quality.prompty"
_SCORE_KEYS = ("routing_appropriateness", "escalation_judgment", "specialist_alignment")
_REASON_KEYS = {
    "routing_appropriateness": "routing_reason",
    "escalation_judgment": "escalation_reason",
    "specialist_alignment": "specialist_reason",
}


def _parse_prompty(path: Path) -> tuple[str, str]:
    """Extract system and user prompt strings from a .prompty file.

    Returns (system_prompt, user_prompt_template) where the user template
    still contains ``{{variable}}`` placeholders.
    """
    text = path.read_text()
    # Strip YAML front matter (between the first two --- markers)
    text = re.sub(r"^---.*?---\s*", "", text, count=1, flags=re.DOTALL)
    system_match = re.search(r"^system:\s*\n(.*?)(?=^user:|\Z)", text,
                             re.MULTILINE | re.DOTALL)
    user_match = re.search(r"^user:\s*\n(.*)", text, re.MULTILINE | re.DOTALL)
    system = system_match.group(1).strip() if system_match else ""
    user = user_match.group(1).strip() if user_match else ""
    return system, user


def _build_openai_client(model_config):
    """Build an openai.OpenAI or openai.AzureOpenAI client from a model config dict."""
    import openai  # noqa: PLC0415

    def _get(key, default=None):
        if isinstance(model_config, dict):
            return model_config.get(key, default)
        return getattr(model_config, key, default)

    if _get("azure_endpoint"):
        return openai.AzureOpenAI(
            azure_endpoint=_get("azure_endpoint"),
            api_key=_get("api_key"),
            api_version=_get("api_version", "2024-02-01"),
        ), _get("azure_deployment")

    client = openai.OpenAI(
        api_key=_get("api_key", "proxy"),
        base_url=str(_get("base_url") or "http://127.0.0.1:4000/v1/"),
    )
    return client, _get("model", "gpt-5")


class TraceQualityEvaluator:
    """Evaluates routing quality, escalation judgment, and specialist alignment
    from a full agent trace span tree.

    Args:
        model_config: An ``AzureOpenAIModelConfiguration`` or
            ``OpenAIModelConfiguration`` from ``azure.ai.evaluation``.

    Returns (per call):
        routing_appropriateness  (int 1-5)
        routing_reason           (str)
        escalation_judgment      (int 1-5)
        escalation_reason        (str)
        specialist_alignment     (int 1-5)
        specialist_reason        (str)
    """

    id = "trace_quality"

    def __init__(self, model_config):
        self._model_config = model_config
        self._system, self._user_template = _parse_prompty(_PROMPTY_PATH)

    def __call__(self, *, trace: "dict | str", **kwargs) -> dict:
        """Evaluate a trace.

        Args:
            trace: The full trace dict (from ``query_app_insights.query_traces``)
                or a JSON string. Passed directly into the prompty as ``{{trace}}``.
        """
        if isinstance(trace, dict):
            trace_json = json.dumps(trace, separators=(",", ":"))
        else:
            trace_json = str(trace)

        user_prompt = self._user_template.replace("{{trace}}", trace_json)

        try:
            client, model = _build_openai_client(self._model_config)
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=1000,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._system},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content
        except Exception as exc:
            logger.warning("TraceQualityEvaluator LLM call failed: %s", exc)
            return _error_result(str(exc))

        return _parse_output(raw)


# ── Output parsing ────────────────────────────────────────────────────────────


def _parse_output(raw) -> dict:
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(str(raw))
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to parse TraceQualityEvaluator output: %s\nRaw: %s", exc, raw)
            return _error_result(f"JSON parse error: {exc}")

    result = {}
    for score_key in _SCORE_KEYS:
        score = data.get(score_key)
        try:
            result[score_key] = int(score)
        except (TypeError, ValueError):
            result[score_key] = None
        reason_key = _REASON_KEYS[score_key]
        result[reason_key] = data.get(reason_key, "")

    return result


def _error_result(message: str) -> dict:
    result = {}
    for score_key in _SCORE_KEYS:
        result[score_key] = None
        result[_REASON_KEYS[score_key]] = f"Evaluation error: {message}"
    return result

