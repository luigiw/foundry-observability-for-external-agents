"""LLM-based evaluator for tool selection quality.

Loads tool_selection.prompty, parses its system/user sections, and calls the
judge LLM directly via the openai client — no promptflow dependency needed.
"""
import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTY_PATH = Path(__file__).parent / "tool_selection.prompty"


def _parse_prompty(path: Path) -> tuple[str, str]:
    text = path.read_text()
    text = re.sub(r"^---.*?---\s*", "", text, count=1, flags=re.DOTALL)
    system_match = re.search(r"^system:\s*\n(.*?)(?=^user:|\Z)", text, re.MULTILINE | re.DOTALL)
    user_match = re.search(r"^user:\s*\n(.*)", text, re.MULTILINE | re.DOTALL)
    system = system_match.group(1).strip() if system_match else ""
    user = user_match.group(1).strip() if user_match else ""
    return system, user


def _build_openai_client(model_config):
    import openai

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

    return openai.OpenAI(
        api_key=_get("api_key", "proxy"),
        base_url=str(_get("base_url") or "http://127.0.0.1:4000/v1/"),
    ), _get("model", "gpt-5")


class ToolSelectionEvaluator:
    """Evaluates whether appropriate tools were selected for the user request.

    Args:
        model_config: AzureOpenAIModelConfiguration or OpenAIModelConfiguration.

    Returns (per call):
        tool_selection        (int 1-5)
        tool_selection_reason (str)
    """

    id = "tool_selection"

    def __init__(self, model_config):
        self._model_config = model_config
        self._system, self._user_template = _parse_prompty(_PROMPTY_PATH)

    def __call__(self, *, user_input: str, tools_used: str, response: str, **kwargs) -> dict:
        # tools_used arrives as a comma-separated string (e.g. "list_files,read_file")
        # or an empty string when no tools were called.
        user_prompt = (
            self._user_template
            .replace("{{user_input}}", user_input)
            .replace("{{tools_used}}", tools_used or "(none)")
            .replace("{{response}}", response)
        )
        try:
            client, model = _build_openai_client(self._model_config)
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_completion_tokens=500,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._system},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = resp.choices[0].message.content
        except Exception as exc:
            logger.warning("ToolSelectionEvaluator LLM call failed: %s", exc)
            return {"tool_selection": None, "tool_selection_reason": f"Error: {exc}"}

        try:
            data = json.loads(raw)
            return {
                "tool_selection": int(data.get("tool_selection")),
                "tool_selection_reason": data.get("tool_selection_reason", ""),
            }
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse ToolSelectionEvaluator output: %s\nRaw: %s", exc, raw)
            return {"tool_selection": None, "tool_selection_reason": f"Parse error: {exc}"}
