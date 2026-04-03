"""LLM-based evaluator for response groundedness in actual tool output."""
import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
_PROMPTY_PATH = Path(__file__).parent / "groundedness.prompty"


def _parse_prompty(path: Path) -> tuple[str, str]:
    text = path.read_text()
    text = re.sub(r"^---.*?---\s*", "", text, count=1, flags=re.DOTALL)
    system_match = re.search(r"^system:\s*\n(.*?)(?=^user:|\Z)", text, re.MULTILINE | re.DOTALL)
    user_match = re.search(r"^user:\s*\n(.*)", text, re.MULTILINE | re.DOTALL)
    return (
        system_match.group(1).strip() if system_match else "",
        user_match.group(1).strip() if user_match else "",
    )


def _build_openai_client(model_config):
    import openai
    def _get(key, default=None):
        return model_config.get(key, default) if isinstance(model_config, dict) else getattr(model_config, key, default)
    if _get("azure_endpoint"):
        return openai.AzureOpenAI(azure_endpoint=_get("azure_endpoint"), api_key=_get("api_key"), api_version=_get("api_version", "2024-02-01")), _get("azure_deployment")
    return openai.OpenAI(api_key=_get("api_key", "proxy"), base_url=str(_get("base_url") or "http://127.0.0.1:4000/v1/")), _get("model", "gpt-5")


class GroundednessEvaluator:
    """Evaluates whether the response is grounded in actual tool output (1-5).

    Returns:
        groundedness        (int 1-5)
        groundedness_reason (str)
    """
    id = "groundedness"

    def __init__(self, model_config):
        self._model_config = model_config
        self._system, self._user_template = _parse_prompty(_PROMPTY_PATH)

    def __call__(self, *, user_input: str, response: str, tools_used: str = "", **kwargs) -> dict:
        user_prompt = (
            self._user_template
            .replace("{{user_input}}", user_input)
            .replace("{{response}}", response)
            .replace("{{tools_used}}", tools_used or "(none)")
        )
        try:
            client, model = _build_openai_client(self._model_config)
            resp = client.chat.completions.create(
                model=model, temperature=0.0, max_completion_tokens=500,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": self._system}, {"role": "user", "content": user_prompt}],
            )
            data = json.loads(resp.choices[0].message.content)
            return {"groundedness": int(data.get("groundedness")), "groundedness_reason": data.get("groundedness_reason", "")}
        except Exception as exc:
            logger.warning("GroundednessEvaluator failed: %s", exc)
            return {"groundedness": None, "groundedness_reason": f"Error: {exc}"}
