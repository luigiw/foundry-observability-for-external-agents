"""Run evaluation on the Copilot SDK local agent using azure-ai-evaluation.

Modes
-----
dataset (default)
    Evaluate a pre-collected JSONL file. No agent credentials needed.
    Default data file: data/sample_traces.jsonl

app-insights
    Fetch traces from Azure Application Insights and evaluate them.
    Requires APPLICATIONINSIGHTS_APP_ID and APPLICATIONINSIGHTS_QUERY_API_KEY
    env vars. Traces must have been generated with OTEL_CAPTURE_CONTENT=true.
    Use --hours to control the lookback window (default: 24h).

Model providers
---------------
azure-openai (default)
    Standard Azure OpenAI deployment. Requires AZURE_OPENAI_* env vars.

gpt5
    GPT-5 via Azure AI Foundry unified /openai/v1/ endpoint.
    Requires AZURE_GPT5_ENDPOINT, AZURE_GPT5_API_KEY, AZURE_GPT5_DEPLOYMENT.

Remote eval (Azure AI Foundry)
-------------------------------
Pass --azure-ai-project to log results to Azure AI Foundry for experiment
tracking and comparison UI. Requires AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP,
AZURE_AI_PROJECT_NAME env vars.

Usage examples
--------------
# Evaluate pre-collected sample traces (no agent required):
python run_eval.py

# Use GPT-5 as the LLM judge:
python run_eval.py --model-provider gpt5

# Skip AI-assisted evaluators (fast, no LLM needed):
python run_eval.py --skip-ai-evaluators

# Evaluate from App Insights traces (last 24h):
python run_eval.py --mode app-insights --model-provider gpt5

# Evaluate from App Insights and log to Azure AI Foundry:
python run_eval.py --mode app-insights --model-provider gpt5 --azure-ai-project

# Use a custom data file:
python run_eval.py --mode dataset --data data/my_traces.jsonl
"""

import os
import sys
import json
import tempfile
import argparse
from pathlib import Path

from dotenv import load_dotenv

eval_dir = Path(__file__).parent
# Load parent .env (copilot-local-agent/.env) first for shared vars like AZURE_GPT5_*,
# then eval-local .env with override=True so eval-specific values take priority.
load_dotenv(eval_dir.parent / ".env")
load_dotenv(eval_dir / ".env", override=True)

_EVAL_FIELDS = frozenset({"user_input", "response", "tools_used", "model"})


# ── Model config ──────────────────────────────────────────────────────────────


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {name}\n"
            "Set it in eval/.env — see .env.example for instructions."
        )
    return value


def _build_model_config(provider: str):
    if provider == "gpt5":
        import proxy
        import atexit
        from azure.ai.evaluation import OpenAIModelConfiguration
        endpoint = _require_env("AZURE_GPT5_ENDPOINT")
        api_key = _require_env("AZURE_GPT5_API_KEY")
        deployment = os.environ.get("AZURE_GPT5_DEPLOYMENT", "gpt-5")
        port = int(os.environ.get("PROXY_PORT", "4000"))
        print(f"Starting GPT-5 proxy on port {port} → {endpoint} (deployment: {deployment})...")
        _proxy_server = proxy.start_gpt5(port, endpoint, api_key)
        atexit.register(proxy.stop, _proxy_server)
        return OpenAIModelConfiguration(
            type="openai",
            model=deployment,
            api_key="proxy",
            base_url=f"http://127.0.0.1:{port}/v1/",
        )

    # Default: azure-openai
    from azure.ai.evaluation import AzureOpenAIModelConfiguration
    return AzureOpenAIModelConfiguration(
        azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_require_env("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_deployment=_require_env("AZURE_OPENAI_DEPLOYMENT"),
    )


# ── Evaluators ────────────────────────────────────────────────────────────────


def _build_evaluators(skip_ai: bool, provider: str = "azure-openai") -> dict:
    sys.path.insert(0, str(eval_dir))
    from evaluators.command_safety import CommandSafetyEvaluator
    from evaluators.code_correctness import CodeCorrectnessEvaluator
    from evaluators.groundedness import GroundednessEvaluator

    # Deterministic evaluator — always included regardless of --skip-ai-evaluators
    evals = {"command_safety": CommandSafetyEvaluator()}

    if skip_ai:
        return evals

    model_config = _build_model_config(provider)
    evals["code_correctness"] = CodeCorrectnessEvaluator(model_config)
    evals["groundedness"] = GroundednessEvaluator(model_config)
    return evals


def _build_column_mapping() -> dict:
    """Per-evaluator column mapping — all modes read from ${data.*} fields."""
    return {
        "command_safety": {
            "column_mapping": {
                "response": "${data.response}",
                "tools_used": "${data.tools_used}",
            }
        },
        "code_correctness": {
            "column_mapping": {
                "user_input": "${data.user_input}",
                "response": "${data.response}",
            }
        },
        "groundedness": {
            "column_mapping": {
                "user_input": "${data.user_input}",
                "response": "${data.response}",
                "tools_used": "${data.tools_used}",
            }
        },
    }


# ── Azure AI Foundry project config ──────────────────────────────────────────


def _build_azure_ai_project() -> str:
    return _require_env("AZURE_AI_PROJECT_ENDPOINT")


# ── Eval runners ──────────────────────────────────────────────────────────────


def _run_evaluate(
    data_path: str,
    evaluators: dict,
    evaluator_config: dict,
    output_path: str,
    azure_ai_project: dict | None,
) -> dict:
    from azure.ai.evaluation import evaluate

    kwargs = dict(
        data=data_path,
        evaluators=evaluators,
        evaluator_config=evaluator_config,
        output_path=output_path,
    )
    if azure_ai_project:
        kwargs["azure_ai_project"] = azure_ai_project

    return evaluate(**kwargs)


def _eval_dataset(args, evaluators, evaluator_config, azure_ai_project):
    data_path = args.data or str(eval_dir / "data" / "sample_traces.jsonl")
    output_path = args.output or str(eval_dir / "results" / "dataset_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating dataset: {data_path}")
    result = _run_evaluate(data_path, evaluators, evaluator_config, output_path, azure_ai_project)
    return result


def _eval_app_insights(args, evaluators, evaluator_config, azure_ai_project):
    sys.path.insert(0, str(eval_dir))
    from query_app_insights import query_traces

    print(f"Querying App Insights (last {args.hours}h, limit {args.limit})...")
    traces = query_traces(args.hours, args.limit)
    if not traces:
        print("No traces found. Ensure OTEL_CAPTURE_CONTENT=true when running the agent.")
        return None
    print(f"  Found {len(traces)} trace(s).")

    output_path = args.output or str(eval_dir / "results" / "app_insights_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, dir=eval_dir / "results"
    ) as tmp:
        for t in traces:
            row = {k: v for k, v in t.items() if k in _EVAL_FIELDS}
            tmp.write(json.dumps(row) + "\n")
        tmp_path = tmp.name

    try:
        result = _run_evaluate(tmp_path, evaluators, evaluator_config, output_path, azure_ai_project)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


# ── Result printing ───────────────────────────────────────────────────────────


def _print_results(result: dict) -> None:
    if not result:
        return
    metrics = result.get("metrics", {})
    if not metrics:
        print("  (no metrics returned)")
        return
    print("\n=== Evaluation Results ===")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Copilot SDK local agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["dataset", "app-insights"], default="dataset",
        help="Evaluation mode (default: dataset).",
    )
    parser.add_argument(
        "--model-provider", choices=["azure-openai", "gpt5"], default="azure-openai",
        help="LLM judge provider (default: azure-openai).",
    )
    parser.add_argument(
        "--skip-ai-evaluators", action="store_true",
        help="Skip AI-assisted evaluators (fast, no LLM needed).",
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to JSONL dataset (dataset mode only).",
    )
    parser.add_argument(
        "--hours", type=int, default=24,
        help="Lookback window in hours for app-insights mode (default: 24).",
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Max traces to evaluate in app-insights mode (default: 100).",
    )
    parser.add_argument(
        "--azure-ai-project", action="store_true",
        help="Log results to Azure AI Foundry (requires AZURE_* env vars).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for results JSON (default: results/<mode>_results.json).",
    )
    args = parser.parse_args()

    evaluators = _build_evaluators(args.skip_ai_evaluators, args.model_provider)
    evaluator_config = _build_column_mapping()
    # Only include config entries for evaluators that are active
    evaluator_config = {k: v for k, v in evaluator_config.items() if k in evaluators}

    azure_ai_project = _build_azure_ai_project() if args.azure_ai_project else None
    if azure_ai_project:
        print(f"Remote logging → Azure AI Foundry project: {azure_ai_project}")

    if args.mode == "dataset":
        result = _eval_dataset(args, evaluators, evaluator_config, azure_ai_project)
    else:
        result = _eval_app_insights(args, evaluators, evaluator_config, azure_ai_project)

    _print_results(result)


if __name__ == "__main__":
    main()
