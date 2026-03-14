"""Run local evaluation on the customer support agents using azure-ai-evaluation.

Modes
-----
dataset (default)
    Evaluate a pre-collected JSONL file. No agent credentials needed.
    Requires AZURE_OPENAI_* env vars for AI-assisted evaluators.
    Default data file: data/sample_traces.jsonl

live
    Call the agent for each query row, then evaluate the outputs.
    Requires the respective agent's dependencies and credentials.
    Default data file: data/eval_queries.jsonl

Model providers
---------------
azure-openai (default)
    Standard Azure OpenAI deployment.  Requires AZURE_OPENAI_* env vars.

foundry
    Azure AI Foundry with Anthropic-native endpoint (e.g. claude-haiku-4-5).
    Requires AZURE_FOUNDRY_RESOURCE and AZURE_FOUNDRY_API_KEY env vars.
    Automatically starts a local OpenAI-compatible proxy on PROXY_PORT (default 4000).

gpt5
    GPT-5 via Azure AI Foundry unified /openai/v1/ endpoint.
    Requires AZURE_GPT5_ENDPOINT, AZURE_GPT5_API_KEY, AZURE_GPT5_DEPLOYMENT env vars.

Usage examples
--------------
# Evaluate pre-collected sample traces (no agent required):
python run_eval.py

# Use Azure AI Foundry as the LLM judge (no Azure OpenAI needed):
python run_eval.py --model-provider foundry

# Use GPT-5 as the LLM judge:
python run_eval.py --model-provider gpt5

# Evaluate the GCP agent live with Foundry as judge:
python run_eval.py --agent gcp --mode live --model-provider foundry

# Skip AI-assisted evaluators (only routing accuracy, no LLM needed):
python run_eval.py --skip-ai-evaluators

# Use a custom data file:
python run_eval.py --mode dataset --data data/gcp_traces.jsonl --agent gcp
"""
import os
import sys
import json
import argparse
from pathlib import Path

from dotenv import load_dotenv

eval_dir = Path(__file__).parent
repo_root = eval_dir.parent

# Load eval-local .env first; agent-specific envs are loaded per agent below.
load_dotenv(eval_dir / ".env")


# ── Agent target loaders ──────────────────────────────────────────────────────


def _get_aws_target():
    """Return a target callable for the AWS Bedrock agent."""
    aws_path = repo_root / "aws" / "langgraph-customer-support"
    if str(aws_path) not in sys.path:
        sys.path.insert(0, str(aws_path))
    from src.graph import invoke_support  # noqa: PLC0415

    def aws_target(query: str, **kwargs) -> dict:
        result = invoke_support(query)
        return {
            "response": result["response"],
            "query_type": result["query_type"],
            "handled_by": result["handled_by"],
            "needs_escalation": result["needs_escalation"],
        }

    return aws_target


def _get_gcp_target():
    """Return a target callable for the GCP / Azure AI Foundry agent."""
    gcp_path = repo_root / "gcp" / "langgraph-customer-support"
    if str(gcp_path) not in sys.path:
        sys.path.insert(0, str(gcp_path))
    load_dotenv(gcp_path / ".env", override=False)
    from src.graph import invoke_support  # noqa: PLC0415

    def gcp_target(query: str, **kwargs) -> dict:
        result = invoke_support(query)
        return {
            "response": result["response"],
            "query_type": result["query_type"],
            "handled_by": result["handled_by"],
            "needs_escalation": result["needs_escalation"],
        }

    return gcp_target


# ── Evaluator setup ───────────────────────────────────────────────────────────


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {name}\n"
            "Set it in eval/.env or export it before running this script."
        )
    return value


def _build_model_config(provider: str):
    """Build the LLM judge model configuration for the given provider.

    provider='azure-openai'
        Uses AzureOpenAIModelConfiguration with AZURE_OPENAI_* env vars.
    provider='foundry'
        Starts a local OpenAI-compatible proxy (proxy.py) that forwards to the
        Azure AI Foundry Anthropic endpoint, then uses OpenAIModelConfiguration
        pointing to the proxy.
    provider='gpt5'
        Uses OpenAIModelConfiguration pointing directly to the Azure AI Foundry
        unified /openai/v1/ endpoint with a GPT-5 deployment.
    """
    if provider == "foundry":
        import proxy  # noqa: PLC0415

        port = int(os.environ.get("PROXY_PORT", "4000"))
        model = os.environ.get("PROXY_MODEL", "claude-haiku-4-5")
        print(f"Starting local proxy on port {port} → Azure AI Foundry ({model})...")
        _proxy_server = proxy.start(port)
        import atexit  # noqa: PLC0415

        atexit.register(proxy.stop, _proxy_server)

        from azure.ai.evaluation import OpenAIModelConfiguration  # noqa: PLC0415

        return OpenAIModelConfiguration(
            type="openai",  # required by the SDK's internal connection type check
            model=model,
            api_key="proxy",  # proxy ignores the key
            base_url=f"http://127.0.0.1:{port}/v1/",
        )

    if provider == "gpt5":
        import proxy  # noqa: PLC0415
        import atexit  # noqa: PLC0415

        endpoint = _require_env("AZURE_GPT5_ENDPOINT")
        api_key = _require_env("AZURE_GPT5_API_KEY")
        deployment = os.environ.get("AZURE_GPT5_DEPLOYMENT", "gpt-5")
        port = int(os.environ.get("PROXY_PORT", "4000"))
        print(f"Starting GPT-5 proxy on port {port} → {endpoint} (deployment: {deployment})...")
        _proxy_server = proxy.start_gpt5(port, endpoint, api_key)
        atexit.register(proxy.stop, _proxy_server)

        from azure.ai.evaluation import OpenAIModelConfiguration  # noqa: PLC0415

        return OpenAIModelConfiguration(
            type="openai",
            model=deployment,
            api_key="proxy",  # proxy ignores the key
            base_url=f"http://127.0.0.1:{port}/v1/",
        )

    # Default: azure-openai
    from azure.ai.evaluation import AzureOpenAIModelConfiguration  # noqa: PLC0415

    return AzureOpenAIModelConfiguration(
        azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_require_env("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_deployment=_require_env("AZURE_OPENAI_DEPLOYMENT"),
    )


def _build_evaluators(skip_ai: bool, provider: str = "azure-openai") -> dict:
    from evaluators.routing_accuracy import RoutingAccuracyEvaluator  # noqa: PLC0415

    evaluators = {"routing_accuracy": RoutingAccuracyEvaluator()}

    if not skip_ai:
        from azure.ai.evaluation import (  # noqa: PLC0415
            IntentResolutionEvaluator,
            RelevanceEvaluator,
            CoherenceEvaluator,
            FluencyEvaluator,
        )

        model_config = _build_model_config(provider)
        evaluators.update(
            {
                "intent_resolution": IntentResolutionEvaluator(model_config),
                "relevance": RelevanceEvaluator(model_config),
                "coherence": CoherenceEvaluator(model_config),
                "fluency": FluencyEvaluator(model_config),
            }
        )

    return evaluators


def _build_column_mapping(mode: str, skip_ai: bool) -> dict:
    """Build per-evaluator column mappings.

    In live mode, agent outputs are referenced as ${outputs.<field>}.
    In dataset mode, all fields come from the data file: ${data.<field>}.
    """
    resp = "${outputs.response}" if mode == "live" else "${data.response}"
    qtype = "${outputs.query_type}" if mode == "live" else "${data.query_type}"

    config = {
        "routing_accuracy": {
            "column_mapping": {
                "expected": "${data.expected_query_type}",
                "actual": qtype,
            }
        },
    }

    if not skip_ai:
        config.update(
            {
                "intent_resolution": {
                    "column_mapping": {
                        "query": "${data.query}",
                        "response": resp,
                    }
                },
                "relevance": {
                    "column_mapping": {
                        "query": "${data.query}",
                        "response": resp,
                    }
                },
                "coherence": {
                    "column_mapping": {
                        "query": "${data.query}",
                        "response": resp,
                    }
                },
                "fluency": {
                    "column_mapping": {
                        "response": resp,
                    }
                },
            }
        )

    return config


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate customer support agents with azure-ai-evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-provider",
        choices=["azure-openai", "foundry", "gpt5"],
        default="azure-openai",
        help=(
            "LLM judge provider: "
            "azure-openai (requires AZURE_OPENAI_* vars), "
            "foundry (Azure AI Foundry Anthropic endpoint via local proxy, requires AZURE_FOUNDRY_* vars), "
            "gpt5 (GPT-5 via Azure AI Foundry /openai/v1/ endpoint, requires AZURE_GPT5_* vars). "
            "Default: azure-openai."
        ),
    )
    parser.add_argument(
        "--agent",
        choices=["aws", "gcp"],
        default="aws",
        help="Which agent to evaluate (default: aws).",
    )
    parser.add_argument(
        "--mode",
        choices=["live", "dataset"],
        default="dataset",
        help="live = call agent per row; dataset = evaluate pre-collected JSONL (default: dataset).",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to evaluation JSONL (default: data/sample_traces.jsonl or data/eval_queries.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(eval_dir / "results"),
        help="Directory to write result JSON (default: results/).",
    )
    parser.add_argument(
        "--skip-ai-evaluators",
        action="store_true",
        help="Skip AI-assisted evaluators. Only routing_accuracy runs (no Azure OpenAI needed).",
    )
    args = parser.parse_args()

    # Resolve data file
    if args.data:
        data_path = args.data
    elif args.mode == "live":
        data_path = str(eval_dir / "data" / "eval_queries.jsonl")
    else:
        data_path = str(eval_dir / "data" / "sample_traces.jsonl")

    # Resolve output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"{args.agent}_{args.mode}_results.json")

    print(f"Agent  : {args.agent.upper()}")
    print(f"Mode   : {args.mode}")
    print(f"Provider: {args.model_provider}")
    print(f"Data   : {data_path}")
    print(f"Output : {output_path}")
    if args.skip_ai_evaluators:
        print("AI-assisted evaluators: SKIPPED")
    print()

    # Build evaluators and column mapping
    evaluators = _build_evaluators(args.skip_ai_evaluators, args.model_provider)
    evaluator_config = _build_column_mapping(args.mode, args.skip_ai_evaluators)

    # Load agent target for live mode
    target = None
    if args.mode == "live":
        print(f"Loading {args.agent.upper()} agent...")
        target = _get_aws_target() if args.agent == "aws" else _get_gcp_target()
        print("Agent loaded.\n")

    # Run evaluation
    from azure.ai.evaluation import evaluate  # noqa: PLC0415

    print("Running evaluation…")
    result = evaluate(
        data=data_path,
        target=target,
        evaluators=evaluators,
        evaluator_config=evaluator_config,
        output_path=output_path,
    )

    # Print aggregate metrics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    metrics = result.get("metrics", {})
    if metrics:
        for metric, value in sorted(metrics.items()):
            formatted = f"{value:.3f}" if isinstance(value, float) else str(value)
            print(f"  {metric:<40} {formatted}")
    else:
        print("  (no aggregate metrics returned)")

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
