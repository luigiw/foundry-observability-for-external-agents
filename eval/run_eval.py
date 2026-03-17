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

app-insights
    Fetch traces from Azure Application Insights and evaluate them.
    Both agents can be evaluated in one run (--agent both) for side-by-side comparison.
    Requires APPLICATIONINSIGHTS_APP_ID and APPLICATIONINSIGHTS_QUERY_API_KEY env vars.
    Use --hours to control the lookback window (default: 24h).
    Note: routing_accuracy is skipped in this mode (no ground-truth labels in traces).

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

# Evaluate both agents from App Insights traces (last 24h), compare results:
python run_eval.py --mode app-insights --agent both --model-provider gpt5

# Evaluate only the AWS agent from the last 48h of traces:
python run_eval.py --mode app-insights --agent aws --hours 48 --model-provider foundry
"""
import os
import sys
import json
import tempfile
import argparse
from pathlib import Path

from dotenv import load_dotenv

eval_dir = Path(__file__).parent
repo_root = eval_dir.parent

# Load eval-local .env first; agent-specific envs are loaded per agent below.
load_dotenv(eval_dir / ".env")

# Fields included in evaluation JSONL rows — excludes timestamp and other metadata
# fields that pandas may coerce to non-serializable types (e.g. pandas.Timestamp).
_TRACE_EVAL_FIELDS = frozenset({
    "query", "response", "query_type", "handled_by", "needs_escalation",
    "agent", "llm_calls",
})


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
        from evaluators.trace_quality import TraceQualityEvaluator  # noqa: PLC0415

        model_config = _build_model_config(provider)
        evaluators.update(
            {
                "intent_resolution": IntentResolutionEvaluator(model_config),
                "relevance": RelevanceEvaluator(model_config),
                "coherence": CoherenceEvaluator(model_config),
                "fluency": FluencyEvaluator(model_config),
                "trace_quality": TraceQualityEvaluator(model_config),
            }
        )

    return evaluators


def _build_column_mapping(mode: str, skip_ai: bool) -> dict:
    """Build per-evaluator column mappings.

    In live mode, agent outputs are referenced as ${outputs.<field>}.
    In dataset and app-insights modes, all fields come from the data file: ${data.<field>}.
    """
    resp = "${outputs.response}" if mode == "live" else "${data.response}"
    qtype = "${outputs.query_type}" if mode == "live" else "${data.query_type}"

    config = {}

    # routing_accuracy requires ground-truth labels — only available in live/dataset modes.
    if mode != "app-insights":
        config["routing_accuracy"] = {
            "column_mapping": {
                "expected": "${data.expected_query_type}",
                "actual": qtype,
            }
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
                "trace_quality": {
                    "column_mapping": {
                        "trace": "${data.trace_json}",
                    }
                },
            }
        )
        # trace_quality requires trace_json which is only present in app-insights mode
        if mode != "app-insights":
            del config["trace_quality"]

    return config


# ── App Insights mode helpers ─────────────────────────────────────────────────


def _eval_from_app_insights(
    agent: str,
    hours: int,
    evaluators: dict,
    evaluator_config: dict,
    output_path: str,
) -> dict | None:
    """Fetch traces from App Insights for *agent* and run evaluation.

    Returns the evaluation result dict, or None if no traces were found.
    """
    sys.path.insert(0, str(eval_dir))
    from query_app_insights import query_traces  # noqa: PLC0415

    print(f"Querying App Insights for {agent.upper()} traces (last {hours}h)...")
    traces = query_traces(agent, hours)
    if not traces:
        print(f"  No traces found for {agent.upper()} in the last {hours}h.")
        return None
    print(f"  Found {len(traces)} trace(s).")

    # Write traces to a temp JSONL file for the evaluate() call.
    # Only include fields needed for evaluation — exclude timestamp and other
    # metadata that pandas may coerce to non-serializable types (e.g. Timestamp).
    # Each row also gets trace_json = the full trace dict as a compact JSON string
    # so TraceQualityEvaluator can receive it via column_mapping.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, dir=eval_dir / "results"
    ) as tmp:
        for t in traces:
            row = {k: v for k, v in t.items() if k in _TRACE_EVAL_FIELDS}
            row["trace_json"] = json.dumps(t, separators=(",", ":"))
            tmp.write(json.dumps(row) + "\n")
        tmp_path = tmp.name

    try:
        from azure.ai.evaluation import evaluate  # noqa: PLC0415

        return evaluate(
            data=tmp_path,
            evaluators=evaluators,
            evaluator_config=evaluator_config,
            output_path=output_path,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _print_comparison(aws_metrics: dict, gcp_metrics: dict) -> None:
    """Print a side-by-side comparison table of AWS vs GCP evaluation metrics."""
    all_keys = sorted(set(aws_metrics) | set(gcp_metrics))
    if not all_keys:
        print("  (no metrics to compare)")
        return

    col_w = 38
    print(f"\n{'=' * 78}")
    print("AGENT COMPARISON")
    print(f"{'=' * 78}")
    print(f"  {'Metric':<{col_w}} {'AWS':>8}  {'GCP':>8}  {'Delta':>8}")
    print(f"  {'-' * col_w} {'--------':>8}  {'--------':>8}  {'--------':>8}")
    for key in all_keys:
        aws_val = aws_metrics.get(key)
        gcp_val = gcp_metrics.get(key)
        aws_str = f"{aws_val:.3f}" if isinstance(aws_val, float) else (str(aws_val) if aws_val is not None else "n/a")
        gcp_str = f"{gcp_val:.3f}" if isinstance(gcp_val, float) else (str(gcp_val) if gcp_val is not None else "n/a")
        if isinstance(aws_val, float) and isinstance(gcp_val, float):
            delta = gcp_val - aws_val
            delta_str = f"{delta:+.3f}"
        else:
            delta_str = "n/a"
        print(f"  {key:<{col_w}} {aws_str:>8}  {gcp_str:>8}  {delta_str:>8}")
    print(f"{'=' * 78}")


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
        choices=["aws", "gcp", "both"],
        default="aws",
        help=(
            "Which agent to evaluate. "
            "'both' is only valid with --mode app-insights and runs a side-by-side comparison. "
            "Default: aws."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["live", "dataset", "app-insights"],
        default="dataset",
        help=(
            "live = call agent per row; "
            "dataset = evaluate pre-collected JSONL; "
            "app-insights = fetch traces from Azure Application Insights. "
            "Default: dataset."
        ),
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Lookback window in hours for --mode app-insights (default: 24).",
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

    # Validate --agent both
    if args.agent == "both" and args.mode != "app-insights":
        parser.error("--agent both is only supported with --mode app-insights")

    # Resolve output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build evaluators (routing_accuracy excluded in app-insights mode — no ground truth;
    # trace_quality excluded in live/dataset modes — no trace_json column)
    evaluators = _build_evaluators(args.skip_ai_evaluators, args.model_provider)
    if args.mode == "app-insights" and "routing_accuracy" in evaluators:
        del evaluators["routing_accuracy"]
    if args.mode != "app-insights" and "trace_quality" in evaluators:
        del evaluators["trace_quality"]
    evaluator_config = _build_column_mapping(args.mode, args.skip_ai_evaluators)

    # ── app-insights mode ──────────────────────────────────────────────────────
    if args.mode == "app-insights":
        agents_to_eval = ["aws", "gcp"] if args.agent == "both" else [args.agent]
        results_by_agent: dict[str, dict] = {}

        for ag in agents_to_eval:
            output_path = str(output_dir / f"{ag}_app_insights_results.json")
            print(f"\nAgent  : {ag.upper()}")
            print(f"Mode   : app-insights  (last {args.hours}h)")
            print(f"Provider: {args.model_provider}")
            print(f"Output : {output_path}")
            if args.skip_ai_evaluators:
                print("AI-assisted evaluators: SKIPPED")
            print()

            result = _eval_from_app_insights(
                ag, args.hours, evaluators, evaluator_config, output_path
            )
            if result is None:
                continue
            results_by_agent[ag] = result

            metrics = result.get("metrics", {})
            print(f"\n{'=' * 60}")
            print(f"EVALUATION SUMMARY — {ag.upper()}")
            print(f"{'=' * 60}")
            for metric, value in sorted(metrics.items()):
                formatted = f"{value:.3f}" if isinstance(value, float) else str(value)
                print(f"  {metric:<40} {formatted}")
            print(f"\nFull results saved to: {output_path}")

        if args.agent == "both" and len(results_by_agent) == 2:
            _print_comparison(
                results_by_agent["aws"].get("metrics", {}),
                results_by_agent["gcp"].get("metrics", {}),
            )
        return

    # ── live / dataset modes ───────────────────────────────────────────────────

    # Resolve data file
    if args.data:
        data_path = args.data
    elif args.mode == "live":
        data_path = str(eval_dir / "data" / "eval_queries.jsonl")
    else:
        data_path = str(eval_dir / "data" / "sample_traces.jsonl")

    output_path = str(output_dir / f"{args.agent}_{args.mode}_results.json")

    print(f"Agent  : {args.agent.upper()}")
    print(f"Mode   : {args.mode}")
    print(f"Provider: {args.model_provider}")
    print(f"Data   : {data_path}")
    print(f"Output : {output_path}")
    if args.skip_ai_evaluators:
        print("AI-assisted evaluators: SKIPPED")
    print()

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
