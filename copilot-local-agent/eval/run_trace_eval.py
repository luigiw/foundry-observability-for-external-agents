"""Batch evaluation of Copilot CLI agent traces using Azure AI Foundry.

Queries App Insights for trace IDs tagged with gen_ai.agent.id, then submits
them to the Foundry evaluation API.  All evaluators run remotely in Foundry —
no local LLM calls needed.

Built-in evaluators (always included):
    task_adherence, task_completion, tool_call_accuracy

Registered custom evaluators (included when register_evaluators.py has been run):
    hw-copilot-cli-command-safety   — flags dangerous shell commands (boolean)
    hw-copilot-cli-code-correctness — technical accuracy of the response (1-5)
    hw-copilot-cli-groundedness     — response grounded in tool output (1-5)

Authentication:
    Uses DefaultAzureCredential — run `az login` before executing.

Required env vars (copilot-local-agent/.env):
    AZURE_AI_PROJECT_ENDPOINT      — Foundry project endpoint
    APPINSIGHTS_RESOURCE_ID        — Full App Insights resource ID
    AZURE_AI_MODEL_DEPLOYMENT_NAME — Model deployment used as judge (falls back
                                     to AZURE_GPT5_DEPLOYMENT, then "gpt-5")
    AGENT_ID                       — gen_ai.agent.id to filter traces by
                                     (default: "hw-copilot-cli")

Usage:
    cd copilot-local-agent
    source .venv/bin/activate
    python eval/register_evaluators.py          # one-time: register custom evaluators
    python eval/run_trace_eval.py               # run all evaluators
    python eval/run_trace_eval.py --hours 6
    python eval/run_trace_eval.py --builtin-only  # skip custom evaluators
    python eval/run_trace_eval.py --no-wait       # submit and exit without polling
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from pprint import pprint
from typing import Any

from dotenv import load_dotenv

eval_dir = Path(__file__).parent
load_dotenv(eval_dir.parent / ".env")
load_dotenv(eval_dir / ".env", override=True)

# ── Config ────────────────────────────────────────────────────────────────────

ENDPOINT = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
APPINSIGHTS_RESOURCE_ID = os.environ.get("APPINSIGHTS_RESOURCE_ID", "")
AGENT_ID = os.environ.get("AGENT_ID", "hw-copilot-cli")
MODEL_DEPLOYMENT = (
    os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME")
    or os.environ.get("AZURE_GPT5_DEPLOYMENT")
    or "gpt-5"
)


# ── Trace ID query ────────────────────────────────────────────────────────────


def get_trace_ids(
    appinsights_resource_id: str,
    agent_id: str,
    start_time: datetime,
    end_time: datetime,
) -> list[str]:
    """Query App Insights for distinct operation_Id values matching agent_id."""
    from azure.identity import DefaultAzureCredential
    from azure.monitor.query import LogsQueryClient, LogsQueryStatus

    kql = f"""
dependencies
| where timestamp between (datetime({start_time.isoformat()}) .. datetime({end_time.isoformat()}))
| extend agent_id = tostring(customDimensions["gen_ai.agent.id"])
| where agent_id == "{agent_id}"
| distinct operation_Id
""".strip()

    print(f"Querying App Insights for traces (agent_id={agent_id})...")
    with DefaultAzureCredential() as credential:
        client = LogsQueryClient(credential)
        try:
            response = client.query_resource(
                appinsights_resource_id,
                query=kql,
                timespan=None,
            )
        except Exception as exc:
            print(f"Error querying App Insights: {exc}")
            return []

    if response.status != LogsQueryStatus.SUCCESS:
        print(f"Query failed: {response.status}")
        if getattr(response, "partial_error", None):
            print(f"  Partial error: {response.partial_error}")
        return []

    trace_ids: list[str] = []
    for table in response.tables:
        for row in table.rows:
            trace_ids.append(row[0])
    return trace_ids


# ── Evaluator config ─────────────────────────────────────────────────────────

# Names must match what was registered via register_evaluators.py
COMMAND_SAFETY_NAME   = "hw-copilot-cli-command-safety"
CODE_CORRECTNESS_NAME = "hw-copilot-cli-code-correctness"
GROUNDEDNESS_NAME     = "hw-copilot-cli-groundedness"

REGISTRY_FILE = eval_dir / ".evaluator_registry.json"


def _is_registered() -> bool:
    """Return True if custom evaluators have been registered (registry file exists)."""
    return REGISTRY_FILE.exists()



def _builtin_evaluator_config(name: str, evaluator_name: str) -> dict[str, Any]:
    """Config for Foundry built-in evaluators (use tool_definitions from traces)."""
    return {
        "type": "azure_ai_evaluator",
        "name": name,
        "evaluator_name": evaluator_name,
        "data_mapping": {
            "query":            "{{query}}",
            "response":         "{{response}}",
            "tool_definitions": "{{tool_definitions}}",
        },
        "initialization_parameters": {
            "deployment_name": MODEL_DEPLOYMENT,
        },
    }


def _custom_evaluator_config(name: str, evaluator_name: str, threshold: float = 3.0) -> dict[str, Any]:
    """Config for registered custom evaluators (query + response from traces)."""
    cfg: dict[str, Any] = {
        "type": "azure_ai_evaluator",
        "name": name,
        "evaluator_name": evaluator_name,
        "data_mapping": {
            "query":    "{{query}}",
            "response": "{{response}}",
        },
        "initialization_parameters": {
            "deployment_name": MODEL_DEPLOYMENT,
        },
    }
    if evaluator_name != COMMAND_SAFETY_NAME:
        cfg["initialization_parameters"]["threshold"] = threshold
    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch eval of Copilot CLI agent traces via Azure AI Foundry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--hours", type=int, default=1,
                        help="Lookback window in hours (default: 1).")
    parser.add_argument("--agent-id", default=AGENT_ID,
                        help=f"gen_ai.agent.id to filter by (default: {AGENT_ID}).")
    parser.add_argument("--no-wait", action="store_true",
                        help="Submit the eval run and exit without polling for completion.")
    parser.add_argument("--builtin-only", action="store_true",
                        help="Use only Foundry built-in evaluators; skip registered custom ones.")
    args = parser.parse_args()

    # Validate required env vars
    missing = [v for v in ("AZURE_AI_PROJECT_ENDPOINT", "APPINSIGHTS_RESOURCE_ID") if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing required env vars: {', '.join(missing)}")
        print("Set them in copilot-local-agent/.env — see eval/.env.example for reference.")
        sys.exit(1)

    end_time = datetime.now(tz=timezone.utc)
    start_time = end_time - timedelta(hours=args.hours)

    print(f"Time range : {start_time.isoformat()} → {end_time.isoformat()}")
    print(f"Agent ID   : {args.agent_id}")
    print(f"LLM judge  : {MODEL_DEPLOYMENT}")
    print(f"Project    : {ENDPOINT}\n")

    trace_ids = get_trace_ids(APPINSIGHTS_RESOURCE_ID, args.agent_id, start_time, end_time)
    if not trace_ids:
        print("No traces found. Make sure:")
        print("  1. The OTel Collector is running (docker compose up -d)")
        print("  2. You have used the agent within the lookback window")
        print("  3. AGENT_ID matches the gen_ai.agent.id set by the transform processor")
        sys.exit(0)

    print(f"Found {len(trace_ids)} trace(s):")
    for tid in trace_ids:
        print(f"  {tid}")

    from azure.identity import DefaultAzureCredential
    from azure.ai.projects import AIProjectClient

    with DefaultAzureCredential() as credential, \
         AIProjectClient(endpoint=ENDPOINT, credential=credential) as project_client:
        oai_client = project_client.get_openai_client()

        testing_criteria = [
            _builtin_evaluator_config("task_adherence",     "builtin.task_adherence"),
            _builtin_evaluator_config("task_completion",    "builtin.task_completion"),
            _builtin_evaluator_config("tool_call_accuracy", "builtin.tool_call_accuracy"),
        ]

        if not args.builtin_only:
            if _is_registered():
                print("  Custom evaluators registered — adding to criteria")
                testing_criteria += [
                    _custom_evaluator_config("command_safety",   COMMAND_SAFETY_NAME),
                    _custom_evaluator_config("code_correctness", CODE_CORRECTNESS_NAME, threshold=3.0),
                    _custom_evaluator_config("groundedness",     GROUNDEDNESS_NAME,     threshold=3.0),
                ]
            else:
                print("  NOTE: Custom evaluators not registered.")
                print("  Run `python eval/register_evaluators.py` to add them.")

        print(f"\nCreating evaluation group ({len(testing_criteria)} evaluators)...")
        eval_obj = oai_client.evals.create(
            name="copilot_cli_trace_eval",
            data_source_config={"type": "azure_ai_source", "scenario": "traces"},
            testing_criteria=testing_criteria,
        )
        print(f"  Created  id={eval_obj.id}  name={eval_obj.name}")

        run_name = f"copilot_cli_{args.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\nSubmitting eval run: {run_name}")
        run = oai_client.evals.runs.create(
            eval_id=eval_obj.id,
            name=run_name,
            metadata={
                "agent_id": args.agent_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            },
            data_source={
                "type": "azure_ai_traces",
                "trace_ids": trace_ids,
                "lookback_hours": args.hours,
            },
        )
        print(f"  Run submitted  id={run.id}  status={run.status}")

        if args.no_wait:
            print("\nResults will appear in Azure AI Foundry → Evaluation tab.")
        else:
            print("\nWaiting for eval run to complete...")
            while True:
                run = oai_client.evals.runs.retrieve(run_id=run.id, eval_id=eval_obj.id)
                print(f"  Status: {run.status}")
                if run.status in {"completed", "failed", "canceled"}:
                    break
                time.sleep(5)

            print("\n=== Eval Result ===")
            pprint(run.model_dump() if hasattr(run, "model_dump") else run)

        print(f"\nEval group retained. ID: {eval_obj.id}")


if __name__ == "__main__":
    main()
