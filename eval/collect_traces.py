"""Collect traces from a live agent and save them to a JSONL file.

Usage:
    python collect_traces.py --agent aws --output data/aws_traces.jsonl
    python collect_traces.py --agent gcp --output data/gcp_traces.jsonl
    python collect_traces.py --agent gcp --input data/eval_queries.jsonl --output data/gcp_traces.jsonl

The output JSONL has the same schema as sample_traces.jsonl and can be passed
directly to run_eval.py with --mode dataset.
"""
import os
import sys
import json
import argparse
from pathlib import Path

eval_dir = Path(__file__).parent
repo_root = eval_dir.parent


def _load_aws_agent():
    """Lazily load the AWS Bedrock agent."""
    aws_path = repo_root / "aws" / "langgraph-customer-support"
    if str(aws_path) not in sys.path:
        sys.path.insert(0, str(aws_path))
    from src.graph import invoke_support  # noqa: PLC0415
    return invoke_support


def _load_gcp_agent():
    """Lazily load the GCP / Azure AI Foundry agent."""
    from dotenv import load_dotenv  # noqa: PLC0415

    gcp_path = repo_root / "gcp" / "langgraph-customer-support"
    if str(gcp_path) not in sys.path:
        sys.path.insert(0, str(gcp_path))
    # Load the GCP-specific .env so AZURE_FOUNDRY_* vars are set
    load_dotenv(gcp_path / ".env", override=False)
    from src.graph import invoke_support  # noqa: PLC0415
    return invoke_support


def collect_traces(
    invoke_support,
    input_file: str,
    output_file: str,
    verbose: bool = False,
) -> None:
    with open(input_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    traces = []
    for i, row in enumerate(rows, 1):
        query = row["query"]
        if verbose:
            print(f"[{i}/{len(rows)}] {query[:80]}...")

        result = invoke_support(query)
        trace = {
            **row,
            "response": result["response"],
            "query_type": result["query_type"],
            "handled_by": result["handled_by"],
            "needs_escalation": result["needs_escalation"],
        }
        traces.append(trace)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")

    print(f"Saved {len(traces)} traces → {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect agent traces and save to JSONL for offline evaluation."
    )
    parser.add_argument(
        "--agent",
        choices=["aws", "gcp"],
        required=True,
        help="Which agent to collect traces from.",
    )
    parser.add_argument(
        "--input",
        default=str(eval_dir / "data" / "eval_queries.jsonl"),
        help="Input JSONL file with query rows (default: data/eval_queries.jsonl).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL file (default: data/<agent>_traces.jsonl).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress.")
    args = parser.parse_args()

    output = args.output or str(eval_dir / "data" / f"{args.agent}_traces.jsonl")

    print(f"Loading {args.agent.upper()} agent...")
    if args.agent == "aws":
        invoke_support = _load_aws_agent()
    else:
        invoke_support = _load_gcp_agent()
    print("Agent loaded. Collecting traces...\n")

    collect_traces(invoke_support, args.input, output, verbose=args.verbose)


if __name__ == "__main__":
    main()
