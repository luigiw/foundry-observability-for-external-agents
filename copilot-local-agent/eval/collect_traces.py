"""Collect traces from the live Copilot SDK agent to build an offline eval dataset.

Loads queries from data/eval_queries.jsonl, runs each through the agent, and
writes the results (with tools_used) to a JSONL file for offline evaluation.

Requires:
    - The agent dependencies installed (pip install -r ../requirements.txt)
    - GitHub Copilot authentication (gh auth login or GITHUB_TOKEN)
    - OTEL_CAPTURE_CONTENT=true is NOT required — this script captures content
      directly, not via OTel spans.

Usage:
    cd copilot-local-agent
    python eval/collect_traces.py
    python eval/collect_traces.py --data eval/data/eval_queries.jsonl --output eval/data/sample_traces.jsonl
"""

import asyncio
import json
import os
import sys
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

# Allow importing from the parent src/ package
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))
load_dotenv(_root / ".env")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a local coding assistant. You can help with:
- Reading and understanding code files
- Running shell commands (tests, builds, git operations)
- Listing files in a project
- Answering coding questions

Always prefer using tools to look at actual code rather than guessing.
Be concise and practical."""


async def run_query(client, query: str, model: str) -> dict:
    """Run a single query through the agent and return the trace dict."""
    from copilot import PermissionHandler
    from src.tools import ALL_TOOLS, reset_tool_tracker, get_tools_used

    reset_tool_tracker()

    async with await client.create_session(
        model=model,
        tools=ALL_TOOLS,
        system_message={"content": SYSTEM_PROMPT},
        on_permission_request=PermissionHandler.approve_all,
    ) as session:
        response_parts: list[str] = []
        done = asyncio.Event()

        def on_event(event):
            etype = event.type.value if hasattr(event.type, "value") else str(event.type)
            if etype == "assistant.message":
                content = getattr(event.data, "content", "")
                if content:
                    response_parts.append(content)
            elif etype == "session.idle":
                done.set()

        session.on(on_event)
        import time
        t0 = time.monotonic()
        await session.send(query)
        await asyncio.wait_for(done.wait(), timeout=120)
        duration_ms = int((time.monotonic() - t0) * 1000)

        return {
            "user_input": query,
            "response": "".join(response_parts),
            "tools_used": ",".join(get_tools_used()),
            "model": model,
            "duration_ms": duration_ms,
        }


async def collect(data_path: Path, output_path: Path, model: str) -> None:
    from copilot import CopilotClient, SubprocessConfig

    queries = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    print(f"Collecting {len(queries)} traces → {output_path}")

    config = SubprocessConfig()
    client = CopilotClient(config)
    await client.start()

    results = []
    try:
        for i, row in enumerate(queries, 1):
            query = row["user_input"]
            print(f"  [{i}/{len(queries)}] {query[:60]}...")
            try:
                trace = await run_query(client, query, model)
                # Carry over expected_tools if present in the input
                if "expected_tools" in row:
                    trace["expected_tools"] = row["expected_tools"]
                results.append(trace)
            except Exception as exc:
                logger.warning("Query %d failed: %s", i, exc)
                results.append({
                    "user_input": query,
                    "response": "",
                    "tools_used": "",
                    "model": model,
                    "duration_ms": 0,
                    "error": str(exc),
                })
    finally:
        await client.stop()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. {len(results)} traces written to {output_path}")


def main():
    eval_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Collect Copilot SDK agent traces for eval.")
    parser.add_argument("--data", default=str(eval_dir / "data" / "eval_queries.jsonl"),
                        help="Input JSONL of queries.")
    parser.add_argument("--output", default=str(eval_dir / "data" / "sample_traces.jsonl"),
                        help="Output JSONL for collected traces.")
    parser.add_argument("--model", default=os.environ.get("COPILOT_MODEL", "gpt-5"),
                        help="Copilot model to use.")
    args = parser.parse_args()

    asyncio.run(collect(Path(args.data), Path(args.output), args.model))


if __name__ == "__main__":
    main()
