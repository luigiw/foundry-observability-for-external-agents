"""Smoke tests for the Copilot SDK local agent.

Runs 3 scripted prompts against the agent and verifies:
  - No exceptions
  - Non-empty responses
  - Traces are flushed without errors

Usage:
    cd copilot-local-agent
    python test_local.py
"""
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TEST_PROMPTS = [
    {
        "name": "list files",
        "prompt": "List the files in the current directory.",
        "expect_keywords": [],  # just non-empty is enough
    },
    {
        "name": "read file",
        "prompt": "Read the file README.md and give me a one-sentence summary.",
        "expect_keywords": [],
    },
    {
        "name": "run command",
        "prompt": "Run the command 'echo hello from copilot agent' and tell me the output.",
        "expect_keywords": ["hello"],
    },
]


async def run_test(prompt_info: dict, client, model: str) -> bool:
    from copilot import PermissionHandler
    from opentelemetry import trace
    from src.tracing import get_tracer
    from src.tools import ALL_TOOLS
    from src.agent import SYSTEM_PROMPT

    tracer = get_tracer("copilot-local-agent.test")

    async with await client.create_session(
        model=model,
        tools=ALL_TOOLS,
        system_message={"content": SYSTEM_PROMPT},
        on_permission_request=PermissionHandler.approve_all,
    ) as session:
        with tracer.start_as_current_span(f"test.{prompt_info['name']}") as span:
            span.set_attribute("test.name", prompt_info["name"])
            span.set_attribute("test.prompt", prompt_info["prompt"])

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
            await session.send(prompt_info["prompt"])
            await asyncio.wait_for(done.wait(), timeout=120)

            response = "".join(response_parts)
            span.set_attribute("test.response_length", len(response))

            if not response:
                logger.error("[%s] FAIL — empty response", prompt_info["name"])
                span.set_attribute("test.passed", False)
                return False

            for kw in prompt_info.get("expect_keywords", []):
                if kw.lower() not in response.lower():
                    logger.error("[%s] FAIL — expected keyword '%s' not found", prompt_info["name"], kw)
                    span.set_attribute("test.passed", False)
                    return False

            logger.info("[%s] PASS — response length=%d", prompt_info["name"], len(response))
            span.set_attribute("test.passed", True)
            return True


async def main() -> None:
    from copilot import CopilotClient, SubprocessConfig, PermissionHandler
    from src.tracing import setup_tracer_provider, flush_traces

    setup_tracer_provider()

    model = os.environ.get("COPILOT_MODEL", "gpt-5")
    logger.info("Running smoke tests with model=%s", model)

    config = SubprocessConfig()
    results = []

    client = CopilotClient(config)
    await client.start()
    try:
        for prompt_info in TEST_PROMPTS:
            logger.info("--- Test: %s ---", prompt_info["name"])
            try:
                passed = await run_test(prompt_info, client, model)
                results.append((prompt_info["name"], passed))
            except Exception as exc:
                logger.exception("[%s] FAIL — exception: %s", prompt_info["name"], exc)
                results.append((prompt_info["name"], False))
    finally:
        await client.stop()

    flush_traces()

    print("\n=== Test Results ===")
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
