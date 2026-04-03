"""GitHub Copilot SDK local coding assistant with Azure Application Insights tracing.

Trace architecture:
  - App-level spans (session.send) → AzureMonitorTraceExporter → App Insights
  - CLI internal spans (LLM, routing) → local JSONL file (COPILOT_TRACE_FILE)
  - Tool handler spans → App Insights (auto-parented to CLI span via W3C context)
"""
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a local coding assistant. You can help with:
- Reading and understanding code files
- Running shell commands (tests, builds, git operations)
- Listing files in a project
- Answering coding questions

You have access to three tools:
- read_file: read the contents of any file
- list_files: list files in a directory
- run_command: run a shell command

Always prefer using tools to look at actual code rather than guessing.
Be concise and practical."""


async def run_agent() -> None:
    """Start the interactive coding assistant."""
    from copilot import CopilotClient, SubprocessConfig, PermissionHandler
    from opentelemetry import trace

    from .tracing import setup_tracer_provider, get_tracer, flush_traces, OTLP_ENDPOINT, EXPORTER_MODE
    from .tools import ALL_TOOLS, reset_tool_tracker, get_tools_used

    # Initialize tracing before anything else
    setup_tracer_provider()
    tracer = get_tracer("copilot-local-agent")

    model = os.environ.get("COPILOT_MODEL", "gpt-5")
    capture_content = os.environ.get("OTEL_CAPTURE_CONTENT", "false").lower() == "true"

    # CLI subprocess sends its spans (LLM calls, routing, tool invocations) to
    # the same OTLP endpoint — either the local OTel Collector (default) or
    # direct OTLP if OTEL_EXPORTER=azure_monitor is set (no CLI OTLP then).
    if EXPORTER_MODE == "otlp":
        cli_telemetry: dict = {
            "otlp_endpoint": OTLP_ENDPOINT,
            "capture_content": True,
        }
    else:
        cli_telemetry = {}  # Direct Azure Monitor mode — CLI spans go to local file only

    trace_file = os.environ.get("COPILOT_TRACE_FILE", "")
    if trace_file:
        cli_telemetry["file_path"] = trace_file

    config = SubprocessConfig(telemetry=cli_telemetry if cli_telemetry else None)

    collector_note = f"OTel Collector ({OTLP_ENDPOINT}) → App Insights" if EXPORTER_MODE == "otlp" else "Azure Monitor (direct)"
    print(f"🤖 Copilot coding assistant (model: {model})")
    print(f"   Traces → {collector_note}")
    if trace_file:
        print(f"   CLI traces also → {trace_file}")
    print("   Type 'quit' or 'exit' to stop.\n")

    client = CopilotClient(config)
    await client.start()
    try:
        async with await client.create_session(
            model=model,
            tools=ALL_TOOLS,
            system_message={"content": SYSTEM_PROMPT},
            on_permission_request=PermissionHandler.approve_all,
        ) as session:
            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break

                # Wrap each turn in an app-level span so it shows in App Insights
                with tracer.start_as_current_span("copilot.session.turn") as span:
                    span.set_attribute("gen_ai.operation.name", "invoke_agent")
                    span.set_attribute("gen_ai.system", "github.copilot")
                    span.set_attribute("gen_ai.request.model", model)
                    span.set_attribute("user.input_length", len(user_input))
                    if capture_content:
                        span.set_attribute("user.input", user_input)

                    response_parts: list[str] = []
                    done = asyncio.Event()
                    reset_tool_tracker()

                    def on_event(event):
                        etype = event.type.value if hasattr(event.type, "value") else str(event.type)
                        if etype == "assistant.message":
                            content = getattr(event.data, "content", "")
                            if content:
                                response_parts.append(content)
                        elif etype == "session.idle":
                            done.set()

                    session.on(on_event)
                    await session.send(user_input)
                    await done.wait()

                    full_response = "".join(response_parts)
                    tools_used = get_tools_used()
                    span.set_attribute("assistant.response_length", len(full_response))
                    span.set_attribute("tools.used", ",".join(tools_used))
                    if capture_content:
                        span.set_attribute("assistant.response", full_response)

                    if full_response:
                        print(f"\nCopilot: {full_response}\n")
                    else:
                        print("\nCopilot: (no response)\n")
    finally:
        await client.stop()

    flush_traces()


def main() -> None:
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()
