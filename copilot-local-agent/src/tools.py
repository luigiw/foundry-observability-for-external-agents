"""Custom tools for the Copilot SDK coding assistant.

Each tool creates an OpenTelemetry child span. The Python Copilot SDK
automatically restores the CLI's W3C trace context around tool handlers,
so these spans are parented to the CLI's span in the distributed trace.
"""
import asyncio
import os
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Per-turn tool usage tracker (reset by agent.py before each session.send())
# ---------------------------------------------------------------------------
_current_turn_tools: list[str] = []


def reset_tool_tracker() -> None:
    """Clear the tool usage list. Call at the start of each agent turn."""
    _current_turn_tools.clear()


def get_tools_used() -> list[str]:
    """Return a snapshot of tool names called during the current turn."""
    return list(_current_turn_tools)

from pydantic import BaseModel, Field
from copilot import define_tool
from copilot.tools import ToolResult


# ---------------------------------------------------------------------------
# Parameter models
# ---------------------------------------------------------------------------

class RunCommandParams(BaseModel):
    command: str = Field(description="Shell command to execute")
    cwd: str = Field(default=".", description="Working directory (default: current dir)")
    timeout: int = Field(default=30, description="Timeout in seconds (max 60)")


class ReadFileParams(BaseModel):
    path: str = Field(description="Path to the file to read")
    max_lines: int = Field(default=200, description="Maximum number of lines to return")


class ListFilesParams(BaseModel):
    directory: str = Field(default=".", description="Directory to list (default: current dir)")
    pattern: str = Field(default="**/*", description="Glob pattern to filter files")
    max_results: int = Field(default=50, description="Maximum number of results")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@define_tool(description="Run a shell command and return its stdout/stderr. Use for running tests, git commands, build steps, etc.")
async def run_command(params: RunCommandParams) -> str:
    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)
    timeout = min(params.timeout, 60)

    with tracer.start_as_current_span("tool.run_command") as span:
        _current_turn_tools.append("run_command")
        span.set_attribute("tool.name", "run_command")
        span.set_attribute("shell.command", params.command)
        span.set_attribute("shell.cwd", params.cwd)
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    params.command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=params.cwd,
                    timeout=timeout,
                ),
            )
            output = result.stdout or ""
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            span.set_attribute("shell.exit_code", result.returncode)
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            span.set_attribute("error", True)
            return f"Command timed out after {timeout}s"
        except Exception as exc:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(exc))
            return f"Error running command: {exc}"


@define_tool(description="Read the contents of a file. Returns up to max_lines lines.")
async def read_file(params: ReadFileParams) -> str:
    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("tool.read_file") as span:
        _current_turn_tools.append("read_file")
        span.set_attribute("tool.name", "read_file")
        span.set_attribute("file.path", params.path)
        try:
            p = Path(params.path).expanduser().resolve()
            span.set_attribute("file.resolved_path", str(p))
            if not p.exists():
                return f"File not found: {params.path}"
            if not p.is_file():
                return f"Not a file: {params.path}"
            lines = p.read_text(errors="replace").splitlines()
            truncated = len(lines) > params.max_lines
            content = "\n".join(lines[: params.max_lines])
            span.set_attribute("file.lines_read", min(len(lines), params.max_lines))
            if truncated:
                content += f"\n\n... (truncated at {params.max_lines} lines, {len(lines)} total)"
            return content
        except Exception as exc:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(exc))
            return f"Error reading file: {exc}"


@define_tool(description="List files in a directory matching a glob pattern.")
async def list_files(params: ListFilesParams) -> str:
    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("tool.list_files") as span:
        _current_turn_tools.append("list_files")
        span.set_attribute("tool.name", "list_files")
        span.set_attribute("fs.directory", params.directory)
        span.set_attribute("fs.pattern", params.pattern)
        try:
            base = Path(params.directory).expanduser().resolve()
            if not base.exists():
                return f"Directory not found: {params.directory}"
            matches = sorted(base.glob(params.pattern))
            files = [str(p.relative_to(base)) for p in matches if p.is_file()]
            files = files[: params.max_results]
            span.set_attribute("fs.file_count", len(files))
            if not files:
                return "No files found."
            return "\n".join(files)
        except Exception as exc:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(exc))
            return f"Error listing files: {exc}"


ALL_TOOLS = [run_command, read_file, list_files]
