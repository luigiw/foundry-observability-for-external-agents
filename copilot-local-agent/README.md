# Copilot SDK Local Agent

A local coding assistant built with the [GitHub Copilot SDK](https://github.com/github/copilot-sdk) (Python), with **all traces** sent to **Azure Application Insights** via an OpenTelemetry Collector sidecar.

> **Note:** The Copilot SDK is in technical preview and may change.

## Architecture

```
You ──► src/agent.py (CopilotClient)
            │
            ├─ app spans (per turn) ──────────────────────────┐
            │                                                  │
            └─ Copilot CLI subprocess                          │
                   │                                           │
                   ├─ LLM / routing spans ────────────────────┤
                   │   (gen_ai.* semantic conventions)         │
                   │                                           ▼
                   └─ tool handler spans          OTel Collector (Docker)
                      (run_command, read_file,          :4317 gRPC
                       list_files)                      :4318 HTTP
                                                             │
                                                             ▼
                                                  Azure Application Insights
```

**Everything** — Python app spans, Copilot CLI LLM spans, tool handler spans — flows through the local OTel Collector container to App Insights via the `azuremonitor` exporter.

A direct Azure Monitor mode is also available (no Docker needed) by setting `OTEL_EXPORTER=azure_monitor`, but in that mode only the Python app spans and tool handler spans reach App Insights (CLI internal spans are excluded).

## Setup

### 1. Prerequisites

- Python 3.10+
- Docker (for the OTel Collector sidecar)
- GitHub Copilot access (`gh auth login` or a GitHub token)
- An Azure Application Insights resource

### 2. Install Python dependencies

```bash
cd copilot-local-agent
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set APPLICATIONINSIGHTS_CONNECTION_STRING at minimum
```

Get your connection string from:
**Azure Portal → App Insights resource → Overview → Connection string**

### 4. Start the OTel Collector sidecar

```bash
cd copilot-local-agent
docker compose up -d
```

This starts `otel/opentelemetry-collector-contrib` on ports `4317` (gRPC) and `4318` (HTTP). It reads your `APPLICATIONINSIGHTS_CONNECTION_STRING` from `.env` and forwards all OTLP traces to App Insights.

## Run

### Interactive chat

```bash
cd copilot-local-agent
python -m src.agent
```

### Smoke tests

```bash
python test_local.py
```

Runs 3 scripted prompts (list files, read README, run a command).

## Viewing Traces in App Insights

1. Open your App Insights resource in the Azure Portal
2. Go to **Investigate → Transaction search** or **Investigate → Performance**
3. Filter by **Role name**: `copilot-local-agent`
4. Each user turn appears as a `copilot.session.turn` span with nested:
   - Copilot CLI LLM invocation spans (`gen_ai.*` attributes)
   - Tool call spans (`tool.run_command`, `tool.read_file`, `tool.list_files`)

**Useful Kusto query:**
```kusto
dependencies
| where cloud_RoleName == "copilot-local-agent"
| project timestamp, name, duration, success, customDimensions
| order by timestamp desc
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | ✅ | — | App Insights connection string |
| `OTEL_EXPORTER` | No | `otlp` | `otlp` (sidecar) or `azure_monitor` (direct) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | `http://localhost:4318` | OTel Collector HTTP endpoint |
| `OTEL_SERVICE_NAME` | No | `copilot-local-agent` | Service name in App Insights |
| `COPILOT_MODEL` | No | `gpt-5` | Model for the Copilot session |
| `GITHUB_TOKEN` | No | gh auth | GitHub token for authentication |
| `COPILOT_TRACE_FILE` | No | *(disabled)* | Also dump CLI spans to a local JSONL file |

## Custom Tools

| Tool | Description |
|---|---|
| `run_command` | Run a shell command (stdout + stderr, configurable timeout) |
| `read_file` | Read a file with line limit |
| `list_files` | List files in a directory with glob filtering |

Add new tools in `src/tools.py` using the `@define_tool` decorator with a Pydantic params model.

## Tracing Design Notes

- **OTel Collector sidecar** (`otelcol-contrib`) bridges OTLP → Azure Monitor, so both Python app spans and Copilot CLI internal spans (LLM calls, routing) all appear in one distributed trace in App Insights
- **W3C trace context propagation** — the Python Copilot SDK automatically restores the CLI's trace context around tool handlers, linking tool spans to CLI spans across the process boundary
- **Direct mode** (`OTEL_EXPORTER=azure_monitor`) bypasses the collector using `AzureMonitorTraceExporter` from the Python app directly; CLI spans are not forwarded in this mode
