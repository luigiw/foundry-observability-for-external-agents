# Hurdles: Getting GitHub Copilot CLI Traces into Azure Application Insights

This document captures the friction points encountered while trying to get
OpenTelemetry traces from the **GitHub Copilot CLI** to flow into Azure Application
Insights. Written to help teammates avoid the same pitfalls and to surface product
gaps for the Copilot SDK / CLI teams.

---

## 1. PyPI `[telemetry]` extra is not published

**What we tried:**
```bash
pip install "github-copilot-sdk[telemetry]"
```
**Symptom:** The install silently succeeds but no OTel packages are added.  
**Root cause:** The `[telemetry]` optional extra is declared in the development
source tree but never shipped to PyPI.  
**Workaround:** Manually add all OTel dependencies to `requirements.txt`:
```
opentelemetry-sdk
opentelemetry-exporter-otlp-proto-http
azure-monitor-opentelemetry-exporter
```
**Product gap:** The published package extras are out of sync with the source.
Either remove the undeclared extra from the README or actually ship it.

---

## 2. CLI subprocess does not inherit the parent `TracerProvider`

**What we tried:** Setting up a `TracerProvider` in the Python wrapper and
calling `await client.create_session(...)` — expecting CLI spans to appear.  
**Symptom:** Only the wrapper's own spans appeared; the CLI's internal LLM spans
(`gen_ai.operation.name = "chat"`, model calls, token counts) were missing.  
**Root cause:** The Copilot CLI is a subprocess with its own telemetry pipeline.
It does not inherit the parent process's `TracerProvider`.  
**Workaround:** Pass explicit telemetry config to `SubprocessConfig`:
```python
from copilot import SubprocessConfig

SubprocessConfig(telemetry={
    "otlp_endpoint": "http://localhost:4318",
    "capture_content": True,
})
```
**Product gap:** This configuration is entirely undocumented. There is no
`TelemetryConfig` type, no SDK docstring, and no example. Developers have no way
to discover this without reading source code or getting help.

---

## 3. No way to verify traces are actually flowing without manual inspection

**What we tried:** Starting the agent, sending a prompt, then checking App Insights.  
**Symptom:** App Insights has a ~2–5 minute ingestion lag. We couldn't tell if
traces were lost in the pipeline or just delayed.  
**Workaround:** Sent a synthetic OTLP span directly to the collector using a raw
HTTP request to `localhost:4318` with a known marker string, then searched for
it in App Insights:
```python
import urllib.request, json, struct, time

# Build a minimal OTLP protobuf-like JSON payload and POST to :4318/v1/traces
```
**Product gap:** There should be a `copilot telemetry verify` or `copilot telemetry status`
command that confirms the OTLP pipeline is reachable and emitting. The collector
logs alone are not enough.

---

## 4. `configure_azure_monitor()` creates duplicate spans

**What we tried:** Using `configure_azure_monitor(connection_string=...)` for
convenience alongside a manually-created `BatchSpanProcessor`.  
**Symptom:** Every span appeared twice in App Insights.  
**Root cause:** `configure_azure_monitor()` registers its own `BatchSpanProcessor`
on the global `TracerProvider`; the manual processor adds a second one exporting
the same spans.  
**Workaround:** Never use `configure_azure_monitor()` when you also configure the
provider manually. Set up the `TracerProvider` entirely by hand:
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

provider = TracerProvider(resource=resource)
bsp = BatchSpanProcessor(AzureMonitorTraceExporter(connection_string=...))
provider.add_span_processor(bsp)
trace.set_tracer_provider(provider)
```
**Product gap:** `configure_azure_monitor()` should detect and warn when a
`TracerProvider` is already configured, or accept an `existing_provider` parameter.

---

## 5. `TracerProvider.force_flush()` silently drops spans on exit

**What we tried:** Calling `trace.get_tracer_provider().force_flush()` at agent
shutdown to ensure in-flight spans were exported before exit.  
**Symptom:** Spans from the final turn consistently did not appear in App Insights
after the process exited.  
**Root cause:** `azure-monitor-opentelemetry`'s internal `_QuickpulseSpanProcessor`
has a `force_flush()` that returns `None` instead of a boolean. The OTel SDK
interprets this as a failed flush and **short-circuits** — it never calls
`BatchSpanProcessor.force_flush()`.  
**Workaround:** Hold a direct reference to the `BatchSpanProcessor` and call
`force_flush()` on it explicitly at shutdown:
```python
bsp = BatchSpanProcessor(exporter)
provider.add_span_processor(bsp)

# At shutdown — do NOT rely on TracerProvider.force_flush():
bsp.force_flush(timeout_millis=10_000)
```
**Product gap:** Bug in `azure-monitor-opentelemetry`. `_QuickpulseSpanProcessor`
must return `True` from `force_flush()` to comply with the OTel spec. This causes
silent data loss that is very hard to diagnose.

---

## 6. OTel Collector: must use the `contrib` image

**What we tried:** `otel/opentelemetry-collector:latest` in `docker-compose.yml`.  
**Symptom:** Collector failed to start — `azuremonitor` exporter not recognised.  
**Root cause:** The `azuremonitor` exporter is only included in the `contrib`
distribution, not the core image.  
**Workaround:**
```yaml
image: otel/opentelemetry-collector-contrib:latest
```
**Product gap:** Azure Monitor documentation and "getting started" guides that
reference the OTel Collector should specify `contrib` explicitly.

---

## 7. OTel Collector: processor order in the pipeline matters

**What we tried:** Adding a `transform/tag_agent_spans` processor to stamp
`gen_ai.agent.id` on spans, listed after `batch` in the pipeline.  
**Symptom:** The tag never appeared on spans in App Insights.  
**Root cause:** Processors execute in declaration order. Listing `transform` after
`batch` means the transform never runs.  
**Fix:**
```yaml
service:
  pipelines:
    traces:
      processors: [transform/tag_agent_spans, batch]   # transform BEFORE batch
```
**Product gap:** The OTel Collector config provides no validation or warning for
pipeline ordering mistakes. A lint step or debug log would surface this immediately.

---

## 8. No built-in agent identity on CLI spans — must tag manually

**What we tried:** Querying App Insights for Copilot CLI spans by agent name to
run evaluation.  
**Symptom:** There was no reliable field to distinguish Copilot CLI spans from
other OTLP sources in the same App Insights workspace.  
**Root cause:** The CLI does not emit `gen_ai.agent.id` or `gen_ai.agent.name`
on its spans.  
**Workaround:** Added an OTTL transform in the OTel Collector config to stamp the
attribute on root `invoke_agent` spans:
```yaml
transform/tag_agent_spans:
  trace_statements:
    - context: span
      statements:
        - set(attributes["gen_ai.agent.id"], "hw-copilot-cli")
          where IsRootSpan() and attributes["gen_ai.operation.name"] == "invoke_agent"
```
**Product gap:** The CLI should emit `gen_ai.agent.id` and `gen_ai.agent.name`
as standard span attributes so downstream tooling (evaluation, dashboards, alerting)
can filter by agent identity without requiring a custom collector transform.

---

## Summary of Product Gaps

| # | Component | Gap | Severity |
|---|---|---|---|
| 1 | Copilot SDK (PyPI) | `[telemetry]` extra not published | Medium |
| 2 | Copilot CLI | Subprocess telemetry config (`SubprocessConfig.telemetry`) is undocumented | **High** |
| 3 | Copilot CLI | No built-in command to verify the OTLP pipeline is working | Medium |
| 4 | azure-monitor-opentelemetry | `configure_azure_monitor()` creates duplicate spans with manual setup | Medium |
| 5 | azure-monitor-opentelemetry | `_QuickpulseSpanProcessor.force_flush()` returns `None` → spans silently dropped on exit | **High** |
| 6 | OTel Collector docs | Azure Monitor docs don't specify `contrib` image is required | Low |
| 7 | OTel Collector | No pipeline ordering validation / warning | Low |
| 8 | Copilot CLI | No `gen_ai.agent.id` / `gen_ai.agent.name` emitted on spans | **High** |

---

*From a hands-on session integrating GitHub Copilot CLI → OTel Collector sidecar → Azure Application Insights.*
