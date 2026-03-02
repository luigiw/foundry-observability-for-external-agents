# Bug Report: Duplicate Spans in AzureAIOpenTelemetryTracer

## Before Filing
- [ ] Check if issue already exists in langchain-azure repo
- [ ] Test with latest version of langchain-azure-ai
- [ ] Create minimal reproduction case
- [ ] Verify it's not our configuration issue

## Summary
Every span created by `AzureAIOpenTelemetryTracer` appears **twice** in Azure Application Insights with identical span IDs, timestamps, and durations.

## Environment
- **Library**: `langchain-azure-ai>=0.1.0`
- **Azure Monitor**: `azure-monitor-opentelemetry>=1.0.0`
- **OpenTelemetry SDK**: `opentelemetry-sdk>=1.20.0`
- **Python**: 3.11+
- **LangChain**: 0.3.0+
- **LangGraph**: 1.0.0+

## Reproduction

### Setup
```python
from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer

# Initialize tracer
tracer = AzureAIOpenTelemetryTracer(
    connection_string="InstrumentationKey=...",
    enable_content_recording=True,
    provider_name="aws.bedrock",  # or "anthropic"
)
```

### Code
```python
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

# Create LLM
llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    temperature=0.2,
)

# Invoke with tracer callback
config = {
    "callbacks": [tracer],
    "metadata": {
        "agent_name": "Test Agent",
        "gen_ai.operation.name": "chat",
    }
}

response = llm.invoke([HumanMessage(content="Hello")], config=config)
```

### Query Azure Application Insights
```kusto
dependencies
| where timestamp > ago(5m)
| where name contains "chat"
| project timestamp, name, operation_Id, id, duration
| order by timestamp desc
```

### Expected Result
One span per LLM call.

### Actual Result
**Two identical spans** with:
- Same `operation_Id`
- Same span `id`
- Same `timestamp`
- Same `duration`

Example:
```
timestamp                   | name | operation_Id               | id               | duration
2026-03-02T06:50:03.798Z   | chat | abc-123                    | span-xyz         | 3236ms
2026-03-02T06:50:03.798Z   | chat | abc-123                    | span-xyz         | 3236ms
```

## Analysis

### Root Cause Theory
Looking at the source code, `AzureAIOpenTelemetryTracer._configure_azure_monitor()`:

```python
@classmethod
def _configure_azure_monitor(cls, connection_string: str) -> None:
    with cls._configure_lock:
        if cls._azure_monitor_configured:
            return
        configure_azure_monitor(connection_string=connection_string)
        cls._azure_monitor_configured = True
```

**Hypothesis**: The `configure_azure_monitor()` call creates a global TracerProvider with Azure Monitor exporters, but the tracer callback ALSO creates spans that get exported. This results in spans being exported twice:
1. Once by the global TracerProvider set up by `configure_azure_monitor()`
2. Once by the tracer callback mechanism

### Additional Evidence
- Issue occurs with both `ChatBedrock` (AWS Bedrock) and `ChatAnthropic` (Anthropic)
- Issue occurs in both AWS Lambda and local development
- `invoke_agent` spans also duplicated
- All span types affected: `chat`, `POST`, `invoke_agent`

## Impact
- **Data volume**: 2x telemetry data → higher costs
- **Query accuracy**: Counts/aggregations are 2x the actual value
- **Trace analysis**: Confusing to see duplicate spans in traces
- **Dashboards**: Metrics show double the actual numbers

## Additional Issues (May be Related)

### 1. Type = "N/A" for LLM spans
`chat` and `POST` spans show type="N/A" instead of expected "GenAI | <provider>"

Expected: `type = "GenAI | aws.bedrock"`
Actual: `type = "N/A"`

Note: `invoke_agent` spans correctly show `type = "InProc"`

### 2. Gen AI Attributes Not Queryable
Cannot query Gen AI semantic convention attributes from App Insights API:

```kusto
dependencies
| extend agent_name = tostring(customDimensions.['gen_ai.agent.name'])
| where isnotempty(agent_name)
```

Returns: `No rows returned`

Possible causes:
- Attributes not being set
- Attributes filtered by exporter
- Stored in different location

## Workarounds Attempted
None successful. The duplicate issue persists regardless of:
- Removing custom OTel TracerProvider setup
- Using ONLY `AzureAIOpenTelemetryTracer` callbacks
- Different LLM providers (Bedrock vs Anthropic)

## Questions
1. Is `configure_azure_monitor()` supposed to be called by the tracer, or should users call it separately?
2. Should we create a TracerProvider manually or rely on the tracer to do it?
3. Is there a recommended way to use `AzureAIOpenTelemetryTracer` to avoid duplicates?

## Related
- OpenTelemetry Gen AI Semantic Conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- Azure Monitor OpenTelemetry: https://learn.microsoft.com/en-us/azure/azure-monitor/app/opentelemetry-enable

