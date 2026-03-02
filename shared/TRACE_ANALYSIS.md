# GCP Agent Trace Analysis

## Test Execution
- **Date**: 2026-03-02
- **Agent**: GCP Customer Support (using Anthropic via Microsoft Foundry)
- **Tracing**: AzureAIOpenTelemetryTracer → Azure Application Insights

## Test Results

### ✅ Agent Functionality
- Agent invoked successfully locally
- Proper routing to specialists (General, Billing)
- Response generation working
- Tracer initialized correctly: `AzureAIOpenTelemetryTracer`

### ✅ Traces Appearing in App Insights
**Query Results (last 30 min):**
- `trace`: 3534 entries
- `dependency`: 126 entries  
- `customMetric`: 60 entries

**Recent Dependencies:**
```
timestamp                   | name                            | type   | duration
----------------------------------------------------------------------------------
2026-03-02T06:50:03.800Z   | POST                            | N/A    | 3233ms
2026-03-02T06:50:03.798Z   | chat                            | N/A    | 3236ms
2026-03-02T06:50:03.798Z   | invoke_agent Billing Specialist | InProc | 3237ms
2026-03-02T06:47:24.499Z   | invoke_agent Billing Specialist | InProc | 2821ms
```

### ❌ Issues Observed

#### 1. Duplicate Spans
**Every span appears TWICE** with same timestamp and duration:
- POST span appears 2x
- chat span appears 2x
- invoke_agent spans appear 2x (in earlier AWS tests)

**Root Cause**: Multiple TracerProviders exporting to Azure Monitor
- `AzureAIOpenTelemetryTracer` calls `configure_azure_monitor()` → creates TracerProvider #1
- Custom OTel setup (if any) creates TracerProvider #2
- Both export the same spans with same IDs

#### 2. Missing Gen AI Attributes in Query
Query for Gen AI semantic conventions returns: `No rows returned`

```sql
dependencies
| extend genai_agent_name = tostring(customDimensions.['gen_ai.agent.name'])
| where isnotempty(genai_agent_name)
```

**Possible Causes:**
1. Attributes not being set correctly
2. Attributes stored differently in customDimensions
3. API key permissions insufficient to access customDimensions
4. Attributes filtered out by Azure Monitor exporter

#### 3. Type = "N/A" for Some Spans
- `POST` spans: type = "N/A" (should be HTTP)
- `chat` spans: type = "N/A" (should be "GenAI | anthropic" or similar)
- `invoke_agent` spans: type = "InProc" ✅ (correct!)

## Comparison: AWS vs GCP

### AWS Agent (Bedrock)
- Uses `ChatBedrock` from `langchain-aws`
- Tracing: `AzureAIOpenTelemetryTracer` with `provider_name="aws.bedrock"`
- **Same duplicate issue**
- **Same Gen AI attribute issue**

### GCP Agent (Microsoft Foundry)
- Uses `ChatAnthropic` from `langchain-anthropic`  
- Tracing: `AzureAIOpenTelemetryTracer` with `provider_name="anthropic"`
- **Same duplicate issue**
- **Same Gen AI attribute issue**

## Expected vs Actual

### Expected Traces (per OTel Gen AI Spec)
```
Span: invoke_agent Router Agent
├─ type: InProc ✅
├─ gen_ai.operation.name: invoke_agent
├─ gen_ai.provider.name: anthropic
├─ gen_ai.agent.name: Router Agent
├─ gen_ai.request.model: claude-haiku-4-5
└─ gen_ai.usage.{input|output}_tokens: <values>
  
Span: chat (LLM call)
├─ type: GenAI | anthropic ❌ (showing "N/A")
├─ gen_ai.operation.name: chat
├─ gen_ai.request.model: claude-haiku-4-5
└─ gen_ai.usage.*: <values>
```

### Actual Traces
- ✅ Spans created with correct names
- ✅ `invoke_agent` spans have correct type (InProc)
- ❌ `chat` and `POST` spans have type="N/A"  
- ❌ Duplicate spans (each appears 2x)
- ❓ Gen AI attributes (unable to verify via API due to permissions)

## Recommendations

1. **Fix Duplicate Spans**:
   - Ensure only ONE TracerProvider is created
   - `AzureAIOpenTelemetryTracer` already calls `configure_azure_monitor()` 
   - Don't create additional TracerProviders
   - Check if issue is in `langchain-azure-ai` library itself

2. **Verify Gen AI Attributes**:
   - Use Azure Portal to manually inspect span details
   - Check if attributes are in customDimensions or different location
   - Verify `langchain-azure-ai` version supports Gen AI semantic conventions

3. **Fix Span Types**:
   - Ensure `gen_ai.system` attribute is set (Azure Monitor uses this for type)
   - May need to contribute fixes to `langchain-azure-ai` library

## Next Steps

- [ ] Manually inspect traces in Azure Portal UI
- [ ] Check if newer version of `langchain-azure-ai` fixes issues
- [ ] Consider filing issues with `langchain-azure-ai` project
- [ ] Implement workaround if library issues confirmed
