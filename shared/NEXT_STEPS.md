# Next Steps: Filing Bug Report

## ✅ Completed
- [x] Identified duplicate span issue affecting both AWS and GCP agents
- [x] Verified issue exists on latest version (langchain-azure-ai 1.0.61)
- [x] Confirmed no existing issues in langchain-azure repo
- [x] Created detailed bug report draft
- [x] Created minimal reproduction case
- [x] Documented trace analysis

## 🎯 Recommended Actions

### 1. File GitHub Issue (Recommended)

**Repository**: https://github.com/langchain-ai/langchain-azure/issues

**Title**: `AzureAIOpenTelemetryTracer creates duplicate spans in Azure Application Insights`

**Labels**: `bug`, `tracing`, `azure-ai`

**Content**: Use `BUG_REPORT_DRAFT.md` as template

**Attachments**:
- `minimal_repro.py` - Minimal reproduction case
- `TRACE_ANALYSIS.md` - Detailed trace analysis
- Screenshots from Azure Portal showing duplicate spans (if you have access)

### 2. Before Filing - Quick Checks

#### A. Test with Minimal Repro
```bash
cd /Users/hanchiwang/Code/hw-agent-playground-refactor/shared
python minimal_repro.py
# Wait 2-3 minutes, check App Insights
```

#### B. Check Azure Portal Manually
Since the API has permission issues, manually verify in Azure Portal:
1. Go to Azure Portal → Application Insights
2. Navigate to Transaction search or Logs
3. Run query:
   ```kusto
   dependencies
   | where timestamp > ago(10m)
   | where name contains "chat"
   | summarize count() by name, operation_Id, id
   | where count_ > 1
   ```
4. Take screenshots showing duplicates

#### C. Inspect Span Details
Check if Gen AI attributes are actually there:
1. Click on a `chat` span in Azure Portal
2. Look at Custom Properties / Custom Dimensions
3. Check if `gen_ai.*` attributes are present
4. Take screenshots

### 3. Alternative: Contact Microsoft Azure Support
If this is an Azure Monitor issue rather than langchain-azure-ai:
- Azure Support Portal
- Provide same reproduction case
- Reference OpenTelemetry integration

### 4. Workaround While Waiting
If you need a temporary fix, you could:

**Option A**: Deduplicate in queries
```kusto
dependencies
| where timestamp > ago(1h)
| summarize arg_max(timestamp, *) by operation_Id, id
| project timestamp, name, duration, ...
```

**Option B**: Create custom TracerProvider (advanced)
- Don't use `AzureAIOpenTelemetryTracer`
- Manually create OTel spans with proper attributes
- More control but more code

**Option C**: Wait for fix
- Continue using current setup
- Accept 2x data volume
- Adjust queries to account for duplicates

## 📝 Issue Template

```markdown
**Bug Description**
AzureAIOpenTelemetryTracer exports every span twice to Azure Application Insights with identical span IDs, timestamps, and durations.

**Environment**
- langchain-azure-ai: 1.0.61
- azure-monitor-opentelemetry: 1.0.0+
- Python: 3.11+
- LangChain: 0.3.0+

**Reproduction**
[See minimal_repro.py]

**Expected Behavior**
One span per LLM invocation in Application Insights

**Actual Behavior**
Two identical spans per invocation (same operation_Id, span id, timestamp)

**Impact**
- 2x telemetry costs
- Incorrect metrics/aggregations
- Confusing trace analysis

**Additional Context**
- Occurs with both ChatBedrock (AWS) and ChatAnthropic
- Occurs in Lambda and local development
- Affects all span types: chat, POST, invoke_agent
- Related issue: chat/POST spans show type="N/A" instead of "GenAI | <provider>"

**Hypothesis**
configure_azure_monitor() creates global TracerProvider, but spans are also exported via callback mechanism, resulting in duplication.

**Files**
[Attach minimal_repro.py and screenshots]
```

## 🔍 Investigation Questions

If maintainers ask for more info:

1. **"Can you share your full code?"**
   → Point to minimal_repro.py

2. **"Are you creating multiple TracerProviders?"**
   → No, only using AzureAIOpenTelemetryTracer

3. **"Does it happen with standard OpenTelemetry?"**
   → Haven't tested outside of langchain-azure-ai

4. **"What version of azure-monitor-opentelemetry?"**
   → Check with `pip show azure-monitor-opentelemetry`

5. **"Can you share Application Insights query results?"**
   → Provide output from query_traces.py

## 📅 Timeline Estimate
- **File issue**: 30 minutes
- **Maintainer response**: 1-7 days
- **Fix (if confirmed)**: 1-4 weeks
- **Release**: Depends on release cycle

## 🤝 Contributing
If you want to help fix it:
1. Fork langchain-azure repo
2. Investigate inference_tracing.py
3. Look at _configure_azure_monitor() and span export logic
4. Submit PR with fix

