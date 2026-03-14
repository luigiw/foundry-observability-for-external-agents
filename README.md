# Multi-Cloud LangGraph Customer Support Agents

A multi-agent customer support system built with [LangGraph](https://github.com/langchain-ai/langgraph), deployable to **AWS Lambda** and **GCP Cloud Run** with a shared, cloud-agnostic core. Both deployments export OpenTelemetry traces to **Azure Application Insights** with full W3C Trace Context propagation, so agent spans appear as children of the caller's trace. Includes a **Streamlit UI** for chatting with agents and viewing traces, and an **evaluation framework** for measuring agent quality.

## Architecture

```
                        ┌────────────────────────┐
                        │     Streamlit UI       │
                        │  (Chat / Traces)       │
                        └───────────┬────────────┘
                                    │ HTTP
                   ┌────────────────┴─────────────────┐
                   │   Azure API Management (APIM)    │
                   │  (optional gateway + tracing)     │
                   └────────┬─────────────┬────────────┘
                            │             │
     ┌──────────────────────▼──┐   ┌──────▼──────────────────────┐
     │  AWS Lambda + API GW   │   │  GCP Cloud Run (FastAPI)    │
     │  (Amazon Bedrock)      │   │  (Azure AI Foundry)         │
     └──────────────┬──────────┘   └──────────────┬──────────────┘
                    │                              │
                    └──────────────┬───────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │   Shared Agent Core     │
                      │   (cloud-agnostic)      │
                      └────────────┬────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │     Router Agent        │
                      │  (Haiku — fast/cheap)   │
                      └────────────┬────────────┘
                                   │
           ┌───────────┬───────────┼───────────┐
           │           │           │           │
      ┌────▼───┐  ┌────▼───┐ ┌────▼───┐ ┌─────▼──────┐
      │Billing │  │  Tech  │ │General │ │ Escalation │
      │  Spec. │  │  Spec. │ │ Spec.  │ │  Handler   │
      └────────┘  └────────┘ └────────┘ └────────────┘
        (Sonnet)    (Sonnet)   (Sonnet)    (Sonnet)
```

**Agent flow:** A lightweight Router Agent (Claude Haiku) classifies incoming queries by type and confidence, then routes to one of four specialist agents (Claude Sonnet). If the router's confidence is below 0.4, the query is automatically escalated.

## Project Structure

```
├── shared/customer-support-agents/   # Cloud-agnostic core (agents, graph, state)
├── aws/langgraph-customer-support/   # AWS Lambda + Bedrock deployment
├── gcp/langgraph-customer-support/   # GCP Cloud Run + Azure AI Foundry deployment
├── eval/                             # Evaluation framework (routing accuracy, quality)
└── ui/                               # Streamlit frontend
```

| Component | LLM Provider | Router Model | Specialist Model | Runtime |
|-----------|-------------|--------------|------------------|---------|
| **AWS** | Amazon Bedrock | Claude 3 Haiku | Claude 3 Sonnet | Python 3.11, Lambda (ARM64) |
| **GCP** | Azure AI Foundry (Anthropic) | Claude Haiku 4.5 | Claude Sonnet 4.5 | Python 3.11, Cloud Run |

## Shared Core

The `shared/customer-support-agents/` package contains all cloud-agnostic logic:

- **Agents** — `CustomerSupportAgents` class with 5 agent methods (router, billing, technical, general, escalation)
- **Graph** — `build_support_graph()` wires the LangGraph state machine; `route_to_specialist()` handles conditional routing
- **State** — `AgentState` TypedDict with fields: `messages`, `query_type`, `confidence`, `needs_escalation`, `customer_id`, `handled_by`, `final_response`

Cloud implementations inject their own LLM provider via a factory function:

```python
from customer_support_agents.agents import CustomerSupportAgents
from customer_support_agents.graph import build_support_graph

agents = CustomerSupportAgents(llm_factory=my_cloud_llm_factory)
graph = build_support_graph(agents)
result = graph.invoke(initial_state)
```

## Getting Started

### Prerequisites

- Python 3.11+
- AWS CLI + [SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html) (for AWS deployment)
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) (for GCP deployment)

### AWS Deployment

```bash
cd aws/langgraph-customer-support
pip install -r requirements.txt

# Test locally
python test_local.py

# Deploy
sam build
sam deploy
```

The SAM template deploys a Lambda function behind API Gateway at `/prod/support` with API key authentication (managed by API Gateway). Retrieve the key value with the CLI command shown in the stack outputs.

<details>
<summary>Environment variables</summary>

| Variable | Description |
|----------|-------------|
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Azure Application Insights connection string |
| `OTEL_SERVICE_NAME` | Defaults to `aws-langgraph-customer-support` |

AWS Bedrock access is configured via IAM role (defined in `template.yaml`).
</details>

### GCP Deployment

```bash
cd gcp/langgraph-customer-support
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env  # Edit with your credentials

# Test locally
uvicorn src.server:app --reload --port 8080
python test_local.py        # Agent tests
./test_local.sh             # Integration tests via curl

# Deploy to Cloud Run
export GCP_PROJECT_ID="your-project-id"
./deploy.sh
```

<details>
<summary>Environment variables (set in <code>.env</code>)</summary>

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_FOUNDRY_RESOURCE` | Yes | Azure AI Foundry resource name |
| `AZURE_FOUNDRY_API_KEY` | Yes | Azure AI Foundry API key |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | No | Azure Application Insights connection string |
| `SUPPORT_API_KEY` | No | Enables `x-api-key` header authentication when set |
| `AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED` | No | Set `true` to record prompt/response content in traces |
| `OTEL_SERVICE_NAME` | No | Defaults to `gcp-langgraph-customer-support` |

> **Note:** `.env` is git-ignored. Copy `.env.example` and fill in your values. Quote values containing semicolons (e.g. the App Insights connection string) to prevent shell parsing issues.
</details>

### Streamlit UI

```bash
cd ui
pip install -r requirements.txt
streamlit run app.py
```

The UI provides:
- **Chat** — Interactive chat with any deployed agent, showing routing metadata (handled_by, query_type, escalation status)
- **Traces** — Query and view OpenTelemetry traces from Azure Application Insights

Agent endpoints are configured in `ui/config.yaml`.

## API

Both deployments expose the same API contract:

### `POST /support`

```bash
curl -X POST https://<endpoint>/support \
  -H "Content-Type: application/json" \
  -H "x-api-key: <your-api-key>" \
  -d '{"message": "I was charged twice for my subscription", "customer_id": "CUST-123"}'
```

**Response:**
```json
{
  "response": "I understand your concern about the double charge...",
  "metadata": {
    "handled_by": "Billing Specialist",
    "query_type": "billing",
    "needs_escalation": false
  }
}
```

GCP also exposes `GET /health` for Cloud Run health checks (exempt from API key requirement).

### Authentication

| Deployment | Mechanism | Header |
|------------|-----------|--------|
| **AWS** | API Gateway managed API key | `x-api-key` (required) |
| **GCP** | FastAPI middleware (`SUPPORT_API_KEY` env var) | `x-api-key` (optional — disabled when env var unset) |

## Observability

Both cloud deployments send telemetry to **Azure Application Insights** via OpenTelemetry, using [Gen AI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

### Trace Attributes

| Attribute | Description |
|-----------|-------------|
| `gen_ai.operation.name` | `"invoke_agent"` or `"chat"` |
| `gen_ai.agent.name` | Agent that handled the query |
| `gen_ai.request.model` | Model used (e.g. `claude-haiku-4-5-20251001`) |
| `gen_ai.usage.input_tokens` | Input token count |
| `gen_ai.usage.output_tokens` | Output token count |
| `cloud_RoleName` | `aws-langgraph-customer-support` or `gcp-langgraph-customer-support` |

Traces can be viewed in the Azure Portal under **Application Insights → Transaction Search**, or through the Streamlit UI's Traces page.

### W3C Trace Context Propagation

Both deployments extract `traceparent` / `tracestate` / `baggage` headers from incoming requests so that agent spans appear as **children** of the caller's trace.

**AWS Lambda** extracts headers directly from the API Gateway event (normalising to lowercase).

**GCP Cloud Run** requires a workaround because Cloud Run's HTTP proxy rewrites the `traceparent` header, replacing the span-id with its own infrastructure span (exported to Google Cloud Trace, not App Insights). This breaks the parent→child link. The server recovers the correct parent using a priority chain:

1. **`request-id`** — Application Insights legacy header (`|<traceId>.<spanId>.`) sent automatically by Azure API Management. Contains the APIM dependency span-id. Cloud Run does not modify this header.
2. **`x-ms-traceparent`** — Custom header a caller can set explicitly (Cloud Run leaves it untouched).
3. **`traceparent`** — Standard W3C header (fallback; may contain Cloud Run's rewritten span-id).

No special APIM policy is needed — the `request-id` header is sent by default when APIM has Application Insights enabled.

### Client-Side Tracing

Both `aws/…/client.py` and `gcp/…/client.py` demonstrate how to set up a client-side `TracerProvider`, create a CLIENT span, and inject W3C trace headers into the outgoing request so that end-to-end traces appear in App Insights.

## Evaluation

The `eval/` directory contains a framework for measuring agent quality using [Azure AI Evaluation](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/evaluate-sdk).

```bash
cd eval
pip install -r requirements.txt
cp .env.example .env  # Configure evaluator credentials

# Collect agent responses
python collect_traces.py --agent gcp --output data/sample_traces.jsonl

# Run evaluation on collected data
python run_eval.py --mode dataset --data data/sample_traces.jsonl

# Or run live evaluation (calls agent per query)
python run_eval.py --mode live --agent gcp
```

### Evaluators

| Evaluator | Type | Description |
|-----------|------|-------------|
| `routing_accuracy` | Custom (no LLM) | Checks if agent's `query_type` matches expected label |
| `intent_resolution` | AI-assisted (1–5) | Did the response address the user's intent? |
| `relevance` | AI-assisted (1–5) | Is the response relevant to the query? |
| `coherence` | AI-assisted (1–5) | Is the response logically coherent? |
| `fluency` | AI-assisted (1–5) | Is the response fluent and well-written? |

AI-assisted evaluators require an Azure OpenAI, Azure AI Foundry, or GPT-5 endpoint. A local proxy (`proxy.py`) translates between the OpenAI API format and these providers. See `eval/README.md` for full details.

## Cost Estimates

| Service | Free Tier | Pay-as-you-go |
|---------|-----------|---------------|
| AWS Lambda | 1M requests/month | — |
| API Gateway | 1M requests/month (12 months) | — |
| Bedrock (Haiku) | — | ~$0.25/M input, $1.25/M output tokens |
| Bedrock (Sonnet) | — | ~$3/M input, $15/M output tokens |
| Cloud Run | 2M requests/month, 360K vCPU-sec | — |
| Azure AI Foundry (Haiku) | — | ~$0.80/M input, $4/M output tokens |
| Azure AI Foundry (Sonnet) | — | ~$3/M input, $15/M output tokens |

## License

This project does not currently include a license.
