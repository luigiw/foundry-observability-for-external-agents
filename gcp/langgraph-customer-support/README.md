# LangGraph Customer Support Multi-Agent System (GCP)

A customer support chatbot using LangGraph with Anthropic Claude models via **Microsoft Foundry** (Azure AI Foundry), deployed to Google Cloud Run.

## Architecture

```
                    ┌──────────────┐
                    │  Cloud Run   │
                    │  (FastAPI)   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │    Router    │ (Claude Haiku 4.5 - fast)
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │   Billing   │ │  Technical  │ │   General   │
    │  Specialist │ │  Specialist │ │  Specialist │
    └─────────────┘ └─────────────┘ └─────────────┘
           │               │               │
           └───────────────┼───────────────┘
                           │
                    ┌──────▼───────┐
                    │  Escalation  │ (if needed)
                    └──────────────┘
```

**LLM Provider:** Anthropic Claude via Microsoft Foundry (Azure AI Foundry)
- Router: `claude-haiku-4-5` (fast, low cost)
- Specialists: `claude-sonnet-4-5` (high quality)

## Setup

### 1. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
# Microsoft Foundry (required)
export AZURE_FOUNDRY_RESOURCE="your-foundry-resource-name"
export AZURE_FOUNDRY_API_KEY="your-foundry-api-key"

# Optional: Azure Application Insights tracing
export APPLICATIONINSIGHTS_CONNECTION_STRING="your-connection-string"
```

### 3. Test locally
```bash
# Run the server
uvicorn src.server:app --reload --port 8080

# In another terminal, run tests
python test_local.py
```

### 4. Deploy to GCP Cloud Run
```bash
export GCP_PROJECT_ID="your-project-id"
export AZURE_FOUNDRY_RESOURCE="your-foundry-resource-name"
export AZURE_FOUNDRY_API_KEY="your-foundry-api-key"
./deploy.sh
```

## API

### POST /support
```bash
curl -X POST http://localhost:8080/support \
  -H "Content-Type: application/json" \
  -d '{"message": "I need help with my billing"}'
```

### Response
```json
{
  "response": "I'd be happy to help with your billing inquiry...",
  "metadata": {
    "handled_by": "Billing Specialist",
    "query_type": "billing",
    "needs_escalation": false
  }
}
```

### GET /health
Health check endpoint for Cloud Run.

## Cost Estimate
- Cloud Run: Free tier (2M requests/month, 360K vCPU-sec)
- Anthropic API (via Foundry): ~$0.80/M input tokens, ~$4/M output tokens (Haiku); ~$3/M input, ~$15/M output (Sonnet)

## Observability
Traces are sent to Azure Application Insights with Gen AI semantic conventions via `langchain-azure-ai`. View traces in the Azure portal under Application Insights → Transaction Search.
