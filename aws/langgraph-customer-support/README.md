# LangGraph Customer Support Multi-Agent System

A customer support chatbot using LangGraph with AWS Bedrock and Lambda deployment.

## Architecture

```
                    ┌──────────────┐
                    │   API GW     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │    Lambda    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │    Router    │ (Claude Haiku - fast)
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

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure AWS credentials
```bash
aws configure
```

### 3. Test locally
```bash
python test_local.py
```

### 4. Deploy to AWS
```bash
sam build
sam deploy
```

## Usage

### API Request
```bash
curl -X POST https://YOUR_API_ENDPOINT/prod/support \
  -H "Content-Type: application/json" \
  -d '{"message": "I need help with my billing"}'
```

### Response
```json
{
  "response": "I'd be happy to help with your billing inquiry...",
  "metadata": {
    "handled_by": "billing_specialist",
    "query_type": "billing",
    "needs_escalation": false
  }
}
```

## Cost Estimate
- Lambda: Free tier (1M requests/month)
- API Gateway: Free tier (1M requests/month for 12 months)
- Bedrock Claude: ~$0.003/1K input tokens, ~$0.015/1K output tokens
