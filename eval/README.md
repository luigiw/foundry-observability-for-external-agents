# Evaluation

Local evaluation for the AWS and GCP customer support agents using [`azure-ai-evaluation`](https://pypi.org/project/azure-ai-evaluation/).

## Directory layout

```
eval/
├── data/
│   ├── eval_queries.jsonl    # 16 queries + expected_query_type labels (live mode input)
│   └── sample_traces.jsonl   # Pre-collected responses for offline evaluation (dataset mode demo)
├── evaluators/
│   └── routing_accuracy.py   # Custom evaluator: checks agent routing classification
├── results/                  # Evaluation output JSON files (git-ignored)
├── collect_traces.py         # Collect agent outputs to JSONL for dataset mode
├── run_eval.py               # Main evaluation runner
├── requirements.txt
└── .env.example
```

## Setup

```bash
cd eval
python -m venv .venv && source .venv/bin/activate   # or use an existing venv
pip install -r requirements.txt
cp .env.example .env
# Fill in AZURE_OPENAI_* values in .env
```

## Environment variables

| Variable | Required for | Description |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | AI-assisted evaluators | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | AI-assisted evaluators | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | AI-assisted evaluators | API version (default: `2024-02-01`) |
| `AZURE_OPENAI_DEPLOYMENT` | AI-assisted evaluators | Deployment name used as LLM judge (e.g. `gpt-4o`) |
| `AWS_DEFAULT_REGION` | Live mode, AWS agent | AWS region for Bedrock (default: `us-east-1`) |
| `AZURE_FOUNDRY_RESOURCE` | Live mode, GCP agent | Azure AI Foundry resource name |
| `AZURE_FOUNDRY_API_KEY` | Live mode, GCP agent | Azure AI Foundry API key |

## Evaluators

| Evaluator | Type | Measures |
|---|---|---|
| `routing_accuracy` | Custom (no LLM) | Whether `query_type` matches the expected label |
| `intent_resolution` | AI-assisted | How well the agent resolved the user's intent (1–5 scale) |
| `relevance` | AI-assisted | Relevance of the response to the query (1–5 scale) |
| `coherence` | AI-assisted | Coherence and logical flow of the response (1–5 scale) |
| `fluency` | AI-assisted | Language quality of the response (1–5 scale) |

## Usage

### Dataset mode (offline — no agent credentials needed)

Evaluate the provided `sample_traces.jsonl` with pre-collected agent responses:

```bash
# Evaluate sample traces with AI-assisted evaluators
python run_eval.py

# Evaluate without Azure OpenAI (routing accuracy only)
python run_eval.py --skip-ai-evaluators

# Evaluate your own collected traces
python run_eval.py --mode dataset --data data/gcp_traces.jsonl --agent gcp
```

### Live mode (call the agent per query)

Requires the respective agent's Python dependencies and cloud credentials.

```bash
# AWS agent (requires langchain-aws, AWS credentials)
pip install -r ../aws/langgraph-customer-support/requirements.txt
python run_eval.py --agent aws --mode live

# GCP agent (requires langchain-anthropic, Azure Foundry credentials in gcp/.env)
pip install -r ../gcp/langgraph-customer-support/requirements.txt
python run_eval.py --agent gcp --mode live
```

### Collecting traces for offline evaluation

Run the agents and save results to JSONL for later evaluation:

```bash
# Collect AWS agent traces
python collect_traces.py --agent aws --output data/aws_traces.jsonl --verbose

# Collect GCP agent traces
python collect_traces.py --agent gcp --output data/gcp_traces.jsonl --verbose

# Use custom input queries
python collect_traces.py --agent gcp --input data/eval_queries.jsonl --output data/my_traces.jsonl
```

Then evaluate the collected traces:

```bash
python run_eval.py --mode dataset --data data/aws_traces.jsonl --agent aws
```

## Output

Results are written to `results/<agent>_<mode>_results.json`. The file contains:
- Aggregate metrics (mean scores across all rows)
- Per-row scores for each evaluator
- The original query and response for each row

## Switching to remote evaluation

To log results to Azure AI Foundry for tracking and comparison, add the `azure_ai_project` parameter to the `evaluate()` call in `run_eval.py`:

```python
result = evaluate(
    ...
    azure_ai_project={
        "subscription_id": os.environ["AZURE_SUBSCRIPTION_ID"],
        "resource_group_name": os.environ["AZURE_RESOURCE_GROUP"],
        "project_name": os.environ["AZURE_AI_PROJECT_NAME"],
    },
)
```
