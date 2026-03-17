# Evaluation

Local evaluation for the AWS and GCP customer support agents using [`azure-ai-evaluation`](https://pypi.org/project/azure-ai-evaluation/).

## Directory layout

```
eval/
├── data/
│   ├── eval_queries.jsonl      # 16 queries + expected_query_type labels (live mode input)
│   └── sample_traces.jsonl     # Pre-collected responses for offline evaluation (dataset mode demo)
├── evaluators/
│   ├── routing_accuracy.py     # Custom evaluator: checks agent routing classification
│   └── trace_quality.py        # Custom evaluator: evaluates routing, escalation & specialist fit from a full trace
├── results/                    # Evaluation output JSON files (git-ignored)
├── collect_traces.py           # Collect agent outputs to JSONL for dataset mode
├── query_app_insights.py       # Fetch evaluation traces (+ child LLM spans) from Application Insights
├── run_eval.py                 # Main evaluation runner
├── requirements.txt
└── .env.example
```

## Setup

```bash
cd eval
python -m venv .venv && source .venv/bin/activate   # or use an existing venv
pip install -r requirements.txt
cp .env.example .env
# Fill in the required values in .env (see sections below)
```

## Environment variables

| Variable | Required for | Description |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | AI-assisted evaluators | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | AI-assisted evaluators | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | AI-assisted evaluators | API version (default: `2024-02-01`) |
| `AZURE_OPENAI_DEPLOYMENT` | AI-assisted evaluators | Deployment name used as LLM judge (e.g. `gpt-4o`) |
| `APPLICATIONINSIGHTS_APP_ID` | `--mode app-insights` | App Insights Application ID (API Access page) |
| `APPLICATIONINSIGHTS_QUERY_API_KEY` | `--mode app-insights` | Read-only query API key (API Access → Create API key) |
| `AWS_DEFAULT_REGION` | Live mode, AWS agent | AWS region for Bedrock (default: `us-east-1`) |
| `AZURE_FOUNDRY_RESOURCE` | Live mode, GCP agent | Azure AI Foundry resource name |
| `AZURE_FOUNDRY_API_KEY` | Live mode, GCP agent | Azure AI Foundry API key |

## Evaluators

| Evaluator | Type | Measures | Modes |
|---|---|---|---|
| `routing_accuracy` | Custom (no LLM) | Whether `query_type` matches the expected label | `live`, `dataset` |
| `intent_resolution` | AI-assisted | How well the agent resolved the user's intent (1–5 scale) | all |
| `relevance` | AI-assisted | Relevance of the response to the query (1–5 scale) | all |
| `coherence` | AI-assisted | Coherence and logical flow of the response (1–5 scale) | all |
| `fluency` | AI-assisted | Language quality of the response (1–5 scale) | all |

> **Note:** `routing_accuracy` is skipped in `app-insights` mode because traces don't carry ground-truth labels.

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

### App Insights mode (evaluate from stored traces)

Fetch `customer_support_evaluation` spans emitted by the deployed agents and evaluate them.
Both agents can be compared side-by-side with `--agent both`.

#### Setup

1. Open **Azure Portal → Application Insights → API Access**.
2. Copy the **Application ID** → set as `APPLICATIONINSIGHTS_APP_ID` in `eval/.env`.
3. Click **Create API key**, grant *Read telemetry*, copy the key → set as `APPLICATIONINSIGHTS_QUERY_API_KEY`.

#### Run

```bash
# Evaluate both agents from the last 24h of traces and compare:
python run_eval.py --mode app-insights --agent both --model-provider gpt5

# Evaluate only the AWS agent using the last 48h of traces:
python run_eval.py --mode app-insights --agent aws --hours 48 --model-provider foundry

# Fetch traces directly (without running evaluation):
python query_app_insights.py --agent both --hours 24
python query_app_insights.py --agent aws --hours 48 --output data/aws_traces.jsonl
```

#### How it works

Each call to `invoke_support()` in both agents wraps the graph execution in an
OpenTelemetry span named `customer_support_evaluation`. The span records:

| Attribute | Contents |
|---|---|
| `customer_support.query` | The user's original message |
| `customer_support.query_type` | Routing decision (`billing`, `technical`, `general`, `escalation`) |
| `customer_support.handled_by` | Name of the specialist agent that handled the query |
| `customer_support.needs_escalation` | Whether the query was escalated |
| `customer_support.response` | The agent's final response text |
| `customer_support.session_id` | Unique session UUID |

These attributes appear in Application Insights under
**Dependencies → customDimensions** and can be queried with KQL.

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
