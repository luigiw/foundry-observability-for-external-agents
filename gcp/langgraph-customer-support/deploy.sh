#!/bin/bash
set -euo pipefail

# ============================================================
# Deploy LangGraph Customer Support Agent to Google Cloud Run
# ============================================================

# Configuration - update these for your project
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID environment variable}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="langgraph-customer-support"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Secrets / env vars — Microsoft Foundry (Azure AI Foundry) for Anthropic models
AZURE_FOUNDRY_RESOURCE="${AZURE_FOUNDRY_RESOURCE:?Set AZURE_FOUNDRY_RESOURCE environment variable (your Foundry resource name)}"
AZURE_FOUNDRY_API_KEY="${AZURE_FOUNDRY_API_KEY:?Set AZURE_FOUNDRY_API_KEY environment variable}"
APPINSIGHTS_CONN_STR="${APPLICATIONINSIGHTS_CONNECTION_STRING:-}"

echo "=== Deploying ${SERVICE_NAME} to Cloud Run ==="
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Image:    ${IMAGE_NAME}"
echo ""

# 1. Authenticate (skip if already logged in)
echo "--- Checking gcloud auth ---"
gcloud auth print-access-token > /dev/null 2>&1 || gcloud auth login

# 2. Set project
gcloud config set project "${PROJECT_ID}"

# 3. Enable required APIs
echo "--- Enabling required APIs ---"
gcloud services enable run.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com

# 4. Build and push container image
echo "--- Building container image with Cloud Build ---"
gcloud builds submit --tag "${IMAGE_NAME}" .

# 5. Deploy to Cloud Run
echo "--- Deploying to Cloud Run ---"
ENV_VARS="AZURE_FOUNDRY_RESOURCE=${AZURE_FOUNDRY_RESOURCE},AZURE_FOUNDRY_API_KEY=${AZURE_FOUNDRY_API_KEY}"
ENV_VARS="${ENV_VARS},OTEL_SERVICE_NAME=${OTEL_SERVICE_NAME:-gcp-langgraph-customer-support}"
if [ -n "${SUPPORT_API_KEY:-}" ]; then
    ENV_VARS="${ENV_VARS},SUPPORT_API_KEY=${SUPPORT_API_KEY}"
fi
if [ -n "${APPINSIGHTS_CONN_STR}" ]; then
    ENV_VARS="${ENV_VARS},APPLICATIONINSIGHTS_CONNECTION_STRING=${APPINSIGHTS_CONN_STR}"
    ENV_VARS="${ENV_VARS},AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED=true"
fi

gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}" \
    --platform managed \
    --region "${REGION}" \
    --allow-unauthenticated \
    --memory 512Mi \
    --timeout 300 \
    --set-env-vars "${ENV_VARS}" \
    --port 8080

# 6. Get the service URL
echo ""
echo "=== Deployment complete ==="
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format "value(status.url)")
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test with:"
echo "  curl -X POST ${SERVICE_URL}/support \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"message\": \"I need help with my billing\"}'"
