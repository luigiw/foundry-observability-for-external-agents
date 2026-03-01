#!/bin/bash
set -euo pipefail

# ============================================================
# Local test script for LangGraph Customer Support Agent
# Starts the server, runs test queries, then shuts down.
# ============================================================

cd "$(dirname "$0")"
PORT=8090

# Activate venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -q -r requirements.txt
else
    source .venv/bin/activate
fi

# Start server in background
echo "Starting server on port ${PORT}..."
uvicorn src.server:app --port ${PORT} --log-level warning &
SERVER_PID=$!
trap "kill ${SERVER_PID} 2>/dev/null; wait ${SERVER_PID} 2>/dev/null; exit" EXIT INT TERM

# Wait for server to be ready
for i in $(seq 1 15); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Health check
echo ""
HEALTH=$(curl -s http://localhost:${PORT}/health)
if [ "$HEALTH" != '{"status":"healthy"}' ]; then
    echo "❌ Health check failed: ${HEALTH}"
    exit 1
fi
echo "✅ Health check passed"
echo ""

# Helper function
run_test() {
    local label="$1"
    local message="$2"
    local extra_fields="${3:-}"

    echo "--- ${label} ---"
    local payload="{\"message\": \"${message}\"${extra_fields}}"
    curl -s -X POST "http://localhost:${PORT}/support" \
        -H "Content-Type: application/json" \
        -d "${payload}" | python3 -c "
import json, sys, textwrap
d = json.load(sys.stdin)
m = d.get('metadata', {})
print(f'  Query type:  {m.get(\"query_type\", \"N/A\")}')
print(f'  Handled by:  {m.get(\"handled_by\", \"N/A\")}')
print(f'  Escalation:  {m.get(\"needs_escalation\", \"N/A\")}')
print(f'  Response:    {textwrap.shorten(d.get(\"response\", \"\"), width=120)}')
"
    echo ""
}

# Run tests
echo "========================================="
echo " LangGraph Customer Support Agent Tests"
echo "========================================="
echo ""

run_test "Test 1: General inquiry" \
    "Hi! What are your business hours?"

run_test "Test 2: Billing issue" \
    "I was charged twice for my subscription last month. Can you help me get a refund?" \
    ", \"customer_id\": \"CUST-42\""

run_test "Test 3: Technical problem" \
    "The app keeps crashing when I try to upload files larger than 10MB. Error code: ERR_FILE_TOO_LARGE"

run_test "Test 4: Escalation" \
    "This is unacceptable! I have been waiting 3 weeks and no one has helped. I want to speak to a manager NOW!"

echo "========================================="
echo " All tests complete ✅"
echo "========================================="
