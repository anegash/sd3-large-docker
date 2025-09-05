#!/bin/bash
# SD3.5 Large - RunPod Startup Script
set -e

echo "🚀 Starting SD3.5 Large API..."

# Set environment variables
export HF_HOME=/workspace/huggingface_cache
export PATH="/root/.local/bin:$PATH"

# Install Poetry if missing
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Change to project directory
cd /workspace/sd3-large-docker

# Load environment
[ -f .env ] && export $(cat .env | grep -v '^#' | xargs)

# Install dependencies
poetry install --only main

# Create logs directory
mkdir -p /workspace/logs

# Start server in background
echo "Starting server..."
nohup poetry run python main.py > /workspace/logs/sd3_server.log 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > /workspace/logs/server.pid

sleep 3
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "✅ Server running at http://localhost:8000"
    echo "📜 Logs: tail -f /workspace/logs/sd3_server.log"
else
    echo "❌ Server failed to start. Check logs."
    exit 1
fi