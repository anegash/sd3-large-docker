#!/bin/bash
# SDXL - RunPod Startup Script
set -e

echo "🚀 Starting SDXL API..."

# Set environment variables
export HF_HOME=/workspace/huggingface_cache
export PATH="/root/.local/bin:$PATH"

# Install Poetry if missing
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Change to project directory (adapt to your actual path)
if [ -d "/workspace/sd3-large-docker" ]; then
    cd /workspace/sd3-large-docker
elif [ -d "/workspace/sdxl-api" ]; then
    cd /workspace/sdxl-api  
elif [ -d "/workspace" ] && [ -f "/workspace/main.py" ]; then
    cd /workspace
else
    echo "❌ Cannot find project directory. Current dir: $(pwd)"
    ls -la /workspace/
    exit 1
fi

# Load environment
[ -f .env ] && export $(cat .env | grep -v '^#' | xargs)

# Clear Python cache to ensure new code loads
echo "🧹 Clearing Python cache..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Install dependencies (skip if already installed)
if [ ! -f ".poetry_installed" ]; then
    echo "📦 Installing dependencies..."
    poetry install --only main
    touch .poetry_installed
else
    echo "✅ Dependencies already installed"
fi

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