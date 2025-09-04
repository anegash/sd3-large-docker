#!/bin/bash

# Start SD3.5 Large LoRA Training System in background
cd /workspace/sd3-large-docker

# Load environment variables from multiple sources
export PATH="/root/.local/bin:$PATH"
export POETRY_CACHE_DIR=/workspace/poetry_cache
export POETRY_VENV_PATH=/workspace/poetry_venvs
export HF_HOME=/workspace/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/workspace/huggingface_cache

# Load HuggingFace token if available
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

echo "🚀 Starting SD3.5 Large LoRA Training System in background..."
echo "📁 Working directory: $(pwd)"
echo "📄 Logs will be saved to: /workspace/logs/app.log"

# Create logs directory if it doesn't exist
mkdir -p /workspace/logs

# Start the server in background
nohup poetry run python main.py > /workspace/logs/app.log 2>&1 &

# Get the PID
APP_PID=$!

echo "✅ Application started in background"
echo "   PID: $APP_PID"
echo "   Logs: tail -f /workspace/logs/app.log"
echo "   Stop: kill $APP_PID"
echo "   Status: ps aux | grep $APP_PID"

# Save PID to file for easy management
echo $APP_PID > /workspace/logs/app.pid

echo ""
echo "🔧 Useful commands:"
echo "   tail -f /workspace/logs/app.log     # View logs"
echo "   kill \$(cat /workspace/logs/app.pid)  # Stop app"
echo "   curl http://localhost:8000/         # Test health"