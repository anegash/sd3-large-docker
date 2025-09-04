#!/bin/bash
# SD3.5 Large LoRA Training System - RunPod Startup Script
# 
# This script starts the SD3 API server with persistent storage
# configured for RunPod environment.

set -e  # Exit on any error

echo "🚀 SD3.5 Large LoRA Training System - Starting..."
echo "=" * 60

# Set environment variables for persistent storage
export WORKSPACE_DIR=/workspace
export HF_HOME=/workspace/huggingface_cache
export TRANSFORMERS_CACHE=/workspace/huggingface_cache/transformers
export HF_DATASETS_CACHE=/workspace/huggingface_cache/datasets
export PATH="/root/.local/bin:$PATH"

# Check if Poetry is available, install if missing
if ! command -v poetry &> /dev/null; then
    echo "🔧 Poetry not found, installing..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="/root/.local/bin:$PATH"
fi

# Change to project directory  
cd /workspace/sd3-large-docker

echo "📋 Environment Status:"
echo "   Workspace: $WORKSPACE_DIR"
echo "   Project Dir: $(pwd)"
echo "   HuggingFace Cache: $HF_HOME"
echo "   Python: $(which python3)"
echo "   Poetry: $(which poetry)"

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "📄 Loading environment from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "   ✅ Environment loaded"
else
    echo "⚠️  No .env file found - make sure HUGGINGFACE_TOKEN is set"
fi

# Verify Poetry installation and activate environment
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found! Please run setup_runpod.py first"
    exit 1
fi

echo "🔍 Checking Poetry environment..."
poetry env info

# Install any missing dependencies (in case of updates)
echo "📦 Ensuring dependencies are up to date..."
poetry install --only main

# Check if model cache exists
if [ -d "/workspace/huggingface_cache" ]; then
    echo "📁 HuggingFace cache found in workspace"
    cache_size=$(du -sh /workspace/huggingface_cache | cut -f1)
    echo "   Cache size: $cache_size"
else
    echo "📁 No existing model cache - first run will download models"
fi

# Check if LoRA weights exist
if [ -d "/workspace/lora_weights" ] && [ "$(ls -A /workspace/lora_weights)" ]; then
    echo "🎯 Existing LoRA weights found:"
    ls -la /workspace/lora_weights/
else
    echo "🎯 No existing LoRA weights - ready for training"
fi

# Start the server
echo ""
echo "🔥 Starting FastAPI server..."
echo "   Server will be available at: http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Health Check: http://localhost:8000/"
echo ""
echo "📝 Logs will be saved to: /workspace/logs/"
echo ""

# Create logs directory if it doesn't exist
mkdir -p /workspace/logs

# Start the server in background with logging
echo "🔄 Starting server in background..."
nohup poetry run python main.py > /workspace/logs/sd3_server.log 2>&1 &

# Get the process ID
SERVER_PID=$!
echo "   ✅ Server started with PID: $SERVER_PID"
echo "   📋 Server process ID saved to: /workspace/logs/server.pid"
echo "$SERVER_PID" > /workspace/logs/server.pid

# Wait a moment and check if server started successfully
sleep 3
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "   ✅ Server is running successfully"
    echo "   🌐 Server URL: http://localhost:8000"
    echo "   📚 API Docs: http://localhost:8000/docs"
    echo "   📜 Logs: tail -f /workspace/logs/sd3_server.log"
    echo ""
    echo "🔧 Useful commands:"
    echo "   • Check status: curl http://localhost:8000/"
    echo "   • View logs: tail -f /workspace/logs/sd3_server.log"
    echo "   • Stop server: kill \$(cat /workspace/logs/server.pid)"
else
    echo "   ❌ Server failed to start. Check logs:"
    echo "   tail -f /workspace/logs/sd3_server.log"
    exit 1
fi