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

# Change to project directory
cd /workspace/sd3-project

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

# Start the server with logging
poetry run python main.py 2>&1 | tee /workspace/logs/sd3_server.log