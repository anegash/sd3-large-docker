#!/bin/bash

# RunPod Pod Startup Script for SD3.5 Large LoRA Training System
# Starts API server, Celery worker, and Redis in a single container

# Don't exit on error immediately - we want to see what's failing
set +e

echo "🚀 Starting SD3.5 Large LoRA Training Pod..."
echo "📊 System Information:"
echo "   Hostname: $(hostname)"
echo "   Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "   Python: $(python --version)"
echo "   Current directory: $(pwd)"
echo "   GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "   No NVIDIA GPU detected"

# Set environment variables
export PYTHONPATH="/app/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export PYTHONUNBUFFERED=1  # Force Python to flush output immediately

# Set HuggingFace cache directory to persistent workspace location
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/hub"
export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"

echo "📦 Environment Variables:"
echo "   PYTHONPATH: $PYTHONPATH"
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "   HF_TOKEN present: $(if [ -n "$HF_TOKEN" ]; then echo "Yes"; else echo "No"; fi)"
echo "   HUGGINGFACE_TOKEN present: $(if [ -n "$HUGGINGFACE_TOKEN" ]; then echo "Yes"; else echo "No"; fi)"
echo "   HF_HOME: $HF_HOME"
echo "   TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"

# Create required directories
mkdir -p /workspace/data/children /workspace/logs /workspace/.cache/huggingface /app/logs
# Create symlinks from app to workspace for compatibility
ln -sf /workspace/data /app/data 2>/dev/null || true
ln -sf /workspace/logs /app/workspace_logs 2>/dev/null || true

echo "📁 Created data directories"

# Start Redis server in background
echo "🔴 Starting Redis server..."
redis-server --daemonize yes --port 6379 --bind 0.0.0.0 --protected-mode no

# Wait for Redis to be ready
sleep 2

# Test Redis connection
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis server is running"
else
    echo "❌ Redis server failed to start"
    exit 1
fi

# Initialize database (create tables if they don't exist)
echo "🗄️  Initializing database..."
cd /app
python -c "
import sys
sys.path.insert(0, '/app/src')
from sd3_api.database.connection import get_engine
from sd3_api.database.models import Base

# Create tables
engine = get_engine()
Base.metadata.create_all(engine)
print('✅ Database initialized')
" 2>/dev/null || echo "⚠️  Database initialization skipped"

# Start Celery worker in background for LoRA training
echo "👷 Starting Celery worker for LoRA training..."
cd /app
celery -A src.sd3_api.tasks.celery_app worker \
    --loglevel=info \
    --concurrency=1 \
    --queues=training,default \
    --logfile=/workspace/logs/celery.log \
    --detach

# Note: Skipping Celery Flower as it's not installed in this version
echo "📝 Note: Celery Flower monitoring skipped (not installed)"

# Wait for Celery to start
sleep 3

# Check if Celery worker is running
if pgrep -f "celery.*worker" > /dev/null; then
    echo "✅ Celery worker is running"
else
    echo "❌ Celery worker failed to start"
fi

# Start the main API server (blocking)
echo "🌐 Starting FastAPI server with LoRA endpoints..."
echo "   Working directory: $(pwd)"
echo "   Python modules:"
python -c "import sys; print('   Python path:', sys.path[:3])"

# Check if the API module exists
echo "   Checking API module..."
python -c "
import sys
sys.path.insert(0, '/app/src')
try:
    from sd3_api import api
    print('   ✅ API module found')
except Exception as e:
    print(f'   ❌ Failed to import API module: {e}')
"

# Start uvicorn with full logging
echo "📡 Starting uvicorn server..."
cd /app

# Run with full error output
python -m uvicorn src.sd3_api.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level debug \
    --access-log 2>&1 | tee /workspace/logs/uvicorn.log

# If uvicorn fails, show the error and keep container alive for debugging
if [ $? -ne 0 ]; then
    echo "❌ Uvicorn failed to start. Error details above."
    echo "📝 Last 50 lines of log:"
    tail -50 /workspace/logs/uvicorn.log 2>/dev/null || echo "No log file found"
    echo "🔄 Keeping container alive for debugging..."
    sleep 3600
fi