#!/bin/bash

# Dummy RunPod startup script for manual development
# Keeps container alive for SSH access and manual git operations

echo "🔧 DUMMY MODE: Container staying alive for manual development"
echo "================================================="
echo "You can now:"
echo "  1. SSH into the container"
echo "  2. git pull latest changes" 
echo "  3. Manually start/stop the server"
echo "  4. Debug in real-time"
echo ""

# Set up environment
export PYTHONPATH="/app/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export PYTHONUNBUFFERED=1

# Set HuggingFace cache directory to persistent workspace location
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/hub"
export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"

# Create required directories
mkdir -p /workspace/data/children /workspace/logs /workspace/.cache/huggingface /app/logs
ln -sf /workspace/data /app/data 2>/dev/null || true
ln -sf /workspace/logs /app/workspace_logs 2>/dev/null || true

# Start Redis server
echo "🔴 Starting Redis server..."
redis-server --daemonize yes --port 6379 --bind 0.0.0.0 --protected-mode no

# Wait for Redis to be ready
sleep 2
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis server is running"
else
    echo "❌ Redis server failed to start"
fi

# Initialize database
echo "🗄️  Initializing database..."
cd /app
python -c "
import sys
sys.path.insert(0, '/app/src')
from sd3_api.database.connection import get_engine
from sd3_api.database.models import Base
engine = get_engine()
Base.metadata.create_all(engine)
print('✅ Database initialized')
" 2>/dev/null || echo "⚠️  Database initialization skipped"

echo ""
echo "🚀 READY FOR MANUAL DEVELOPMENT!"
echo "===============================

Commands to use:

# Pull latest changes
cd /app && git pull origin live-debug-runpod

# Start Celery worker
celery -A src.sd3_api.tasks.celery_app worker --loglevel=info --concurrency=1 --queues=training,default --logfile=/workspace/logs/celery.log --detach

# Start API server
python -m uvicorn src.sd3_api.api:app --host 0.0.0.0 --port 8000 --workers 1 --log-level debug

# Check processes
ps aux | grep -E '(celery|uvicorn)'

# Kill processes
pkill -f celery
pkill -f uvicorn
"

# Keep container alive forever
while true; do
    echo "[$(date)] Container alive - ready for manual development"
    sleep 60
done