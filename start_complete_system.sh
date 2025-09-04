#!/bin/bash

# Complete SD3.5 LoRA System Startup
# Starts Redis, API server, and Celery worker in correct order

echo "🚀 Starting Complete SD3.5 LoRA Training System"
echo "================================================"

cd /workspace/sd3-large-docker

# Load environment variables
export PATH="/root/.local/bin:$PATH"
export POETRY_CACHE_DIR=/workspace/poetry_cache
export POETRY_VENV_PATH=/workspace/poetry_venvs
export HF_HOME=/workspace/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/workspace/huggingface_cache

# Load HuggingFace token if available
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

mkdir -p /workspace/logs

# 1. Start Redis Server (required for Celery)
echo "🔴 Starting Redis server..."
redis-server --daemonize yes --port 6379 --bind 0.0.0.0 --protected-mode no

sleep 2

if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis server started successfully"
else
    echo "❌ Redis server failed to start - Celery will not work!"
    echo "   Trying to start anyway..."
fi

# 2. Start API Server in background
echo "🌐 Starting SD3.5 API server in background..."
nohup poetry run python main.py > /workspace/logs/app.log 2>&1 &
APP_PID=$!
echo $APP_PID > /workspace/logs/app.pid

echo "✅ API server started (PID: $APP_PID)"

# Wait for API server to be ready
echo "⏳ Waiting for API server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "✅ API server is ready!"
        break
    fi
    echo "   Waiting... ($i/30)"
    sleep 2
done

# 3. Start Celery Worker in background
echo "🔄 Starting Celery worker in background..."
nohup poetry run celery -A src.sd3_api.tasks.celery_app worker --loglevel=info -Q training > /workspace/logs/celery.log 2>&1 &
CELERY_PID=$!
echo $CELERY_PID > /workspace/logs/celery.pid

echo "✅ Celery worker started (PID: $CELERY_PID)"

echo ""
echo "🎉 Complete SD3.5 LoRA Training System Started!"
echo "================================================"
echo "📊 System Status:"
echo "   Redis:        $(redis-cli ping 2>/dev/null || echo 'FAILED')"
echo "   API Server:   PID $APP_PID (check: curl http://localhost:8000/)"
echo "   Celery Worker: PID $CELERY_PID"
echo ""
echo "📁 Log Files:"
echo "   API:    tail -f /workspace/logs/app.log"
echo "   Celery: tail -f /workspace/logs/celery.log"
echo ""
echo "🛑 Stop Commands:"
echo "   All:    ./stop_complete_system.sh"
echo "   API:    kill \$(cat /workspace/logs/app.pid)"
echo "   Celery: kill \$(cat /workspace/logs/celery.pid)"
echo "   Redis:  redis-cli shutdown"
echo ""
echo "🧪 Test Training:"
echo "   python test_fresh_lora.py"