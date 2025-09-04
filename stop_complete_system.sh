#!/bin/bash

# Stop Complete SD3.5 LoRA Training System

echo "🛑 Stopping Complete SD3.5 LoRA Training System"
echo "==============================================="

# Stop Celery Worker
if [ -f /workspace/logs/celery.pid ]; then
    CELERY_PID=$(cat /workspace/logs/celery.pid)
    if kill -0 $CELERY_PID 2>/dev/null; then
        echo "🔄 Stopping Celery worker (PID: $CELERY_PID)..."
        kill $CELERY_PID
        sleep 2
        
        # Force kill if still running
        if kill -0 $CELERY_PID 2>/dev/null; then
            echo "   Force stopping Celery..."
            kill -9 $CELERY_PID
        fi
    fi
    rm -f /workspace/logs/celery.pid
    echo "✅ Celery worker stopped"
else
    # Fallback: kill all celery processes
    pkill -f "celery.*worker" && echo "✅ Celery processes stopped"
fi

# Stop API Server
if [ -f /workspace/logs/app.pid ]; then
    APP_PID=$(cat /workspace/logs/app.pid)
    if kill -0 $APP_PID 2>/dev/null; then
        echo "🌐 Stopping API server (PID: $APP_PID)..."
        kill $APP_PID
        sleep 2
        
        # Force kill if still running
        if kill -0 $APP_PID 2>/dev/null; then
            echo "   Force stopping API server..."
            kill -9 $APP_PID
        fi
    fi
    rm -f /workspace/logs/app.pid
    echo "✅ API server stopped"
else
    # Fallback: kill all uvicorn/python processes
    pkill -f "python.*main.py" && echo "✅ Python processes stopped"
    pkill -f "uvicorn.*sd3_api" && echo "✅ Uvicorn processes stopped"
fi

# Stop Redis Server
echo "🔴 Stopping Redis server..."
redis-cli shutdown 2>/dev/null && echo "✅ Redis server stopped" || echo "   Redis was not running"

echo ""
echo "🔍 Checking for remaining processes..."
REMAINING=$(ps aux | grep -E "(python.*main.py|celery.*worker|redis-server)" | grep -v grep)
if [ -n "$REMAINING" ]; then
    echo "⚠️  Some processes may still be running:"
    echo "$REMAINING"
else
    echo "✅ All system processes stopped successfully"
fi

echo ""
echo "📁 Log files preserved:"
echo "   API:    /workspace/logs/app.log"
echo "   Celery: /workspace/logs/celery.log"