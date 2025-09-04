#!/bin/bash

# Stop SD3.5 Large LoRA Training System

echo "🛑 Stopping SD3.5 Large LoRA Training System..."

# Stop by PID file if it exists
if [ -f /workspace/logs/app.pid ]; then
    APP_PID=$(cat /workspace/logs/app.pid)
    if kill -0 $APP_PID 2>/dev/null; then
        echo "   Stopping process with PID: $APP_PID"
        kill $APP_PID
        sleep 2
        
        # Force kill if still running
        if kill -0 $APP_PID 2>/dev/null; then
            echo "   Force stopping..."
            kill -9 $APP_PID
        fi
        
        rm /workspace/logs/app.pid
        echo "✅ Application stopped"
    else
        echo "   PID file exists but process not running"
        rm /workspace/logs/app.pid
    fi
else
    echo "   No PID file found, looking for processes..."
    
    # Find and kill python main.py processes
    PYTHON_PIDS=$(pgrep -f "python.*main.py")
    if [ -n "$PYTHON_PIDS" ]; then
        echo "   Found Python processes: $PYTHON_PIDS"
        kill $PYTHON_PIDS
        sleep 2
        
        # Force kill if still running
        kill -9 $(pgrep -f "python.*main.py") 2>/dev/null || true
        echo "✅ Python processes stopped"
    fi
    
    # Find and kill uvicorn processes
    UVICORN_PIDS=$(pgrep -f "uvicorn.*src.sd3_api.api")
    if [ -n "$UVICORN_PIDS" ]; then
        echo "   Found Uvicorn processes: $UVICORN_PIDS"
        kill $UVICORN_PIDS
        sleep 2
        
        # Force kill if still running  
        kill -9 $(pgrep -f "uvicorn.*src.sd3_api.api") 2>/dev/null || true
        echo "✅ Uvicorn processes stopped"
    fi
fi

echo ""
echo "🔍 Checking remaining processes..."
REMAINING=$(ps aux | grep -E "(main.py|uvicorn.*sd3_api)" | grep -v grep)
if [ -n "$REMAINING" ]; then
    echo "⚠️  Some processes may still be running:"
    echo "$REMAINING"
else
    echo "✅ All application processes stopped"
fi