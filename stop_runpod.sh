#!/bin/bash
# SD3.5 Large LoRA Training System - Stop Server Script

echo "🛑 Stopping SD3 server..."

PID_FILE="/workspace/logs/server.pid"

if [ -f "$PID_FILE" ]; then
    SERVER_PID=$(cat $PID_FILE)
    
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "   🔄 Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID
        
        # Wait for graceful shutdown
        sleep 2
        
        # Force kill if still running
        if kill -0 $SERVER_PID 2>/dev/null; then
            echo "   ⚡ Force stopping server..."
            kill -9 $SERVER_PID
        fi
        
        echo "   ✅ Server stopped successfully"
        rm -f $PID_FILE
    else
        echo "   ⚠️  Server process not running (PID: $SERVER_PID)"
        rm -f $PID_FILE
    fi
else
    echo "   ⚠️  No PID file found - server may not be running"
    
    # Try to find and kill any python main.py processes
    PIDS=$(pgrep -f "python main.py")
    if [ ! -z "$PIDS" ]; then
        echo "   🔍 Found SD3 processes, stopping them..."
        echo "$PIDS" | xargs kill
        echo "   ✅ Processes stopped"
    else
        echo "   ℹ️  No SD3 server processes found"
    fi
fi

echo "🏁 Stop command completed"