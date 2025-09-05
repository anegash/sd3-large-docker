#!/bin/bash
# SDXL - Stop Server Script

echo "🛑 Stopping SDXL server..."

PID_FILE="/workspace/logs/server.pid"

if [ -f "$PID_FILE" ]; then
    SERVER_PID=$(cat $PID_FILE)
    if kill -0 $SERVER_PID 2>/dev/null; then
        kill $SERVER_PID
        sleep 2
        kill -0 $SERVER_PID 2>/dev/null && kill -9 $SERVER_PID
        echo "✅ Server stopped"
    fi
    rm -f $PID_FILE
else
    # Kill any remaining processes more thoroughly
    echo "🔍 Searching for running Python processes..."
    PIDS=$(pgrep -f "python.*main.py\|uvicorn.*api:app")
    if [ ! -z "$PIDS" ]; then
        echo "Found PIDs: $PIDS"
        echo "$PIDS" | xargs kill
        sleep 2
        # Force kill if still running
        PIDS=$(pgrep -f "python.*main.py\|uvicorn.*api:app")
        [ ! -z "$PIDS" ] && echo "$PIDS" | xargs kill -9
    fi
fi

echo "🏁 Done"