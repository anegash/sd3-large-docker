#!/bin/bash
# SD3.5 Large - Stop Server Script

echo "🛑 Stopping SD3 server..."

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
    # Kill any remaining processes
    PIDS=$(pgrep -f "python main.py")
    [ ! -z "$PIDS" ] && echo "$PIDS" | xargs kill
fi

echo "🏁 Done"