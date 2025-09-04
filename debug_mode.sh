#!/bin/bash

# Debug Mode Script for RunPod Live Debugging
# Keeps container alive and provides restart functionality

echo "🔧 ENTERING DEBUG MODE"
echo "====================="
echo "Container will stay alive for debugging"
echo ""
echo "Available commands:"
echo "  /app/restart_server.sh   - Restart just the API server"
echo "  /app/restart_celery.sh   - Restart just Celery worker"  
echo "  /app/show_logs.sh        - Show recent logs"
echo "  /app/test_fix.sh         - Test the current train_lora fix"
echo ""
echo "Files you can edit directly:"
echo "  /app/src/sd3_api/tasks/training_tasks.py  - Main file with train_lora"
echo "  /app/src/sd3_api/lora_api.py              - API endpoints"
echo "  /app/src/sd3_api/api.py                   - Main API"
echo ""

# Create restart scripts
cat > /app/restart_server.sh << 'EOF'
#!/bin/bash
echo "🔄 Restarting API server..."

# Kill existing uvicorn processes
pkill -f uvicorn || true
sleep 2

# Start server again
cd /app
echo "📡 Starting uvicorn server..."
python -m uvicorn src.sd3_api.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level debug \
    --access-log 2>&1 | tee /workspace/logs/uvicorn.log &

echo "✅ Server restart initiated. Check /app/show_logs.sh for status"
EOF

cat > /app/restart_celery.sh << 'EOF'
#!/bin/bash
echo "🔄 Restarting Celery worker..."

# Kill existing celery processes
pkill -f celery || true
sleep 2

# Start Celery worker again
cd /app
celery -A src.sd3_api.tasks.celery_app worker \
    --loglevel=info \
    --concurrency=1 \
    --queues=training,default \
    --logfile=/workspace/logs/celery.log \
    --detach

sleep 2
if pgrep -f "celery.*worker" > /dev/null; then
    echo "✅ Celery worker restarted successfully"
else
    echo "❌ Celery worker failed to restart"
fi
EOF

cat > /app/show_logs.sh << 'EOF'
#!/bin/bash
echo "📝 Recent API logs:"
tail -20 /workspace/logs/uvicorn.log 2>/dev/null || echo "No API logs yet"
echo ""
echo "📝 Recent Celery logs:"
tail -20 /workspace/logs/celery.log 2>/dev/null || echo "No Celery logs yet"
echo ""
echo "📝 Process status:"
echo "API server: $(pgrep -f uvicorn > /dev/null && echo "✅ Running" || echo "❌ Stopped")"
echo "Celery worker: $(pgrep -f "celery.*worker" > /dev/null && echo "✅ Running" || echo "❌ Stopped")"
echo "Redis server: $(redis-cli ping > /dev/null 2>&1 && echo "✅ Running" || echo "❌ Stopped")"
EOF

cat > /app/test_fix.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing the Celery serialization fix..."

# Test basic Celery functionality
echo "1. Testing basic Celery..."
curl -s -X POST "http://localhost:8000/lora/debug/test-celery" | python -m json.tool || echo "❌ Basic test failed"

echo ""
echo "2. Testing training imports..."  
curl -s -X POST "http://localhost:8000/lora/debug/test-training-imports" \
    -H "Content-Type: application/json" \
    -d '{"child_id": "test", "model_id": 1}' | python -m json.tool || echo "❌ Import test failed"

echo ""
echo "3. Creating test child..."
curl -s -X POST "http://localhost:8000/lora/children" \
    -H "Content-Type: application/json" \
    -d '{"id": "debug_test", "name": "Debug Test", "description": "Testing fix"}' | python -m json.tool || echo "❌ Child creation failed"

echo ""
echo "4. Starting train_lora (should not stay in PENDING)..."
RESPONSE=$(curl -s -X POST "http://localhost:8000/lora/children/debug_test/train" \
    -H "Content-Type: application/json" \
    -d '{"training_config": {"training_steps": 5, "lora_rank": 16}}')

echo "$RESPONSE" | python -m json.tool || echo "❌ Training start failed"

# Extract task ID from response
TASK_ID=$(echo "$RESPONSE" | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('task_id', 'unknown'))" 2>/dev/null)

if [ "$TASK_ID" != "unknown" ]; then
    echo ""
    echo "5. Monitoring task status (should progress beyond PENDING)..."
    for i in {1..10}; do
        echo "Check $i/10:"
        curl -s "http://localhost:8000/lora/debug/task-status/$TASK_ID" | python -m json.tool || echo "❌ Status check failed"
        sleep 2
    done
fi
EOF

# Make scripts executable
chmod +x /app/restart_server.sh /app/restart_celery.sh /app/show_logs.sh /app/test_fix.sh

# Start a keep-alive loop that does periodic activity
echo "🔄 Starting keep-alive background process..."
(
    while true; do
        # Ping ourselves every 30 seconds to show activity
        curl -s http://localhost:8000/ > /dev/null 2>&1 || true
        
        # Write a heartbeat to logs
        echo "[$(date)] Debug mode heartbeat" >> /workspace/logs/debug_heartbeat.log
        
        # Sleep for 30 seconds
        sleep 30
    done
) &

KEEPALIVE_PID=$!
echo "✅ Keep-alive process started (PID: $KEEPALIVE_PID)"

echo ""
echo "🎯 DEBUG MODE READY!"
echo "Container will stay alive indefinitely."
echo "You can now SSH into RunPod and run the commands above."
echo ""
echo "💡 Quick start:"
echo "   ssh into your pod"
echo "   /app/test_fix.sh              # Test current fix" 
echo "   edit /app/src/sd3_api/tasks/training_tasks.py"
echo "   /app/restart_server.sh        # Apply changes"
echo "   /app/test_fix.sh              # Test again"
echo ""

# Keep the main process alive forever
while true; do
    sleep 60
    echo "[$(date)] Debug mode active - container staying alive" >> /workspace/logs/debug_mode.log
done