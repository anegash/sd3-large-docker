#!/bin/bash
# SDXL - RunPod Startup Script
set -e

echo "🚀 Starting SDXL API..."

# Set environment variables
export HF_HOME=/workspace/huggingface_cache
export PATH="/root/.local/bin:$PATH"

# Install Poetry if missing
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Find project directory with our latest code and fixes
PROJECT_DIR=""
CANDIDATE_DIRS=("/workspace/sdxl-api" "/workspace/sd3-large-docker" "/workspace")

echo "🔍 Searching for project directory with latest fixes..."
for dir in "${CANDIDATE_DIRS[@]}"; do
    if [ -d "$dir" ] && [ -f "$dir/main.py" ] && [ -d "$dir/src/sd3_api" ]; then
        echo "Found project at: $dir"
        # Check if this directory has our fixes
        if grep -q "Use either CPU offloading OR manual GPU placement" "$dir/src/sd3_api/pipeline.py" 2>/dev/null; then
            echo "✅ Directory contains our fixes!"
            PROJECT_DIR="$dir"
            break
        else
            echo "⚠️  Directory missing our fixes, continuing search..."
        fi
    fi
done

if [ -z "$PROJECT_DIR" ]; then
    echo "❌ Cannot find project directory with fixes!"
    echo "Available directories in /workspace:"
    ls -la /workspace/ | grep -E "sd3|sdxl"
    exit 1
fi

echo "Using project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# Load environment
[ -f .env ] && export $(cat .env | grep -v '^#' | xargs)

# Clear Python cache to ensure new code loads
echo "🧹 Clearing Python cache..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Install dependencies (skip if already installed)
if [ ! -f ".poetry_installed" ]; then
    echo "📦 Installing dependencies..."
    poetry install --only main
    touch .poetry_installed
else
    echo "✅ Dependencies already installed"
fi

# Create logs directory
mkdir -p /workspace/logs

# Start server in background
echo "Starting server..."
nohup poetry run python main.py > /workspace/logs/sd3_server.log 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > /workspace/logs/server.pid

sleep 3
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "✅ Server running at http://localhost:8000"
    echo "📜 Logs: tail -f /workspace/logs/sd3_server.log"
else
    echo "❌ Server failed to start. Check logs."
    exit 1
fi