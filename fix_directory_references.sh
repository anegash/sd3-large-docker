#!/bin/bash
# Fix Directory References and Run from Correct Location
# This script ensures everything runs from /workspace/sdxl-api instead of /workspace/sd3-large-docker

set -e

echo "🔧 FIXING DIRECTORY REFERENCES FOR RUNPOD"
echo "=========================================="

# Detect current working directory
CURRENT_DIR=$(pwd)
echo "Current directory: $CURRENT_DIR"

# Define the correct directories
OLD_DIR="/workspace/sd3-large-docker" 
NEW_DIR="/workspace/sdxl-api"
EXPECTED_DIRS=("$NEW_DIR" "$OLD_DIR" "/workspace")

echo "Target directory: $NEW_DIR"
echo

# 1. Find the correct directory with our code
echo "📁 LOCATING PROJECT DIRECTORY"
echo "-----------------------------"
PROJECT_DIR=""

for dir in "${EXPECTED_DIRS[@]}"; do
    if [ -d "$dir" ] && [ -f "$dir/main.py" ] && [ -d "$dir/src/sd3_api" ]; then
        echo "✅ Found project at: $dir"
        if grep -q "Use either CPU offloading OR manual GPU placement" "$dir/src/sd3_api/pipeline.py" 2>/dev/null; then
            echo "✅ Directory contains our fixes!"
            PROJECT_DIR="$dir"
            break
        else
            echo "❌ Directory missing our fixes"
        fi
    else
        echo "❌ Not found: $dir"
    fi
done

if [ -z "$PROJECT_DIR" ]; then
    echo "❌ Cannot find project directory with fixes!"
    exit 1
fi

echo "Using project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"
echo

# 2. Stop any running servers
echo "🛑 STOPPING EXISTING SERVERS"
echo "----------------------------"
echo "Killing Python processes..."
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "uvicorn.*api:app" 2>/dev/null || true
sleep 2

# Force kill if still running
if pgrep -f "python.*main.py" > /dev/null; then
    echo "Force killing remaining processes..."
    pkill -9 -f "python.*main.py" 2>/dev/null || true
fi

echo "✅ Servers stopped"
echo

# 3. Fix script references to old directory
echo "📝 FIXING SCRIPT REFERENCES"
echo "---------------------------"

# Fix start_runpod.sh if it exists
if [ -f "start_runpod.sh" ]; then
    echo "Updating start_runpod.sh..."
    
    # Create backup
    cp start_runpod.sh start_runpod.sh.backup
    
    # Update paths in the script
    sed -i "s|/workspace/sd3-large-docker|$PROJECT_DIR|g" start_runpod.sh 2>/dev/null || \
    sed -i '' "s|/workspace/sd3-large-docker|$PROJECT_DIR|g" start_runpod.sh 2>/dev/null || \
    echo "Could not update start_runpod.sh automatically"
    
    echo "✅ Updated start_runpod.sh"
else
    echo "❌ start_runpod.sh not found"
fi

# Fix any hardcoded paths in Python files
echo "Checking for hardcoded paths in Python files..."
if grep -r "/workspace/sd3-large-docker" src/ 2>/dev/null; then
    echo "❌ Found hardcoded old paths in Python files!"
    echo "Files with old paths:"
    grep -r "/workspace/sd3-large-docker" src/ | cut -d: -f1 | sort | uniq
    echo "Consider updating these manually"
else
    echo "✅ No hardcoded old paths in Python files"
fi
echo

# 4. Clear Python cache thoroughly
echo "🧹 CLEARING PYTHON CACHE"
echo "------------------------"
echo "Clearing .pyc files and __pycache__ directories..."
find "$PROJECT_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$PROJECT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Clear system-wide cache for this project
find /workspace -path "*sdxl*" -name "*.pyc" -delete 2>/dev/null || true
find /workspace -path "*sd3*" -name "*.pyc" -delete 2>/dev/null || true
find /workspace -path "*sdxl*" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find /workspace -path "*sd3*" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "✅ Python cache cleared"
echo

# 5. Setup environment and dependencies
echo "📦 SETTING UP ENVIRONMENT"
echo "-------------------------"

# Set environment variables
export HF_HOME=/workspace/huggingface_cache
export PATH="/root/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

echo "Environment variables set:"
echo "  HF_HOME=$HF_HOME"
echo "  PROJECT_DIR=$PROJECT_DIR"
echo "  PYTHONPATH includes PROJECT_DIR"

# Load .env if it exists
if [ -f .env ]; then
    echo "Loading .env file..."
    set -a && source .env && set +a
    echo "✅ Environment loaded"
else
    echo "⚠️  No .env file found"
fi

# Check for dependencies
echo "Checking Python dependencies..."
if command -v poetry > /dev/null; then
    echo "Using Poetry"
    # Install dependencies if not already installed
    if [ ! -f ".poetry_installed" ]; then
        echo "Installing dependencies..."
        poetry install --only main
        touch .poetry_installed
    else
        echo "Dependencies already installed"
    fi
    PYTHON_CMD="poetry run python"
elif [ -d "/workspace/venv" ]; then
    echo "Using existing venv at /workspace/venv"
    # Find the actual venv path dynamically
    VENV_PYTHON=$(find /workspace/venv -name "python" -type f | head -1)
    if [ -n "$VENV_PYTHON" ]; then
        PYTHON_CMD="$VENV_PYTHON"
    else
        echo "Could not find Python in venv, falling back to system Python"
        PYTHON_CMD="python"
    fi
else
    echo "Using system Python"
    PYTHON_CMD="python"
fi
echo

# 6. Verify our fixes are accessible
echo "🔍 VERIFYING FIXES"
echo "------------------"
echo "Checking if Python can import our modules..."

$PYTHON_CMD -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
try:
    import src.sd3_api.pipeline
    import src.sd3_api.lora_trainer
    print('✅ Successfully imported modules')
    print(f'Pipeline location: {src.sd3_api.pipeline.__file__}')
    print(f'LoRA trainer location: {src.sd3_api.lora_trainer.__file__}')
    
    # Check for our specific fixes
    with open(src.sd3_api.pipeline.__file__, 'r') as f:
        if 'Use either CPU offloading OR manual GPU placement' in f.read():
            print('✅ GPU offloading fix found')
        else:
            print('❌ GPU offloading fix NOT found')
    
    with open(src.sd3_api.lora_trainer.__file__, 'r') as f:
        content = f.read()
        if 'target_modules = list(target_modules)' in content and 'pytorch_lora_weights.bin' in content:
            print('✅ LoRA training fixes found')
        else:
            print('❌ LoRA training fixes NOT found')
            
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Module verification failed!"
    exit 1
fi
echo

# 7. Create logs directory
echo "📁 SETTING UP LOGS"
echo "------------------"
mkdir -p /workspace/logs
echo "✅ Logs directory ready"
echo

# 8. Start the server
echo "🚀 STARTING SERVER FROM CORRECT DIRECTORY"
echo "==========================================="
echo "Starting server from: $PROJECT_DIR"
echo "Using Python: $PYTHON_CMD"
echo "Server will bind to: http://0.0.0.0:8000"
echo

# Start server in background with proper logging
echo "Starting server..."
nohup $PYTHON_CMD main.py > /workspace/logs/sd3_server.log 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > /workspace/logs/server.pid

# Wait a moment and check if it started
sleep 3

if kill -0 $SERVER_PID 2>/dev/null; then
    echo "✅ Server started successfully!"
    echo "   PID: $SERVER_PID"
    echo "   Working directory: $PROJECT_DIR"
    echo "   Logs: tail -f /workspace/logs/sd3_server.log"
    echo "   URL: https://your-runpod-id-8000.proxy.runpod.net/"
    echo
    echo "🎯 VERIFICATION STEPS:"
    echo "1. Check logs for GPU offloading warning (should be GONE)"
    echo "2. Test LoRA training (should work without JSON errors)"
    echo "3. Verify weight files are created properly"
    echo
    echo "Monitor startup:"
    echo "tail -f /workspace/logs/sd3_server.log"
else
    echo "❌ Server failed to start!"
    echo "Check logs: cat /workspace/logs/sd3_server.log"
    exit 1
fi