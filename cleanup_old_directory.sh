#!/bin/bash
# Clean up old directory references and ensure single source of truth

echo "🧹 CLEANING UP DUPLICATE DIRECTORIES"
echo "=================================="

# Current working directory
CURRENT_DIR=$(pwd)
echo "Current directory: $CURRENT_DIR"

# Check if we're in the right place
if [[ "$CURRENT_DIR" =~ sdxl-api$ ]]; then
    echo "✅ Running from correct directory: sdxl-api"
    OLD_DIR="/workspace/sd3-large-docker"
elif [[ "$CURRENT_DIR" =~ sd3-large-docker$ ]]; then
    echo "⚠️  Running from old directory: sd3-large-docker"
    echo "Moving to correct directory..."
    
    # Check if sdxl-api exists
    if [ -d "/workspace/sdxl-api" ]; then
        echo "✅ Found /workspace/sdxl-api - switching to it"
        cd /workspace/sdxl-api
    else
        echo "📁 Creating /workspace/sdxl-api and moving content"
        mkdir -p /workspace/sdxl-api
        cp -r /workspace/sd3-large-docker/* /workspace/sdxl-api/
        cp -r /workspace/sd3-large-docker/.git /workspace/sdxl-api/
        cd /workspace/sdxl-api
    fi
    
    OLD_DIR="/workspace/sd3-large-docker"
else
    echo "❌ Not in expected directory. Please cd to /workspace/sdxl-api first"
    exit 1
fi

# Stop any processes that might be running from the old directory
echo "🛑 Stopping processes from old directory..."
pkill -f "python.*sd3-large-docker.*main.py" 2>/dev/null || true
pkill -f "uvicorn.*sd3-large-docker" 2>/dev/null || true

# Clear cache from old directory
if [ -d "$OLD_DIR" ]; then
    echo "🧹 Clearing cache from old directory..."
    find "$OLD_DIR" -name "*.pyc" -delete 2>/dev/null || true
    find "$OLD_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
fi

echo "✅ Directory cleanup complete!"
echo "Current working directory: $(pwd)"
echo "Run ./fix_directory_references.sh next to complete the setup"