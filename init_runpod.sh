#!/bin/bash
# SD3.5 Large LoRA Training System - RunPod Initialization Script
# This script sets up the environment for pod restarts

set -e

echo "🚀 SD3 LoRA Training System - RunPod Initialization"
echo "=================================================="

# Set up environment variables
export PATH="/root/.local/bin:$PATH"
export WORKSPACE_DIR=/workspace
export HF_HOME=/workspace/huggingface_cache
export TRANSFORMERS_CACHE=/workspace/huggingface_cache/transformers
export HF_DATASETS_CACHE=/workspace/huggingface_cache/datasets

# Add environment variables to current session and future sessions
cat > /tmp/sd3_env.sh << 'EOF'
# SD3 Environment Variables
export PATH="/root/.local/bin:$PATH"
export WORKSPACE_DIR=/workspace
export HF_HOME=/workspace/huggingface_cache
export TRANSFORMERS_CACHE=/workspace/huggingface_cache/transformers
export HF_DATASETS_CACHE=/workspace/huggingface_cache/datasets

# SD3 LoRA Training System aliases
alias sd3-start="/workspace/start_sd3.sh"
alias sd3-stop="/workspace/stop_sd3.sh"
alias sd3-logs="tail -f /workspace/logs/sd3_server.log"
alias sd3-status="curl -s http://localhost:8000/ | jq . || curl -s http://localhost:8000/"
alias sd3-env="cd /workspace/sd3-large-docker && source /tmp/sd3_env.sh && poetry shell"
EOF

# Source the environment for current session
source /tmp/sd3_env.sh

# Add to bashrc if not already there
if ! grep -q "SD3 Environment Variables" ~/.bashrc; then
    echo "" >> ~/.bashrc
    cat /tmp/sd3_env.sh >> ~/.bashrc
    echo "✅ Environment variables added to ~/.bashrc"
else
    echo "✅ Environment variables already in ~/.bashrc"
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "🔧 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo "✅ Poetry installed"
else
    echo "✅ Poetry already available"
fi

# Check if Python environment exists
cd /workspace/sd3-large-docker
if [ ! -d "/workspace/venv" ] || ! poetry env list | grep -q sd3-large-api; then
    echo "🐍 Setting up Python environment..."
    poetry config virtualenvs.path /workspace/venv
    poetry config virtualenvs.create true
    poetry config virtualenvs.in-project false
    poetry install --only main
    echo "✅ Python environment ready"
else
    echo "✅ Python environment already exists"
fi

# Make sure startup scripts are executable and in the right place
if [ -f "/workspace/sd3-large-docker/start_runpod.sh" ]; then
    cp /workspace/sd3-large-docker/start_runpod.sh /workspace/start_sd3.sh
    chmod +x /workspace/start_sd3.sh
    echo "✅ Startup script ready at /workspace/start_sd3.sh"
fi

if [ -f "/workspace/sd3-large-docker/stop_runpod.sh" ]; then
    cp /workspace/sd3-large-docker/stop_runpod.sh /workspace/stop_sd3.sh
    chmod +x /workspace/stop_sd3.sh
    echo "✅ Stop script ready at /workspace/stop_sd3.sh"
fi

echo ""
echo "🎉 RunPod initialization complete!"
echo ""
echo "🔧 Available commands:"
echo "   • /workspace/start_sd3.sh  - Start server"
echo "   • /workspace/stop_sd3.sh   - Stop server"
echo "   • source ~/.bashrc         - Load aliases"
echo ""
echo "🚀 To start the server:"
echo "   /workspace/start_sd3.sh"