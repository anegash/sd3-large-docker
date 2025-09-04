#!/bin/bash

# RunPod Setup Script for SD3.5 Large LoRA Training System
# This script sets up the complete environment on RunPod network volume

set -e  # Exit on any error

echo "🚀 Starting RunPod setup for SD3.5 Large LoRA Training System..."

# Navigate to network volume
cd /workspace/sd3-large-docker || {
    echo "❌ Error: /workspace/sd3-large-docker directory not found"
    echo "Make sure you've cloned the repository to the network volume"
    exit 1
}

echo "📁 Working directory: $(pwd)"

# Check available space
echo "💾 Checking disk space..."
df -h /workspace

# Install Poetry
echo "📦 Installing Poetry..."
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="/root/.local/bin:$PATH"
    echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
else
    echo "✅ Poetry already installed"
fi

# Verify Poetry installation
poetry --version || {
    echo "❌ Poetry installation failed"
    exit 1
}

# Configure Poetry to use network volume
echo "⚙️ Configuring Poetry to use network volume..."
export POETRY_CACHE_DIR=/workspace/poetry_cache
export POETRY_VENV_PATH=/workspace/poetry_venvs
mkdir -p /workspace/poetry_cache /workspace/poetry_venvs

poetry config cache-dir /workspace/poetry_cache
poetry config virtualenvs.path /workspace/poetry_venvs

# Set HuggingFace cache to network volume
echo "🤗 Configuring HuggingFace cache..."
export HF_HOME=/workspace/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/workspace/huggingface_cache
mkdir -p /workspace/huggingface_cache

# Add environment variables to bashrc for persistence
echo 'export POETRY_CACHE_DIR=/workspace/poetry_cache' >> ~/.bashrc
echo 'export POETRY_VENV_PATH=/workspace/poetry_venvs' >> ~/.bashrc
echo 'export HF_HOME=/workspace/huggingface_cache' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=/workspace/huggingface_cache' >> ~/.bashrc

# Create directories for training data and models
echo "📂 Creating application directories..."
mkdir -p /workspace/training_data
mkdir -p /workspace/models
mkdir -p /workspace/generated_images

# Install dependencies
echo "📥 Installing Python dependencies..."
poetry install --only main

# Check GPU availability
echo "🖥️ Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ NVIDIA GPU detected"
else
    echo "⚠️ NVIDIA drivers not found - CPU mode will be used"
fi

# Set up HuggingFace authentication
echo "🔐 Setting up HuggingFace authentication..."
if [ -n "$HF_TOKEN" ]; then
    echo "✅ Found HF_TOKEN environment variable"
    export HUGGINGFACE_TOKEN=$HF_TOKEN
    echo "HUGGINGFACE_TOKEN=$HF_TOKEN" > .env
    echo 'export HUGGINGFACE_TOKEN=$HF_TOKEN' >> ~/.bashrc
    echo "✅ HuggingFace authentication configured automatically"
else
    echo "⚠️ HF_TOKEN not found - you'll need to run setup_huggingface.py manually"
    echo "Next step: Run 'poetry run python setup_huggingface.py' to authenticate with HuggingFace"
fi

echo ""
echo "🎉 RunPod setup completed successfully!"
echo ""
if [ -n "$HF_TOKEN" ]; then
    echo "Ready to start! Next steps:"
    echo "1. Start server: poetry run python main.py"
    echo "2. Start Celery worker: poetry run celery -A src.sd3_api.tasks.celery_app worker --loglevel=info -Q training"
else
    echo "Next steps:"
    echo "1. Run: poetry run python setup_huggingface.py"
    echo "2. Start server: poetry run python main.py"
    echo "3. Start Celery worker: poetry run celery -A src.sd3_api.tasks.celery_app worker --loglevel=info -Q training"
fi
echo ""
echo "Environment variables set:"
echo "- POETRY_CACHE_DIR: /workspace/poetry_cache"
echo "- POETRY_VENV_PATH: /workspace/poetry_venvs"
echo "- HF_HOME: /workspace/huggingface_cache"
echo "- HUGGINGFACE_HUB_CACHE: /workspace/huggingface_cache"
echo ""
echo "Directories created:"
echo "- /workspace/training_data (for LoRA training images)"
echo "- /workspace/models (for trained LoRA models)"
echo "- /workspace/generated_images (for generated outputs)"