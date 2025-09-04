#!/bin/bash

# SD3.5 Large LoRA Training System - RunPod Startup Script
# Keeps container alive and initializes everything needed for development

echo "============================================================"
echo "🚀 SD3.5 LARGE LORA TRAINING SYSTEM - RUNPOD CONTAINER"
echo "============================================================"
echo "Container ready for SD3.5 Large LoRA training development"
echo ""

# Set up environment variables
export PYTHONPATH="/workspace/sd3-large-docker/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export PYTHONUNBUFFERED=1
export POETRY_CACHE_DIR=/workspace/poetry_cache
export POETRY_VENV_PATH=/workspace/poetry_venvs
export HF_HOME=/workspace/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/workspace/huggingface_cache

# Display system information
echo "🔧 System Information:"
echo "  CUDA Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null || echo 'Not available')"
echo "  GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
echo "  Python: $(python --version)"
echo "  Poetry: $(poetry --version 2>/dev/null || echo 'Not found')"
echo "  Working Dir: $(pwd)"
echo ""

# Create required directories
mkdir -p /workspace/logs \
         /workspace/training_data \
         /workspace/models \
         /workspace/generated_images

# Start essential services
echo "🔴 Starting Redis server for Celery..."
service redis-server start
sleep 2

if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis server is running"
else
    echo "❌ Redis server failed to start"
fi

# Start SSH service
echo "🔐 Starting SSH service..."
service ssh start
echo "✅ SSH server is running (password: runpod)"

# Navigate to project directory
cd /workspace/sd3-large-docker

# Set up environment in bashrc for SSH sessions
if ! grep -q "SD3 LORA SYSTEM" /root/.bashrc; then
    echo "" >> /root/.bashrc
    echo "# SD3 LORA SYSTEM ENVIRONMENT" >> /root/.bashrc
    echo "export PATH=\"/root/.local/bin:\$PATH\"" >> /root/.bashrc
    echo "export POETRY_CACHE_DIR=/workspace/poetry_cache" >> /root/.bashrc
    echo "export POETRY_VENV_PATH=/workspace/poetry_venvs" >> /root/.bashrc
    echo "export HF_HOME=/workspace/huggingface_cache" >> /root/.bashrc
    echo "export HUGGINGFACE_HUB_CACHE=/workspace/huggingface_cache" >> /root/.bashrc
    echo "cd /workspace/sd3-large-docker" >> /root/.bashrc
fi

# Initialize database
echo "🗄️  Initializing database..."
poetry run python -c "
import sys
sys.path.insert(0, '/workspace/sd3-large-docker/src')
from sd3_api.database.connection import get_engine
from sd3_api.database.models import Base
try:
    engine = get_engine()
    Base.metadata.create_all(engine)
    print('✅ Database initialized successfully')
except Exception as e:
    print(f'⚠️  Database initialization failed: {e}')
" 2>/dev/null || echo "⚠️  Database initialization skipped"

echo ""
echo "📁 SD3.5 LoRA System Directories:"
echo "  /workspace/sd3-large-docker    - Main project code"
echo "  /workspace/huggingface_cache   - Model cache (network volume)"
echo "  /workspace/training_data       - LoRA training images"
echo "  /workspace/models              - Trained LoRA models"
echo "  /workspace/generated_images    - Generated outputs"
echo "  /workspace/logs               - System logs"
echo ""
echo "🛠️  Development Commands:"
echo "  ./start_app.sh                  - Start SD3.5 API server"
echo "  ./start_worker.sh               - Start Celery LoRA training worker"
echo "  git pull origin live-debug-runpod - Update code"
echo "  poetry run python main.py       - Start server manually"
echo "  nvidia-smi                      - Check GPU status"
echo "  htop                            - System monitor"
echo ""
echo "🤗 HuggingFace Setup:"
echo "  Set HF_TOKEN environment variable in RunPod dashboard"
echo "  Or run: export HF_TOKEN=your_token_here"
echo ""
echo "🔗 Exposed Ports:"
echo "  22   - SSH access"
echo "  8000 - SD3.5 API server"
echo "  6006 - TensorBoard (if needed)"
echo "  8080 - Alternative HTTP"
echo ""

# Show quick status
echo "🔍 Quick Status Check:"
echo "  Project files: $(ls -la 2>/dev/null | wc -l) items"
echo "  GPU memory: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)MB free"
echo "  Disk space: $(df -h /workspace | tail -1 | awk '{print $4}') available"
echo ""
echo "✅ Container ready! SSH in and run ./start_app.sh to begin."
echo "============================================================"

# Create log files and keep container alive
touch /workspace/logs/heartbeat.log /workspace/logs/container.log
echo "[$(date)] SD3.5 LoRA container initialized successfully" >> /workspace/logs/container.log

COUNTER=0
while true; do
    COUNTER=$((COUNTER + 1))
    TIMESTAMP=$(date)
    
    # Heartbeat every 5 minutes
    if [ $((COUNTER % 5)) -eq 0 ]; then
        echo "[${TIMESTAMP}] SD3.5 LoRA Container Heartbeat #${COUNTER}" >> /workspace/logs/heartbeat.log
    fi
    
    # System status every 30 minutes
    if [ $((COUNTER % 30)) -eq 0 ]; then
        echo "[${TIMESTAMP}] System Status Check:" >> /workspace/logs/container.log
        nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv >> /workspace/logs/container.log 2>/dev/null
        free -h >> /workspace/logs/container.log
        df -h /workspace >> /workspace/logs/container.log
        echo "---" >> /workspace/logs/container.log
    fi
    
    sleep 60
done