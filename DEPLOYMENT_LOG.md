# SD3.5 Large Docker + LoRA Training System - Complete Setup Log

## What We Built
Successfully created a production-ready Docker-based SD3.5 Large image generation API with LoRA training capabilities, deployed to RunPod with persistent storage.

## Key Features Implemented
- **SD3.5 Large image generation** with 48GB VRAM support
- **LoRA training system** with Celery background workers  
- **Multi-child management** with token replacement (`{child_id}`)
- **Persistent volume storage** to avoid re-downloading 8GB model
- **FastAPI with full documentation** at `/docs`
- **Real-time progress tracking** and training status
- **HuggingFace token support** (both `HF_TOKEN` and `HUGGINGFACE_TOKEN`)

## Issues Fixed During Development

### 1. **Celery Startup Loop (Critical Fix)**
- **Problem**: Container kept restarting due to `--queue` flag error
- **Solution**: Changed `--queue=training` to `--queues=training` in `start_pod.sh`
- **Location**: `start_pod.sh:78`

### 2. **HuggingFace Token Recognition**
- **Problem**: Code only looked for `HUGGINGFACE_TOKEN` but RunPod used `HF_TOKEN`
- **Solution**: Modified to support both environment variables
- **Location**: `src/sd3_api/pipeline.py:43`
- **Code Change**:
  ```python
  # Before
  hf_token = os.getenv("HUGGINGFACE_TOKEN")
  
  # After  
  hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
  ```

### 3. **Volume Persistence Setup**
- **Problem**: Models downloading on every restart (8GB each time)
- **Solution**: Configured proper volume paths for RunPod's `/volume` mount
- **Changes**: Updated all cache directories to use `/volume/` instead of `/app/`

### 4. **Celery Flower Compatibility**
- **Problem**: `celery flower` command not available in installed version
- **Solution**: Removed Celery Flower monitoring to eliminate startup errors
- **Status**: Optional monitoring removed, core functionality preserved

## Docker Image Publishing Workflow

### Step 1: Local Development
```bash
# Working directory: /Users/antenehnegash/development/ml/sd3-large-docker/
```

### Step 2: File Transfer to EC2 Build Server
```bash
# Copy entire project to EC2 (excluding git/cache files)
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' --exclude='poetry.lock' /Users/antenehnegash/development/ml/sd3-large-docker/ ubuntu@ec2-35-173-103-83.compute-1.amazonaws.com:~/sd3-large-docker/

# Copy individual updated files as needed
rsync -avz /Users/antenehnegash/development/ml/sd3-large-docker/start_pod.sh ubuntu@ec2-35-173-103-83.compute-1.amazonaws.com:~/sd3-large-docker/
rsync -avz /Users/antenehnegash/development/ml/sd3-large-docker/src/sd3_api/pipeline.py ubuntu@ec2-35-173-103-83.compute-1.amazonaws.com:~/sd3-large-docker/src/sd3_api/
```

### Step 3: Docker Build on EC2
```bash
# SSH to EC2 build server
ssh ubuntu@ec2-35-173-103-83.compute-1.amazonaws.com

# Docker system setup (first time only)
sudo apt update && sudo apt install -y docker.io
sudo usermod -aG docker ubuntu
sudo systemctl start docker && sudo systemctl enable docker

# Build Docker image
cd sd3-large-docker
sudo docker build -f Dockerfile.fixed -t sd3-large-api:latest .
```

### Step 4: Docker Hub Authentication & Push
```bash
# Login to Docker Hub
echo '<docker_hub_token>' | sudo docker login --username <username> --password-stdin

# Tag for Docker Hub
sudo docker tag sd3-large-api:latest antenehmtk/sd3-docker-lazy-lora:latest

# Push to registry  
sudo docker push antenehmtk/sd3-docker-lazy-lora:latest
```

### Version History Published
- `antenehmtk/sd3-docker-lazy-lora:v1.1` - Fixed Celery command
- `antenehmtk/sd3-docker-lazy-lora:v1.2` - Added HF_TOKEN support  
- `antenehmtk/sd3-docker-lazy-lora:latest` - Volume persistence + logging

## RunPod Deployment

### Initial Setup
1. **Create RunPod Pod** with:
   - Image: `antenehmtk/sd3-docker-lazy-lora:latest`
   - GPU: 48GB VRAM (NVIDIA A40)
   - Environment: `HF_TOKEN=<your-token>`
   - Ports: 8000, 5555

### Current Deployment Status
- **API URL**: https://kz09j0vpybpa60-8000.proxy.runpod.net/
- **SSH**: `ssh kz09j0vpybpa60-64411b5c@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **Status**: ✅ Running successfully
- **Data**: 2 children, 10 training images persisted

## Final Architecture

### Storage Layout
```
/volume/                           # Persistent RunPod volume
├── .cache/huggingface/           # SD3.5 Large model cache (8GB)
│   └── hub/                      # HuggingFace transformers cache
├── data/                         # Application data
│   └── children/                 # LoRA training data
│       ├── child_001/
│       │   ├── images/          # Training images  
│       │   └── models/          # LoRA weights (.safetensors)
│       └── test_child/
└── logs/                         # Application logs
    ├── uvicorn.log              # API server logs
    └── celery.log               # Background worker logs

/app/                             # Application container
├── src/sd3_api/                 # Python package
├── start_pod.sh                 # Startup script
├── pyproject.toml               # Poetry config
└── data -> /volume/data         # Symlink to persistent storage
```

### Key Environment Variables
```bash
# HuggingFace Authentication  
HF_TOKEN=<your-token>                    # Primary (RunPod standard)
HUGGINGFACE_TOKEN=<your-token>           # Fallback support

# Python Environment
PYTHONPATH="/app/src:$PYTHONPATH" 
PYTHONUNBUFFERED=1

# HuggingFace Cache Directories
HF_HOME="/volume/.cache/huggingface"
TRANSFORMERS_CACHE="/volume/.cache/huggingface/hub"
HF_DATASETS_CACHE="/volume/.cache/huggingface/datasets"
```

### Startup Sequence
1. ✅ System info logging (GPU, memory, environment)
2. ✅ Redis server startup  
3. ✅ Database initialization
4. ✅ Celery worker startup (LoRA training queue)
5. ✅ FastAPI server with eager model loading
6. ✅ SD3.5 Large model download/cache (first run only)
7. ✅ API ready for requests

## Testing & Verification

### API Health Check
```bash
curl "https://kz09j0vpybpa60-8000.proxy.runpod.net/" | jq .
# Response: {"message": "Stable Diffusion 3.5 API is running!", "device": "Ready on CUDA GPU: NVIDIA A40"}
```

### Child Management  
```bash
# Create child
curl -X POST "https://kz09j0vpybpa60-8000.proxy.runpod.net/lora/children" \
  -H "Content-Type: application/json" \
  -d '{"id": "test_child", "name": "Test", "description": "Testing"}'

# List children
curl "https://kz09j0vpybpa60-8000.proxy.runpod.net/lora/children" | jq .
```

### System Statistics
```bash
curl "https://kz09j0vpybpa60-8000.proxy.runpod.net/lora/system/statistics" | jq .
# Shows: total_children: 2, total_training_images: 10, persistent storage working
```

## Key Success Metrics
- ✅ **Zero startup failures** after fixes
- ✅ **Fast restart times** (model cached in volume)  
- ✅ **Data persistence** across pod restarts
- ✅ **Full LoRA workflow** operational
- ✅ **Production-ready logging** for debugging

## Next Steps for Future Development
- Add Celery Flower monitoring (install proper version)
- Implement model version management
- Add training progress webhooks
- Optimize model loading for faster startup
- Add batch processing capabilities