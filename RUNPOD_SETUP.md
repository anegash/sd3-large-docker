# SD3.5 Large LoRA Training System - RunPod Deployment

## 🎯 Project Overview

Successfully deployed SD3.5 Large API with LoRA (Low-Rank Adaptation) training functionality on RunPod A40 GPU Pod with persistent storage.

### Features Implemented:
- ✅ **SD3.5 Large image generation** via FastAPI 
- ✅ **LoRA training** for personalized image generation (5-20 images)
- ✅ **RunPod persistent storage** - all data survives pod restarts
- ✅ **Background server execution** with process management
- ✅ **Automatic environment setup** for pod restarts

## 🚀 Deployment History

### Branch: `feature/lora-training-runpod`
- Added LoRA training functionality using PEFT library
- Extended API with new endpoints: `/train-lora`, `/lora`, `/lora/{id}`
- Configured persistent storage using `/workspace/` directories
- Created RunPod-specific setup and initialization scripts

### Key Components:
1. **LoRA Training Module** (`src/sd3_api/lora_trainer.py`)
2. **Extended Pipeline** (`src/sd3_api/pipeline.py`) - LoRA weight management
3. **API Endpoints** (`src/sd3_api/api.py`) - Training and generation
4. **RunPod Scripts** (`setup_runpod.py`, `start_runpod.sh`, `stop_runpod.sh`)

## 🗂️ Persistent Storage Structure

```
/workspace/
├── sdxl-api/                 # Project code (git repo)
├── huggingface_cache/        # SD3.5 model cache (~8GB)
├── lora_weights/             # Trained LoRA weights per person
├── logs/                     # Server logs and PID files
├── start_sdxl.sh             # Background server startup
└── stop_sdxl.sh              # Server stop script
```

## 📋 RunPod Setup Commands

### Initial Deployment:
```bash
# Clone the repository  
git clone https://github.com/anegash/sd3-large-docker.git /workspace/sdxl-api

# One-time comprehensive setup
cd /workspace/sdxl-api
python3 setup_runpod.py  # Uses HF_TOKEN from RunPod environment
```

### After Pod Restart:
```bash
# Start server
cd /workspace/sdxl-api
./start_runpod.sh
```

## 🔧 Server Management

### Commands:
```bash
# Start server in background
./start_runpod.sh

# Stop server  
./stop_runpod.sh

# Check server health
curl http://localhost:8000/

# View logs
tail -f /workspace/logs/sd3_server.log
```

## 🎨 API Endpoints

### Image Generation:
```bash
# GET method with LoRA
curl "http://localhost:8000/generate?prompt=photo%20of%20john&person_id=john_doe&steps=20"

# POST method with LoRA
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "portrait of john_doe", "person_id": "john_doe", "steps": 20}'
```

### LoRA Training:
```bash
# Train LoRA model with 5-20 images
curl -X POST "http://localhost:8000/train-lora" \
  -F "person_id=john_doe" \
  -F "num_train_epochs=100" \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg" \
  -F "files=@photo3.jpg"

# List trained models
curl "http://localhost:8000/lora"

# Delete LoRA model
curl -X DELETE "http://localhost:8000/lora/john_doe"
```

## 🐛 Troubleshooting

### Common Issues Fixed:
1. **Poetry not found**: Fixed by `start_runpod.sh` auto-installation
2. **Corrupted model cache**: `rm -rf /workspace/huggingface_cache/*`
3. **TaskType.DIFFUSION error**: Fixed by using `TaskType.FEATURE_EXTRACTION`
4. **Lock file conflicts**: Updated `poetry.lock` with new dependencies

### Environment Variables:
- `HF_TOKEN`: HuggingFace token (set in RunPod environment)
- `WORKSPACE_DIR=/workspace`
- `HF_HOME=/workspace/huggingface_cache`

## ✅ Successful Deployment

**Hardware**: RunPod A40 GPU Pod  
**Status**: ✅ Successfully deployed and tested  
**Model**: SD3.5 Large (~8GB) with LoRA training capability  
**Storage**: Persistent `/workspace/` storage configured  
**Server**: Running in background at `http://localhost:8000`

### Key Success Factors:
- Persistent storage prevents re-downloading models
- Background execution allows terminal usage
- Simplified setup scripts reduce complexity
- LoRA weights persist across sessions
- Comprehensive error handling and logging