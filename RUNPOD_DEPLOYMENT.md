# RunPod Pod Deployment Guide
## SD3.5 Large LoRA Training System

Complete guide for deploying the full LoRA training system to RunPod Pods with A40 GPU.

## 🎯 Overview

This deployment provides a **persistent environment** with:
- ✅ **Full API server** with all LoRA endpoints 
- ✅ **Background Celery worker** for async training
- ✅ **Redis message broker** for job queue
- ✅ **Flower monitoring** for training progress
- ✅ **Persistent storage** for models and data
- ✅ **A40 GPU optimization** with 48GB VRAM

## 🚀 Quick Deploy

### Step 1: Push to Container Registry

```bash
# Build the container locally
docker build -t your-registry/sd3-lora-training:latest .

# Push to your container registry (Docker Hub, etc.)
docker push your-registry/sd3-lora-training:latest
```

### Step 2: Create RunPod Pod

1. **Go to RunPod Console**: https://console.runpod.io/
2. **Create Pod** → Select **A40** GPU
3. **Container Image**: `your-registry/sd3-lora-training:latest`
4. **Container Disk**: 50GB minimum
5. **Volume Disk**: 100GB minimum (for models/data)
6. **Environment Variables**:
   ```
   HUGGINGFACE_TOKEN=your_hf_token_here
   CUDA_VISIBLE_DEVICES=0
   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

### Step 3: Access Your API

- **Main API**: `https://your-pod-id-8000.proxy.runpod.net/`
- **Celery Monitor**: `https://your-pod-id-5555.proxy.runpod.net/`
- **API Docs**: `https://your-pod-id-8000.proxy.runpod.net/docs`

## 📋 Complete Setup Steps

### Prerequisites

1. **HuggingFace Token**: Get access to SD3.5 Large model
2. **Container Registry**: Docker Hub, AWS ECR, or similar
3. **RunPod Account**: With sufficient credits for A40 usage

### Detailed Deployment

#### 1. Prepare Container

```bash
# Clone repository
git clone <your-repo>
cd sd3-large-docker

# Build with correct tag
docker build -t your-registry/sd3-lora-training:latest .

# Test locally (optional)
docker run --gpus all \
  -e HUGGINGFACE_TOKEN=your_token \
  -p 8000:8000 \
  -p 5555:5555 \
  your-registry/sd3-lora-training:latest

# Push to registry
docker push your-registry/sd3-lora-training:latest
```

#### 2. Configure RunPod Pod

**Pod Configuration:**
- **Name**: `sd3-lora-training`
- **Image**: `your-registry/sd3-lora-training:latest`
- **GPU**: RTX A40 (48GB VRAM recommended)
- **CPU**: 8+ cores
- **RAM**: 32GB minimum
- **Container Disk**: 50GB (for system and cache)
- **Volume Disk**: 100GB (for models and training data)

**Environment Variables:**
```bash
HUGGINGFACE_TOKEN=your_actual_token_here
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**Port Mapping:**
- `8000`: Main API server
- `5555`: Celery Flower monitoring

#### 3. Verify Deployment

```bash
# Test API health
curl https://your-pod-id-8000.proxy.runpod.net/

# Check Celery workers
curl https://your-pod-id-5555.proxy.runpod.net/
```

## 🔧 Pod Management

### Starting/Stopping

The Pod will **automatically start** all services:
1. Redis server
2. Database initialization  
3. Celery worker (for training)
4. Celery Flower (monitoring)
5. FastAPI server (main API)

### Monitoring

- **API Health**: `GET /` endpoint shows device info and model status
- **Celery Monitor**: Flower UI at port 5555
- **Logs**: Available through RunPod console
- **Training Status**: `GET /lora/children/{child_id}/training-status`

### Persistent Data

Data is stored on the Volume Disk:
- `/app/data/children/`: Child profiles and training images
- `/app/data/lora_models/`: Trained LoRA models
- `/app/.cache/huggingface/`: Downloaded SD3.5 models

## 🎨 Using the LoRA Training System

### 1. Create a Child

```bash
curl -X POST "https://your-pod-8000.proxy.runpod.net/lora/children" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "child_001", 
    "name": "Alice",
    "description": "5-year-old girl with brown hair"
  }'
```

### 2. Upload Training Images

```bash
curl -X POST "https://your-pod-8000.proxy.runpod.net/lora/children/child_001/images" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "files=@image4.jpg" \
  -F "files=@image5.jpg"
```

### 3. Start LoRA Training

```bash
curl -X POST "https://your-pod-8000.proxy.runpod.net/lora/children/child_001/train" \
  -H "Content-Type: application/json" \
  -d '{
    "training_config": {
      "training_steps": 1000,
      "learning_rate": 1e-4,
      "lora_rank": 64,
      "batch_size": 1
    }
  }'
```

### 4. Monitor Training Progress

```bash
# Check training status
curl "https://your-pod-8000.proxy.runpod.net/lora/children/child_001/training-status"

# Monitor via Flower UI
open https://your-pod-5555.proxy.runpod.net/
```

### 5. Generate Images with Trained LoRA

```bash
curl -X POST "https://your-pod-8000.proxy.runpod.net/generate/with-children" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a happy {child_001} playing in the park",
    "steps": 20,
    "guidance": 7.5
  }'
```

## 💰 Cost Optimization

### A40 GPU Pricing (Approximate)
- **On-Demand**: ~$0.79/hour
- **Spot**: ~$0.39/hour (50% savings)

### Optimization Tips
1. **Use Spot Instances**: 50% cost reduction
2. **Stop when not training**: Pause Pod between training sessions
3. **Batch training jobs**: Train multiple children in sequence
4. **Use smaller models for testing**: Switch to lighter configs during development

## 🔧 Troubleshooting

### Common Issues

#### 1. Container Won't Start
```bash
# Check Pod logs in RunPod console
# Common cause: HUGGINGFACE_TOKEN not set or invalid
```

#### 2. GPU Not Detected
```bash
# SSH into Pod and check
nvidia-smi
# Ensure CUDA_VISIBLE_DEVICES=0 is set
```

#### 3. Training Jobs Stuck
```bash
# Check Celery worker status
curl https://your-pod-5555.proxy.runpod.net/
# Restart Pod if needed
```

#### 4. Out of Memory
```bash
# Reduce batch_size in training config
# Or reduce image resolution
# A40 has 48GB VRAM - should handle most workloads
```

### Debug Commands

```bash
# SSH into running Pod
# Check processes
ps aux | grep -E "(celery|uvicorn|redis)"

# Check GPU usage
nvidia-smi

# Check disk space
df -h

# View logs
tail -f /app/logs/celery.log
```

## 🎯 Production Recommendations

### For Production Use:
1. **Use A40 GPU**: 48GB VRAM for large batch training
2. **100GB+ Volume**: Store multiple trained models
3. **Set up monitoring**: Use Flower + custom metrics
4. **Backup models**: Export trained LoRAs regularly
5. **Use Spot instances**: Significant cost savings

### For Development/Testing:
1. **RTX 4090**: 24GB VRAM, lower cost
2. **50GB Volume**: Sufficient for testing
3. **On-demand pricing**: More reliable for short sessions

## 🚀 Next Steps

Once deployed, you can:

1. **Train multiple children**: Create different character LoRAs
2. **Experiment with hyperparameters**: Optimize training quality
3. **Scale up**: Add more workers for parallel training
4. **Integrate**: Connect to your applications via the API

The A40's 48GB VRAM will allow you to train high-quality LoRAs with larger batch sizes and better convergence than smaller GPUs!

## 📚 API Reference

Full API documentation available at:
`https://your-pod-8000.proxy.runpod.net/docs`

All endpoints from the main application are available, including:
- Child management (`/lora/children/*`)
- Image upload (`/lora/children/{id}/images`)
- Training control (`/lora/children/{id}/train`)  
- Generation with LoRAs (`/generate/with-children`)
- System statistics (`/lora/system/statistics`)