# SD3.5 Large LoRA Training System - RunPod Deployment Guide

Complete guide for deploying the SD3.5 Large LoRA training system on RunPod GPU instances.

## 🚀 Quick Deploy

### 1. Build and Push Container
```bash
# Build the optimized RunPod container
docker build -f Dockerfile.dummy -t yourusername/sd3-lora:latest .

# Push to Docker Hub
docker push yourusername/sd3-lora:latest
```

### 2. Deploy on RunPod
1. Go to [RunPod.io](https://runpod.io) and create account
2. Navigate to **Pods** → **Deploy**
3. **Container Image**: `yourusername/sd3-lora:latest`
4. **GPU**: Select A40, RTX 4090, or better (≥24GB VRAM recommended)
5. **Volume**: Network Volume ≥100GB for model storage
6. **Environment Variables**: Set `HF_TOKEN=your_huggingface_token`
7. **Ports**: 22, 8000, 6006, 8080 (auto-configured)
8. Click **Deploy**

### 3. Connect and Start
```bash
# SSH into your pod (get details from RunPod dashboard)
ssh root@your-pod-id.runpod.io
# Password: runpod

# Pull latest code and start
git pull origin live-debug-runpod
./start_app.sh
```

## 📋 System Requirements

### Minimum GPU Requirements
- **VRAM**: 24GB+ (for SD3.5 Large)
- **Recommended**: NVIDIA A40, A100, RTX 4090, RTX 6000 Ada
- **Budget Option**: RTX 3090 (24GB) works but slower

### Storage Requirements
- **Network Volume**: 100GB+ recommended
- **Model Cache**: ~30GB for SD3.5 Large + dependencies
- **Training Data**: Variable based on your datasets
- **Generated Images**: Plan for 1-10GB depending on usage

## 🛠️ Container Features

### Pre-installed Components
- **CUDA 12.1** with PyTorch 2.2.1
- **Python 3.10** with Poetry package manager
- **Redis Server** for Celery task queue
- **SSH Server** for remote access
- **All dependencies** pre-installed and cached

### Directory Structure
```
/workspace/
├── sd3-large-docker/          # Main project code
├── huggingface_cache/         # Model cache (persistent)
├── poetry_cache/              # Poetry cache (persistent)
├── poetry_venvs/              # Virtual environments
├── training_data/             # Your LoRA training images
├── models/                    # Trained LoRA adapters
├── generated_images/          # Output images
└── logs/                      # System logs
```

### Environment Variables
- `HF_TOKEN`: HuggingFace access token (required for SD3.5 Large)
- `POETRY_CACHE_DIR`: `/workspace/poetry_cache`
- `HF_HOME`: `/workspace/huggingface_cache`
- `CUDA_VISIBLE_DEVICES`: Auto-detected

## 🔧 Development Workflow

### Starting the System
```bash
# Method 1: Use convenience script (recommended)
./start_app.sh

# Method 2: Manual startup
poetry run python main.py
```

### Starting LoRA Training Worker
```bash
# In a separate terminal/tmux session
./start_worker.sh

# Or manually
poetry run celery -A src.sd3_api.tasks.celery_app worker --loglevel=info -Q training
```

### Basic API Usage
```bash
# Health check
curl http://localhost:8000/

# Generate image
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset", "steps": 20}'

# Create child for LoRA training
curl -X POST "http://localhost:8000/lora/children" \
  -H "Content-Type: application/json" \
  -d '{"id": "child001", "name": "Test Child"}'
```

## 📊 LoRA Training Workflow

### 1. Create a Child
```bash
curl -X POST "http://localhost:8000/lora/children" \
  -H "Content-Type: application/json" \
  -d '{"id": "alice", "name": "Alice", "description": "5-year-old girl"}'
```

### 2. Upload Training Images
```bash
curl -X POST "http://localhost:8000/lora/children/alice/images" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

### 3. Start Training
```bash
curl -X POST "http://localhost:8000/lora/children/alice/train" \
  -H "Content-Type: application/json" \
  -d '{"training_config": {"training_steps": 1000, "lora_rank": 64}}'
```

### 4. Monitor Training
```bash
# Check training status
curl "http://localhost:8000/lora/children/alice/training-status"

# Watch Celery logs
tail -f /workspace/logs/celery.log
```

### 5. Generate with Trained Child
```bash
curl -X POST "http://localhost:8000/generate/with-children" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a happy {alice} playing in the park", "steps": 20}'
```

## 🔍 Monitoring and Debugging

### System Logs
```bash
# Container heartbeat
tail -f /workspace/logs/heartbeat.log

# System status
tail -f /workspace/logs/container.log

# Application logs
tail -f /workspace/logs/app.log

# Celery worker logs
tail -f /workspace/logs/celery.log
```

### Resource Monitoring
```bash
# GPU usage
nvidia-smi

# System resources
htop

# Disk usage
df -h /workspace
du -sh /workspace/*
```

### Common Debug Commands
```bash
# Check processes
ps aux | grep -E "(uvicorn|celery|redis)"

# Test Redis connection
redis-cli ping

# Check database
ls -la data/

# View API docs
# Navigate to http://your-pod-url:8000/docs
```

## 💰 Cost Optimization

### GPU Selection Strategy
- **Development/Testing**: RTX 3090 (24GB) - ~$0.34/hr
- **Production Training**: A40 (48GB) - ~$0.79/hr  
- **Heavy Workloads**: A100 (80GB) - ~$1.89/hr

### Cost-Saving Tips
1. **Use Spot Instances**: 50-90% cheaper, can be interrupted
2. **Right-size GPU**: Don't over-provision for your needs
3. **Network Volume**: Persist data between pod sessions
4. **Auto-pause**: Stop pods when not actively training
5. **Monitor Usage**: Check GPU utilization regularly

### Billing Optimization
```bash
# Check GPU utilization every 10 minutes
watch -n 600 nvidia-smi

# Auto-shutdown after idle (optional script)
# Add to crontab: check for idle and shutdown
```

## 🚨 Troubleshooting

### Container Won't Start
- **Issue**: Image pull fails
- **Fix**: Verify image name and Docker Hub access
- **Check**: RunPod dashboard logs

### SSH Connection Failed
- **Issue**: Can't connect via SSH
- **Fix**: Check RunPod dashboard for correct hostname
- **Password**: Default is `runpod`, change after first login

### Model Loading Errors
- **Issue**: "incomplete metadata" or download errors
- **Fix**: Clear model cache: `rm -rf /workspace/huggingface_cache/hub/models--stabilityai--stable-diffusion-3.5-large/`
- **Retry**: Restart with `./start_app.sh`

### Out of Memory
- **Issue**: CUDA OOM during training
- **Fix**: Reduce batch size, use gradient accumulation
- **Check**: GPU memory with `nvidia-smi`

### Training Stuck
- **Issue**: Celery worker not processing jobs  
- **Fix**: Check Redis: `redis-cli ping`
- **Restart**: `./start_worker.sh`

### Slow Performance
- **Check**: Network I/O to /workspace
- **Solution**: Use local SSD for temporary files
- **Monitor**: `iotop` for disk I/O

## 🔒 Security Notes

### Change Default Password
```bash
# Immediately after first SSH login
passwd root
```

### HuggingFace Token Security
- Set `HF_TOKEN` in RunPod environment variables (encrypted)
- Never commit tokens to git repositories
- Use read-only tokens when possible

### Network Security
- RunPod pods are isolated by default
- Only exposed ports are accessible
- Use SSH key authentication for better security

## 📚 API Documentation

### Interactive Docs
Once running, visit: `http://your-pod-url:8000/docs`

### Key Endpoints
- `GET /` - Health check and status
- `POST /generate` - Basic image generation
- `POST /generate/with-children` - Generation with LoRA adapters
- `POST /lora/children` - Create child for training
- `POST /lora/children/{id}/train` - Start training
- `GET /lora/children/{id}/training-status` - Check progress

### WebSocket Support (Future)
- Real-time training progress updates
- Live generation status
- Resource monitoring dashboard

---

## 🆘 Support

For issues specific to this deployment:
1. Check logs in `/workspace/logs/`
2. Review RunPod dashboard status
3. Verify environment variables
4. Test with minimal examples

For RunPod platform issues:
- [RunPod Discord](https://discord.gg/runpod)
- [RunPod Documentation](https://docs.runpod.io/)

---

**Ready to train LoRA adapters on SD3.5 Large!** 🚀