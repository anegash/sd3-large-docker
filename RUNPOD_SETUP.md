# RunPod Setup Guide

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anegash/sd3-large-docker.git /workspace/sdxl-api
   cd /workspace/sdxl-api
   ```

2. **Run setup** (one command does everything):
   ```bash
   python3 setup_runpod.py
   ```

3. **Start the server**:
   ```bash
   ./start_runpod.sh
   ```

## What This Sets Up

- ✅ **Poetry environment** with all dependencies
- ✅ **HuggingFace authentication** (uses HF_TOKEN from RunPod environment)
- ✅ **Persistent storage** for models and LoRA weights
- ✅ **Background server** at http://localhost:8000

## Server Management

```bash
# Start server
./start_runpod.sh

# Stop server  
./stop_runpod.sh

# Check health
curl http://localhost:8000/

# View logs
tail -f /workspace/logs/sd3_server.log
```

## API Usage

```bash
# Generate image
curl "http://localhost:8000/generate?prompt=a%20beautiful%20sunset&steps=20"

# Train LoRA (upload images first via /upload-images endpoint)
curl -X POST "http://localhost:8000/train-lora" \
  -H "Content-Type: application/json" \
  -d '{"person_id": "test_person"}'
```

## Requirements

- Set `HF_TOKEN` in RunPod environment variables
- Persistent storage recommended (50GB+)