# SDXL LoRA Training API

Professional SDXL image generation API with LoRA (Low-Rank Adaptation) training capabilities, optimized for RunPod GPU deployment.

## Features

- 🎨 **SDXL Image Generation**: High-quality text-to-image generation
- 🧠 **LoRA Training**: Train personalized models with 5-20 images
- 🚀 **RunPod Optimized**: Easy deployment on GPU pods
- 🔄 **Persistent Storage**: Models and weights survive pod restarts
- 📊 **REST API**: FastAPI with automatic documentation

## Quick Start (RunPod)

1. **Clone and setup** (one-time):
   ```bash
   git clone https://github.com/yourusername/sd3-large-docker.git /workspace/sdxl-api
   cd /workspace/sdxl-api
   python3 setup_runpod.py
   ```

2. **Start server**:
   ```bash
   ./start_runpod.sh
   ```

3. **API available at**: `http://localhost:8000`

## API Usage

### Generate Images
```bash
# Basic generation
curl "http://localhost:8000/generate?prompt=a%20beautiful%20sunset&steps=20"

# With trained LoRA
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "portrait of person123", "person_id": "person123", "steps": 25}'
```

### Train LoRA Models
```bash
# Upload training images
curl -X POST "http://localhost:8000/upload-images" \
  -F "person_id=person123" \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg" \
  -F "files=@photo3.jpg"

# Start training
curl -X POST "http://localhost:8000/train-lora" \
  -H "Content-Type: application/json" \
  -d '{"person_id": "person123"}'

# List trained models
curl "http://localhost:8000/lora"
```

## Requirements

- **GPU**: NVIDIA GPU with 12GB+ VRAM recommended
- **Storage**: 50GB+ persistent storage for RunPod
- **Token**: Set `HF_TOKEN` in RunPod environment variables

## Server Management

```bash
# Start server
./start_runpod.sh

# Stop server
./stop_runpod.sh

# Check status
curl http://localhost:8000/

# View logs
tail -f /workspace/logs/sd3_server.log
```

## Project Structure

```
src/sd3_api/
├── api.py              # FastAPI application
├── pipeline.py         # SDXL pipeline management
├── lora_trainer.py     # LoRA training functionality
├── models.py           # Request/response models
└── config.py           # Configuration

Setup files:
├── setup_runpod.py     # One-time RunPod setup
├── start_runpod.sh     # Server startup
└── stop_runpod.sh      # Server shutdown
```

## Development

```bash
# Local setup
poetry install --with dev

# Code formatting
poetry run black src/ main.py
poetry run isort src/ main.py

# Run locally
poetry run python main.py
```