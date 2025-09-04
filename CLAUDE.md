# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a professionally structured Docker-based Stable Diffusion 3.5 Large image generation API service with **LoRA (Low-Rank Adaptation) training capabilities**. The project follows modern Python packaging standards with Poetry dependency management:

- **src/sd3_api/**: Main package with modular architecture including LoRA training
- **main.py**: Entry point for the server
- **pyproject.toml**: Poetry configuration with dependencies and dev tools
- **poetry.lock**: Locked dependencies for reproducible builds
- **setup_huggingface.py**: HuggingFace authentication setup script
- **setup_env.sh**: Basic Poetry environment setup
- **setup_lora_env.sh**: Complete LoRA training environment setup
- **Dockerfile**: Container configuration using Poetry for dependency management

## Architecture

The application uses a modular FastAPI architecture with LoRA training capabilities:

### Core Components
- **api.py**: Main FastAPI application with base endpoints
- **lora_api.py**: LoRA-specific API endpoints for training and child management
- **pipeline.py**: Base SD3 pipeline management
- **lora/pipeline.py**: Extended pipeline with multi-LoRA adapter support
- **device.py**: Multi-platform device detection (CUDA/MPS/CPU)
- **models.py**: Pydantic models for base API
- **lora_models.py**: Pydantic models for LoRA API
- **config.py**: Configuration constants and settings

### LoRA Training System
- **lora/trainer.py**: Core LoRA training logic with SD3.5 Large
- **tasks/**: Celery-based background job system for async training
- **database/**: SQLAlchemy models and management for training data
- **utils/**: File upload, storage, and validation utilities

### Key Features
- **Eager model loading**: Downloads and loads SD3.5 Large at server startup
- **Real-time progress**: Health endpoint shows loading status with live updates
- **HuggingFace authentication**: Automatic token handling for gated models
- **Multi-platform GPU support**: CUDA (NVIDIA) or MPS (Apple Silicon) with CPU fallback
- **LoRA training**: Custom child training with async background jobs
- **Multi-child generation**: Generate images with multiple children in single image
- **Token replacement**: Smart prompt processing with `{child_id}` tokens
- **Model**: `stabilityai/stable-diffusion-3.5-large` with automatic precision handling

## Common Commands

### Development Setup
```bash
# Complete LoRA training setup (recommended)
./setup_lora_env.sh

# Basic setup (original functionality only)
./setup_env.sh

# Set up HuggingFace authentication for SD3.5 Large
poetry run python setup_huggingface.py

# Manual setup
poetry install --only main
poetry install --with dev  # for development tools

# Run the server (eager loads model at startup)
poetry run python main.py

# Alternative: Direct uvicorn
poetry run uvicorn src.sd3_api.api:app --host 0.0.0.0 --port 8000

# Start Celery worker for LoRA training (required for training)
poetry run celery -A src.sd3_api.tasks.celery_app worker --loglevel=info -Q training
```

### Code Quality
```bash
# Install development dependencies
poetry install --with dev

# Format code
poetry run black src/ main.py

# Sort imports
poetry run isort src/ main.py

# Lint code
poetry run flake8 src/ main.py

# Type checking
poetry run mypy src/ main.py

# Run all quality checks
poetry run black src/ main.py && poetry run isort src/ main.py && poetry run flake8 src/ main.py
```

### Docker
```bash
# Build the container (uses Poetry internally)
docker build -t sd3-large-api .

# Run with GPU (NVIDIA) - requires HuggingFace token as env var
docker run --gpus all -e HUGGINGFACE_TOKEN=your_token -p 8000:8000 sd3-large-api

# Run with Apple Silicon MPS or CPU fallback
docker run -e HUGGINGFACE_TOKEN=your_token -p 8000:8000 sd3-large-api

# Without token (will fail for SD3.5 Large)
docker run -p 8000:8000 sd3-large-api
```

### API Usage

#### Basic Image Generation
```bash
# Health check
curl "http://localhost:8000/"

# Generate image (GET)
curl "http://localhost:8000/generate?prompt=a%20beautiful%20sunset&steps=20&guidance=7.5"

# Generate image (POST)
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset", "steps": 20, "guidance": 7.5}'
```

#### LoRA Training and Child Management
```bash
# Create a new child
curl -X POST "http://localhost:8000/lora/children" \
  -H "Content-Type: application/json" \
  -d '{"id": "child_001", "name": "Alice", "description": "5-year-old girl"}'

# Upload training images (multipart form)
curl -X POST "http://localhost:8000/lora/children/child_001/images" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"

# Start LoRA training
curl -X POST "http://localhost:8000/lora/children/child_001/train" \
  -H "Content-Type: application/json" \
  -d '{"training_config": {"training_steps": 1000, "lora_rank": 64}}'

# Check training status
curl "http://localhost:8000/lora/children/child_001/training-status"

# Generate image with trained child
curl -X POST "http://localhost:8000/generate/with-children" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a happy {child_001} playing in the park", "steps": 20, "guidance": 7.5}'

# Generate image with multiple children
curl -X POST "http://localhost:8000/generate/with-children" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "{child_001} and {child_002} having a picnic", "steps": 20}'
```

## Key Technical Details

### Device Support
- Automatic detection: CUDA → MPS → CPU fallback
- Smart dtype handling: float16 for GPU, float32 for CPU
- Different model variants loaded based on device capabilities

### API Features
- **Eager loading**: Model loads at server startup, not on first request
- **Loading progress**: Health endpoint shows real-time loading status
- **HuggingFace auth**: Automatic token handling for gated models
- **Comprehensive validation**: Pydantic request/response models
- **Proper error handling**: HTTP status codes with meaningful messages
- **Dual endpoints**: Both GET and POST for `/generate`
- **OpenAPI docs**: Available at `/docs` with interactive testing

### Project Structure
```
src/sd3_api/
├── __init__.py          # Package initialization
├── api.py               # Main FastAPI application with base endpoints
├── lora_api.py          # LoRA-specific API endpoints
├── config.py            # Configuration settings
├── device.py            # Device detection utilities
├── models.py            # Base Pydantic request/response models
├── lora_models.py       # LoRA-specific Pydantic models
├── pipeline.py          # Base SD3 pipeline management
├── database/            # Database layer
│   ├── __init__.py
│   ├── models.py        # SQLAlchemy database models
│   ├── connection.py    # Database connection management
│   └── manager.py       # CRUD operations and database manager
├── lora/                # LoRA training system
│   ├── __init__.py
│   ├── trainer.py       # Core LoRA training logic
│   └── pipeline.py      # Extended pipeline with LoRA support
├── tasks/               # Background job system
│   ├── __init__.py
│   ├── celery_app.py    # Celery configuration
│   └── training_tasks.py # Celery tasks for training
└── utils/               # Utility modules
    ├── __init__.py
    ├── storage.py       # File storage management
    ├── upload.py        # File upload handling
    └── validation.py    # Input validation utilities
```

### API Endpoints

#### Base Endpoints
- `GET /` - Health check and device info
- `GET/POST /generate` - Basic image generation
- `POST /generate/with-children` - Enhanced generation with LoRA

#### Child Management (`/lora/children`)
- `POST /lora/children` - Create a new child
- `GET /lora/children` - List all children
- `GET /lora/children/{child_id}` - Get child details
- `PUT /lora/children/{child_id}` - Update child information
- `DELETE /lora/children/{child_id}` - Delete child and all data

#### Training Images (`/lora/children/{child_id}/images`)
- `POST /lora/children/{child_id}/images` - Upload training images (batch)
- `GET /lora/children/{child_id}/images` - List training images
- `DELETE /training-images/{image_id}` - Delete specific image

#### Training Management
- `POST /lora/children/{child_id}/train` - Start LoRA training
- `GET /lora/children/{child_id}/training-status` - Get training status
- `GET /training-tasks/{task_id}/status` - Get Celery task status

#### Model Management
- `GET /lora/children/{child_id}/models` - List all models for child

#### Statistics and Monitoring
- `GET /lora/children/{child_id}/statistics` - Child-specific statistics
- `GET /lora/system/statistics` - Overall system statistics

### Configuration
All configuration is centralized in `config.py` including model settings, API defaults, and server configuration. Device-specific optimizations are handled automatically.

### Authentication Requirements
- **SD3.5 Large**: Requires HuggingFace account + token + model access approval
- **Setup script**: `poetry run python setup_huggingface.py` guides through process
- **Token storage**: Saved in `.env` file (gitignored) for local development
- **Docker deployment**: Pass token via `HUGGINGFACE_TOKEN` environment variable

### Loading Process
1. **Server startup**: FastAPI starts in ~3 seconds
2. **Background loading**: Model downloads (~8GB) and loads automatically
3. **Progress monitoring**: Check health endpoint for real-time status
4. **Ready state**: API accepts generation requests once loading completes
- https://i2rnjbyutc996h-8000.proxy.runpod.net/ we deployed the app on runpod with A40 GPU