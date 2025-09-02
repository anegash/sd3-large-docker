# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a professionally structured Docker-based Stable Diffusion 3.5 Large image generation API service. The project follows modern Python packaging standards with:

- **src/sd3_api/**: Main package with modular architecture
- **main.py**: Entry point for the server
- **pyproject.toml**: Modern Python packaging configuration
- **requirements.txt**: Dependencies specification
- **Dockerfile**: Container configuration using RunPod PyTorch base image

## Architecture

The application uses a modular FastAPI architecture:
- **api.py**: FastAPI application and endpoint definitions
- **pipeline.py**: SD3 pipeline management with device detection
- **device.py**: Multi-platform device detection (CUDA/MPS/CPU)
- **models.py**: Pydantic models for request/response validation
- **config.py**: Configuration constants and settings
- Model loading happens at startup with automatic device optimization
- Supports both GET and POST endpoints for image generation
- GPU acceleration via CUDA (NVIDIA) or MPS (Apple Silicon) with CPU fallback
- Model: `stabilityai/stable-diffusion-3.5-large` with automatic precision handling

## Common Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run the server
python main.py

# Alternative: Run with uvicorn directly
uvicorn src.sd3_api.api:app --host 0.0.0.0 --port 8000
```

### Code Quality
```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black src/ main.py

# Sort imports
isort src/ main.py

# Lint code
flake8 src/ main.py

# Type checking
mypy src/ main.py
```

### Docker
```bash
# Build the container
docker build -t sd3-large-api .

# Run with GPU (NVIDIA)
docker run --gpus all -p 8000:8000 sd3-large-api

# Run with Apple Silicon MPS or CPU fallback
docker run -p 8000:8000 sd3-large-api
```

### API Usage
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

## Key Technical Details

### Device Support
- Automatic detection: CUDA → MPS → CPU fallback
- Smart dtype handling: float16 for GPU, float32 for CPU
- Different model variants loaded based on device capabilities

### API Features
- Comprehensive request/response validation with Pydantic
- Proper HTTP status codes and error handling
- Both GET and POST endpoints for flexibility
- OpenAPI documentation available at `/docs`

### Project Structure
```
src/sd3_api/
├── __init__.py          # Package initialization
├── api.py               # FastAPI application
├── config.py            # Configuration settings
├── device.py            # Device detection utilities
├── models.py            # Pydantic request/response models
└── pipeline.py          # SD3 pipeline management
```

### Configuration
All configuration is centralized in `config.py` including model settings, API defaults, and server configuration. Device-specific optimizations are handled automatically.