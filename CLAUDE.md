# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a professionally structured Docker-based Stable Diffusion 3.5 Large image generation API service. The project follows modern Python packaging standards with Poetry dependency management:

- **src/sd3_api/**: Main package with modular architecture
- **main.py**: Entry point for the server
- **pyproject.toml**: Poetry configuration with dependencies and dev tools
- **poetry.lock**: Locked dependencies for reproducible builds
- **setup_huggingface.py**: HuggingFace authentication setup script
- **setup_env.sh**: Automated Poetry environment setup
- **Dockerfile**: Container configuration using Poetry for dependency management
- **handler.py**: RunPod serverless deployment handler

## Architecture

The application uses a modular FastAPI architecture:
- **api.py**: FastAPI application and endpoint definitions
- **pipeline.py**: SD3 pipeline management with device detection and HuggingFace auth
- **device.py**: Multi-platform device detection (CUDA/MPS/CPU)
- **models.py**: Pydantic models for request/response validation
- **config.py**: Configuration constants and settings
- **Asynchronous model loading**: Downloads and loads SD3.5 Large in background at server startup
- **Real-time progress**: Health endpoint shows loading status with live updates
- **HuggingFace authentication**: Automatic token handling for gated models
- **Multi-platform GPU support**: CUDA (NVIDIA) or MPS (Apple Silicon) with CPU fallback
- **Model**: `stabilityai/stable-diffusion-3.5-large` with automatic precision handling
- **RunPod integration**: Serverless deployment support via handler.py

## Common Commands

### Development Setup
```bash
# Automatic setup (recommended)
./setup_env.sh

# Set up HuggingFace authentication for SD3.5 Large
poetry run python setup_huggingface.py

# Manual setup
poetry install --only main
poetry install --with dev  # for development tools

# Run the server (loads model asynchronously at startup)
poetry run python main.py

# Alternative: Direct uvicorn
poetry run uvicorn src.sd3_api.api:app --host 0.0.0.0 --port 8000
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

### RunPod Deployment
```bash
# Deploy to RunPod Serverless
# The Dockerfile is configured to run handler.py by default for serverless deployment
# Requires HUGGINGFACE_TOKEN environment variable in RunPod settings
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
- **Asynchronous loading**: Model loads in background at server startup, not on first request
- **Loading progress**: Health endpoint shows real-time loading status
- **HuggingFace auth**: Automatic token handling for gated models
- **Comprehensive validation**: Pydantic request/response models with prompt, steps, guidance parameters
- **Proper error handling**: HTTP status codes with meaningful messages
- **Dual endpoints**: Both GET and POST for `/generate`
- **OpenAPI docs**: Available at `/docs` with interactive testing
- **RunPod serverless**: Compatible with RunPod serverless deployment

### Project Structure
```
src/sd3_api/
├── __init__.py          # Package initialization
├── api.py               # FastAPI application and endpoints
├── config.py            # Configuration settings
├── device.py            # Device detection utilities  
├── models.py            # Pydantic request/response models
└── pipeline.py          # SD3 pipeline management

Additional files:
├── handler.py           # RunPod serverless handler
├── main.py              # FastAPI server entry point
├── setup_huggingface.py # HuggingFace authentication setup
└── setup_env.sh         # Environment setup script
```

### Configuration
All configuration is centralized in `config.py` including model settings, API defaults, and server configuration. Device-specific optimizations are handled automatically.

### Authentication Requirements
- **SD3.5 Large**: Requires HuggingFace account + token + model access approval
- **Setup script**: `poetry run python setup_huggingface.py` guides through process
- **Token storage**: Saved in `.env` file (gitignored) for local development
- **Docker deployment**: Pass token via `HUGGINGFACE_TOKEN` environment variable

### Loading Process
1. **Server startup**: FastAPI starts in ~3 seconds
2. **Background loading**: Model downloads (~8GB) and loads asynchronously in background task
3. **Progress monitoring**: Check health endpoint for real-time status via `pipeline.status` property
4. **Ready state**: API accepts generation requests once loading completes

### API Request/Response Format
- **GenerateRequest**: `prompt` (required), `steps` (1-150, default 15), `guidance` (1.0-15.0, default 7.5)
- **GenerateResponse**: Base64-encoded PNG image
- **HealthResponse**: Server status and device information
- **ErrorResponse**: Error messages with appropriate HTTP status codes

Note: The current API does not support `width`, `height`, or `seed` parameters.