# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a professionally structured Docker-based Stable Diffusion XL (SDXL) image generation API service with LoRA training capabilities. The project follows modern Python packaging standards with Poetry dependency management:

- **src/sd3_api/**: Main package with modular architecture (renamed for compatibility)
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
- **pipeline.py**: SDXL pipeline management with device detection and LoRA support
- **device.py**: Multi-platform device detection (CUDA/MPS/CPU)
- **models.py**: Pydantic models for request/response validation with width/height support
- **config.py**: Configuration constants and settings
- **lora_trainer.py**: LoRA training functionality for personalization
- **Asynchronous model loading**: Downloads and loads SDXL models in background at server startup
- **Real-time progress**: Health endpoint shows loading status with live updates
- **HuggingFace authentication**: Automatic token handling (no special access required for SDXL)
- **Multi-platform GPU support**: CUDA (NVIDIA) or MPS (Apple Silicon) with CPU fallback
- **Models**: `stabilityai/stable-diffusion-xl-base-1.0` with optional refiner
- **LoRA training**: Support for personalized image generation with custom training
- **RunPod integration**: Optimized for RunPod GPU deployment

## Common Commands

### Development Setup
```bash
# Automatic setup (recommended)
./setup_env.sh

# Set up HuggingFace authentication for SDXL (optional, public models)
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
docker build -t sdxl-api .

# Run with GPU (NVIDIA) - HuggingFace token recommended but not required
docker run --gpus all -e HUGGINGFACE_TOKEN=your_token -p 8000:8000 sdxl-api

# Run with Apple Silicon MPS or CPU fallback
docker run -e HUGGINGFACE_TOKEN=your_token -p 8000:8000 sdxl-api

# Without token (will work for SDXL as models are public)
docker run -p 8000:8000 sdxl-api
```

### RunPod Deployment
```bash
# Setup RunPod environment (run once)
python setup_runpod.py

# Start server on RunPod
./start_runpod.sh

# Stop server
./stop_runpod.sh

# Deploy to RunPod Serverless
# The Dockerfile is configured to run handler.py by default for serverless deployment
# Requires HUGGINGFACE_TOKEN environment variable in RunPod settings
```

### Deployment Workflow (IMPORTANT)
**The application runs on RunPod, NOT locally. Always follow this workflow:**

1. **Local Development**: Make code changes locally
2. **Commit Changes**: `git add . && git commit -m "description"`  
3. **Push to Remote**: `git push` (if working on shared branch)
4. **RunPod Deployment**: 
   - SSH/connect to RunPod instance
   - `git pull` to get latest changes
   - `./stop_runpod.sh && ./start_runpod.sh` to restart server
5. **Test**: Use RunPod URL (e.g., `https://xyz-8000.proxy.runpod.net/`) for testing

**Never run deployment scripts locally** - they are meant for RunPod environment.

## Debugging Deployment Issues (Reference)

### Common Deployment Problems & Solutions

When code changes don't take effect after `git pull && restart`:

#### 1. **Verify Code Actually Updated**
```bash
# Check git status and recent commits
git status
git log --oneline -3

# Verify specific fixes are present in files
grep -n "specific_change_text" path/to/file.py

# Check for any uncommitted local changes
git diff HEAD
```

#### 2. **Server Process Issues**
```bash
# Check what processes are actually running
ps aux | grep python
ps aux | grep uvicorn

# Verify server is running from correct directory
ls -la /proc/$(pgrep -f main.py)/cwd

# Check server startup logs for errors
tail -f /workspace/logs/sd3_server.log
```

#### 3. **Python Cache Problems**
```bash
# Clear Python bytecode cache (most common issue)
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Clear system-wide Python cache if needed
find /workspace -name "*.pyc" -delete 2>/dev/null
find /root -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Clear Poetry cache
poetry cache clear --all pypi
```

#### 4. **Import Path Conflicts**
```bash
# Check Python import paths
python -c "import sys; print('\n'.join(sys.path))"

# Verify module location
python -c "import src.sd3_api.module; print(module.__file__)"

# Check for multiple code copies
find /workspace -name "target_file.py" -type f
```

#### 5. **Script Path Issues**
```bash
# Verify working directory in scripts
pwd
ls -la  # Should show main.py, src/, pyproject.toml

# Check if scripts reference correct paths
grep -n "cd " *.sh
```

#### 6. **Environment & Dependencies**
```bash
# Check Poetry environment
poetry env info

# Verify dependencies are current
poetry show --outdated

# Check for conflicting virtual environments
which python
```

### Systematic Debugging Workflow

1. **Pre-flight Checks**
   - `git status` - Verify clean working directory
   - `git log --oneline -3` - Confirm latest commits present
   - `grep -n "fix_text" file.py` - Verify specific changes exist

2. **Process Management**
   - `./stop_runpod.sh` - Clean shutdown
   - `ps aux | grep python` - Verify all processes stopped
   - Clear Python cache (see commands above)
   - `./start_runpod.sh` - Restart with fresh cache

3. **Validation**
   - Check startup logs for expected behavior changes
   - Test API endpoints to confirm fixes work
   - Monitor logs during testing for error patterns

4. **Escalation Steps** (if basic restart fails)
   - Force kill all Python processes: `pkill -9 -f python`
   - System-wide cache clear (see commands above)
   - Manual server start: `poetry run python main.py`
   - Check for process conflicts or port binding issues

### Error Pattern Reference

| Error Pattern | Likely Cause | Solution |
|---------------|--------------|----------|
| Same errors after git pull + restart | Python cache not cleared | Clear .pyc files and __pycache__ |
| "Module not found" errors | Import path issues | Check PYTHONPATH and working directory |
| "Address already in use" | Previous process not killed | pkill -f python, check port 8000 |
| Old code behavior persists | Multiple code copies | find duplicate files, verify import paths |
| Dependencies missing | Poetry env issues | poetry install, check virtual env |

### LoRA Training Specific Issues

| Error | Root Cause | Fix Applied |
|-------|------------|-------------|
| `Object of type set is not JSON serializable` | target_modules set → JSON | Convert to list before JSON.dump |
| `pytorch_lora_weights.bin not found` | Training simulation, no real files | Create placeholder weight files |
| GPU offloading warning | enable_model_cpu_offload + .to(device) | Use either offloading OR manual placement |

### API Usage
```bash
# Health check
curl "http://localhost:8000/"

# Generate image (GET) with dimensions
curl "http://localhost:8000/generate?prompt=a%20beautiful%20sunset&steps=20&guidance=7.5&width=1024&height=1024"

# Generate image (POST) with LoRA
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a portrait of person123", "steps": 25, "guidance": 7.5, "width": 768, "height": 1024, "person_id": "person123"}'
```

## Key Technical Details

### Device Support
- Automatic detection: CUDA → MPS → CPU fallback
- Smart dtype handling: float16 for GPU, float32 for CPU
- Different model variants loaded based on device capabilities

### API Features
- **Asynchronous loading**: SDXL model loads in background at server startup
- **Loading progress**: Health endpoint shows real-time loading status
- **HuggingFace auth**: Automatic token handling (SDXL models are public)
- **Comprehensive validation**: Pydantic request/response models with prompt, steps, guidance, width, height
- **LoRA support**: Train and use personalized models via `/train-lora` and `person_id` parameter
- **Flexible dimensions**: Support for custom width/height (512-2048, divisible by 8)
- **Proper error handling**: HTTP status codes with meaningful messages
- **Dual endpoints**: Both GET and POST for `/generate`
- **OpenAPI docs**: Available at `/docs` with interactive testing
- **RunPod optimized**: Memory-efficient settings for GPU deployment

### Project Structure
```
src/sd3_api/
├── __init__.py          # Package initialization
├── api.py               # FastAPI application and endpoints
├── config.py            # Configuration settings
├── device.py            # Device detection utilities  
├── models.py            # Pydantic request/response models
├── pipeline.py          # SDXL pipeline management
├── lora_trainer.py      # LoRA training functionality
└── image_manager.py     # Image storage and management

Setup & deployment files:
├── main.py              # FastAPI server entry point
├── handler.py           # RunPod serverless handler
├── setup_env.sh         # Local development setup
├── setup_huggingface.py # HuggingFace authentication setup
├── setup_runpod.py      # RunPod environment setup
├── start_runpod.sh      # RunPod server startup script
└── stop_runpod.sh       # RunPod server shutdown script
```

### Configuration
All configuration is centralized in `config.py` including model settings, API defaults, and server configuration. Device-specific optimizations are handled automatically.

### Authentication Requirements
- **SDXL Models**: Public models, no special access required
- **Setup script**: `poetry run python setup_huggingface.py` for better download speeds
- **Token storage**: Saved in `.env` file (gitignored) for local development
- **Docker deployment**: Pass token via `HUGGINGFACE_TOKEN` environment variable (optional)

### Loading Process
1. **Server startup**: FastAPI starts in ~3 seconds
2. **Background loading**: SDXL model downloads (~6.5GB) and loads asynchronously
3. **Progress monitoring**: Check health endpoint for real-time status via `pipeline.status` property
4. **Ready state**: API accepts generation requests once loading completes

### API Request/Response Format
- **GenerateRequest**: `prompt` (required), `steps` (1-150, default 20), `guidance` (1.0-15.0, default 7.5), `width` (512-2048, default 1024), `height` (512-2048, default 1024), `person_id` (optional)
- **GenerateResponse**: Base64-encoded PNG image
- **HealthResponse**: Server status and device information
- **ErrorResponse**: Error messages with appropriate HTTP status codes

### LoRA Training API
- **POST /train-lora**: Train personalized model with 5-20 images
- **GET /lora**: List available trained models
- **DELETE /lora/{person_id}**: Remove trained model