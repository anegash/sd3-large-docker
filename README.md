# SD3 Large API

A professional FastAPI service for generating images using Stable Diffusion 3.5 Large with multi-platform GPU support.

## Features

- 🚀 **Multi-platform GPU support**: Automatic detection and optimization for CUDA (NVIDIA), MPS (Apple Silicon), and CPU
- 🎨 **Stable Diffusion 3.5 Large**: State-of-the-art image generation using the latest SD3.5 model
- 🔑 **HuggingFace integration**: Automated authentication handling for gated models
- 🔧 **Professional architecture**: Modular design with proper separation of concerns
- 📊 **Type safety**: Full type hints and Pydantic validation
- 🐳 **Docker ready**: Containerized deployment with GPU support
- 📚 **API documentation**: Automatic OpenAPI/Swagger documentation
- ⚡ **Performance optimized**: Smart device detection and precision handling

## Prerequisites

### HuggingFace Access

SD3.5 Large is a gated model requiring HuggingFace authentication:

1. **Create account**: [Sign up at HuggingFace](https://huggingface.co/join)
2. **Request access**: Visit [SD3.5 Large model page](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) and request access
3. **Get token**: Create an access token at [HuggingFace Settings](https://huggingface.co/settings/tokens)

The setup script will guide you through this process.

## Quick Start

### Installation

#### Option 1: Automatic Setup with Poetry (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd sd3-large-docker

# Run the setup script (installs Poetry if needed, creates environment, installs dependencies)
./setup_env.sh

# Set up HuggingFace authentication for SD3.5 Large access
poetry run python setup_huggingface.py
```

#### Option 2: Manual Setup with Poetry

```bash
# Clone the repository
git clone <your-repo-url>
cd sd3-large-docker

# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies (Poetry automatically creates and manages virtual environment)
poetry install --only main

# Set up HuggingFace authentication
poetry run python setup_huggingface.py

# Install with development dependencies (optional)
poetry install --with dev
```

#### Option 3: Legacy pip Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd sd3-large-docker

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Running the Server

#### With Poetry (Recommended)

```bash
# Start the API server
poetry run python main.py

# Alternative: Activate Poetry shell and run normally
poetry shell
python main.py

# The API will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

#### With pip/venv

```bash
# Activate your virtual environment first
source venv/bin/activate

# Start the API server
python main.py
```

### Using Docker

The Dockerfile now uses Poetry for dependency management:

```bash
# Build the container (now uses Poetry internally)
docker build -t sd3-large-api .

# Run with GPU support (NVIDIA)
docker run --gpus all -p 8000:8000 sd3-large-api

# Run with CPU/MPS fallback
docker run -p 8000:8000 sd3-large-api
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/
```

### Generate Image (GET)

```bash
curl "http://localhost:8000/generate?prompt=a%20beautiful%20sunset&steps=20&guidance=7.5"
```

### Generate Image (POST)

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over the ocean",
    "steps": 20,
    "guidance": 7.5
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and device info |
| `/generate` | GET/POST | Generate image from text prompt |
| `/docs` | GET | Interactive API documentation |

### Parameters

- **prompt** (required): Text description of the image to generate
- **steps** (optional): Number of inference steps (1-150, default: 15)
- **guidance** (optional): Guidance scale (1.0-15.0, default: 7.5)

### Response

```json
{
  "image": "base64-encoded-png-data"
}
```

## Project Structure

```
sd3-large-docker/
├── src/sd3_api/
│   ├── __init__.py          # Package initialization
│   ├── api.py               # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── device.py            # Device detection utilities
│   ├── models.py            # Pydantic request/response models
│   └── pipeline.py          # SD3 pipeline management
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
├── pyproject.toml          # Modern Python packaging
├── Dockerfile              # Container configuration
├── CLAUDE.md               # Development guidance
└── README.md               # This file
```

## Development

### Setup Development Environment

#### With Poetry (Recommended)

```bash
# Install with development dependencies
poetry install --with dev
```

#### With pip/venv

```bash
# Install with development dependencies  
pip install -e ".[dev]"
```

### Code Quality Tools

#### With Poetry (Recommended)

```bash
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

#### With pip/venv

```bash
# Format code
black src/ main.py

# Sort imports
isort src/ main.py

# Lint code
flake8 src/ main.py

# Type checking
mypy src/ main.py
```

### Device Support

The application automatically detects and uses the best available hardware:

1. **NVIDIA GPU (CUDA)**: Uses float16 precision with fp16 model variant
2. **Apple Silicon (MPS)**: Uses float16 precision with fp16 model variant  
3. **CPU**: Uses float32 precision with standard model variant

## Configuration

Key settings in `src/sd3_api/config.py`:

```python
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
DEFAULT_STEPS = 15
DEFAULT_GUIDANCE = 7.5
HOST = "0.0.0.0" 
PORT = 8000
```

## Requirements

### System Requirements

- Python 3.10+
- 8GB+ RAM (16GB+ recommended)
- GPU with 8GB+ VRAM (optional but recommended)

### GPU Requirements

- **NVIDIA**: CUDA-compatible GPU with 8GB+ VRAM
- **Apple Silicon**: M1/M2/M3 Mac with 16GB+ unified memory
- **CPU**: Any modern CPU (slow but functional)

## Docker Deployment

The included Dockerfile uses the RunPod PyTorch base image with CUDA support:

```bash
# Build
docker build -t sd3-large-api .

# Run with all GPUs
docker run --gpus all -p 8000:8000 sd3-large-api

# Run with specific GPU
docker run --gpus device=0 -p 8000:8000 sd3-large-api

# Run CPU-only
docker run -p 8000:8000 sd3-large-api
```

## Performance Notes

- **Server startup**: Server starts in ~3 seconds, model loads in background (30-60 seconds)
- **Model loading**: Check `/` endpoint to monitor loading progress
- **Image generation**: ~2-10 seconds per image depending on hardware
- **Memory usage**: ~8-12GB VRAM/RAM for model storage
- **Steps vs Quality**: More steps = higher quality but slower generation

### Loading States

The server will show different statuses during startup:

1. **"Initializing..."** - Server just started, model loading beginning
2. **"Loading model..."** - Model actively downloading and loading (~8GB download)
3. **"Ready on [device]"** - Model loaded and ready for image generation
4. **"Error: [message]"** - Model failed to load (check logs for details)

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the code quality tools
5. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check the API documentation at `/docs`
- Review the `CLAUDE.md` file for development guidance