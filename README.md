# Stable Diffusion 3.5 Large - RunPod Hub

[![Runpod](https://api.runpod.io/badge/anegash/sd3-large-docker)](https://console.runpod.io/hub/anegash/sd3-large-docker)

Professional-grade text-to-image generation using Stability AI's Stable Diffusion 3.5 Large model, optimized for RunPod serverless deployment and local development.

## Features

### RunPod Serverless
- ⚡ **Serverless deployment**: Ready for RunPod Hub with automatic scaling
- 🔧 **Handler integration**: Pre-built RunPod serverless handler
- 📋 **Complete metadata**: Hub.json with presets and configuration
- 🧪 **Test suite**: Comprehensive test cases for validation

### Core Features  
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

## RunPod Deployment

### RunPod Pod Deployment (Persistent Storage)

For persistent LoRA training with storage that survives pod restarts:

#### Initial Setup (One-time)

1. **Create RunPod Pod**: Launch a pod with persistent storage (recommended: 50GB+ network volume)
2. **Clone Repository**: 
   ```bash
   git clone https://github.com/yourusername/sd3-large-docker.git
   cd sd3-large-docker
   ```

3. **Run Setup Script**: 
   ```bash
   python3 setup_runpod.py
   ```
   This will:
   - Install Poetry and dependencies in `/workspace/venv` 
   - Set up persistent directories in `/workspace`
   - Configure HuggingFace cache in `/workspace/huggingface_cache`
   - Create startup scripts and environment variables

4. **Set HuggingFace Token**: 
   - Either set `HUGGINGFACE_TOKEN` in RunPod environment variables, or
   - Run the setup script and enter your token when prompted

#### Starting the Service

After initial setup, start the server:
```bash
/workspace/start_sd3.sh
# OR use the alias:
sd3-run
```

The server will be available at `http://localhost:8000` with:
- **API Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/`
- **LoRA Training**: `POST /train-lora`
- **Image Generation**: `POST /generate` with `person_id` parameter

#### Persistent Storage Structure
```
/workspace/
├── sd3-project/          # Project files
├── venv/                 # Poetry virtual environment  
├── huggingface_cache/    # Model cache (8GB+ SD3.5 Large)
├── lora_weights/         # Trained LoRA weights
├── logs/                 # Server logs
└── start_sd3.sh         # Startup script
```

### Quick Deploy to RunPod Hub (Serverless)

1. **Fork/Clone Repository**: Get your own copy of this repository
2. **Configure Secrets**: Set `HUGGINGFACE_TOKEN` in RunPod environment  
3. **Build Container**: RunPod will build from the included Dockerfile
4. **Deploy**: Use the pre-configured hub.json settings

### RunPod Serverless Usage

```json
{
  "input": {
    "prompt": "a beautiful sunset over mountains",
    "steps": 20,
    "guidance": 7.5,
    "width": 1024,
    "height": 1024
  }
}
```

**Response:**
```json
{
  "image": "base64_encoded_png_data",
  "seed": 42,
  "format": "png"
}
```

### Available Presets

- **High Quality**: 30 steps, guidance 7.5 for detailed artwork
- **Fast Generation**: 20 steps, guidance 5.0 for quick results  
- **Portrait**: 768×1024 optimized for characters/faces
- **Landscape**: 1024×768 optimized for scenic views

## Local Development

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

### LoRA Training and Personalized Generation

#### Train LoRA Model

Train a personalized LoRA model using 5-20 images of a person:

```bash
curl -X POST "http://localhost:8000/train-lora" \
  -F "person_id=john_doe" \
  -F "num_train_epochs=100" \
  -F "learning_rate=0.0001" \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg" \
  -F "files=@photo3.jpg" \
  -F "files=@photo4.jpg" \
  -F "files=@photo5.jpg"
```

#### Generate with LoRA Weights

Once trained, generate images using the person's LoRA weights:

```bash
# GET method
curl "http://localhost:8000/generate?prompt=a%20photo%20of%20john_doe%20in%20a%20suit&person_id=john_doe"

# POST method
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "portrait of john_doe smiling in a garden",
    "person_id": "john_doe",
    "steps": 20,
    "guidance": 7.5
  }'
```

#### List Available LoRA Models

```bash
curl "http://localhost:8000/lora"
```

#### Delete LoRA Model

```bash
curl -X DELETE "http://localhost:8000/lora/john_doe"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and device info |
| `/generate` | GET/POST | Generate image from text prompt |
| `/train-lora` | POST | Train LoRA model with uploaded images |
| `/lora` | GET | List available LoRA person IDs |
| `/lora/{person_id}` | DELETE | Delete LoRA weights for person |
| `/docs` | GET | Interactive API documentation |

### Generation Parameters

- **prompt** (required): Text description of the image to generate
- **steps** (optional): Number of inference steps (1-150, default: 15)
- **guidance** (optional): Guidance scale (1.0-15.0, default: 7.5)
- **person_id** (optional): Person ID to load LoRA weights for

### LoRA Training Parameters

- **person_id** (required): Unique identifier for the person
- **files** (required): 5-20 image files for training
- **num_train_epochs** (optional): Training epochs (10-500, default: 100)
- **learning_rate** (optional): Learning rate (default: 0.0001)

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