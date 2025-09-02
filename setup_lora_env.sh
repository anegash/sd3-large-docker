#!/bin/bash

# LoRA Training Environment Setup Script
# This script sets up the complete environment for SD3.5 Large with LoRA training

set -e  # Exit on any error

echo "🚀 Setting up SD3.5 Large LoRA Training Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    print_status "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    
    # Check if installation was successful
    if ! command -v poetry &> /dev/null; then
        print_error "Failed to install Poetry. Please install manually."
        exit 1
    fi
    
    print_success "Poetry installed successfully"
else
    print_success "Poetry is already installed"
fi

# Install Python dependencies
print_status "Installing Python dependencies with Poetry..."
poetry install --only main
print_success "Main dependencies installed"

# Install development dependencies (optional)
if [[ "$1" == "--dev" ]]; then
    print_status "Installing development dependencies..."
    poetry install --with dev
    print_success "Development dependencies installed"
fi

# Create data directories
print_status "Creating data directories..."
mkdir -p data/training_images
mkdir -p data/lora_models
mkdir -p data/generated_images
mkdir -p data/temp
print_success "Data directories created"

# Set up database
print_status "Setting up database..."
poetry run python -c "
from src.sd3_api.database.connection import create_tables
create_tables()
print('Database tables created successfully')
"
print_success "Database setup complete"

# Check for HuggingFace token
if [[ -z "${HUGGINGFACE_TOKEN}" ]]; then
    print_warning "HUGGINGFACE_TOKEN not found in environment"
    print_status "Setting up HuggingFace authentication..."
    poetry run python setup_huggingface.py
else
    print_success "HuggingFace token found in environment"
fi

# Check for Redis (required for Celery)
if ! command -v redis-server &> /dev/null; then
    print_warning "Redis not found. Redis is required for background training tasks."
    print_status "Please install Redis:"
    echo "  Ubuntu/Debian: sudo apt-get install redis-server"
    echo "  macOS: brew install redis"
    echo "  Windows: Download from https://redis.io/download"
else
    print_success "Redis is available"
    
    # Check if Redis is running
    if ! redis-cli ping &> /dev/null; then
        print_warning "Redis is not running. Starting Redis..."
        if command -v systemctl &> /dev/null; then
            sudo systemctl start redis
        elif command -v brew &> /dev/null; then
            brew services start redis
        else
            print_warning "Please start Redis manually: redis-server"
        fi
    else
        print_success "Redis is running"
    fi
fi

# Create .env file if it doesn't exist
if [[ ! -f ".env" ]]; then
    print_status "Creating .env file..."
    cat > .env << EOL
# Database Configuration
DATABASE_URL=sqlite:///./sd3_lora.db
SQL_DEBUG=false

# Redis Configuration (for Celery)
REDIS_URL=redis://localhost:6379/0

# HuggingFace Configuration
HUGGINGFACE_TOKEN=

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Training Configuration
MAX_CONCURRENT_TRAININGS=2
DEFAULT_LORA_RANK=64
DEFAULT_TRAINING_STEPS=1000

# Storage Configuration
MAX_TRAINING_IMAGES_PER_CHILD=50
MAX_FILE_SIZE_MB=10

# Logging Configuration
LOG_LEVEL=INFO
EOL
    print_success ".env file created. Please update HUGGINGFACE_TOKEN"
else
    print_success ".env file already exists"
fi

# Create systemd service file (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]] && command -v systemctl &> /dev/null; then
    if [[ "$2" == "--service" ]]; then
        print_status "Creating systemd service file..."
        WORKING_DIR=$(pwd)
        POETRY_PATH=$(which poetry)
        
        sudo tee /etc/systemd/system/sd3-lora-api.service > /dev/null << EOL
[Unit]
Description=SD3.5 Large LoRA API Service
After=network.target redis.service

[Service]
Type=exec
User=$USER
WorkingDirectory=$WORKING_DIR
Environment=PATH=$PATH
ExecStart=$POETRY_PATH run python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL
        
        sudo systemctl daemon-reload
        print_success "Systemd service created. Enable with: sudo systemctl enable sd3-lora-api"
    fi
fi

# Test installation
print_status "Testing installation..."
if poetry run python -c "
import torch
from src.sd3_api.database.connection import create_tables
from src.sd3_api.utils.storage import storage_manager
from src.sd3_api.database.manager import db_manager

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available()}' if hasattr(torch.backends, 'mps') else 'MPS not available')

# Test database
create_tables()
print('Database connection: OK')

# Test storage
stats = storage_manager.get_storage_stats()
print(f'Storage system: OK')

print('Installation test: PASSED')
"; then
    print_success "Installation test passed"
else
    print_error "Installation test failed"
    exit 1
fi

echo ""
echo "🎉 Setup complete! Your SD3.5 Large LoRA training environment is ready."
echo ""
print_status "Next steps:"
echo "1. Update your HUGGINGFACE_TOKEN in the .env file"
echo "2. Start Redis if not already running: redis-server"
echo "3. Start the API server: poetry run python main.py"
echo "4. Start a Celery worker for training: poetry run celery -A src.sd3_api.tasks.celery_app worker --loglevel=info -Q training"
echo "5. Visit http://localhost:8000/docs for the API documentation"
echo ""
print_status "Key endpoints:"
echo "• POST /lora/children - Create a new child"
echo "• POST /lora/children/{child_id}/images - Upload training images"
echo "• POST /lora/children/{child_id}/train - Start LoRA training"
echo "• POST /generate/with-children - Generate images with LoRA"
echo ""
print_success "Happy training! 🎨"