"""Configuration settings for the SDXL API."""

import os
from pathlib import Path
from typing import Literal

# Model configuration
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_MODEL_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
MODEL_VARIANT = "fp16"
TORCH_DTYPE = "float16"

# API configuration
DEFAULT_STEPS = 20
MAX_STEPS = 150
DEFAULT_GUIDANCE = 7.5
MAX_GUIDANCE = 15.0

# Server configuration
HOST = "0.0.0.0"
PORT = 8000

# Device configuration
DeviceType = Literal["cuda", "mps", "cpu"]

# RunPod persistent storage configuration
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
LORA_WEIGHTS_DIR = WORKSPACE_DIR / "lora_weights" 
MODELS_CACHE_DIR = WORKSPACE_DIR / "models"
HUGGINGFACE_CACHE_DIR = WORKSPACE_DIR / "huggingface_cache"
LOGS_DIR = WORKSPACE_DIR / "logs"
VENV_DIR = WORKSPACE_DIR / "venv"

# Ensure directories exist
def ensure_workspace_dirs():
    """Ensure all workspace directories exist."""
    for dir_path in [LORA_WEIGHTS_DIR, MODELS_CACHE_DIR, HUGGINGFACE_CACHE_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

# Set environment variables for HuggingFace
os.environ["HF_HOME"] = str(HUGGINGFACE_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HUGGINGFACE_CACHE_DIR / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(HUGGINGFACE_CACHE_DIR / "datasets")