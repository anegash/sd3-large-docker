"""Configuration settings for the SD3 API."""

from typing import Literal

# Model configuration
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
MODEL_VARIANT = "fp16"
TORCH_DTYPE = "float16"

# API configuration
DEFAULT_STEPS = 15
MAX_STEPS = 150
DEFAULT_GUIDANCE = 7.5
MAX_GUIDANCE = 15.0

# Server configuration
HOST = "0.0.0.0"
PORT = 8000

# Device configuration
DeviceType = Literal["cuda", "mps", "cpu"]