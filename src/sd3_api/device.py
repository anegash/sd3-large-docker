"""Device detection and management utilities."""

import logging
from typing import Tuple

import torch

from .config import DeviceType

logger = logging.getLogger(__name__)


def detect_device() -> Tuple[DeviceType, str]:
    """
    Detect the best available device for inference.
    
    Returns:
        Tuple of (device_type, device_description)
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA GPU: {device_name}")
        return "cuda", f"CUDA GPU: {device_name}"
    
    elif torch.backends.mps.is_available():
        logger.info("Using Apple Silicon GPU (MPS)")
        return "mps", "Apple Silicon GPU (MPS)"
    
    else:
        logger.info("Using CPU")
        return "cpu", "CPU"


def get_torch_dtype_for_device(device: DeviceType) -> torch.dtype:
    """Get the appropriate torch dtype for the given device."""
    if device == "cpu":
        # CPU doesn't support float16
        return torch.float32
    else:
        # GPU devices support float16 for better performance
        return torch.float16