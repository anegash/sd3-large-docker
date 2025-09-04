"""Stable Diffusion pipeline management."""

import logging
import os
from typing import Optional

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from huggingface_hub import login

from .config import MODEL_ID, MODEL_VARIANT
from .device import DeviceType, detect_device, get_torch_dtype_for_device

logger = logging.getLogger(__name__)


class SD3Pipeline:
    """Manages the Stable Diffusion 3.5 pipeline."""
    
    def __init__(self, eager_load: bool = False):
        self.pipeline: Optional[StableDiffusion3Pipeline] = None
        self.device: Optional[DeviceType] = None
        self.device_description: str = ""
        self.is_loading: bool = False
        self.load_error: Optional[str] = None
        
        if eager_load:
            self._initialize_pipeline()
    
    def _initialize_pipeline(self) -> None:
        """Initialize the diffusion pipeline."""
        if self.is_loading:
            logger.warning("Pipeline is already loading")
            return
            
        self.is_loading = True
        self.load_error = None
        logger.info("Initializing Stable Diffusion 3.5 Large pipeline...")
        
        try:
            # Check for HuggingFace authentication (support both HF_TOKEN and HUGGINGFACE_TOKEN)
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            if hf_token:
                logger.info("Using HuggingFace token from environment")
                login(token=hf_token, add_to_git_credential=True)
            else:
                logger.info("No HuggingFace token found, attempting anonymous access")
            
            # Detect device
            self.device, self.device_description = detect_device()
            torch_dtype = get_torch_dtype_for_device(self.device)
            
            # Load pipeline
            # SD3.5 Large doesn't support variant parameter properly, load without it
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch_dtype
            )
            
            # Move to device
            self.pipeline.to(self.device)
            logger.info(f"Pipeline loaded successfully on {self.device_description}")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            self.load_error = str(e)
            raise
        finally:
            self.is_loading = False
    
    def initialize_async(self) -> None:
        """Initialize pipeline if not already done."""
        if not self.is_ready and not self.is_loading and self.load_error is None:
            self._initialize_pipeline()
    
    def generate_image(
        self, 
        prompt: str, 
        num_inference_steps: int = 15, 
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text prompt for image generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            
        Returns:
            Generated PIL Image
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized")
        
        logger.info(f"Generating image with prompt: '{prompt[:50]}...' "
                   f"(steps={num_inference_steps}, guidance={guidance_scale})")
        
        try:
            result = self.pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            return result.images[0]
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
    
    @property
    def is_ready(self) -> bool:
        """Check if the pipeline is ready for inference."""
        return self.pipeline is not None
    
    @property 
    def status(self) -> str:
        """Get the current status of the pipeline."""
        if self.load_error:
            return f"Error: {self.load_error}"
        elif self.is_loading:
            return "Loading model..."
        elif self.is_ready:
            return f"Ready on {self.device_description}"
        else:
            return "Not initialized"