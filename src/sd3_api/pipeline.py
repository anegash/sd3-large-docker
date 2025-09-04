"""Stable Diffusion pipeline management."""

import json
import logging
import os
from typing import Optional

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from huggingface_hub import login
from peft import PeftModel

from .config import MODEL_ID, MODEL_VARIANT
from .device import DeviceType, detect_device, get_torch_dtype_for_device
from .lora_trainer import LoRATrainer

logger = logging.getLogger(__name__)


class SD3Pipeline:
    """Manages the Stable Diffusion 3.5 pipeline."""
    
    def __init__(self, eager_load: bool = False):
        self.pipeline: Optional[StableDiffusion3Pipeline] = None
        self.device: Optional[DeviceType] = None
        self.device_description: str = ""
        self.is_loading: bool = False
        self.load_error: Optional[str] = None
        self.lora_trainer = LoRATrainer()
        self.current_lora_id: Optional[str] = None
        
        # Import config to ensure workspace directories are created
        from .config import ensure_workspace_dirs
        ensure_workspace_dirs()
        
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
            # Check for HuggingFace authentication
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if hf_token:
                logger.info("Using HuggingFace token from environment")
                login(token=hf_token, add_to_git_credential=True)
            else:
                logger.info("No HuggingFace token found, attempting anonymous access")
            
            # Detect device
            self.device, self.device_description = detect_device()
            torch_dtype = get_torch_dtype_for_device(self.device)
            
            # Load pipeline
            if self.device == "cpu":
                # CPU doesn't support fp16 variant
                self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                    MODEL_ID, 
                    torch_dtype=torch_dtype
                )
            else:
                # GPU devices can use fp16 variant
                self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                    MODEL_ID, 
                    torch_dtype=torch_dtype, 
                    variant=MODEL_VARIANT
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
        guidance_scale: float = 7.5,
        person_id: Optional[str] = None
    ) -> Image.Image:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text prompt for image generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            person_id: Optional person ID to load LoRA weights for
            
        Returns:
            Generated PIL Image
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized")
        
        # Load LoRA weights if person_id provided
        if person_id and person_id != self.current_lora_id:
            self.load_lora_weights(person_id)
        
        logger.info(f"Generating image with prompt: '{prompt[:50]}...' "
                   f"(steps={num_inference_steps}, guidance={guidance_scale})")
        if person_id:
            logger.info(f"Using LoRA weights for person_id: {person_id}")
        
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
    
    def load_lora_weights(self, person_id: str) -> None:
        """Load LoRA weights for a specific person."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized")
        
        # Check if LoRA exists
        lora_dir = self.lora_trainer.lora_weights_dir / person_id
        metadata_path = self.lora_trainer.lora_weights_dir / f"{person_id}_metadata.json"
        
        if not metadata_path.exists():
            raise ValueError(f"No LoRA weights found for person_id: {person_id}")
        
        try:
            # Read metadata to check if it's actually trained
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if metadata.get("model_type") == "sd3_lora_trained" and lora_dir.exists():
                logger.info(f"Loading trained LoRA weights for {person_id}")
                
                # Load the actual LoRA weights
                from peft import PeftModel
                self.pipeline.text_encoder = PeftModel.from_pretrained(
                    self.pipeline.text_encoder,
                    str(lora_dir),
                    adapter_name=person_id
                )
                
                # Set active adapter
                self.pipeline.text_encoder.set_adapter(person_id)
                self.current_lora_id = person_id
                
                logger.info(f"Successfully loaded trained LoRA weights for {person_id}")
            else:
                logger.info(f"Loading placeholder LoRA for {person_id}")
                self.current_lora_id = person_id
                
        except Exception as e:
            logger.error(f"Failed to load LoRA weights for {person_id}: {e}")
            # Fall back to placeholder mode
            self.current_lora_id = person_id
            logger.info(f"Using placeholder mode for {person_id}")
    
    def unload_lora_weights(self) -> None:
        """Unload current LoRA weights."""
        if self.current_lora_id and self.pipeline:
            try:
                logger.info(f"Unloading LoRA weights for {self.current_lora_id}")
                # This would disable the adapter - implementation depends on peft version
                self.current_lora_id = None
            except Exception as e:
                logger.warning(f"Failed to cleanly unload LoRA weights: {e}")
    
    def get_available_loras(self) -> list:
        """Get list of available LoRA person IDs."""
        return self.lora_trainer.list_available_loras()