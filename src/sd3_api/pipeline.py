"""SDXL pipeline management."""

import json
import logging
import os
from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from PIL import Image
from huggingface_hub import login
from peft import PeftModel

from .config import MODEL_ID, REFINER_MODEL_ID, MODEL_VARIANT
from .device import DeviceType, detect_device, get_torch_dtype_for_device
from .lora_trainer import LoRATrainer

logger = logging.getLogger(__name__)


class SDXLPipeline:
    """Manages the Stable Diffusion XL pipeline."""
    
    def __init__(self, eager_load: bool = False):
        self.pipeline: Optional[StableDiffusionXLPipeline] = None
        self.refiner: Optional[DiffusionPipeline] = None
        self.device: Optional[DeviceType] = None
        self.device_description: str = ""
        self.is_loading: bool = False
        self.load_error: Optional[str] = None
        self.lora_trainer = LoRATrainer()
        self.current_lora_id: Optional[str] = None
        self.use_refiner: bool = False  # Disable refiner for RunPod to save memory
        
        # Import config to ensure workspace directories are created
        from .config import ensure_workspace_dirs
        ensure_workspace_dirs()
        
        if eager_load:
            self._initialize_pipeline()
    
    def _initialize_pipeline(self) -> None:
        """Initialize the SDXL pipeline."""
        if self.is_loading:
            logger.warning("Pipeline is already loading")
            return
            
        self.is_loading = True
        self.load_error = None
        logger.info("Initializing Stable Diffusion XL pipeline...")
        
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
            
            # Load SDXL base pipeline
            if self.device == "cpu":
                # CPU doesn't support fp16 variant
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    MODEL_ID, 
                    torch_dtype=torch_dtype,
                    use_safetensors=True
                )
            else:
                # GPU devices can use fp16 variant
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    MODEL_ID, 
                    torch_dtype=torch_dtype, 
                    variant=MODEL_VARIANT,
                    use_safetensors=True
                )
            
            # Enable memory efficient attention and CPU offloading for RunPod
            if hasattr(self.pipeline, 'enable_memory_efficient_attention'):
                self.pipeline.enable_memory_efficient_attention()
            
            # Use either CPU offloading OR manual GPU placement, not both
            if hasattr(self.pipeline, 'enable_model_cpu_offload') and self.device != "cpu":
                # Use automatic offloading for memory efficiency
                self.pipeline.enable_model_cpu_offload()
            else:
                # Manual GPU placement when offloading not available or on CPU
                self.pipeline.to(self.device)
            
            # Optionally load refiner (disabled for memory efficiency)
            if self.use_refiner and self.device != "cpu":
                try:
                    logger.info("Loading SDXL refiner...")
                    self.refiner = DiffusionPipeline.from_pretrained(
                        REFINER_MODEL_ID,
                        text_encoder_2=self.pipeline.text_encoder_2,
                        vae=self.pipeline.vae,
                        torch_dtype=torch_dtype,
                        variant=MODEL_VARIANT,
                        use_safetensors=True
                    )
                    self.refiner.to(self.device)
                    logger.info("SDXL refiner loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load refiner: {e}")
                    self.refiner = None
            
            logger.info(f"SDXL pipeline loaded successfully on {self.device_description}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SDXL pipeline: {e}")
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
        num_inference_steps: int = 20, 
        guidance_scale: float = 7.5,
        person_id: Optional[str] = None,
        width: int = 1024,
        height: int = 1024
    ) -> Image.Image:
        """
        Generate an image from a text prompt using SDXL.
        
        Args:
            prompt: Text prompt for image generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            person_id: Optional person ID to load LoRA weights for
            width: Image width (must be divisible by 8)
            height: Image height (must be divisible by 8)
            
        Returns:
            Generated PIL Image
        """
        if self.pipeline is None:
            raise RuntimeError("SDXL pipeline not initialized")
        
        # Ensure dimensions are divisible by 8 for SDXL
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        # Load LoRA weights if person_id provided
        if person_id and person_id != self.current_lora_id:
            self.load_lora_weights(person_id)
        
        # Modify prompt to use unique token if person_id is provided
        original_prompt = prompt
        if person_id:
            # Replace references to the person with the unique token
            unique_token = f"sks {person_id}"
            # Common replacements
            prompt = prompt.replace(person_id, unique_token)
            prompt = prompt.replace("person", unique_token)
            # If prompt doesn't contain the token, prepend it
            if unique_token not in prompt and "sks" not in prompt:
                prompt = f"{unique_token}, {prompt}"
            logger.info(f"Modified prompt for LoRA: '{prompt[:100]}...'")
        
        logger.info(f"Generating SDXL image with prompt: '{prompt[:50]}...' "
                   f"(steps={num_inference_steps}, guidance={guidance_scale}, {width}x{height})")
        if person_id:
            logger.info(f"Using LoRA weights for person_id: {person_id}")
        
        try:
            # Generate with SDXL base model
            result = self.pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                output_type="latent" if self.use_refiner and self.refiner else "pil"
            )
            
            # Apply refiner if available and enabled
            if self.use_refiner and self.refiner:
                logger.info("Applying SDXL refiner...")
                # Use the refiner for high-res pass
                refined_result = self.refiner(
                    prompt=prompt,
                    image=result.images,
                    num_inference_steps=num_inference_steps // 2,
                    guidance_scale=guidance_scale,
                )
                return refined_result.images[0]
            else:
                return result.images[0]
            
        except Exception as e:
            logger.error(f"SDXL image generation failed: {e}")
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
        """Load SDXL LoRA weights for a specific person."""
        if self.pipeline is None:
            raise RuntimeError("SDXL pipeline not initialized")
        
        # Check if LoRA exists
        lora_dir = self.lora_trainer.lora_weights_dir / person_id
        metadata_path = self.lora_trainer.lora_weights_dir / f"{person_id}_metadata.json"
        
        if not metadata_path.exists():
            raise ValueError(f"No LoRA weights found for person_id: {person_id}")
        
        try:
            # Unload current LoRA if any
            if self.current_lora_id:
                try:
                    self.pipeline.unload_lora_weights()
                except:
                    pass  # Ignore errors when unloading
            
            # Read metadata to check if it's actually trained
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check for real trained model (v2) or legacy
            if lora_dir.exists():
                # Check if it has the adapter_model.safetensors file (real training)
                adapter_file = lora_dir / "adapter_model.safetensors"
                bin_file = lora_dir / "adapter_model.bin"
                
                if adapter_file.exists() or bin_file.exists():
                    logger.info(f"Loading real trained SDXL LoRA weights for {person_id}")
                    
                    # Load LoRA weights using PEFT format
                    from peft import PeftModel
                    self.pipeline.unet = PeftModel.from_pretrained(
                        self.pipeline.unet,
                        str(lora_dir)
                    )
                    self.current_lora_id = person_id
                    
                    logger.info(f"Successfully loaded real LoRA weights for {person_id}")
                else:
                    logger.warning(f"No real LoRA weights found for {person_id}, using base model")
                    self.current_lora_id = person_id
            else:
                logger.info(f"No LoRA directory for {person_id}, using base model")
                self.current_lora_id = person_id
                
        except Exception as e:
            logger.error(f"Failed to load SDXL LoRA weights for {person_id}: {e}")
            # Fall back to base model
            self.current_lora_id = person_id
            logger.info(f"Using base model for {person_id}")
    
    def unload_lora_weights(self) -> None:
        """Unload current SDXL LoRA weights."""
        if self.current_lora_id and self.pipeline:
            try:
                logger.info(f"Unloading SDXL LoRA weights for {self.current_lora_id}")
                self.pipeline.unload_lora_weights()
                self.current_lora_id = None
            except Exception as e:
                logger.warning(f"Failed to cleanly unload SDXL LoRA weights: {e}")
    
    def get_available_loras(self) -> list:
        """Get list of available LoRA person IDs."""
        return self.lora_trainer.list_available_loras()