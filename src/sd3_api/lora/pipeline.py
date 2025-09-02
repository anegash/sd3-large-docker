"""Extended SD3 Pipeline with LoRA adapter support."""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel

from ..pipeline import SD3Pipeline
from ..database.manager import db_manager
from ..utils.storage import storage_manager

logger = logging.getLogger(__name__)


class LoRASD3Pipeline(SD3Pipeline):
    """Extended SD3 Pipeline with LoRA adapter support."""
    
    def __init__(self, eager_load: bool = False):
        super().__init__(eager_load)
        self.loaded_loras: Dict[str, PeftModel] = {}
        self.active_loras: Set[str] = set()
        self.child_token_pattern = re.compile(r'\{([^}]+)\}')
    
    def load_lora_adapter(self, child_id: str) -> bool:
        """
        Load LoRA adapter for a specific child.
        
        Args:
            child_id: Child identifier
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if child_id in self.loaded_loras:
                logger.info(f"LoRA adapter for {child_id} already loaded")
                return True
            
            # Get active LoRA model for child
            lora_model = db_manager.get_lora_model(child_id, active_only=True)
            if not lora_model:
                logger.warning(f"No active LoRA model found for {child_id}")
                return False
            
            if not Path(lora_model.model_path).exists():
                logger.error(f"LoRA model file not found: {lora_model.model_path}")
                return False
            
            # Ensure base pipeline is loaded
            if not self.is_ready:
                self._initialize_pipeline()
            
            # Load LoRA adapter
            logger.info(f"Loading LoRA adapter for {child_id} from {lora_model.model_path}")
            
            # Load LoRA weights into a copy of the UNet
            lora_unet = PeftModel.from_pretrained(
                self.pipeline.transformer,
                Path(lora_model.model_path).parent,
                adapter_name=child_id
            )
            
            self.loaded_loras[child_id] = lora_unet
            logger.info(f"Successfully loaded LoRA adapter for {child_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter for {child_id}: {e}")
            return False
    
    def unload_lora_adapter(self, child_id: str) -> bool:
        """
        Unload LoRA adapter for a specific child.
        
        Args:
            child_id: Child identifier
            
        Returns:
            True if unloaded successfully, False otherwise
        """
        try:
            if child_id not in self.loaded_loras:
                logger.info(f"LoRA adapter for {child_id} not loaded")
                return True
            
            # Remove from active set
            self.active_loras.discard(child_id)
            
            # Delete the adapter
            del self.loaded_loras[child_id]
            
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Successfully unloaded LoRA adapter for {child_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload LoRA adapter for {child_id}: {e}")
            return False
    
    def _extract_child_ids_from_prompt(self, prompt: str) -> List[str]:
        """
        Extract child IDs from prompt tokens like {child_001}.
        
        Args:
            prompt: Text prompt
            
        Returns:
            List of child IDs found in prompt
        """
        matches = self.child_token_pattern.findall(prompt)
        return list(set(matches))  # Remove duplicates
    
    def _prepare_multi_lora_pipeline(self, child_ids: List[str]) -> Optional[torch.nn.Module]:
        """
        Prepare pipeline with multiple LoRA adapters.
        
        Args:
            child_ids: List of child IDs to load
            
        Returns:
            Combined UNet with multiple LoRA adapters or None if failed
        """
        if not child_ids:
            return self.pipeline.transformer
        
        # Load all required LoRA adapters
        loaded_adapters = []
        for child_id in child_ids:
            if child_id not in self.loaded_loras:
                if not self.load_lora_adapter(child_id):
                    logger.warning(f"Failed to load LoRA for {child_id}, skipping")
                    continue
            loaded_adapters.append(child_id)
        
        if not loaded_adapters:
            logger.warning("No LoRA adapters could be loaded, using base model")
            return self.pipeline.transformer
        
        # For multiple LoRAs, we'll use adapter merging or sequential application
        # This is a simplified version - in practice, you might want more sophisticated merging
        try:
            if len(loaded_adapters) == 1:
                # Single adapter case
                child_id = loaded_adapters[0]
                lora_unet = self.loaded_loras[child_id]
                self.active_loras = {child_id}
                return lora_unet
            else:
                # Multiple adapters case - this is complex and might require custom implementation
                # For now, we'll use the first adapter as primary
                logger.warning(f"Multiple LoRA adapters requested: {loaded_adapters}. Using first adapter as primary.")
                primary_child = loaded_adapters[0]
                lora_unet = self.loaded_loras[primary_child]
                self.active_loras = {primary_child}
                return lora_unet
                
        except Exception as e:
            logger.error(f"Failed to prepare multi-LoRA pipeline: {e}")
            return self.pipeline.transformer
    
    def _replace_child_tokens_in_prompt(self, prompt: str, child_ids: List[str]) -> str:
        """
        Replace child ID tokens in prompt with appropriate descriptors.
        
        Args:
            prompt: Original prompt
            child_ids: List of child IDs found in prompt
            
        Returns:
            Modified prompt with tokens replaced
        """
        modified_prompt = prompt
        
        for child_id in child_ids:
            # Get child information from database
            child = db_manager.get_child(child_id)
            if child:
                # Replace {child_id} with more descriptive text
                token = f"{{{child_id}}}"
                replacement = f"person {child_id}"  # Simple replacement - could be more sophisticated
                modified_prompt = modified_prompt.replace(token, replacement)
                logger.info(f"Replaced {token} with '{replacement}' in prompt")
            else:
                logger.warning(f"Child {child_id} not found in database")
        
        return modified_prompt
    
    def generate_image_with_lora(
        self,
        prompt: str,
        num_inference_steps: int = 15,
        guidance_scale: float = 7.5,
        child_ids: Optional[List[str]] = None
    ) -> Tuple[Image.Image, List[str]]:
        """
        Generate image with LoRA adapters.
        
        Args:
            prompt: Text prompt (may contain child ID tokens like {child_001})
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            child_ids: Optional explicit list of child IDs to use
            
        Returns:
            Tuple of (generated image, list of child IDs used)
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized")
        
        # Extract child IDs from prompt if not explicitly provided
        if child_ids is None:
            child_ids = self._extract_child_ids_from_prompt(prompt)
        
        logger.info(f"Generating image with LoRA adapters for children: {child_ids}")
        
        # Prepare multi-LoRA pipeline
        current_unet = self._prepare_multi_lora_pipeline(child_ids)
        
        # Replace child tokens in prompt
        modified_prompt = self._replace_child_tokens_in_prompt(prompt, child_ids)
        
        try:
            # Temporarily replace UNet in pipeline
            original_unet = self.pipeline.transformer
            self.pipeline.transformer = current_unet
            
            # Generate image
            logger.info(f"Generating with modified prompt: '{modified_prompt[:100]}...'")
            result = self.pipeline(
                modified_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            
            return result.images[0], child_ids
            
        except Exception as e:
            logger.error(f"Image generation with LoRA failed: {e}")
            raise
        finally:
            # Restore original UNet
            self.pipeline.transformer = original_unet
    
    def generate_image(
        self,
        prompt: str,
        num_inference_steps: int = 15,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        Generate image (override parent method to support LoRA).
        
        This method automatically detects child tokens in the prompt and uses
        appropriate LoRA adapters if available.
        """
        # Check if prompt contains child tokens
        child_ids = self._extract_child_ids_from_prompt(prompt)
        
        if child_ids:
            # Use LoRA generation
            image, used_child_ids = self.generate_image_with_lora(
                prompt, num_inference_steps, guidance_scale
            )
            logger.info(f"Generated image using LoRA adapters for: {used_child_ids}")
            return image
        else:
            # Use parent method for regular generation
            return super().generate_image(prompt, num_inference_steps, guidance_scale)
    
    def get_loaded_loras(self) -> List[str]:
        """Get list of currently loaded LoRA adapters."""
        return list(self.loaded_loras.keys())
    
    def get_active_loras(self) -> List[str]:
        """Get list of currently active LoRA adapters."""
        return list(self.active_loras)
    
    def preload_child_loras(self, child_ids: List[str]) -> Dict[str, bool]:
        """
        Preload LoRA adapters for multiple children.
        
        Args:
            child_ids: List of child IDs to preload
            
        Returns:
            Dictionary mapping child_id to success status
        """
        results = {}
        for child_id in child_ids:
            results[child_id] = self.load_lora_adapter(child_id)
        return results
    
    def cleanup_unused_loras(self, keep_child_ids: Optional[List[str]] = None) -> int:
        """
        Clean up unused LoRA adapters to free memory.
        
        Args:
            keep_child_ids: Optional list of child IDs to keep loaded
            
        Returns:
            Number of adapters cleaned up
        """
        if keep_child_ids is None:
            keep_child_ids = []
        
        cleaned_count = 0
        child_ids_to_remove = []
        
        for child_id in self.loaded_loras:
            if child_id not in keep_child_ids:
                child_ids_to_remove.append(child_id)
        
        for child_id in child_ids_to_remove:
            if self.unload_lora_adapter(child_id):
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} unused LoRA adapters")
        return cleaned_count
    
    @property
    def lora_status(self) -> Dict[str, any]:
        """Get status information about LoRA adapters."""
        return {
            "loaded_count": len(self.loaded_loras),
            "active_count": len(self.active_loras),
            "loaded_child_ids": list(self.loaded_loras.keys()),
            "active_child_ids": list(self.active_loras),
        }