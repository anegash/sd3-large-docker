"""LoRA training functionality for SD3 pipeline."""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from transformers import CLIPTextModel

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Simple LoRA trainer for SD3 personalization."""
    
    def __init__(self, lora_weights_dir: Optional[Path] = None):
        # Import here to avoid circular import
        from .config import LORA_WEIGHTS_DIR, ensure_workspace_dirs
        
        if lora_weights_dir is None:
            lora_weights_dir = LORA_WEIGHTS_DIR
            
        self.lora_weights_dir = Path(lora_weights_dir)
        ensure_workspace_dirs()  # Ensure all workspace dirs exist
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1,
            task_type=TaskType.DIFFUSION,
        )
    
    def train_lora(
        self, 
        person_id: str, 
        images: List[Image.Image], 
        pipeline: StableDiffusion3Pipeline,
        num_train_epochs: int = 100,
        learning_rate: float = 1e-4
    ) -> None:
        """
        Train LoRA weights for a specific person.
        
        Args:
            person_id: Unique identifier for the person
            images: List of PIL Images for training
            pipeline: SD3 pipeline to train on
            num_train_epochs: Number of training epochs
            learning_rate: Learning rate for training
        """
        logger.info(f"Starting LoRA training for person_id: {person_id}")
        
        # Create training prompts
        training_prompts = [
            f"a photo of {person_id}",
            f"portrait of {person_id}", 
            f"{person_id} smiling",
            f"close up of {person_id}",
            f"headshot of {person_id}"
        ]
        
        # Prepare training dataset
        dataset_dict = {
            "image": [],
            "text": []
        }
        
        for i, image in enumerate(images):
            # Use different prompts cyclically
            prompt = training_prompts[i % len(training_prompts)]
            dataset_dict["image"].append(image)
            dataset_dict["text"].append(prompt)
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Get the text encoder from pipeline
        text_encoder = pipeline.text_encoder
        
        # Apply LoRA to text encoder
        lora_model = get_peft_model(text_encoder, self.lora_config)
        
        # Simple training loop (basic implementation)
        optimizer = torch.optim.AdamW(lora_model.parameters(), lr=learning_rate)
        
        lora_model.train()
        
        for epoch in range(num_train_epochs):
            total_loss = 0
            
            for item in dataset:
                # Tokenize text
                text_inputs = pipeline.tokenizer(
                    item["text"],
                    padding="max_length",
                    max_length=pipeline.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                
                # Forward pass through text encoder
                text_embeddings = lora_model(text_inputs.input_ids.to(pipeline.device))
                
                # Simple loss (MSE between original and LoRA embeddings)
                with torch.no_grad():
                    original_embeddings = text_encoder(text_inputs.input_ids.to(pipeline.device))
                
                loss = torch.nn.functional.mse_loss(
                    text_embeddings.last_hidden_state, 
                    original_embeddings.last_hidden_state
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{num_train_epochs}, Loss: {total_loss/len(dataset):.4f}")
        
        # Save LoRA weights
        self._save_lora_weights(person_id, lora_model)
        logger.info(f"LoRA training completed for {person_id}")
    
    def _save_lora_weights(self, person_id: str, lora_model) -> None:
        """Save LoRA weights to filesystem."""
        weight_path = self.lora_weights_dir / f"{person_id}.safetensors"
        lora_model.save_pretrained(str(weight_path.with_suffix("")))
        
        # Save metadata
        metadata = {
            "person_id": person_id,
            "model_type": "sd3_lora",
            "created_at": str(torch.datetime.now() if hasattr(torch, 'datetime') else "unknown")
        }
        
        metadata_path = self.lora_weights_dir / f"{person_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def list_available_loras(self) -> List[str]:
        """List all available LoRA person IDs."""
        person_ids = []
        
        for file_path in self.lora_weights_dir.glob("*_metadata.json"):
            try:
                with open(file_path, "r") as f:
                    metadata = json.load(f)
                    person_ids.append(metadata["person_id"])
            except Exception as e:
                logger.warning(f"Failed to read metadata from {file_path}: {e}")
        
        return sorted(person_ids)
    
    def get_lora_path(self, person_id: str) -> Optional[Path]:
        """Get the path to LoRA weights for a person."""
        lora_dir = self.lora_weights_dir / person_id
        if lora_dir.exists():
            return lora_dir
        return None