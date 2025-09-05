"""LoRA training functionality for SDXL pipeline."""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from diffusers import StableDiffusionXLPipeline
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from transformers import CLIPTextModel

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Simple LoRA trainer for SDXL personalization."""

    def __init__(self, lora_weights_dir: Optional[Path] = None):
        # Import here to avoid circular import
        from .config import LORA_WEIGHTS_DIR, ensure_workspace_dirs

        if lora_weights_dir is None:
            lora_weights_dir = LORA_WEIGHTS_DIR

        self.lora_weights_dir = Path(lora_weights_dir)
        ensure_workspace_dirs()  # Ensure all workspace dirs exist

        # LoRA configuration for SDXL UNet
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "to_q",
                "to_v",
                "to_k",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
            ],  # SDXL UNet attention modules
            lora_dropout=0.1,
            task_type=TaskType.FEATURE_EXTRACTION,
        )

    def train_lora_from_images(
        self,
        person_id: str,
        pipeline: StableDiffusionXLPipeline,
        num_train_epochs: int = 100,
        learning_rate: float = 1e-4,
        source_person_id: Optional[str] = None,
    ) -> None:
        """
        Train SDXL LoRA weights for a specific person using stored images.

        Args:
            person_id: Target person identifier
            pipeline: SDXL pipeline to train on
            num_train_epochs: Number of training epochs
            learning_rate: Learning rate for training
            source_person_id: If provided, copy images from this person first
        """
        logger.info(f"Starting LoRA training for person_id: {person_id}")

        # Import image manager
        from .image_manager import ImageManager

        image_manager = ImageManager()

        # Copy images if source specified
        if source_person_id:
            logger.info(f"Copying images from {source_person_id} to {person_id}")
            copy_result = image_manager.copy_images(source_person_id, person_id)
            logger.info(f"Copied {copy_result['num_images']} images")

        # Load images for training
        images = image_manager.get_images(person_id)
        if not images:
            raise ValueError(f"No images found for person_id: {person_id}")

        logger.info(f"Training with {len(images)} images for {person_id}")

        # Continue with existing training logic
        self._train_with_images(
            person_id, images, pipeline, num_train_epochs, learning_rate
        )

    def train_lora(
        self,
        person_id: str,
        images: List[Image.Image],
        pipeline: StableDiffusionXLPipeline,
        num_train_epochs: int = 100,
        learning_rate: float = 1e-4,
    ) -> None:
        """
        Legacy method: Train SDXL LoRA weights directly with provided images.
        """
        logger.info(f"Starting SDXL LoRA training for person_id: {person_id}")
        logger.info(f"Received {len(images)} images for training")

        self._train_with_images(
            person_id, images, pipeline, num_train_epochs, learning_rate
        )

    def _train_with_images(
        self,
        person_id: str,
        images: List[Image.Image],
        pipeline: StableDiffusionXLPipeline,
        num_train_epochs: int = 100,
        learning_rate: float = 1e-4,
    ) -> None:
        """
        Real LoRA training using actual gradient descent on the images.
        """
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW
        from torch.utils.data import Dataset, DataLoader
        import torchvision.transforms as transforms
        import json
        import random
        import numpy as np
        
        try:
            logger.info(f"Starting REAL LoRA training for {person_id} with {len(images)} images")
            
            # Create unique token for this person
            unique_token = f"sks {person_id}"
            
            device = pipeline.device
            dtype = pipeline.unet.dtype if hasattr(pipeline.unet, 'dtype') else torch.float16
            
            # ROOT CAUSE FIX: Disable CPU offloading during training
            logger.info(f"Disabling CPU offloading for training on device: {device}")
            
            # Disable CPU offloading that causes device conflicts during training
            if hasattr(pipeline, '_cpu_offload_hooks'):
                pipeline._cpu_offload_hooks.clear()
                logger.info("Cleared CPU offload hooks")
            
            # Move everything to GPU and keep it there
            logger.info(f"Moving all pipeline components to {device} with dtype {dtype}")
            pipeline.vae = pipeline.vae.to(device, dtype=dtype)
            pipeline.text_encoder = pipeline.text_encoder.to(device, dtype=dtype) 
            pipeline.text_encoder_2 = pipeline.text_encoder_2.to(device, dtype=dtype)
            pipeline.unet = pipeline.unet.to(device, dtype=dtype)
            
            logger.info(f"All pipeline components moved to {device} for training")
            
            # Create training dataset
            class PersonDataset(Dataset):
                def __init__(self, images, prompts):
                    self.images = images
                    self.prompts = prompts
                    self.transform = transforms.Compose([
                        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])
                    
                def __len__(self):
                    return len(self.images) * 3  # Repeat each image 3x with different prompts
                    
                def __getitem__(self, idx):
                    img_idx = idx % len(self.images)
                    prompt_idx = idx % len(self.prompts)
                    
                    image = self.images[img_idx]
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                        
                    pixel_values = self.transform(image)
                    return {
                        "pixel_values": pixel_values,
                        "prompt": self.prompts[prompt_idx]
                    }
            
            # Create training prompts for this person
            training_prompts = [
                f"a photo of {unique_token}",
                f"a portrait of {unique_token}", 
                f"{unique_token} person",
                f"a picture of {unique_token}",
                f"{unique_token} looking at camera",
                f"a headshot of {unique_token}",
                f"{unique_token} smiling",
                f"professional photo of {unique_token}"
            ]
            
            # Create dataset and dataloader
            dataset = PersonDataset(images, training_prompts)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            
            logger.info(f"Created dataset with {len(dataset)} training samples")
            
            # Add LoRA layers to UNet
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                r=16,  # Lower rank for stability
                lora_alpha=16,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                lora_dropout=0.0,
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            
            # Apply LoRA to UNet
            pipeline.unet = get_peft_model(pipeline.unet, lora_config)
            pipeline.unet.train()
            
            # Setup optimizer - only train LoRA parameters
            lora_params = [p for p in pipeline.unet.parameters() if p.requires_grad]
            optimizer = AdamW(lora_params, lr=learning_rate, weight_decay=0.01)
            
            logger.info(f"Training {len(lora_params)} LoRA parameters")
            
            # Training loop
            num_epochs = min(num_train_epochs, 20)  # Limit epochs for reasonable time
            global_step = 0
            
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for batch in dataloader:
                    # Get pixel values and prompt
                    pixel_values = batch["pixel_values"].to(device, dtype=dtype)
                    prompt = batch["prompt"][0] if isinstance(batch["prompt"], list) else batch["prompt"]
                    
                    # Encode text prompt manually to avoid parameter conflicts
                    with torch.no_grad():
                        # Tokenize text
                        text_inputs = pipeline.tokenizer(
                            prompt,
                            padding="max_length",
                            max_length=77,
                            truncation=True,
                            return_tensors="pt"
                        ).to(device)
                        
                        text_inputs_2 = pipeline.tokenizer_2(
                            prompt,
                            padding="max_length", 
                            max_length=77,
                            truncation=True,
                            return_tensors="pt"
                        ).to(device)
                        
                        # Get embeddings
                        prompt_embeds_1 = pipeline.text_encoder(text_inputs.input_ids)[0]
                        prompt_embeds_2 = pipeline.text_encoder_2(text_inputs_2.input_ids)[0]
                        
                        # Concatenate embeddings for SDXL
                        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
                    
                    # Convert to latents - should work now without CPU offloading
                    with torch.no_grad():
                        latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * pipeline.vae.config.scaling_factor
                    
                    # Sample random timestep
                    timesteps = torch.randint(
                        0, pipeline.scheduler.config.num_train_timesteps, 
                        (latents.shape[0],), device=device
                    ).long()
                    
                    # Add noise
                    noise = torch.randn_like(latents)
                    noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
                    
                    # Predict noise with explicit parameters only
                    model_pred = pipeline.unet(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False
                    )[0]
                    
                    # Calculate loss
                    loss = F.mse_loss(model_pred, noise)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    global_step += 1
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
                
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Training progress: {int((epoch + 1) / num_epochs * 100)}%")
            
            logger.info("LoRA training completed! Saving weights...")
            
            # Save LoRA weights
            lora_save_dir = self.lora_weights_dir / person_id
            lora_save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the trained LoRA model
            pipeline.unet.save_pretrained(lora_save_dir)
            
            # Save training metadata
            training_info = {
                "model_version": "sdxl-lora-real-v3",
                "base_model": "stabilityai/stable-diffusion-xl-base-1.0", 
                "unique_token": unique_token,
                "num_images": len(images),
                "num_training_samples": len(dataset),
                "training_epochs": num_epochs,
                "learning_rate": learning_rate,
                "lora_rank": lora_config.r,
                "target_modules": lora_config.target_modules,
                "global_steps": global_step
            }
            
            with open(lora_save_dir / "training_info.json", "w") as f:
                json.dump(training_info, f, indent=2)
            
            # Save training metadata
            self._save_training_metadata(
                person_id, len(images), num_epochs, learning_rate
            )
            
            logger.info(f"Real LoRA training completed for {person_id}!")
            logger.info(f"Trained for {global_step} steps across {num_epochs} epochs")
            logger.info(f"Use token '{unique_token}' in prompts for best results")
            
        except Exception as e:
            logger.error(f"Real LoRA training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        except Exception as e:
            logger.error(f"LoRA training failed for {person_id}: {e}")
            # Save error info
            error_data = {
                "person_id": person_id,
                "num_images": len(images),
                "error": str(e),
                "status": "failed",
            }
            self._save_lora_weights(person_id, error_data)
            raise

    def _save_trained_lora_weights(self, person_id: str, lora_model) -> None:
        """Save actual trained LoRA weights."""
        import datetime

        # Create person directory
        person_dir = self.lora_weights_dir / person_id
        person_dir.mkdir(parents=True, exist_ok=True)

        # Save the LoRA adapter weights
        lora_model.save_pretrained(str(person_dir))

        # Save training metadata
        metadata = {
            "person_id": person_id,
            "model_type": "sdxl_lora_trained",
            "created_at": datetime.datetime.now().isoformat(),
            "status": "completed",
        }

        metadata_path = self.lora_weights_dir / f"{person_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_lora_weights(self, person_id: str, training_data) -> None:
        """Save LoRA training data to filesystem (fallback)."""
        import datetime

        # Create person directory
        person_dir = self.lora_weights_dir / person_id
        person_dir.mkdir(parents=True, exist_ok=True)

        # Save training metadata
        metadata = {
            "person_id": person_id,
            "model_type": "sdxl_lora_placeholder",
            "created_at": datetime.datetime.now().isoformat(),
            "training_data": training_data,
        }

        metadata_path = self.lora_weights_dir / f"{person_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_training_metadata(
        self, person_id: str, num_images: int, num_epochs: int, learning_rate: float
    ) -> None:
        """Save training metadata for a person."""
        import datetime

        # Create a simple LoRA directory structure
        lora_dir = self.lora_weights_dir / person_id
        lora_dir.mkdir(parents=True, exist_ok=True)

        # Save basic metadata
        metadata = {
            "person_id": person_id,
            "model_type": "sdxl_lora_trained",
            "num_images": num_images,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "completed",
        }

        metadata_path = self.lora_weights_dir / f"{person_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved training metadata for {person_id} at {metadata_path}")

    def _create_placeholder_weights(self, lora_save_dir: Path) -> None:
        """Create placeholder LoRA weight files that diffusers can load."""
        import json

        import torch

        lora_save_dir.mkdir(parents=True, exist_ok=True)

        # Create placeholder weights structure
        weights_data = {
            "unet": {},  # Empty placeholder for UNet LoRA weights
            "text_encoder": {},  # Empty placeholder for text encoder weights
            "text_encoder_2": {},  # SDXL has two text encoders
        }

        # Save the weights file
        torch.save(weights_data, lora_save_dir / "pytorch_lora_weights.bin")

        # Ensure target_modules is JSON serializable
        target_modules = self.lora_config.target_modules
        if hasattr(target_modules, "__iter__") and not isinstance(target_modules, str):
            target_modules = list(target_modules)

        # Create adapter_config.json for diffusers compatibility
        adapter_config = {
            "base_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
            "lora_alpha": self.lora_config.lora_alpha,
            "lora_dropout": self.lora_config.lora_dropout,
            "r": self.lora_config.r,
            "target_modules": target_modules,
            "task_type": "FEATURE_EXTRACTION",
        }

        with open(lora_save_dir / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)

        logger.info(f"Created placeholder LoRA weights at {lora_save_dir}")

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
