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
        Internal method to train SDXL LoRA with images using PEFT.
        """
        import time
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW
        from peft import LoraConfig, get_peft_model, TaskType
        import torchvision.transforms as transforms

        try:
            # Create training prompts with unique identifier token
            # Using a unique token helps the model learn the specific person
            unique_token = f"sks {person_id}"  # sks is a common token for personalization
            
            training_prompts = [
                f"a photo of {unique_token}",
                f"a portrait of {unique_token}",
                f"{unique_token} smiling",
                f"a close up photo of {unique_token}",
                f"a headshot of {unique_token}",
                f"a picture of {unique_token}",
                f"{unique_token} looking at camera",
                f"a photo of {unique_token} person",
                f"{unique_token} in professional attire",
                f"a selfie of {unique_token}",
            ]

            logger.info(
                f"Starting real SDXL LoRA training for {person_id} with {len(images)} images..."
            )

            device = pipeline.device
            
            # Configure LoRA for UNet only (most effective for personalization)
            lora_config = LoraConfig(
                r=32,  # Increased rank for better personalization
                lora_alpha=32,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Key attention layers
                lora_dropout=0.0,
            )

            # Apply LoRA to UNet
            pipeline.unet = get_peft_model(pipeline.unet, lora_config)
            pipeline.unet.train()
            
            # Prepare optimizer
            optimizer = AdamW(
                pipeline.unet.parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )

            # Image preprocessing for SDXL
            transform = transforms.Compose([
                transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
            ])

            # Process training images
            processed_data = []
            for i, image in enumerate(images):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                    
                # Get multiple prompts per image for better training
                for j in range(2):  # Use each image with 2 different prompts
                    prompt_idx = (i * 2 + j) % len(training_prompts)
                    processed_data.append({
                        "image": transform(image).unsqueeze(0).to(device),
                        "prompt": training_prompts[prompt_idx]
                    })

            logger.info(f"Prepared {len(processed_data)} training samples")

            # Training parameters
            actual_epochs = min(num_train_epochs, 50)  # Limit for reasonable training time
            gradient_accumulation_steps = 4
            
            # Training loop
            global_step = 0
            for epoch in range(actual_epochs):
                epoch_loss = 0.0
                
                for batch_idx, data in enumerate(processed_data):
                    with torch.set_grad_enabled(True):
                        # Encode text
                        text_inputs = pipeline.tokenizer(
                            data["prompt"],
                            padding="max_length",
                            max_length=77,
                            truncation=True,
                            return_tensors="pt"
                        ).to(device)
                        
                        text_inputs_2 = pipeline.tokenizer_2(
                            data["prompt"],
                            padding="max_length",
                            max_length=77,
                            truncation=True,
                            return_tensors="pt"
                        ).to(device)
                        
                        # Get text embeddings
                        prompt_embeds = pipeline.text_encoder(
                            text_inputs.input_ids,
                            output_hidden_states=True
                        )
                        prompt_embeds_2 = pipeline.text_encoder_2(
                            text_inputs_2.input_ids,
                            output_hidden_states=True
                        )
                        
                        # Combine embeddings (SDXL uses both encoders)
                        prompt_embeds = torch.cat([
                            prompt_embeds.hidden_states[-2],
                            prompt_embeds_2.hidden_states[-2]
                        ], dim=-1)
                        
                        # Add noise to image
                        noise = torch.randn_like(data["image"])
                        timesteps = torch.randint(
                            0, pipeline.scheduler.config.num_train_timesteps,
                            (1,), device=device
                        ).long()
                        
                        # Forward diffusion process
                        noisy_images = pipeline.scheduler.add_noise(
                            data["image"], noise, timesteps
                        )
                        
                        # Predict noise
                        model_pred = pipeline.unet(
                            noisy_images,
                            timesteps,
                            encoder_hidden_states=prompt_embeds,
                            return_dict=False
                        )[0]
                        
                        # Calculate loss
                        loss = F.mse_loss(model_pred, noise)
                        loss = loss / gradient_accumulation_steps
                        loss.backward()
                        
                        epoch_loss += loss.item()
                        
                        # Update weights
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            global_step += 1
                
                avg_loss = epoch_loss / len(processed_data)
                logger.info(f"Epoch {epoch + 1}/{actual_epochs} - Loss: {avg_loss:.4f}")
                
                # Save checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Checkpoint: {int((epoch + 1) / actual_epochs * 100)}% complete")

            # Save the trained LoRA weights
            lora_save_dir = self.lora_weights_dir / person_id
            lora_save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save LoRA adapter
            pipeline.unet.save_pretrained(lora_save_dir)
            
            # Save training metadata
            training_info = {
                "model_version": "sdxl-lora-v2-real",
                "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
                "unique_token": unique_token,
                "num_images": len(images),
                "num_samples": len(processed_data),
                "training_epochs": actual_epochs,
                "learning_rate": learning_rate,
                "lora_rank": lora_config.r,
                "target_modules": lora_config.target_modules,
            }
            
            with open(lora_save_dir / "training_info.json", "w") as f:
                json.dump(training_info, f, indent=2)
                
            logger.info(f"Real LoRA training completed - saved to {lora_save_dir}")
            
            # Reset model to eval mode
            pipeline.unet.eval()
            
            # Save metadata
            self._save_training_metadata(
                person_id, len(images), actual_epochs, learning_rate
            )
            
            logger.info(f"LoRA training completed successfully for {person_id}")

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
