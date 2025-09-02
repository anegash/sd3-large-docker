"""LoRA trainer for Stable Diffusion 3.5 Large."""

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime

import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..config import MODEL_ID
from ..device import detect_device, get_torch_dtype_for_device
from ..database.models import TrainingStatus
from ..database.manager import db_manager
from ..utils.storage import storage_manager

logger = logging.getLogger(__name__)


class LoRATrainingConfig:
    """Configuration for LoRA training."""
    
    def __init__(self, 
                 lora_rank: int = 64,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 learning_rate: float = 1e-4,
                 training_steps: int = 1000,
                 batch_size: int = 1,
                 gradient_accumulation_steps: int = 4,
                 max_grad_norm: float = 1.0,
                 use_8bit_adam: bool = False,
                 mixed_precision: str = "fp16",
                 save_steps: int = 250,
                 validation_steps: int = 100,
                 seed: int = 42):
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_8bit_adam = use_8bit_adam
        self.mixed_precision = mixed_precision
        self.save_steps = save_steps
        self.validation_steps = validation_steps
        self.seed = seed


class ChildImageDataset(Dataset):
    """Dataset for child training images."""
    
    def __init__(self, image_paths: List[str], child_id: str, tokenizer, 
                 size: int = 1024, flip_p: float = 0.5):
        self.image_paths = image_paths
        self.child_id = child_id
        self.tokenizer = tokenizer
        self.size = size
        self.flip_p = flip_p
        
        # Create training prompts - these will be used to teach the model about the child
        self.prompts = [
            f"a photo of {child_id}",
            f"portrait of {child_id}",
            f"{child_id} smiling",
            f"close-up of {child_id}",
            f"picture of {child_id}",
        ]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Resize and center crop
            image = self._preprocess_image(image)
            
            # Convert to tensor
            image_tensor = torch.tensor(image).permute(2, 0, 1).float() / 127.5 - 1.0
            
            # Random prompt selection
            prompt = random.choice(self.prompts)
            
            # Tokenize prompt
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            
            return {
                "pixel_values": image_tensor,
                "input_ids": text_inputs.input_ids.flatten(),
                "attention_mask": text_inputs.attention_mask.flatten(),
                "prompt": prompt
            }
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a dummy item if image loading fails
            return self.__getitem__(0 if idx != 0 else 1)
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image to target size."""
        # Calculate resize dimensions to maintain aspect ratio
        w, h = image.size
        target_size = self.size
        
        if w > h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)
        
        # Resize
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Center crop to target size
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        image = image.crop((left, top, left + target_size, top + target_size))
        
        # Random horizontal flip
        if random.random() < self.flip_p:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        return image


class LoRATrainer:
    """LoRA trainer for Stable Diffusion 3.5 Large."""
    
    def __init__(self, config: LoRATrainingConfig):
        self.config = config
        self.device, self.device_description = detect_device()
        self.torch_dtype = get_torch_dtype_for_device(self.device)
        
        # Initialize pipeline and components
        self.pipeline = None
        self.unet = None
        self.text_encoder = None
        self.tokenizer = None
        self.vae = None
        self.scheduler = None
        
        # Training state
        self.is_training = False
        self.progress_callback: Optional[Callable] = None
        
        # Set random seeds for reproducibility
        self._set_seed(config.seed)
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def load_pipeline(self):
        """Load the base Stable Diffusion 3.5 pipeline."""
        if self.pipeline is not None:
            return
        
        logger.info("Loading Stable Diffusion 3.5 Large pipeline for training...")
        
        # Load the pipeline
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=self.torch_dtype,
            variant="fp16" if self.device != "cpu" else None
        )
        
        # Extract components
        self.unet = self.pipeline.transformer
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.vae = self.pipeline.vae
        self.scheduler = self.pipeline.scheduler
        
        # Move to device
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        
        # Set to evaluation mode
        self.vae.eval()
        self.text_encoder.eval()
        
        logger.info(f"Pipeline loaded successfully on {self.device_description}")
    
    def setup_lora(self, target_modules: Optional[List[str]] = None) -> None:
        """Setup LoRA configuration for the UNet."""
        if target_modules is None:
            # Default LoRA target modules for SD3.5 transformer blocks
            target_modules = [
                "to_k", "to_q", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2"
            ]
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            # task_type=TaskType.DIFFUSION,  # Not available in this PEFT version
        )
        
        # Apply LoRA to UNet
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        logger.info("LoRA configuration applied to UNet")
    
    def prepare_dataset(self, child_id: str) -> DataLoader:
        """Prepare training dataset for a child."""
        # Get training images from storage
        training_images = db_manager.get_training_images(child_id)
        
        if len(training_images) < 5:
            raise ValueError(f"Insufficient training images for {child_id}. Need at least 5, got {len(training_images)}")
        
        image_paths = [img.file_path for img in training_images]
        
        # Create dataset
        dataset = ChildImageDataset(
            image_paths=image_paths,
            child_id=child_id,
            tokenizer=self.tokenizer,
            size=1024
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 for compatibility
            pin_memory=True if self.device == "cuda" else False
        )
        
        logger.info(f"Prepared dataset with {len(dataset)} images for {child_id}")
        return dataloader
    
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set progress callback function."""
        self.progress_callback = callback
    
    def _update_progress(self, progress: float, message: str = ""):
        """Update training progress."""
        if self.progress_callback:
            self.progress_callback(progress, message)
    
    def train(self, child_id: str, model_id: int) -> str:
        """
        Train LoRA adapter for a child.
        
        Args:
            child_id: Child identifier
            model_id: Database model ID for tracking
            
        Returns:
            Path to saved LoRA model
        """
        try:
            self.is_training = True
            
            # Update status to training
            db_manager.update_training_status(model_id, TrainingStatus.TRAINING, progress=0.0)
            self._update_progress(0.0, "Initializing training...")
            
            # Load pipeline if not already loaded
            if self.pipeline is None:
                self.load_pipeline()
            
            # Setup LoRA
            self.setup_lora()
            
            # Prepare dataset
            dataloader = self.prepare_dataset(child_id)
            
            # Setup optimizer
            if self.config.use_8bit_adam:
                try:
                    from bitsandbytes.optim import AdamW8bit
                    optimizer = AdamW8bit(
                        self.unet.parameters(),
                        lr=self.config.learning_rate,
                        betas=(0.9, 0.999),
                        weight_decay=0.01,
                        eps=1e-08,
                    )
                except ImportError:
                    logger.warning("8-bit Adam not available, using regular Adam")
                    optimizer = torch.optim.AdamW(
                        self.unet.parameters(),
                        lr=self.config.learning_rate,
                        betas=(0.9, 0.999),
                        weight_decay=0.01,
                        eps=1e-08,
                    )
            else:
                optimizer = torch.optim.AdamW(
                    self.unet.parameters(),
                    lr=self.config.learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                    eps=1e-08,
                )
            
            # Setup mixed precision scaler
            scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision == "fp16" and self.device == "cuda" else None
            
            # Training loop
            self.unet.train()
            step = 0
            total_loss = 0.0
            
            self._update_progress(0.1, "Starting training loop...")
            
            while step < self.config.training_steps:
                for batch in dataloader:
                    if step >= self.config.training_steps:
                        break
                    
                    # Move batch to device
                    pixel_values = batch["pixel_values"].to(self.device, dtype=self.torch_dtype)
                    input_ids = batch["input_ids"].to(self.device)
                    
                    # Encode images with VAE
                    with torch.no_grad():
                        latents = self.vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * self.vae.config.scaling_factor
                    
                    # Add noise to latents
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                    noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                    
                    # Encode text
                    with torch.no_grad():
                        text_embeddings = self.text_encoder(input_ids)[0]
                    
                    # Forward pass
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
                            loss = F.mse_loss(noise_pred.float(), noise.float())
                    else:
                        noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
                        loss = F.mse_loss(noise_pred.float(), noise.float())
                    
                    # Backward pass
                    loss = loss / self.config.gradient_accumulation_steps
                    
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # Update weights
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.config.max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.config.max_grad_norm)
                            optimizer.step()
                        
                        optimizer.zero_grad()
                    
                    total_loss += loss.item()
                    step += 1
                    
                    # Update progress
                    if step % 10 == 0:
                        progress = step / self.config.training_steps
                        avg_loss = total_loss / step
                        message = f"Step {step}/{self.config.training_steps}, Loss: {avg_loss:.4f}"
                        self._update_progress(progress, message)
                        
                        # Update database
                        db_manager.update_training_status(
                            model_id, TrainingStatus.TRAINING, 
                            progress=progress, loss=avg_loss
                        )
                    
                    # Validation and checkpointing
                    if step % self.config.validation_steps == 0:
                        self._validate_model(child_id, step)
                    
                    if step % self.config.save_steps == 0:
                        # Save intermediate checkpoint
                        checkpoint_path = self._save_checkpoint(child_id, step)
                        logger.info(f"Saved checkpoint at step {step}: {checkpoint_path}")
            
            # Save final model
            model_path = self._save_final_model(child_id)
            
            # Update status to completed
            final_loss = total_loss / max(step, 1)
            db_manager.update_training_status(
                model_id, TrainingStatus.COMPLETED, 
                progress=1.0, loss=final_loss
            )
            
            self._update_progress(1.0, f"Training completed! Model saved: {model_path}")
            logger.info(f"LoRA training completed for {child_id}. Model saved: {model_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Training failed for {child_id}: {e}")
            db_manager.update_training_status(
                model_id, TrainingStatus.FAILED, 
                error_message=str(e)
            )
            raise
        finally:
            self.is_training = False
    
    def _validate_model(self, child_id: str, step: int):
        """Generate validation image to check training progress."""
        try:
            self.unet.eval()
            
            # Generate a test image
            test_prompt = f"portrait of {child_id}"
            
            with torch.no_grad():
                # This is a simplified validation - in practice you'd want
                # to generate a full image and save it for inspection
                logger.info(f"Validation at step {step}: {test_prompt}")
            
            self.unet.train()
            
        except Exception as e:
            logger.warning(f"Validation failed at step {step}: {e}")
    
    def _save_checkpoint(self, child_id: str, step: int) -> str:
        """Save training checkpoint."""
        checkpoint_dir = storage_manager.get_child_model_dir(child_id)
        checkpoint_path = checkpoint_dir / f"checkpoint_{step}.safetensors"
        
        # Save LoRA weights
        self.unet.save_pretrained(checkpoint_path.parent, safe_serialization=True)
        
        return str(checkpoint_path)
    
    def _save_final_model(self, child_id: str) -> str:
        """Save final LoRA model."""
        model_path = storage_manager.get_lora_model_path(child_id)
        
        # Save LoRA adapter
        self.unet.save_pretrained(model_path.parent, safe_serialization=True)
        
        return str(model_path)


def create_training_config(**kwargs) -> LoRATrainingConfig:
    """Create training configuration with custom parameters."""
    return LoRATrainingConfig(**kwargs)