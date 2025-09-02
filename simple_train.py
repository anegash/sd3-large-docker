#!/usr/bin/env python3
"""
Simplified LoRA training that bypasses database issues.
Directly uses files from disk.
"""

import os
import sys
import random
from pathlib import Path
from typing import List

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model

from sd3_api.device import detect_device, get_torch_dtype_for_device
from sd3_api.config import MODEL_ID
from diffusers import StableDiffusion3Pipeline

class SimpleChildDataset(Dataset):
    """Simple dataset that loads images from disk."""
    
    def __init__(self, image_paths: List[str], child_id: str, tokenizer, size: int = 1024):
        self.image_paths = image_paths
        self.child_id = child_id
        self.tokenizer = tokenizer
        self.size = size
        
        # Create training prompts
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
            
            # Convert to tensor
            import numpy as np
            image_array = np.array(image)
            image_tensor = torch.tensor(image_array).permute(2, 0, 1).float() / 127.5 - 1.0
            
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
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy item if image loading fails
            return self.__getitem__(0 if idx != 0 else 1)

def main():
    """Run simplified LoRA training."""
    child_id = "aman"
    
    print("🚀 Simple LoRA Training for Aman")
    print("=" * 40)
    
    # Find training images directly
    training_dir = Path("data/children/aman/training_images")
    if not training_dir.exists():
        print(f"❌ Training directory not found: {training_dir}")
        return False
    
    image_paths = list(training_dir.glob("*.jpeg"))
    if len(image_paths) < 5:
        print(f"❌ Need at least 5 images, found {len(image_paths)}")
        return False
    
    print(f"✅ Found {len(image_paths)} training images")
    
    # Detect device
    device, device_description = detect_device()
    torch_dtype = get_torch_dtype_for_device(device)
    print(f"🖥️  Using device: {device_description}")
    
    # Load pipeline with consistent dtype handling
    print("📦 Loading SD3.5 Large pipeline...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        variant="fp16" if device != "cpu" else None
    )
    pipeline.to(device, dtype=torch_dtype)
    
    # Extract components
    unet = pipeline.transformer
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    vae = pipeline.vae
    scheduler = pipeline.scheduler
    
    # Set to eval mode
    vae.eval()
    text_encoder.eval()
    
    print("🔧 Setting up LoRA...")
    
    # Configure LoRA with simpler target modules
    target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Create dataset and dataloader
    print("📊 Creating dataset...")
    dataset = SimpleChildDataset(
        [str(p) for p in image_paths],
        child_id,
        tokenizer,
        size=512  # Smaller size for faster training
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    
    # Setup optimizer with more conservative settings for SD3.5
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=1e-5,  # Lower learning rate for stability
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # Add learning rate scheduler for better convergence
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=1e-7)
    
    # Training parameters - reduced for testing the loss function
    num_train_steps = 50  # Much smaller for testing
    gradient_accumulation_steps = 2
    
    print(f"🎯 Starting training for {num_train_steps} steps...")
    
    unet.train()
    step = 0
    total_loss = 0.0
    
    # Simple training loop
    while step < num_train_steps:
        for batch in dataloader:
            if step >= num_train_steps:
                break
            
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device, dtype=torch_dtype)
            input_ids = batch["input_ids"].to(device)
            prompt = batch["prompt"][0]  # Get the prompt string
            
            # Encode images with VAE - ensure consistent dtype
            with torch.no_grad():
                # Ensure pixel_values match VAE dtype
                pixel_values = pixel_values.to(dtype=torch_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=torch_dtype, device=device)
            
            # Generate noise matching latents exactly
            noise = torch.randn_like(latents, dtype=torch_dtype, device=device)
            
            # Sample timesteps with logit-normal distribution (SD3 approach)
            # This focuses training on middle timesteps where learning is most effective
            # Avoid extreme timesteps (0, 1) that cause numerical instability
            u = torch.rand((latents.shape[0],), device=device, dtype=torch_dtype)
            # Transform uniform to logit-normal: more weight on middle values
            timesteps = torch.sigmoid(torch.logit(u * 0.998 + 0.001))
            
            # SD3 FlowMatch: x_t = (1-t) * x_0 + t * x_1, where x_1 is noise
            # This creates a straight path from data to noise
            timesteps_broadcast = timesteps.view(-1, 1, 1, 1)
            noisy_latents = (1.0 - timesteps_broadcast) * latents + timesteps_broadcast * noise
            
            # Encode text with all 3 text encoders
            with torch.no_grad():
                prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt,  
                    prompt_3=prompt,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False
                )
            
            # Forward pass - ensure all inputs have consistent dtype
            model_pred = unet(
                hidden_states=noisy_latents.to(dtype=torch_dtype),
                timestep=timesteps.to(dtype=torch_dtype),
                encoder_hidden_states=prompt_embeds.to(dtype=torch_dtype),
                pooled_projections=pooled_prompt_embeds.to(dtype=torch_dtype),
                return_dict=False
            )[0]
            
            # SD3 Rectified Flow target: velocity field v_t = x_1 - x_0
            # Where x_0 is the data (latents) and x_1 is the noise
            # The model learns to predict the velocity from data to noise
            target = noise - latents
            
            # Calculate MSE loss with SD3's reweighting strategy
            # Weight middle timesteps more heavily as they're more informative
            timestep_weights = 1.0 / (timesteps + 0.1)  # Higher weight for middle timesteps
            
            # Calculate element-wise loss
            loss_elements = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction='none')
            
            # Apply timestep weighting and reduce to scalar
            loss = (loss_elements.mean(dim=[1, 2, 3]) * timestep_weights).mean()
            
            # Add comprehensive stability checks
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss {loss.item()}, skipping step")
                continue
                
            # Check for extreme loss values that indicate training instability
            if loss.item() > 10.0:
                print(f"Warning: Very high loss {loss.item():.4f}, skipping step")
                continue
                
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights with conservative gradient clipping
            if (step + 1) % gradient_accumulation_steps == 0:
                # More aggressive gradient clipping for stability
                grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=0.5)
                
                # Skip update if gradients are too large
                if grad_norm > 10.0:
                    print(f"Warning: Large gradient norm {grad_norm:.2f}, skipping update")
                    optimizer.zero_grad()
                    continue
                
                optimizer.step()
                scheduler.step()  # Update learning rate
                optimizer.zero_grad()
            
            total_loss += loss.item()
            step += 1
            
            # Show progress - more frequent updates for shorter training
            if step % 5 == 0 or step == 1:
                current_loss = loss.item() * gradient_accumulation_steps  # Unscale for display
                avg_loss = total_loss / step if step > 0 else current_loss
                progress = step / num_train_steps
                percent = int(progress * 100)
                bar = "█" * (percent // 5) + "░" * (20 - percent // 5)
                lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                print(f"\r🎯 {bar} {percent}% Step {step}/{num_train_steps}, Loss: {current_loss:.4f}, Avg: {avg_loss:.4f}, LR: {lr:.2e}", end="", flush=True)
    
    print(f"\n\n✅ Training completed!")
    
    # Save LoRA model
    save_dir = Path("data/lora_models/aman")
    save_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(save_dir, safe_serialization=True)
    
    model_path = save_dir / "adapter_model.safetensors"
    print(f"💾 LoRA model saved to: {model_path}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("🎉 Training successful! Ready to generate images.")
        else:
            print("❌ Training failed.")
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()