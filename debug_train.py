#!/usr/bin/env python3
"""
Debug version of LoRA training to identify NaN source.
"""

import os
import sys
import random
import math
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
    
    def __init__(self, image_paths: List[str], child_id: str, tokenizer, size: int = 512):
        self.image_paths = image_paths
        self.child_id = child_id
        self.tokenizer = tokenizer
        self.size = size
        
        # Create training prompts
        self.prompts = [
            f"a photo of {child_id}",
            f"portrait of {child_id}",
        ]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Simple resize without cropping
            image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
            
            # Convert to tensor
            import numpy as np
            image_array = np.array(image)
            image_tensor = torch.tensor(image_array).permute(2, 0, 1).float() / 127.5 - 1.0
            
            # Simple prompt
            prompt = f"a photo of {self.child_id}"
            
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
                "prompt": prompt
            }
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__(0 if idx != 0 else 1)

def debug_tensor(tensor, name):
    """Debug tensor for NaN/Inf values."""
    if torch.isnan(tensor).any():
        print(f"❌ {name} contains NaN!")
        return False
    if torch.isinf(tensor).any():
        print(f"❌ {name} contains Inf!")
        return False
    print(f"✅ {name}: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")
    return True

def main():
    """Run debug LoRA training."""
    child_id = "aman"
    
    print("🔍 Debug LoRA Training")
    print("=" * 25)
    
    # Find training images
    training_dir = Path("data/children/aman/training_images")
    if not training_dir.exists():
        print(f"❌ Training directory not found: {training_dir}")
        return False
    
    image_paths = list(training_dir.glob("*.jpeg"))[:5]  # Only 5 images for debugging
    print(f"✅ Using {len(image_paths)} training images for debug")
    
    # Detect device
    device, device_description = detect_device()
    torch_dtype = get_torch_dtype_for_device(device)
    print(f"🖥️  Using device: {device_description}")
    
    # Force CPU for debugging to avoid MPS issues
    if device == "mps":
        print("🔧 Forcing CPU mode for debugging...")
        device = "cpu"
        torch_dtype = torch.float32
    
    # Load pipeline
    print("📦 Loading SD3.5 Large pipeline...")
    try:
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            variant="fp16" if torch_dtype == torch.float16 else None
        )
        pipeline.to(device)
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        return False
    
    # Extract components
    unet = pipeline.transformer
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    vae = pipeline.vae
    
    # Set to eval mode
    vae.eval()
    text_encoder.eval()
    
    print("🔧 Setting up LoRA...")
    
    # Very simple LoRA config for debugging
    target_modules = ["to_q", "to_v"]  # Only attention modules
    
    lora_config = LoraConfig(
        r=8,  # Much smaller rank
        lora_alpha=4,
        target_modules=target_modules,
        lora_dropout=0.0,  # No dropout
        bias="none",
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Create dataset
    dataset = SimpleChildDataset(
        [str(p) for p in image_paths],
        child_id,
        tokenizer,
        size=256  # Smaller images
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,  # No shuffling for debugging
        num_workers=0,
    )
    
    # Very conservative optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=5e-6,  # Very low learning rate
        betas=(0.9, 0.999),
        weight_decay=0.0,
        eps=1e-8,
    )
    
    print("🎯 Starting debug training...")
    
    unet.train()
    
    # Test single forward/backward pass
    for step, batch in enumerate(dataloader):
        if step >= 3:  # Only test 3 steps
            break
            
        print(f"\n--- Step {step + 1} ---")
        
        # Move batch to device
        pixel_values = batch["pixel_values"].to(device, dtype=torch_dtype)
        prompt = batch["prompt"][0]
        
        print(f"📝 Prompt: {prompt}")
        
        # Debug input
        if not debug_tensor(pixel_values, "pixel_values"):
            break
        
        # Encode images with VAE
        print("🎨 Encoding with VAE...")
        with torch.no_grad():
            try:
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                if not debug_tensor(latents, "latents"):
                    break
            except Exception as e:
                print(f"❌ VAE encoding failed: {e}")
                break
        
        # Generate noise
        noise = torch.randn_like(latents, dtype=torch_dtype, device=device)
        if not debug_tensor(noise, "noise"):
            break
        
        # Simple timestep - avoid extremes
        timesteps = torch.tensor([0.5], device=device, dtype=torch_dtype).expand(latents.shape[0])
        if not debug_tensor(timesteps, "timesteps"):
            break
        
        # Linear interpolation (rectified flow)
        timesteps_broadcast = timesteps.view(-1, 1, 1, 1)
        noisy_latents = (1.0 - timesteps_broadcast) * latents + timesteps_broadcast * noise
        if not debug_tensor(noisy_latents, "noisy_latents"):
            break
        
        # Encode text
        print("📝 Encoding text...")
        with torch.no_grad():
            try:
                prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt,
                    prompt_3=prompt,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False
                )
                if not debug_tensor(prompt_embeds, "prompt_embeds"):
                    break
                if not debug_tensor(pooled_prompt_embeds, "pooled_prompt_embeds"):
                    break
            except Exception as e:
                print(f"❌ Text encoding failed: {e}")
                break
        
        # Forward pass
        print("🔮 UNet forward pass...")
        try:
            model_pred = unet(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False
            )[0]
            if not debug_tensor(model_pred, "model_pred"):
                break
        except Exception as e:
            print(f"❌ UNet forward pass failed: {e}")
            break
        
        # Calculate target
        target = noise - latents
        if not debug_tensor(target, "target"):
            break
        
        # Calculate loss
        print("📊 Calculating loss...")
        try:
            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction='mean')
            print(f"📊 Loss: {loss.item():.6f}")
            
            if not torch.isfinite(loss):
                print(f"❌ Loss is not finite: {loss.item()}")
                break
                
        except Exception as e:
            print(f"❌ Loss calculation failed: {e}")
            break
        
        # Backward pass
        print("⬅️  Backward pass...")
        optimizer.zero_grad()
        try:
            loss.backward()
            
            # Check gradients
            total_norm = 0.0
            nan_grads = 0
            for param in unet.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    if torch.isnan(param.grad).any():
                        nan_grads += 1
            
            total_norm = total_norm ** (1. / 2)
            print(f"📊 Gradient norm: {total_norm:.6f}")
            
            if nan_grads > 0:
                print(f"❌ Found {nan_grads} parameters with NaN gradients!")
                break
                
            if total_norm > 10.0:
                print(f"⚠️ Large gradient norm: {total_norm:.2f}")
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=0.1)
            
            # Optimizer step
            optimizer.step()
            print(f"✅ Step {step + 1} completed successfully")
            
        except Exception as e:
            print(f"❌ Backward pass failed: {e}")
            break
    
    print("\n🎉 Debug training completed!")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("✅ Debug completed successfully")
        else:
            print("❌ Debug failed")
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()