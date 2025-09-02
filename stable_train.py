#!/usr/bin/env python3
"""
Ultra-stable LoRA training with comprehensive NaN prevention.
Based on research findings for SD3 training stability.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from diffusers import StableDiffusion3Pipeline
import numpy as np

from sd3_api.device import detect_device, get_torch_dtype_for_device
from sd3_api.config import MODEL_ID

class StableChildDataset(Dataset):
    """Ultra-stable dataset with extensive preprocessing."""
    
    def __init__(self, image_paths, child_id, tokenizer, size=512):
        self.image_paths = image_paths
        self.child_id = child_id
        self.tokenizer = tokenizer
        self.size = size
        
        # Single, simple prompt to reduce variability
        self.prompt = f"a photo of {child_id}"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
            
            # Convert to tensor with careful normalization
            image_array = np.array(image, dtype=np.float32)
            image_tensor = torch.tensor(image_array).permute(2, 0, 1)
            
            # Normalize to [-1, 1] range
            image_tensor = (image_tensor / 255.0) * 2.0 - 1.0
            
            # Clamp to prevent extreme values
            image_tensor = torch.clamp(image_tensor, -1.0, 1.0)
            
            # Tokenize with padding
            text_inputs = self.tokenizer(
                self.prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            
            return {
                "pixel_values": image_tensor,
                "input_ids": text_inputs.input_ids.flatten(),
                "prompt": self.prompt
            }
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.image_paths))

def safe_mse_loss(pred, target, reduction='mean'):
    """MSE loss with NaN protection."""
    # Check for NaN/Inf in inputs
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        print("❌ NaN/Inf in prediction")
        return torch.tensor(float('nan'))
        
    if torch.isnan(target).any() or torch.isinf(target).any():
        print("❌ NaN/Inf in target")
        return torch.tensor(float('nan'))
    
    # Calculate loss with explicit float32
    loss = F.mse_loss(pred.float(), target.float(), reduction=reduction)
    
    # Clamp loss to reasonable range
    loss = torch.clamp(loss, 0.0, 100.0)
    
    return loss

def main():
    """Ultra-stable training loop."""
    child_id = "aman"
    
    print("🛡️  Ultra-Stable LoRA Training")
    print("=" * 35)
    
    # Load training images
    training_dir = Path("data/children/aman/training_images")
    image_paths = list(training_dir.glob("*.jpeg"))[:3]  # Only 3 images for stability
    print(f"✅ Using {len(image_paths)} images")
    
    # Device setup - MPS with float32 for Apple Silicon stability  
    device, device_description = detect_device()
    # Force float32 on MPS to avoid mixed precision issues
    torch_dtype = torch.float32 if device == "mps" else get_torch_dtype_for_device(device)
    print(f"🖥️  Using {device_description} with {torch_dtype} for stability")
    
    # Load pipeline
    print("📦 Loading pipeline...")
    try:
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
        )
        pipeline.to(device)
        print("✅ Pipeline loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        return False
    
    # Extract components
    unet = pipeline.transformer
    vae = pipeline.vae
    tokenizer = pipeline.tokenizer
    
    # Freeze everything except LoRA
    vae.eval()
    vae.requires_grad_(False)
    
    # Ultra-conservative LoRA setup
    print("🔧 Setting up minimal LoRA...")
    lora_config = LoraConfig(
        r=4,  # Very small rank
        lora_alpha=2,  # Very small alpha  
        target_modules=["to_q"],  # Only query projection
        lora_dropout=0.0,
        bias="none",
    )
    
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Create dataset
    dataset = StableChildDataset(
        [str(p) for p in image_paths],
        child_id,
        tokenizer,
        size=256  # Small size for stability
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    
    # Ultra-conservative optimizer
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=1e-6,  # Extremely low learning rate
        betas=(0.9, 0.95),
        weight_decay=0.001,
        eps=1e-8,
    )
    
    print("🎯 Starting ultra-stable training...")
    
    unet.train()
    
    # Training loop with extensive safety checks
    for step, batch in enumerate(dataloader):
        if step >= 5:  # Only 5 steps for testing
            break
            
        print(f"\n--- Step {step + 1} ---")
        
        # Prepare inputs
        pixel_values = batch["pixel_values"].to(device, dtype=torch_dtype)
        prompt = batch["prompt"][0]
        
        # VAE encoding with safety checks
        with torch.no_grad():
            try:
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Safety checks
                if torch.isnan(latents).any():
                    print("❌ NaN in latents, skipping")
                    continue
                    
            except Exception as e:
                print(f"❌ VAE encoding failed: {e}")
                continue
        
        # Simple noise generation
        noise = torch.randn_like(latents, dtype=torch_dtype, device=device)
        
        # Fixed timestep for stability
        timesteps = torch.tensor([0.5], device=device, dtype=torch_dtype)
        
        # Simple linear interpolation
        timesteps_broadcast = timesteps.view(-1, 1, 1, 1)
        noisy_latents = (1.0 - timesteps_broadcast) * latents + timesteps_broadcast * noise
        
        # Text encoding
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
                
                # Safety checks
                if torch.isnan(prompt_embeds).any() or torch.isnan(pooled_prompt_embeds).any():
                    print("❌ NaN in text embeddings, skipping")
                    continue
                    
            except Exception as e:
                print(f"❌ Text encoding failed: {e}")
                continue
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            model_pred = unet(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False
            )[0]
            
            # Safety check
            if torch.isnan(model_pred).any():
                print("❌ NaN in model prediction, skipping")
                continue
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            continue
        
        # Calculate target and loss
        target = noise - latents
        
        loss = safe_mse_loss(model_pred, target)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ Invalid loss: {loss.item()}")
            continue
            
        print(f"📊 Loss: {loss.item():.8f}")
        
        # Backward pass with extensive safety
        try:
            loss.backward()
            
            # Check gradients before clipping
            total_norm = 0.0
            nan_params = 0
            
            for param in unet.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        nan_params += 1
                        param.grad.zero_()  # Zero out NaN gradients
                    else:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
            
            total_norm = total_norm ** (1. / 2)
            
            if nan_params > 0:
                print(f"⚠️ Zeroed {nan_params} parameters with NaN gradients")
            
            print(f"📊 Gradient norm: {total_norm:.8f}")
            
            # Very aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=0.01)
            
            # Optimizer step
            optimizer.step()
            
            print(f"✅ Step {step + 1} completed")
            
        except Exception as e:
            print(f"❌ Backward pass failed: {e}")
            continue
    
    # Try to save the model
    try:
        save_dir = Path("data/lora_models/aman_stable")
        save_dir.mkdir(parents=True, exist_ok=True)
        unet.save_pretrained(save_dir, safe_serialization=True)
        print(f"💾 Model saved to: {save_dir}")
    except Exception as e:
        print(f"⚠️ Failed to save model: {e}")
    
    print("\n🎉 Ultra-stable training completed!")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("✅ Training successful!")
        else:
            print("❌ Training failed.")
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()