#!/usr/bin/env python3
"""
Test LoRA model generation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
import base64
from datetime import datetime

def test_lora_generation():
    """Test generating with the trained LoRA model."""
    
    print("🚀 Testing LoRA Generation")
    print("=" * 30)
    
    # Load base pipeline
    print("📦 Loading SD3.5 Large pipeline...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipeline.to("mps")
    
    # Load LoRA model
    lora_path = "data/lora_models/aman/adapter_model.safetensors"
    if not Path(lora_path).exists():
        print(f"❌ LoRA model not found at {lora_path}")
        return False
    
    print("🎨 Loading LoRA adapter...")
    
    try:
        # Load LoRA into the transformer
        pipeline.transformer = PeftModel.from_pretrained(
            pipeline.transformer,
            "data/lora_models/aman/",
            torch_dtype=torch.float16
        )
        print("✅ LoRA loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load LoRA: {e}")
        return False
    
    # Test generation
    prompts = [
        "a photo of aman",
        "aman smiling",
        "portrait of aman", 
        "aman in natural lighting"
    ]
    
    for prompt in prompts:
        print(f"🎯 Generating: {prompt}")
        
        try:
            image = pipeline(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lora_test_{timestamp}.png"
            image.save(filename)
            print(f"✅ Generated and saved: {filename}")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            continue
    
    return True

if __name__ == "__main__":
    test_lora_generation()