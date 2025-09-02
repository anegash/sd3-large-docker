#!/usr/bin/env python3
"""
Simple test script to start LoRA training directly.
This bypasses the API and database issues for testing.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sd3_api.lora.trainer import LoRATrainer, create_training_config
from sd3_api.utils.storage import storage_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run LoRA training test."""
    child_id = "aman"
    
    print("🚀 Starting LoRA Training Test for Aman")
    print("=" * 50)
    
    # Check training images exist
    training_dir = storage_manager.get_child_training_dir(child_id)
    image_files = list(training_dir.glob("*.jpeg"))
    
    if not image_files:
        print(f"❌ No training images found in {training_dir}")
        return False
    
    print(f"✅ Found {len(image_files)} training images")
    for img in image_files[:3]:  # Show first 3
        print(f"   📷 {img.name}")
    if len(image_files) > 3:
        print(f"   ... and {len(image_files) - 3} more")
    
    # Create training config optimized for Mac
    config = create_training_config(
        training_steps=400,  # Reduced for testing
        batch_size=1,
        gradient_accumulation_steps=2,
        lora_rank=32,
        lora_alpha=16,
        learning_rate=5e-5,
        save_steps=100,
        validation_steps=50,
        seed=42
    )
    
    print(f"\n📋 Training Configuration:")
    print(f"   Steps: {config.training_steps}")
    print(f"   LoRA Rank: {config.lora_rank}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Batch Size: {config.batch_size}")
    
    # Create trainer
    print(f"\n🔧 Initializing LoRA Trainer...")
    trainer = LoRATrainer(config)
    
    # Set up progress callback
    def progress_callback(progress: float, message: str):
        percent = int(progress * 100)
        bar = "█" * (percent // 5) + "░" * (20 - percent // 5)
        print(f"\r🎯 Training: {bar} {percent}% - {message}", end="", flush=True)
    
    trainer.set_progress_callback(progress_callback)
    
    # Get model save path
    model_path = storage_manager.get_lora_model_path(child_id)
    print(f"\n💾 Model will be saved to: {model_path}")
    
    try:
        print(f"\n🎯 Starting Training...")
        print("This may take 15-30 minutes on Mac...")
        
        # Start training (use model_id = 0 for test)
        result_path = trainer.train(child_id, model_id=0)
        
        print(f"\n\n🎉 Training Completed!")
        print(f"✅ LoRA model saved: {result_path}")
        
        return True
        
    except Exception as e:
        print(f"\n\n❌ Training Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎊 Ready to generate images with the trained LoRA!")
    else:
        print("\n💥 Training failed. Check the error messages above.")