#!/usr/bin/env python3
"""
SD3.5 Large LoRA Training Test Script
Tests the complete LoRA training workflow using images from /Users/antenehnegash/Downloads/aman
"""

import os
import sys
import json
import time
import requests
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lora_training_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LoRATrainingTester:
    """Test class for SD3.5 Large LoRA training workflow"""
    
    def __init__(self, 
                 api_base_url: str = "http://localhost:8000",
                 images_dir: str = "/Users/antenehnegash/Downloads/aman",
                 child_id: str = "aman"):
        self.api_base_url = api_base_url
        self.images_dir = Path(images_dir)
        self.child_id = child_id
        self.session = requests.Session()
        
        # Test configuration
        self.training_config = {
            "training_steps": 500,  # Start with fewer steps for testing
            "learning_rate": 1e-4,
            "lora_rank": 64,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "mixed_precision": "fp16",
            "save_steps": 100,
            "logging_steps": 50
        }
        
    def check_server_health(self) -> bool:
        """Check if the API server is running and healthy"""
        try:
            logger.info("🏥 Checking server health...")
            response = self.session.get(f"{self.api_base_url}/")
            response.raise_for_status()
            
            health_data = response.json()
            logger.info(f"✅ Server is healthy: {health_data.get('status', 'unknown')}")
            logger.info(f"   Device: {health_data.get('device_info', 'unknown')}")
            logger.info(f"   Model Status: {health_data.get('model_status', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Server health check failed: {e}")
            return False
    
    def get_training_images(self) -> List[Path]:
        """Get list of training images"""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        images = [
            img for img in self.images_dir.iterdir() 
            if img.is_file() and img.suffix.lower() in image_extensions
        ]
        
        # Filter out test images if any
        images = [img for img in images if 'test' not in img.name.lower()]
        
        logger.info(f"📸 Found {len(images)} training images")
        for img in images[:5]:  # Show first 5
            logger.info(f"   - {img.name} ({img.stat().st_size / 1024 / 1024:.1f}MB)")
        
        if len(images) > 5:
            logger.info(f"   ... and {len(images) - 5} more images")
        
        return sorted(images)
    
    def create_child(self) -> bool:
        """Create a child for LoRA training"""
        try:
            logger.info(f"👶 Creating child: {self.child_id}")
            
            child_data = {
                "id": self.child_id,
                "name": "Aman",
                "description": "Test child for LoRA training using photo dataset"
            }
            
            response = self.session.post(
                f"{self.api_base_url}/lora/children",
                json=child_data
            )
            
            if response.status_code == 409:
                logger.info("   Child already exists, continuing...")
                return True
            
            response.raise_for_status()
            logger.info("✅ Child created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create child: {e}")
            return False
    
    def upload_training_images(self, images: List[Path], batch_size: int = 5) -> bool:
        """Upload training images in batches"""
        try:
            logger.info(f"📤 Uploading {len(images)} training images...")
            
            # Upload in batches to avoid overwhelming the server
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                logger.info(f"   Uploading batch {i//batch_size + 1}/{(len(images)-1)//batch_size + 1}")
                
                files = []
                try:
                    for img_path in batch:
                        files.append(('files', (img_path.name, open(img_path, 'rb'), 'image/jpeg')))
                    
                    response = self.session.post(
                        f"{self.api_base_url}/lora/children/{self.child_id}/images",
                        files=files
                    )
                    response.raise_for_status()
                    
                finally:
                    # Close all file handles
                    for _, (_, file_handle, _) in files:
                        file_handle.close()
                
                logger.info(f"   ✅ Batch {i//batch_size + 1} uploaded successfully")
            
            logger.info("✅ All training images uploaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload training images: {e}")
            return False
    
    def start_training(self) -> str:
        """Start LoRA training and return task ID"""
        try:
            logger.info("🏋️ Starting LoRA training...")
            logger.info(f"   Training config: {json.dumps(self.training_config, indent=2)}")
            
            response = self.session.post(
                f"{self.api_base_url}/lora/children/{self.child_id}/train",
                json={"training_config": self.training_config}
            )
            response.raise_for_status()
            
            result = response.json()
            task_id = result.get('task_id')
            
            logger.info(f"✅ Training started with task ID: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start training: {e}")
            return ""
    
    def monitor_training(self, task_id: str, timeout: int = 3600) -> bool:
        """Monitor training progress"""
        try:
            logger.info(f"👀 Monitoring training progress (timeout: {timeout}s)...")
            
            start_time = time.time()
            last_status = None
            
            while time.time() - start_time < timeout:
                try:
                    # Check training status
                    response = self.session.get(
                        f"{self.api_base_url}/lora/children/{self.child_id}/training-status"
                    )
                    response.raise_for_status()
                    status_data = response.json()
                    
                    current_status = status_data.get('status', 'unknown')
                    
                    # Log status changes
                    if current_status != last_status:
                        logger.info(f"   Status: {current_status}")
                        if 'progress' in status_data:
                            progress = status_data['progress']
                            logger.info(f"   Progress: {progress}")
                        last_status = current_status
                    
                    # Check if training completed
                    if current_status == 'COMPLETED':
                        logger.info("✅ Training completed successfully!")
                        return True
                    elif current_status == 'FAILED':
                        error_msg = status_data.get('error', 'Unknown error')
                        logger.error(f"❌ Training failed: {error_msg}")
                        return False
                    
                    # Also check Celery task status if available
                    if task_id:
                        try:
                            task_response = self.session.get(
                                f"{self.api_base_url}/training-tasks/{task_id}/status"
                            )
                            if task_response.status_code == 200:
                                task_data = task_response.json()
                                task_status = task_data.get('status', 'unknown')
                                
                                if task_status != current_status:
                                    logger.info(f"   Celery Task Status: {task_status}")
                                    if 'result' in task_data:
                                        logger.info(f"   Task Result: {task_data['result']}")
                        except:
                            pass  # Task endpoint might not be available
                    
                except requests.RequestException as e:
                    logger.warning(f"   Failed to check status: {e}")
                
                # Wait before next check
                time.sleep(30)  # Check every 30 seconds
            
            logger.warning(f"⏰ Training monitoring timed out after {timeout}s")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error monitoring training: {e}")
            return False
    
    def test_generation_with_lora(self) -> bool:
        """Test image generation with trained LoRA"""
        try:
            logger.info("🎨 Testing image generation with trained LoRA...")
            
            test_prompts = [
                f"a portrait of {{{self.child_id}}} smiling",
                f"a photo of {{{self.child_id}}} in a park",
                f"{{{self.child_id}}} wearing casual clothes",
                f"a headshot of {{{self.child_id}}} with natural lighting"
            ]
            
            for i, prompt in enumerate(test_prompts):
                logger.info(f"   Generating image {i+1}/{len(test_prompts)}: {prompt}")
                
                response = self.session.post(
                    f"{self.api_base_url}/generate/with-children",
                    json={
                        "prompt": prompt,
                        "steps": 20,
                        "guidance": 7.5,
                        "width": 512,
                        "height": 512
                    }
                )
                
                if response.status_code == 200:
                    logger.info(f"   ✅ Generated image {i+1} successfully")
                    
                    # Save the image
                    output_path = f"generated_test_{self.child_id}_{i+1}.jpg"
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"   💾 Saved to: {output_path}")
                else:
                    logger.error(f"   ❌ Failed to generate image {i+1}: {response.status_code}")
                    logger.error(f"   Response: {response.text}")
            
            logger.info("✅ Generation testing completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to test generation: {e}")
            return False
    
    def get_child_statistics(self) -> Dict[str, Any]:
        """Get training statistics for the child"""
        try:
            response = self.session.get(
                f"{self.api_base_url}/lora/children/{self.child_id}/statistics"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def run_full_test(self) -> bool:
        """Run the complete LoRA training test workflow"""
        logger.info("🚀 Starting complete LoRA training test workflow")
        logger.info("=" * 60)
        
        # Step 1: Check server health
        if not self.check_server_health():
            logger.error("❌ Server health check failed. Make sure the API server is running.")
            return False
        
        # Step 2: Get training images
        try:
            images = self.get_training_images()
            if len(images) < 3:
                logger.error(f"❌ Insufficient training images: {len(images)} (need at least 3)")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to get training images: {e}")
            return False
        
        # Step 3: Create child
        if not self.create_child():
            return False
        
        # Step 4: Upload training images
        if not self.upload_training_images(images):
            return False
        
        # Step 5: Start training
        task_id = self.start_training()
        if not task_id:
            return False
        
        # Step 6: Monitor training
        training_success = self.monitor_training(task_id, timeout=1800)  # 30 minutes timeout
        
        # Step 7: Get statistics
        stats = self.get_child_statistics()
        if stats:
            logger.info("📊 Training Statistics:")
            logger.info(f"   {json.dumps(stats, indent=2)}")
        
        # Step 8: Test generation (even if training failed, to debug)
        if training_success:
            self.test_generation_with_lora()
        else:
            logger.warning("⚠️ Training failed, skipping generation test")
        
        logger.info("=" * 60)
        if training_success:
            logger.info("🎉 COMPLETE LORA TRAINING TEST PASSED! 🎉")
        else:
            logger.error("❌ LORA TRAINING TEST FAILED")
        
        return training_success


def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SD3.5 Large LoRA training")
    parser.add_argument("--api-url", default="http://localhost:8000", 
                       help="API server URL")
    parser.add_argument("--images-dir", default="/Users/antenehnegash/Downloads/aman",
                       help="Directory containing training images")  
    parser.add_argument("--child-id", default="aman",
                       help="Child ID for training")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with fewer training steps")
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = LoRATrainingTester(
        api_base_url=args.api_url,
        images_dir=args.images_dir,
        child_id=args.child_id
    )
    
    # Adjust config for quick test
    if args.quick:
        logger.info("🏃‍♂️ Running quick test mode")
        tester.training_config["training_steps"] = 100
        tester.training_config["save_steps"] = 50
    
    # Run the test
    success = tester.run_full_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()