#!/usr/bin/env python3
"""
Fresh LoRA Training Test - RunPod Internal Version
Creates a new child and tests LoRA training from images in /workspace
"""

import requests
import json
import time
import logging
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreshLoRATestRunPod:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.session = requests.Session()
        # Create a unique child ID to avoid conflicts
        self.child_id = f"aman_{uuid.uuid4().hex[:8]}"
        
    def cleanup_old_child(self, child_id):
        """Try to delete old child if it exists"""
        try:
            logger.info(f"🧹 Cleaning up old child: {child_id}")
            response = self.session.delete(f"{self.api_url}/lora/children/{child_id}")
            if response.status_code == 200:
                logger.info("✅ Old child deleted successfully")
            else:
                logger.info(f"   Child doesn't exist or already cleaned up")
        except Exception as e:
            logger.info(f"   Cleanup not needed: {e}")
    
    def test_health(self):
        """Test API health"""
        logger.info("🏥 Testing API health...")
        try:
            response = self.session.get(f"{self.api_url}/")
            response.raise_for_status()
            data = response.json()
            logger.info(f"✅ API healthy: {data.get('message', 'OK')}")
            logger.info(f"   Device: {data.get('device', 'Unknown')}")
            return True
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False
    
    def create_fresh_child(self):
        """Create a fresh child with unique ID"""
        logger.info(f"👶 Creating fresh child: {self.child_id}")
        try:
            child_data = {
                "id": self.child_id,
                "name": f"Aman Fresh Test {self.child_id[-8:]}",
                "description": "Fresh child for LoRA training test - no conflicts"
            }
            
            response = self.session.post(
                f"{self.api_url}/lora/children",
                json=child_data
            )
            response.raise_for_status()
            logger.info("✅ Fresh child created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create fresh child: {e}")
            if hasattr(e, 'response'):
                logger.error(f"Response: {e.response.text}")
            return False
    
    def upload_images(self):
        """Upload training images from /workspace"""
        logger.info("📤 Uploading training images...")
        
        # Check for images in /workspace/training_images
        images_dir = Path("/workspace/training_images")
        if not images_dir.exists():
            logger.error(f"❌ Images directory not found: {images_dir}")
            logger.error("   Run: scp -r /Users/antenehnegash/Downloads/aman root@<RUNPOD_IP>:/workspace/training_images")
            return False
        
        # Get first 8 images for testing (good number for LoRA)
        images = list(images_dir.glob("*.jpeg"))[:8]
        if not images:
            images = list(images_dir.glob("*.jpg"))[:8]
        
        if len(images) < 3:
            logger.error(f"❌ Need at least 3 images, found {len(images)}")
            return False
        
        logger.info(f"   Uploading {len(images)} images...")
        
        try:
            files = []
            for img_path in images:
                with open(img_path, 'rb') as f:
                    files.append(('files', (img_path.name, f.read(), 'image/jpeg')))
            
            response = self.session.post(
                f"{self.api_url}/lora/children/{self.child_id}/images",
                files=files
            )
            response.raise_for_status()
            
            logger.info("✅ Images uploaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload images: {e}")
            if hasattr(e, 'response'):
                logger.error(f"Response: {e.response.text}")
            return False
    
    def start_minimal_training(self):
        """Start LoRA training with minimal configuration"""
        logger.info("🏋️ Starting minimal LoRA training...")
        
        # Very conservative training config
        training_config = {
            "training_steps": 100,    # Minimum allowed
            "learning_rate": 5e-5,    # Conservative learning rate
            "lora_rank": 16,          # Small rank to reduce memory
            "lora_alpha": 16,         # Match rank for stability
            "lora_dropout": 0.05,     # Low dropout
            "batch_size": 1,          # Single batch
            "gradient_accumulation_steps": 2,  # Minimal accumulation
            "mixed_precision": "fp16",
            "save_steps": 50,         # Save frequently
            "validation_steps": 50,
            "max_grad_norm": 0.5,     # Conservative gradient clipping
            "use_8bit_adam": True,    # Use 8-bit to save memory
            "seed": 42
        }
        
        logger.info(f"   Config: {json.dumps(training_config, indent=2)}")
        
        try:
            response = self.session.post(
                f"{self.api_url}/lora/children/{self.child_id}/train",
                json={"training_config": training_config}
            )
            response.raise_for_status()
            
            result = response.json()
            task_id = result.get('task_id')
            
            logger.info(f"✅ Training started successfully!")
            logger.info(f"   Task ID: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start training: {e}")
            if hasattr(e, 'response'):
                logger.error(f"Response: {e.response.text}")
            return None
    
    def monitor_training_detailed(self, task_id, timeout=1200):  # 20 minutes
        """Monitor training with detailed logging"""
        logger.info(f"👀 Monitoring training progress (timeout: {timeout}s)...")
        
        start_time = time.time()
        last_status = None
        last_progress = None
        check_count = 0
        
        while time.time() - start_time < timeout:
            check_count += 1
            try:
                # Check training status
                response = self.session.get(
                    f"{self.api_url}/lora/children/{self.child_id}/training-status"
                )
                response.raise_for_status()
                status_data = response.json()
                
                current_status = status_data.get('status', 'unknown')
                current_progress = status_data.get('progress', 0)
                
                # Log status changes or progress updates
                if (current_status != last_status or 
                    current_progress != last_progress or 
                    check_count % 6 == 0):  # Log every 3 minutes regardless
                    
                    logger.info(f"   Check #{check_count}: Status={current_status}, Progress={current_progress:.2f}")
                    
                    if 'message' in status_data:
                        logger.info(f"   Message: {status_data['message']}")
                    
                    if 'error' in status_data and status_data['error']:
                        logger.error(f"   Error: {status_data['error']}")
                    
                    last_status = current_status
                    last_progress = current_progress
                
                # Check completion states
                if current_status == 'COMPLETED':
                    logger.info("🎉 Training completed successfully!")
                    return True
                elif current_status == 'FAILED':
                    logger.error(f"❌ Training failed!")
                    logger.error(f"   Error: {status_data.get('error', 'Unknown error')}")
                    return False
                elif 'ERROR' in current_status.upper():
                    logger.error(f"❌ Training in error state: {current_status}")
                    return False
                
                # Check Celery task if available
                if task_id and check_count % 3 == 0:  # Check every 90 seconds
                    try:
                        task_response = self.session.get(
                            f"{self.api_url}/lora/training-tasks/{task_id}/status"
                        )
                        if task_response.status_code == 200:
                            task_data = task_response.json()
                            task_status = task_data.get('status', 'unknown')
                            logger.info(f"   Celery Task: {task_status}")
                            
                            if 'result' in task_data and task_data['result']:
                                logger.info(f"   Task Result: {task_data['result']}")
                    except:
                        pass
                
            except Exception as e:
                logger.warning(f"   Failed to check status (attempt {check_count}): {e}")
            
            # Wait 30 seconds between checks
            time.sleep(30)
        
        logger.warning(f"⏰ Training monitoring timed out after {timeout}s")
        return False
    
    def test_generation(self):
        """Test generation with trained LoRA"""
        logger.info("🎨 Testing generation with trained LoRA...")
        
        test_prompt = f"a portrait photo of {{{self.child_id}}}, high quality, detailed"
        
        try:
            response = self.session.post(
                f"{self.api_url}/generate/with-children",
                json={
                    "prompt": test_prompt,
                    "steps": 20,
                    "guidance": 7.5,
                    "width": 512,
                    "height": 512
                }
            )
            
            if response.status_code == 200:
                logger.info("✅ Generation successful!")
                
                # Save the image
                output_path = f"/workspace/generated_fresh_{self.child_id}.jpg"
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"💾 Saved to: {output_path}")
                return True
            else:
                logger.error(f"❌ Generation failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return False
    
    def run_fresh_test(self):
        """Run complete fresh LoRA test"""
        logger.info("🚀 Starting Fresh LoRA Training Test (RunPod Internal)")
        logger.info("==" * 30)
        
        # Cleanup first
        self.cleanup_old_child("aman_test")
        
        # Step 1: Health check
        if not self.test_health():
            return False
        
        # Step 2: Create fresh child
        if not self.create_fresh_child():
            return False
        
        # Step 3: Upload images
        if not self.upload_images():
            return False
        
        # Step 4: Start training
        task_id = self.start_minimal_training()
        if not task_id:
            return False
        
        # Step 5: Monitor training closely
        training_success = self.monitor_training_detailed(task_id)
        
        # Step 6: Test generation if successful
        if training_success:
            self.test_generation()
        
        logger.info("==" * 30)
        if training_success:
            logger.info("🎉 FRESH LORA TRAINING TEST PASSED! 🎉")
        else:
            logger.error("❌ FRESH LORA TRAINING TEST FAILED")
        
        return training_success

def main():
    tester = FreshLoRATestRunPod()
    success = tester.run_fresh_test()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())