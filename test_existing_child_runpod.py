#!/usr/bin/env python3
"""
Test LoRA training with existing child that already has images uploaded
Uses the "aman" child that already has 27 images from API upload
"""

import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExistingChildLoRATest:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.session = requests.Session()
        self.child_id = "aman"  # Use existing child with 27 images
        
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
    
    def check_existing_child(self):
        """Check existing child and images"""
        logger.info(f"🔍 Checking existing child: {self.child_id}")
        try:
            response = self.session.get(f"{self.api_url}/lora/children/{self.child_id}")
            response.raise_for_status()
            child = response.json()
            
            image_count = child.get('training_image_count', 0)
            logger.info(f"✅ Child found: {child['name']}")
            logger.info(f"   Images: {image_count}")
            logger.info(f"   Has model: {child.get('has_active_model', False)}")
            
            if image_count < 3:
                logger.error(f"❌ Need at least 3 images, found {image_count}")
                logger.info(f"💡 Images should be in: /workspace/sd3-large-docker/data/training_images/{self.child_id}/")
                logger.info(f"💡 Run: mkdir -p /workspace/sd3-large-docker/data/training_images/{self.child_id}")
                logger.info(f"💡 Run: mv /workspace/sd3-large-docker/data/children/{self.child_id}/training_images/* /workspace/sd3-large-docker/data/training_images/{self.child_id}/")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to get child: {e}")
            return False
    
    def start_training(self):
        """Start LoRA training with existing images"""
        logger.info("🏋️ Starting LoRA training with existing images...")
        
        # Optimized training config for A40 GPU
        training_config = {
            "training_steps": 100,
            "learning_rate": 5e-5,
            "lora_rank": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "mixed_precision": "fp16",
            "save_steps": 50,
            "validation_steps": 50,
            "max_grad_norm": 0.5,
            "use_8bit_adam": True,
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
    
    def monitor_training(self, task_id, timeout=1200):  # 20 minutes
        """Monitor training progress"""
        logger.info(f"👀 Monitoring training progress (timeout: {timeout}s)...")
        
        start_time = time.time()
        last_status = None
        check_count = 0
        
        while time.time() - start_time < timeout:
            check_count += 1
            try:
                response = self.session.get(
                    f"{self.api_url}/lora/children/{self.child_id}/training-status"
                )
                response.raise_for_status()
                status_data = response.json()
                
                current_status = status_data.get('status', 'unknown')
                current_progress = status_data.get('progress', 0)
                
                if (current_status != last_status or check_count % 6 == 0):
                    logger.info(f"   Check #{check_count}: Status={current_status}, Progress={current_progress:.2f}")
                    
                    if 'message' in status_data:
                        logger.info(f"   Message: {status_data['message']}")
                    
                    last_status = current_status
                
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
                
                # Check Celery task
                if task_id and check_count % 3 == 0:
                    try:
                        task_response = self.session.get(
                            f"{self.api_url}/lora/training-tasks/{task_id}/status"
                        )
                        if task_response.status_code == 200:
                            task_data = task_response.json()
                            logger.info(f"   Celery Task: {task_data.get('status', 'unknown')}")
                    except:
                        pass
                
            except Exception as e:
                logger.warning(f"   Failed to check status: {e}")
            
            time.sleep(30)  # Check every 30 seconds
        
        logger.warning(f"⏰ Training monitoring timed out after {timeout}s")
        return False
    
    def test_generation(self):
        """Test generation with trained LoRA"""
        logger.info("🎨 Testing generation with trained LoRA...")
        
        test_prompt = f"a portrait photo of {{{self.child_id}}}, professional headshot, high quality"
        
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
                output_path = f"/workspace/generated_aman_trained.jpg"
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
    
    def run_training_test(self):
        """Run complete LoRA training test with existing child"""
        logger.info("🚀 Starting LoRA Training Test (Existing Child)")
        logger.info("=" * 60)
        
        # Step 1: Health check
        if not self.test_health():
            return False
        
        # Step 2: Check existing child
        if not self.check_existing_child():
            return False
        
        # Step 3: Start training
        task_id = self.start_training()
        if not task_id:
            return False
        
        # Step 4: Monitor training
        training_success = self.monitor_training(task_id)
        
        # Step 5: Test generation
        if training_success:
            self.test_generation()
        
        logger.info("=" * 60)
        if training_success:
            logger.info("🎉 LORA TRAINING TEST PASSED! 🎉")
        else:
            logger.error("❌ LORA TRAINING TEST FAILED")
        
        return training_success

def main():
    tester = ExistingChildLoRATest()
    success = tester.run_training_test()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())