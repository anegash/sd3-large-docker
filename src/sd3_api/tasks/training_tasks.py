"""Celery tasks for LoRA training."""

import logging
import traceback
from typing import Dict, Any

from celery import current_task

from .celery_app import celery_app

# Initialize logger first before using it
logger = logging.getLogger(__name__)

try:
    from ..database.manager import db_manager
    from ..database.models import TrainingStatus
    USE_DATABASE = True
except ImportError:
    # Fallback for simple storage mode
    db_manager = None
    USE_DATABASE = False
    
    # Mock TrainingStatus enum
    class TrainingStatus:
        TRAINING = "training"
        COMPLETED = "completed"
        FAILED = "failed"
try:
    from ..lora.trainer import LoRATrainer, create_training_config
    TRAINER_AVAILABLE = True
    logger.info(f"✅ Successfully imported LoRATrainer and create_training_config")
except ImportError as e:
    logger.error(f"❌ Failed to import trainer: {e}")
    TRAINER_AVAILABLE = False
    LoRATrainer = None
    create_training_config = None

try:
    from ..utils.storage import storage_manager
    STORAGE_AVAILABLE = True
    logger.info(f"✅ Successfully imported storage_manager")
except ImportError as e:
    logger.error(f"❌ Failed to import storage_manager: {e}")
    STORAGE_AVAILABLE = False
    storage_manager = None


class TrainingProgressCallback:
    """Progress callback that updates Celery task state."""
    
    def __init__(self, task_id: str, model_id: int):
        self.task_id = task_id
        self.model_id = model_id
    
    def __call__(self, progress: float, message: str = ""):
        """Update training progress."""
        # Update Celery task state
        if current_task:
            current_task.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "message": message,
                    "model_id": self.model_id
                }
            )
        
        # Update database if available
        if USE_DATABASE and db_manager:
            db_manager.update_training_status(
                self.model_id, 
                TrainingStatus.TRAINING, 
                progress=progress
            )
        
        logger.info(f"Training progress for model {self.model_id}: {progress:.1%} - {message}")


@celery_app.task(bind=True, name="sd3_api.tasks.training_tasks.train_lora")
def train_lora(self, child_id, model_id, training_config):
    """
    Train LoRA adapter for a child.
    
    Args:
        child_id: Child identifier
        model_id: Database model ID
        training_config: Training configuration parameters
        
    Returns:
        Training result dictionary
    """
    # FIRST LOG - Check if function is even called
    print(f"🔥 CRITICAL: train_lora FUNCTION CALLED! child_id={child_id}, model_id={model_id}")
    logger.info(f"🔥 CRITICAL: train_lora FUNCTION CALLED! child_id={child_id}, model_id={model_id}")
    
    logger.info(f"🚀 TASK EXECUTION START: LoRA training task for child_id: {child_id}, model_id: {model_id}")
    logger.info(f"📋 Training config: {training_config}")
    logger.info(f"🔧 Task request info:")
    logger.info(f"   Task ID: {self.request.id}")
    logger.info(f"   Task name: {self.request.task}")
    logger.info(f"   Retries: {self.request.retries}")
    logger.info(f"   Delivery info: {self.request.delivery_info}")
    logger.info(f"🌍 Environment check during execution:")
    logger.info(f"   USE_DATABASE: {USE_DATABASE}")
    logger.info(f"   db_manager: {db_manager}")
    logger.info(f"   TRAINER_AVAILABLE: {TRAINER_AVAILABLE}")
    logger.info(f"   STORAGE_AVAILABLE: {STORAGE_AVAILABLE}")
    logger.info(f"   LoRATrainer: {LoRATrainer}")
    logger.info(f"   storage_manager: {storage_manager}")
    
    logger.info(f"🔍 Re-checking imports during task execution...")
    # Re-check imports during execution to see if they work at runtime
    try:
        from ..lora.trainer import LoRATrainer as RuntimeLoRATrainer, create_training_config as runtime_create_training_config
        logger.info(f"✅ RUNTIME: Successfully imported LoRATrainer during execution")
        runtime_trainer_available = True
    except Exception as e:
        logger.error(f"❌ RUNTIME: Failed to import LoRATrainer during execution: {e}")
        logger.error(f"   Import error type: {type(e).__name__}")
        logger.error(f"   Import error details: {str(e)}")
        runtime_trainer_available = False
        
    try:
        from ..utils.storage import storage_manager as runtime_storage_manager
        logger.info(f"✅ RUNTIME: Successfully imported storage_manager during execution")
        runtime_storage_available = True
    except Exception as e:
        logger.error(f"❌ RUNTIME: Failed to import storage_manager during execution: {e}")
        logger.error(f"   Import error type: {type(e).__name__}")
        logger.error(f"   Import error details: {str(e)}")
        runtime_storage_available = False
    
    # Check dependencies before proceeding
    logger.info(f"🔍 Dependency check results:")
    logger.info(f"   Module level - TRAINER_AVAILABLE: {TRAINER_AVAILABLE}")
    logger.info(f"   Module level - STORAGE_AVAILABLE: {STORAGE_AVAILABLE}")  
    logger.info(f"   Runtime level - trainer_available: {runtime_trainer_available}")
    logger.info(f"   Runtime level - storage_available: {runtime_storage_available}")
    
    if not TRAINER_AVAILABLE and not runtime_trainer_available:
        error_msg = "LoRATrainer not available - import failed at both module and runtime level"
        logger.error(f"💥 DEPENDENCY ERROR: {error_msg}")
        raise RuntimeError(error_msg)
        
    if not STORAGE_AVAILABLE and not runtime_storage_available:
        error_msg = "storage_manager not available - import failed at both module and runtime level"  
        logger.error(f"💥 DEPENDENCY ERROR: {error_msg}")
        raise RuntimeError(error_msg)
    
    logger.info(f"⚙️ All dependencies available - starting training logic...")
    
    try:
        # Update task state to started
        self.update_state(
            state="PROGRESS",
            meta={
                "progress": 0.0,
                "message": "Initializing training...",
                "model_id": model_id
            }
        )
        logger.info(f"✅ Task state updated to PROGRESS")
        
        # Validate inputs
        if not child_id:
            raise ValueError("child_id cannot be empty")
        if not isinstance(model_id, int) or model_id <= 0:
            raise ValueError(f"Invalid model_id: {model_id}")
        if not training_config:
            raise ValueError("training_config cannot be empty")
        
        logger.info(f"✅ Input validation passed")
        
        # Create training configuration
        logger.info(f"🔧 Creating training configuration...")
        try:
            config = create_training_config(**training_config)
            logger.info(f"✅ Training config created: rank={config.lora_rank}, steps={config.training_steps}")
        except Exception as e:
            logger.error(f"❌ Failed to create training config: {e}")
            raise ValueError(f"Invalid training configuration: {e}")
        
        # Initialize trainer
        logger.info(f"🤖 Initializing LoRA trainer...")
        try:
            trainer = LoRATrainer(config)
            logger.info(f"✅ LoRA trainer initialized on device: {trainer.device}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize trainer: {e}")
            logger.error(f"   Full error: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize trainer: {e}")
        
        # Set up progress callback
        logger.info(f"📊 Setting up progress callback...")
        progress_callback = TrainingProgressCallback(self.request.id, model_id)
        trainer.set_progress_callback(progress_callback)
        logger.info(f"✅ Progress callback configured")
        
        # Start training
        logger.info(f"🎯 Starting LoRA training...")
        try:
            model_path = trainer.train(child_id, model_id)
            logger.info(f"✅ Training completed successfully. Model saved to: {model_path}")
        except Exception as e:
            logger.error(f"❌ Training failed during execution: {e}")
            logger.error(f"   Full training error: {traceback.format_exc()}")
            raise RuntimeError(f"Training execution failed: {e}")
        
        # Training completed successfully
        result = {
            "status": "completed",
            "child_id": child_id,
            "model_id": model_id,
            "model_path": model_path,
            "task_id": self.request.id
        }
        
        logger.info(f"🎉 LoRA training completed successfully for {child_id}")
        logger.info(f"📁 Final result: {result}")
        return result
        
    except Exception as e:
        error_msg = str(e)
        full_traceback = traceback.format_exc()
        
        logger.error(f"💥 LoRA training failed for {child_id}: {error_msg}")
        logger.error(f"🔍 Full traceback:\n{full_traceback}")
        
        # Update database with error if available
        if USE_DATABASE and db_manager:
            try:
                logger.info(f"💾 Updating database with error status...")
                db_manager.update_training_status(
                    model_id, 
                    TrainingStatus.FAILED, 
                    error_message=error_msg
                )
                logger.info(f"✅ Database updated with error status")
            except Exception as db_error:
                logger.error(f"❌ Failed to update database with error status: {db_error}")
                logger.error(f"   DB error traceback: {traceback.format_exc()}")
        else:
            logger.info(f"📝 Using simple storage - database update skipped")
        
        # Update task state to failure
        try:
            logger.info(f"📋 Updating task state to FAILURE...")
            self.update_state(
                state="FAILURE",
                meta={
                    "error": error_msg,
                    "traceback": full_traceback,
                    "model_id": model_id
                }
            )
            logger.info(f"✅ Task state updated to FAILURE")
        except Exception as task_error:
            logger.error(f"❌ Failed to update task state: {task_error}")
        
        # Re-raise the exception to mark task as failed
        logger.error(f"🚨 Re-raising exception to mark task as failed")
        raise


@celery_app.task(name="sd3_api.tasks.training_tasks.test_celery")
def test_celery():
    """Simple test task to verify Celery worker is functioning."""
    import time
    from datetime import datetime
    
    logger.info("🧪 Test Celery task started")
    
    # Simulate some work
    for i in range(5):
        logger.info(f"🔄 Test task progress: {i+1}/5")
        time.sleep(1)
    
    result = {
        "status": "success",
        "message": "Celery worker is functioning correctly",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    logger.info(f"✅ Test Celery task completed: {result}")
    return result


@celery_app.task(bind=True, name="sd3_api.tasks.training_tasks.test_training_imports")
def test_training_imports(self, child_id, model_id):
    """Test task to isolate import failures without complex dependencies."""
    # CRITICAL: First log to verify function is called
    print(f"🔥 CRITICAL: test_training_imports FUNCTION CALLED! child_id={child_id}")
    logger.info(f"🔥 CRITICAL: test_training_imports FUNCTION CALLED! child_id={child_id}")
    
    logger.info(f"🧪 Testing imports that train_lora needs...")
    
    # Test basic imports first
    try:
        import torch
        logger.info(f"✅ torch import successful: {torch.__version__}")
    except Exception as e:
        logger.error(f"❌ torch import failed: {e}")
        return {"status": "failed", "error": "torch import failed", "details": str(e)}
    
    # Test relative imports like train_lora uses
    try:
        logger.info(f"🔍 Testing relative import: ..lora.trainer")
        from ..lora.trainer import LoRATrainer, create_training_config
        logger.info(f"✅ LoRATrainer import successful")
    except Exception as e:
        logger.error(f"❌ LoRATrainer import failed: {e}")
        import traceback
        logger.error(f"   Full traceback: {traceback.format_exc()}")
        return {"status": "failed", "error": "LoRATrainer import failed", "details": str(e)}
    
    try:
        logger.info(f"🔍 Testing relative import: ..utils.storage")
        from ..utils.storage import storage_manager
        logger.info(f"✅ storage_manager import successful")
    except Exception as e:
        logger.error(f"❌ storage_manager import failed: {e}")
        import traceback
        logger.error(f"   Full traceback: {traceback.format_exc()}")
        return {"status": "failed", "error": "storage_manager import failed", "details": str(e)}
    
    # Test creating trainer instance (lightweight test)
    try:
        logger.info(f"🔍 Testing LoRATrainer instantiation...")
        config = create_training_config(
            training_steps=10, 
            lora_rank=16,
            learning_rate=1e-4,
            batch_size=1
        )
        logger.info(f"✅ create_training_config successful")
        
        # Don't actually create trainer (too heavy), just test config
        logger.info(f"✅ Training config created successfully: {config}")
        
    except Exception as e:
        logger.error(f"❌ Training config creation failed: {e}")
        import traceback
        logger.error(f"   Full traceback: {traceback.format_exc()}")
        return {"status": "failed", "error": "Training config failed", "details": str(e)}
    
    result = {
        "status": "success",
        "child_id": child_id,
        "model_id": model_id,
        "message": "All imports and basic operations successful",
        "torch_version": torch.__version__ if 'torch' in locals() else "unknown"
    }
    
    logger.info(f"🎉 test_training_imports completed successfully: {result}")
    return result


@celery_app.task(name="sd3_api.tasks.training_tasks.cleanup_training_data")
def cleanup_training_data(child_id):
    """
    Clean up training data for a child.
    
    Args:
        child_id: Child identifier
        
    Returns:
        Cleanup result dictionary
    """
    try:
        logger.info(f"Starting cleanup task for child_id: {child_id}")
        
        # Remove files from storage
        storage_manager.cleanup_child_data(child_id)
        
        result = {
            "status": "completed",
            "child_id": child_id,
            "message": "Training data cleaned up successfully"
        }
        
        logger.info(f"Cleanup completed successfully for {child_id}")
        return result
        
    except Exception as e:
        logger.error(f"Cleanup failed for {child_id}: {e}")
        logger.error(traceback.format_exc())
        raise


@celery_app.task(name="sd3_api.tasks.training_tasks.get_training_status")
def get_training_status(model_id):
    """
    Get current training status from database or simple storage.
    
    Args:
        model_id: Database model ID
        
    Returns:
        Status dictionary
    """
    try:
        if USE_DATABASE and db_manager:
            # Get model from database
            with db_manager.get_session() as session:
                from ..database.models import LoRAModel
                model = session.query(LoRAModel).filter(LoRAModel.id == model_id).first()
                
                if not model:
                    return {
                        "status": "not_found",
                        "error": f"Model {model_id} not found"
                    }
                
                return {
                    "status": "success",
                    "model_id": model_id,
                    "training_status": model.training_status.value,
                    "progress": model.training_progress,
                    "training_loss": model.training_loss,
                    "error_message": model.error_message,
                    "created_at": model.created_at.isoformat() if model.created_at else None,
                    "training_started_at": model.training_started_at.isoformat() if model.training_started_at else None,
                    "training_completed_at": model.training_completed_at.isoformat() if model.training_completed_at else None,
                }
        else:
            # Using simple storage - return basic status
            return {
                "status": "success",
                "model_id": model_id,
                "training_status": "unknown",
                "progress": 0.0,
                "training_loss": None,
                "error_message": "Simple storage mode - status tracking limited",
                "created_at": None,
                "training_started_at": None,
                "training_completed_at": None,
            }
            
    except Exception as e:
        logger.error(f"Failed to get training status for model {model_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def start_training_task(child_id, model_id, training_config):
    """
    Start a training task and return task ID.
    
    Args:
        child_id: Child identifier  
        model_id: Database model ID
        training_config: Training configuration
        
    Returns:
        Celery task ID
    """
    try:
        logger.info(f"🔥 start_training_task called with:")
        logger.info(f"   child_id: '{child_id}' (type: {type(child_id)})")
        logger.info(f"   model_id: {model_id} (type: {type(model_id)})")
        logger.info(f"   training_config: {training_config}")
        
        logger.info(f"📋 About to call train_lora.delay()...")
        logger.info(f"🔍 Celery app info: {celery_app}")
        logger.info(f"🔍 train_lora task info: {train_lora}")
        logger.info(f"🔍 train_lora task name: {train_lora.name}")
        
        # Try to dispatch the task
        task = train_lora.delay(child_id, model_id, training_config)
        
        logger.info(f"🎊 SUCCESS! Training task dispatched with ID: {task.id}")
        logger.info(f"   Task state: {task.state}")
        logger.info(f"   Task status: {task.status}")
        logger.info(f"   Task result: {task.result}")
        logger.info(f"   Task info: {task.info}")
        
        # Wait a moment and check task state again
        import time
        time.sleep(2)
        logger.info(f"📊 Task state after 2 seconds: {task.state}")
        logger.info(f"📊 Task info after 2 seconds: {task.info}")
        
        return task.id
        
    except Exception as e:
        logger.error(f"💀 FATAL: start_training_task failed: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.error(f"   Full traceback: {traceback.format_exc()}")
        raise


def get_task_status(task_id):
    """
    Get Celery task status.
    
    Args:
        task_id: Celery task ID
        
    Returns:
        Task status dictionary
    """
    try:
        task = celery_app.AsyncResult(task_id)
        
        if task.state == "PENDING":
            return {
                "state": "PENDING",
                "progress": 0.0,
                "message": "Task is waiting to start..."
            }
        elif task.state == "PROGRESS":
            return {
                "state": "PROGRESS",
                "progress": task.info.get("progress", 0.0),
                "message": task.info.get("message", ""),
                "model_id": task.info.get("model_id")
            }
        elif task.state == "SUCCESS":
            return {
                "state": "SUCCESS",
                "progress": 1.0,
                "message": "Training completed successfully",
                "result": task.result
            }
        elif task.state == "FAILURE":
            return {
                "state": "FAILURE",
                "progress": 0.0,
                "message": "Training failed",
                "error": str(task.info.get("error", "Unknown error")),
                "traceback": task.info.get("traceback", "")
            }
        else:
            return {
                "state": task.state,
                "progress": 0.0,
                "message": f"Task state: {task.state}"
            }
            
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        return {
            "state": "ERROR",
            "progress": 0.0,
            "message": f"Failed to get task status: {e}"
        }