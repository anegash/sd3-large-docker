"""Celery tasks for LoRA training."""

import logging
import traceback
from typing import Dict, Any

from celery import current_task

from .celery_app import celery_app
from ..database.manager import db_manager
from ..database.models import TrainingStatus
from ..lora.trainer import LoRATrainer, create_training_config
from ..utils.storage import storage_manager

logger = logging.getLogger(__name__)


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
        
        # Update database
        db_manager.update_training_status(
            self.model_id, 
            TrainingStatus.TRAINING, 
            progress=progress
        )
        
        logger.info(f"Training progress for model {self.model_id}: {progress:.1%} - {message}")


@celery_app.task(bind=True, name="sd3_api.tasks.training_tasks.train_lora")
def train_lora(self, child_id: str, model_id: int, training_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train LoRA adapter for a child.
    
    Args:
        child_id: Child identifier
        model_id: Database model ID
        training_config: Training configuration parameters
        
    Returns:
        Training result dictionary
    """
    try:
        logger.info(f"Starting LoRA training task for child_id: {child_id}, model_id: {model_id}")
        
        # Update task state to started
        self.update_state(
            state="PROGRESS",
            meta={
                "progress": 0.0,
                "message": "Initializing training...",
                "model_id": model_id
            }
        )
        
        # Create training configuration
        config = create_training_config(**training_config)
        
        # Initialize trainer
        trainer = LoRATrainer(config)
        
        # Set up progress callback
        progress_callback = TrainingProgressCallback(self.request.id, model_id)
        trainer.set_progress_callback(progress_callback)
        
        # Start training
        model_path = trainer.train(child_id, model_id)
        
        # Training completed successfully
        result = {
            "status": "completed",
            "child_id": child_id,
            "model_id": model_id,
            "model_path": model_path,
            "task_id": self.request.id
        }
        
        logger.info(f"LoRA training completed successfully for {child_id}")
        return result
        
    except Exception as e:
        logger.error(f"LoRA training failed for {child_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Update database with error
        try:
            db_manager.update_training_status(
                model_id, 
                TrainingStatus.FAILED, 
                error_message=str(e)
            )
        except Exception as db_error:
            logger.error(f"Failed to update database with error status: {db_error}")
        
        # Update task state to failure
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "traceback": traceback.format_exc(),
                "model_id": model_id
            }
        )
        
        # Re-raise the exception to mark task as failed
        raise


@celery_app.task(name="sd3_api.tasks.training_tasks.cleanup_training_data")
def cleanup_training_data(child_id: str) -> Dict[str, Any]:
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
def get_training_status(model_id: int) -> Dict[str, Any]:
    """
    Get current training status from database.
    
    Args:
        model_id: Database model ID
        
    Returns:
        Status dictionary
    """
    try:
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
            
    except Exception as e:
        logger.error(f"Failed to get training status for model {model_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def start_training_task(child_id: str, model_id: int, training_config: Dict[str, Any]) -> str:
    """
    Start a training task and return task ID.
    
    Args:
        child_id: Child identifier  
        model_id: Database model ID
        training_config: Training configuration
        
    Returns:
        Celery task ID
    """
    task = train_lora.delay(child_id, model_id, training_config)
    logger.info(f"Started training task {task.id} for child {child_id}")
    return task.id


def get_task_status(task_id: str) -> Dict[str, Any]:
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