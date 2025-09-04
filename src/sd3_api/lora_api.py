"""FastAPI endpoints for LoRA training and child management."""

import os
import logging
import traceback
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse

from .simple_storage import simple_storage
from .lora_models import (
    ChildCreateRequest, ChildResponse, TrainingImageResponse, BatchUploadResponse,
    TrainingConfigRequest, StartTrainingRequest, LoRAModelResponse,
    TrainingStatusResponse, GenerateWithChildrenRequest, GenerateWithChildrenResponse,
    ChildStatisticsResponse, SystemStatsResponse, UploadImageResponse
)

logger = logging.getLogger(__name__)

# Create router for LoRA endpoints
lora_router = APIRouter(prefix="/lora", tags=["LoRA Training"])


# Child Management Endpoints

@lora_router.post("/children", response_model=ChildResponse)
async def create_child(request: ChildCreateRequest) -> ChildResponse:
    """Create a new child for training."""
    try:
        # Create child using simple storage
        child_data = simple_storage.create_child(
            child_id=request.id,
            name=request.name,
            description=request.description
        )
        
        return ChildResponse(
            id=child_data["id"],
            name=child_data["name"],
            description=child_data["description"],
            created_at=child_data["created_at"],
            updated_at=child_data["updated_at"],
            training_image_count=child_data.get("training_image_count", 0),
            has_active_model=child_data.get("has_lora_model", False)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create child: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@lora_router.get("/children", response_model=List[ChildResponse])
async def list_children() -> List[ChildResponse]:
    """List all children."""
    try:
        children = simple_storage.list_children()
        results = []
        
        for child in children:
            results.append(ChildResponse(
                id=child["id"],
                name=child["name"],
                description=child["description"],
                created_at=child["created_at"],
                updated_at=child["updated_at"],
                training_image_count=child.get("training_image_count", 0),
                has_active_model=child.get("has_lora_model", False)
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to list children: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@lora_router.get("/children/{child_id}", response_model=ChildResponse)
async def get_child(child_id: str) -> ChildResponse:
    """Get information about a specific child."""
    try:
        logger.info(f"Looking for child {child_id} in {simple_storage.children_dir}")
        child = simple_storage.get_child(child_id)
        if not child:
            logger.info(f"Child {child_id} not found, returning 404")
            raise HTTPException(status_code=404, detail="Child not found")
        
        training_image_count = simple_storage.get_training_image_count(child_id)
        has_lora_model = simple_storage.has_lora_model(child_id)
        
        return ChildResponse(
            id=child["id"],
            name=child["name"],
            description=child["description"],
            created_at=child["created_at"],
            updated_at=child["updated_at"],
            training_image_count=training_image_count,
            has_active_model=has_lora_model
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get child {child_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@lora_router.put("/children/{child_id}", response_model=ChildResponse)
async def update_child(child_id: str, name: Optional[str] = None, 
                      description: Optional[str] = None) -> ChildResponse:
    """Update child information."""
    try:
        # Update child using simple storage
        updates = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        
        child = simple_storage.update_child(child_id, **updates)
        if not child:
            raise HTTPException(status_code=404, detail="Child not found")
        
        training_image_count = simple_storage.get_training_image_count(child_id)
        has_lora_model = simple_storage.has_lora_model(child_id)
        
        return ChildResponse(
            id=child["id"],
            name=child["name"],
            description=child["description"],
            created_at=child["created_at"],
            updated_at=child["updated_at"],
            training_image_count=training_image_count,
            has_active_model=has_lora_model
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update child {child_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@lora_router.delete("/children/{child_id}")
async def delete_child(child_id: str) -> Dict[str, str]:
    """Delete a child and all associated data."""
    try:
        # Check if child exists and delete using simple storage
        success = simple_storage.delete_child(child_id)
        if not success:
            raise HTTPException(status_code=404, detail="Child not found")
        
        return {"message": f"Child {child_id} and all associated data deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete child {child_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Training Image Management

@lora_router.post("/children/{child_id}/images", response_model=BatchUploadResponse)
async def upload_training_images(
    child_id: str,
    files: List[UploadFile] = File(...),
    descriptions: Optional[List[str]] = Form(None)
) -> BatchUploadResponse:
    """Upload training images for a child."""
    try:
        # Check if child exists
        child = simple_storage.get_child(child_id)
        if not child:
            raise HTTPException(status_code=404, detail="Child not found")
        
        # Upload files using simple storage
        upload_results = []
        for i, file in enumerate(files):
            try:
                # Read file data
                image_data = await file.read()
                
                # Get description if provided
                description = descriptions[i] if descriptions and i < len(descriptions) else None
                
                # Save using simple storage
                file_path = simple_storage.save_training_image(child_id, file.filename, image_data)
                
                upload_results.append(UploadImageResponse(
                    filename=file.filename,
                    file_path=str(file_path),
                    description=description,
                    metadata={"size": len(image_data), "original_filename": file.filename},
                    status="uploaded"
                ))
                
            except Exception as e:
                logger.error(f"Failed to upload {file.filename}: {e}")
                continue
        
        return BatchUploadResponse(
            uploaded_images=upload_results,
            total_uploaded=len(upload_results),
            child_id=child_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload images for {child_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@lora_router.get("/children/{child_id}/images", response_model=List[TrainingImageResponse])
async def get_training_images(child_id: str) -> List[TrainingImageResponse]:
    """Get all training images for a child."""
    try:
        # Check if child exists
        child = simple_storage.get_child(child_id)
        if not child:
            raise HTTPException(status_code=404, detail="Child not found")
        
        images = simple_storage.get_training_images(child_id)
        
        return [
            TrainingImageResponse(
                id=hash(img["filename"]),  # Use hash as simple ID
                filename=img["filename"],
                description=None,  # Simple storage doesn't store descriptions per image
                width=None,  # Would need PIL to extract dimensions
                height=None,
                file_size=img["size"],
                uploaded_at=img["created"]
            )
            for img in images
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training images for {child_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@lora_router.delete("/children/{child_id}/images/{filename}")
async def delete_training_image(child_id: str, filename: str) -> Dict[str, str]:
    """Delete a training image."""
    try:
        # Check if child exists
        child = simple_storage.get_child(child_id)
        if not child:
            raise HTTPException(status_code=404, detail="Child not found")
        
        # Delete the specific image file
        training_dir = simple_storage.training_images_dir(child_id)
        image_path = training_dir / filename
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Training image not found")
        
        image_path.unlink()
        return {"message": f"Training image {filename} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete training image {filename} for {child_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Training Management

@lora_router.post("/children/{child_id}/train", response_model=Dict[str, Any])
async def start_training(child_id: str, request: StartTrainingRequest) -> Dict[str, Any]:
    """Start LoRA training for a child."""
    try:
        # Prepare training configuration
        config_dict = {}
        if request.training_config:
            config_dict = request.training_config.model_dump()
        
        # Start training using simple storage (handles validation)
        logger.info(f"🎯 Starting training for child {child_id}")
        logger.info(f"📋 Training config: {config_dict}")
        
        task_id = simple_storage.start_training(child_id, config_dict)
        logger.info(f"✅ Simple storage training started with task_id: {task_id}")
        
        # Start background training task with Celery
        # Generate a model_id from the task_id for compatibility
        model_id = hash(task_id) % 2147483647  # Convert to positive int
        logger.info(f"🔢 Generated model_id: {model_id} from task_id: {task_id}")
        
        try:
            logger.info(f"📦 Importing Celery training tasks...")
            from .tasks.training_tasks import start_training_task
            logger.info(f"✅ Successfully imported start_training_task")
            
            logger.info(f"🚀 Dispatching Celery task with params:")
            logger.info(f"   child_id: {child_id}")
            logger.info(f"   model_id: {model_id}")
            logger.info(f"   config_dict: {config_dict}")
            
            celery_task_id = start_training_task(child_id, model_id, config_dict)
            logger.info(f"🎉 SUCCESS! Celery task dispatched with ID: {celery_task_id}")
            
        except ImportError as e:
            logger.error(f"❌ Import error when loading Celery tasks: {e}")
            logger.error(f"   Full import error: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to import training tasks: {str(e)}")
            
        except Exception as e:
            logger.error(f"💥 CRITICAL: Failed to start Celery task: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Full traceback: {traceback.format_exc()}")
            
            # Don't continue silently - this is critical
            raise HTTPException(status_code=500, detail=f"Failed to start training task: {str(e)}")
        
        return {
            "message": "Training started successfully",
            "child_id": child_id,
            "task_id": task_id,
            "training_config": config_dict
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start training for {child_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@lora_router.get("/children/{child_id}/training-status", response_model=TrainingStatusResponse)
async def get_training_status(child_id: str) -> TrainingStatusResponse:
    """Get current training status for a child."""
    try:
        # Get training status from simple storage
        status_info = simple_storage.get_training_status(child_id)
        
        return TrainingStatusResponse(
            model_id=hash(status_info.get("task_id", "default")) % 2147483647,  # Convert to positive int
            child_id=child_id,
            status=status_info["status"],
            progress=status_info["progress"],
            message="Training in progress" if status_info["status"] == "training" else "Ready",
            loss=None,
            error=None,
            started_at=status_info.get("started_at"),
            estimated_completion=None
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get training status for {child_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@lora_router.get("/training-tasks/{task_id}/status")
async def get_task_status_endpoint(task_id: str) -> Dict[str, Any]:
    """Get status of a specific training task."""
    try:
        # Extract child_id from task_id (format: train_childid_timestamp)
        parts = task_id.split("_")
        if len(parts) >= 2 and parts[0] == "train":
            child_id = parts[1]
            status_info = simple_storage.get_training_status(child_id)
            return {
                "task_id": task_id,
                "status": status_info["status"],
                "child_id": child_id
            }
        else:
            raise HTTPException(status_code=404, detail="Task not found")
        
    except ValueError:
        raise HTTPException(status_code=404, detail="Task not found")
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Model Management

@lora_router.get("/children/{child_id}/models", response_model=List[LoRAModelResponse])
async def get_child_models(child_id: str) -> List[LoRAModelResponse]:
    """Get all LoRA models for a child."""
    try:
        # Check if child exists
        child = simple_storage.get_child(child_id)
        if not child:
            raise HTTPException(status_code=404, detail="Child not found")
        
        # Get status info
        status_info = simple_storage.get_training_status(child_id)
        
        # Return single model info if exists
        models = []
        if status_info["has_model"] or status_info["status"] in ["training", "completed"]:
            models.append(LoRAModelResponse(
                id=status_info.get("task_id", "model_1"),
                child_id=child_id,
                version=child.get("model_version", 1),
                training_status=status_info["status"],
                training_progress=status_info["progress"],
                training_loss=None,
                error_message=None,
                created_at=child["created_at"],
                training_started_at=status_info.get("started_at"),
                training_completed_at=status_info.get("completed_at"),
                is_active=status_info["has_model"]
            ))
        
        return models
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get models for {child_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Statistics and Monitoring

@lora_router.get("/children/{child_id}/statistics", response_model=ChildStatisticsResponse)
async def get_child_statistics(child_id: str) -> ChildStatisticsResponse:
    """Get detailed statistics for a child."""
    try:
        # Check if child exists
        child = simple_storage.get_child(child_id)
        if not child:
            raise HTTPException(status_code=404, detail="Child not found")
        
        # Get training images stats
        images = simple_storage.get_training_images(child_id)
        upload_stats = {
            "total_images": len(images),
            "total_size": sum(img["size"] for img in images)
        }
        
        # Get model info
        status_info = simple_storage.get_training_status(child_id)
        model_info = None
        if status_info["has_model"] or status_info["status"] in ["training", "completed"]:
            model_info = {
                "model_id": status_info.get("task_id", "model_1"),
                "version": child.get("model_version", 1),
                "training_status": status_info["status"],
                "training_progress": status_info["progress"],
                "created_at": child["created_at"]
            }
        
        # Simple generation history (no database tracking)
        history_stats = {
            "total_generations": 0,
            "recent_generations": 0
        }
        
        return ChildStatisticsResponse(
            child_id=child_id,
            training_images=upload_stats,
            model_info=model_info,
            generation_history=history_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get statistics for {child_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@lora_router.get("/system/statistics", response_model=SystemStatsResponse)
async def get_system_statistics() -> SystemStatsResponse:
    """Get overall system statistics."""
    try:
        # Get storage stats from simple storage
        storage_stats = simple_storage.get_storage_stats()
        
        # Count active models
        all_children = simple_storage.list_children()
        active_models = sum(1 for child in all_children if child.get("has_lora_model", False))
        
        # Count training queue (simple - just count "training" status)
        training_queue_size = sum(1 for child in all_children if child.get("training_status") == "training")
        
        return SystemStatsResponse(
            total_children=storage_stats["total_children"],
            total_training_images=storage_stats["total_training_images"],
            total_lora_models=storage_stats["total_lora_models"],
            active_models=active_models,
            storage_stats={
                "total_size_bytes": storage_stats["storage_size_bytes"]
            },
            training_queue_size=training_queue_size,
            loaded_loras=[]  # Would need pipeline instance to get this
        )
        
    except Exception as e:
        logger.error(f"Failed to get system statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Debug Endpoints

@lora_router.post("/debug/test-celery")
async def test_celery_worker():
    """Test if Celery worker is functioning."""
    try:
        from .tasks.training_tasks import test_celery
        
        logger.info("🧪 Starting Celery test task...")
        task = test_celery.delay()
        
        return {
            "message": "Celery test task started",
            "task_id": task.id,
            "status": "pending",
            "instructions": f"Check task status at /lora/debug/task-status/{task.id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start Celery test task: {e}")
        raise HTTPException(status_code=500, detail=f"Celery test failed: {str(e)}")


@lora_router.get("/debug/task-status/{task_id}")
async def get_debug_task_status(task_id: str):
    """Get status of any Celery task."""
    try:
        from .tasks.training_tasks import get_task_status
        
        status = get_task_status(task_id)
        return status
        
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@lora_router.get("/debug/celery-info")
async def get_celery_info():
    """Get Celery worker and queue information."""
    try:
        from .tasks.celery_app import celery_app
        
        # Get active workers
        inspect = celery_app.control.inspect()
        active_workers = inspect.active()
        registered_tasks = inspect.registered()
        
        return {
            "redis_url": celery_app.conf.broker_url,
            "active_workers": active_workers or {},
            "registered_tasks": registered_tasks or {},
            "task_routes": dict(celery_app.conf.task_routes) if celery_app.conf.task_routes else {}
        }
        
    except Exception as e:
        logger.error(f"Failed to get Celery info: {e}")
        return {
            "error": str(e),
            "message": "Failed to inspect Celery workers"
        }