"""Database manager with CRUD operations."""

import os
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session
from PIL import Image

from .connection import get_session
from .models import Child, TrainingImage, LoRAModel, GenerationHistory, TrainingStatus


class DatabaseManager:
    """Database manager for LoRA training system."""
    
    # Children CRUD operations
    
    def create_child(self, child_id: str, name: str, description: Optional[str] = None) -> dict:
        """Create a new child record."""
        with get_session() as session:
            existing = session.query(Child).filter(Child.id == child_id).first()
            if existing:
                raise ValueError(f"Child with ID '{child_id}' already exists")
            
            child = Child(
                id=child_id,
                name=name,
                description=description
            )
            session.add(child)
            session.commit()
            session.refresh(child)
            
            # Return as dictionary to avoid session issues
            return {
                "id": child.id,
                "name": child.name,
                "description": child.description,
                "created_at": child.created_at,
                "updated_at": child.updated_at
            }
    
    def get_child(self, child_id: str) -> Optional[dict]:
        """Get child by ID."""
        with get_session() as session:
            child = session.query(Child).filter(Child.id == child_id).first()
            if not child:
                return None
            return {
                "id": child.id,
                "name": child.name,
                "description": child.description,
                "created_at": child.created_at,
                "updated_at": child.updated_at
            }
    
    def get_all_children(self) -> List[dict]:
        """Get all children."""
        with get_session() as session:
            children = session.query(Child).all()
            return [
                {
                    "id": child.id,
                    "name": child.name,
                    "description": child.description,
                    "created_at": child.created_at,
                    "updated_at": child.updated_at
                }
                for child in children
            ]
    
    def update_child(self, child_id: str, name: Optional[str] = None, 
                    description: Optional[str] = None) -> Optional[Child]:
        """Update child information."""
        with get_session() as session:
            child = session.query(Child).filter(Child.id == child_id).first()
            if not child:
                return None
            
            if name is not None:
                child.name = name
            if description is not None:
                child.description = description
            child.updated_at = datetime.utcnow()
            
            session.commit()
            session.refresh(child)
            return child
    
    def delete_child(self, child_id: str) -> bool:
        """Delete child and all associated data."""
        with get_session() as session:
            child = session.query(Child).filter(Child.id == child_id).first()
            if not child:
                return False
            
            session.delete(child)
            session.commit()
            return True
    
    # Training Images CRUD operations
    
    def add_training_image(self, child_id: str, filename: str, file_path: str,
                          description: Optional[str] = None) -> TrainingImage:
        """Add a training image for a child."""
        with get_session() as session:
            # Verify child exists
            child = session.query(Child).filter(Child.id == child_id).first()
            if not child:
                raise ValueError(f"Child '{child_id}' not found")
            
            # Get image dimensions and file size
            width, height, file_size = None, None, None
            try:
                if os.path.exists(file_path):
                    with Image.open(file_path) as img:
                        width, height = img.size
                    file_size = os.path.getsize(file_path)
            except Exception:
                pass  # Continue without metadata if image processing fails
            
            training_image = TrainingImage(
                child_id=child_id,
                filename=filename,
                file_path=file_path,
                description=description,
                width=width,
                height=height,
                file_size=file_size
            )
            
            session.add(training_image)
            session.commit()
            session.refresh(training_image)
            return training_image
    
    def get_training_images(self, child_id: str) -> List[TrainingImage]:
        """Get all training images for a child."""
        with get_session() as session:
            return session.query(TrainingImage).filter(
                TrainingImage.child_id == child_id
            ).all()
    
    def get_training_image_count(self, child_id: str) -> int:
        """Get count of training images for a child."""
        with get_session() as session:
            return session.query(TrainingImage).filter(
                TrainingImage.child_id == child_id
            ).count()
    
    def delete_training_image(self, image_id: int) -> bool:
        """Delete a training image."""
        with get_session() as session:
            image = session.query(TrainingImage).filter(TrainingImage.id == image_id).first()
            if not image:
                return False
            
            # Delete file if it exists
            if os.path.exists(image.file_path):
                try:
                    os.remove(image.file_path)
                except Exception:
                    pass  # Continue even if file deletion fails
            
            session.delete(image)
            session.commit()
            return True
    
    # LoRA Models CRUD operations
    
    def create_lora_model(self, child_id: str, model_path: str, training_config: dict,
                         version: int = 1) -> LoRAModel:
        """Create a new LoRA model record."""
        with get_session() as session:
            # Verify child exists
            child = session.query(Child).filter(Child.id == child_id).first()
            if not child:
                raise ValueError(f"Child '{child_id}' not found")
            
            # Deactivate existing models for this child
            existing_models = session.query(LoRAModel).filter(
                LoRAModel.child_id == child_id,
                LoRAModel.is_active == "true"
            ).all()
            for model in existing_models:
                model.is_active = "false"
            
            lora_model = LoRAModel(
                child_id=child_id,
                model_path=model_path,
                version=version,
                training_steps=training_config.get("training_steps", 1000),
                learning_rate=training_config.get("learning_rate", 1e-4),
                lora_rank=training_config.get("lora_rank", 64),
                lora_alpha=training_config.get("lora_alpha", 32),
                lora_dropout=training_config.get("lora_dropout", 0.1),
                training_status=TrainingStatus.PENDING,
                is_active="true"
            )
            
            session.add(lora_model)
            session.commit()
            session.refresh(lora_model)
            return lora_model
    
    def get_lora_model(self, child_id: str, active_only: bool = True) -> Optional[LoRAModel]:
        """Get LoRA model for a child."""
        with get_session() as session:
            query = session.query(LoRAModel).filter(LoRAModel.child_id == child_id)
            if active_only:
                query = query.filter(LoRAModel.is_active == "true")
            return query.order_by(LoRAModel.version.desc()).first()
    
    def get_all_lora_models(self, child_id: str) -> List[LoRAModel]:
        """Get all LoRA models for a child."""
        with get_session() as session:
            return session.query(LoRAModel).filter(
                LoRAModel.child_id == child_id
            ).order_by(LoRAModel.version.desc()).all()
    
    def update_training_status(self, model_id: int, status: TrainingStatus,
                              progress: Optional[float] = None,
                              loss: Optional[float] = None,
                              error_message: Optional[str] = None) -> Optional[LoRAModel]:
        """Update training status and progress."""
        with get_session() as session:
            model = session.query(LoRAModel).filter(LoRAModel.id == model_id).first()
            if not model:
                return None
            
            model.training_status = status
            if progress is not None:
                model.training_progress = progress
            if loss is not None:
                model.training_loss = loss
            if error_message is not None:
                model.error_message = error_message
            
            # Update timestamps
            if status == TrainingStatus.TRAINING and not model.training_started_at:
                model.training_started_at = datetime.utcnow()
            elif status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
                model.training_completed_at = datetime.utcnow()
            
            model.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(model)
            return model
    
    def get_models_by_status(self, status: TrainingStatus) -> List[LoRAModel]:
        """Get all models with a specific training status."""
        with get_session() as session:
            return session.query(LoRAModel).filter(
                LoRAModel.training_status == status
            ).all()
    
    # Generation History operations
    
    def add_generation_history(self, prompt: str, children_ids: List[str],
                              image_path: str, steps: int, guidance_scale: float,
                              seed: Optional[int] = None, 
                              generation_time: Optional[float] = None) -> GenerationHistory:
        """Add generation history record."""
        with get_session() as session:
            history = GenerationHistory(
                prompt=prompt,
                children_ids=",".join(children_ids) if children_ids else None,
                image_path=image_path,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                generation_time=generation_time
            )
            
            session.add(history)
            session.commit()
            session.refresh(history)
            return history
    
    def get_generation_history(self, limit: int = 100) -> List[GenerationHistory]:
        """Get recent generation history."""
        with get_session() as session:
            return session.query(GenerationHistory).order_by(
                GenerationHistory.created_at.desc()
            ).limit(limit).all()
    
    def get_child_generation_history(self, child_id: str, limit: int = 50) -> List[GenerationHistory]:
        """Get generation history for a specific child."""
        with get_session() as session:
            return session.query(GenerationHistory).filter(
                GenerationHistory.children_ids.contains(child_id)
            ).order_by(GenerationHistory.created_at.desc()).limit(limit).all()


# Global database manager instance
db_manager = DatabaseManager()