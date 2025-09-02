"""
Simple file-based storage system using directories for each child.
No database required - just organized folders.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class SimpleChildStorage:
    """Simple directory-based storage for children and their data."""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.children_dir = self.base_dir / "children"
        self.children_dir.mkdir(parents=True, exist_ok=True)
    
    def child_dir(self, child_id: str) -> Path:
        """Get the directory path for a child."""
        return self.children_dir / child_id
    
    def child_info_file(self, child_id: str) -> Path:
        """Get the info.json file path for a child."""
        return self.child_dir(child_id) / "info.json"
    
    def training_images_dir(self, child_id: str) -> Path:
        """Get the training images directory for a child."""
        return self.child_dir(child_id) / "training_images"
    
    def lora_models_dir(self, child_id: str) -> Path:
        """Get the LoRA models directory for a child."""
        return self.child_dir(child_id) / "lora_models"
    
    def generated_images_dir(self, child_id: str) -> Path:
        """Get the generated images directory for a child."""
        return self.child_dir(child_id) / "generated_images"
    
    # Child Management
    
    def create_child(self, child_id: str, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Create a new child with directory structure."""
        child_path = self.child_dir(child_id)
        
        if child_path.exists():
            raise ValueError(f"Child '{child_id}' already exists")
        
        # Create directory structure
        child_path.mkdir(parents=True, exist_ok=True)
        self.training_images_dir(child_id).mkdir(exist_ok=True)
        self.lora_models_dir(child_id).mkdir(exist_ok=True)
        self.generated_images_dir(child_id).mkdir(exist_ok=True)
        
        # Create info file
        info = {
            "id": child_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "training_status": "not_started",
            "model_version": 0
        }
        
        with open(self.child_info_file(child_id), "w") as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Created child directory structure for {child_id}")
        return info
    
    def get_child(self, child_id: str) -> Optional[Dict[str, Any]]:
        """Get child information."""
        info_file = self.child_info_file(child_id)
        if not info_file.exists():
            return None
        
        try:
            with open(info_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading child info for {child_id}: {e}")
            return None
    
    def update_child(self, child_id: str, **updates) -> Optional[Dict[str, Any]]:
        """Update child information."""
        info = self.get_child(child_id)
        if not info:
            return None
        
        # Update fields
        for key, value in updates.items():
            if value is not None:
                info[key] = value
        
        info["updated_at"] = datetime.now().isoformat()
        
        # Save updated info
        with open(self.child_info_file(child_id), "w") as f:
            json.dump(info, f, indent=2)
        
        return info
    
    def list_children(self) -> List[Dict[str, Any]]:
        """List all children."""
        children = []
        
        for child_dir in self.children_dir.iterdir():
            if child_dir.is_dir():
                info = self.get_child(child_dir.name)
                if info:
                    # Add computed fields
                    info["training_image_count"] = self.get_training_image_count(child_dir.name)
                    info["has_lora_model"] = self.has_lora_model(child_dir.name)
                    children.append(info)
        
        return sorted(children, key=lambda x: x["created_at"])
    
    def delete_child(self, child_id: str) -> bool:
        """Delete child and all associated data."""
        child_path = self.child_dir(child_id)
        if not child_path.exists():
            return False
        
        try:
            shutil.rmtree(child_path)
            logger.info(f"Deleted child directory for {child_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting child {child_id}: {e}")
            return False
    
    # Training Images
    
    def save_training_image(self, child_id: str, filename: str, image_data: bytes) -> str:
        """Save a training image for a child."""
        if not self.child_dir(child_id).exists():
            raise ValueError(f"Child '{child_id}' does not exist")
        
        training_dir = self.training_images_dir(child_id)
        training_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name, ext = os.path.splitext(filename)
        safe_filename = f"{timestamp}_{name}{ext}"
        file_path = training_dir / safe_filename
        
        # Save image
        with open(file_path, "wb") as f:
            f.write(image_data)
        
        logger.info(f"Saved training image: {file_path}")
        return str(file_path)
    
    def get_training_images(self, child_id: str) -> List[Dict[str, Any]]:
        """Get list of training images for a child."""
        training_dir = self.training_images_dir(child_id)
        if not training_dir.exists():
            return []
        
        images = []
        for img_file in training_dir.glob("*"):
            if img_file.is_file() and img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".mpo"}:
                stat = img_file.stat()
                images.append({
                    "filename": img_file.name,
                    "path": str(img_file),
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                })
        
        return sorted(images, key=lambda x: x["created"])
    
    def get_training_image_count(self, child_id: str) -> int:
        """Get count of training images for a child."""
        return len(self.get_training_images(child_id))
    
    # LoRA Models
    
    def start_training(self, child_id: str, config: Dict = None) -> str:
        """Start training for a child."""
        if not self.child_dir(child_id).exists():
            raise ValueError(f"Child '{child_id}' does not exist")
        
        # Check minimum images
        image_count = self.get_training_image_count(child_id)
        if image_count < 5:
            raise ValueError(f"Need at least 5 images, got {image_count}")
        
        # Update training status
        info = self.get_child(child_id)
        if info.get("training_status") == "training":
            raise ValueError("Training already in progress")
        
        # Create a simple task ID
        task_id = f"train_{child_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.update_child(child_id, 
                         training_status="training", 
                         task_id=task_id,
                         training_started=datetime.now().isoformat())
        
        return task_id
    
    def save_lora_model(self, child_id: str, model_data: bytes, version: int = 1) -> str:
        """Save a LoRA model for a child."""
        if not self.child_dir(child_id).exists():
            raise ValueError(f"Child '{child_id}' does not exist")
        
        models_dir = self.lora_models_dir(child_id)
        models_dir.mkdir(exist_ok=True)
        
        model_file = models_dir / f"lora_v{version}.safetensors"
        
        with open(model_file, "wb") as f:
            f.write(model_data)
        
        # Update child info
        self.update_child(child_id, 
                         model_version=version, 
                         training_status="completed",
                         training_completed=datetime.now().isoformat())
        
        logger.info(f"Saved LoRA model: {model_file}")
        return str(model_file)
    
    def get_lora_model_path(self, child_id: str, version: Optional[int] = None) -> Optional[str]:
        """Get path to LoRA model for a child."""
        models_dir = self.lora_models_dir(child_id)
        
        if version:
            model_file = models_dir / f"lora_v{version}.safetensors"
        else:
            # Get latest version
            model_files = list(models_dir.glob("lora_v*.safetensors"))
            if not model_files:
                return None
            model_file = max(model_files, key=lambda f: f.stat().st_mtime)
        
        return str(model_file) if model_file.exists() else None
    
    def has_lora_model(self, child_id: str) -> bool:
        """Check if child has a LoRA model."""
        return self.get_lora_model_path(child_id) is not None
    
    def get_training_status(self, child_id: str) -> Dict[str, Any]:
        """Get training status for a child."""
        info = self.get_child(child_id)
        if not info:
            raise ValueError(f"Child '{child_id}' not found")
        
        status = info.get("training_status", "not_started")
        task_id = info.get("task_id")
        started = info.get("training_started")
        completed = info.get("training_completed")
        
        return {
            "child_id": child_id,
            "status": status,
            "task_id": task_id,
            "started_at": started,
            "completed_at": completed,
            "progress": 1.0 if status == "completed" else 0.5 if status == "training" else 0.0,
            "has_model": self.has_lora_model(child_id)
        }
    
    # Generated Images
    
    def save_generated_image(self, child_id: str, image_data: bytes, prompt: str, metadata: Optional[Dict] = None) -> str:
        """Save a generated image for a child."""
        gen_dir = self.generated_images_dir(child_id)
        gen_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_file = gen_dir / f"generated_{timestamp}.png"
        metadata_file = gen_dir / f"generated_{timestamp}.json"
        
        # Save image
        with open(image_file, "wb") as f:
            f.write(image_data)
        
        # Save metadata
        meta_data = {
            "prompt": prompt,
            "child_id": child_id,
            "generated_at": datetime.now().isoformat(),
            "image_file": str(image_file),
            **(metadata or {})
        }
        
        with open(metadata_file, "w") as f:
            json.dump(meta_data, f, indent=2)
        
        return str(image_file)
    
    # Statistics
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "total_children": 0,
            "total_training_images": 0,
            "total_lora_models": 0,
            "total_generated_images": 0,
            "storage_size_bytes": 0
        }
        
        for child_dir in self.children_dir.iterdir():
            if child_dir.is_dir():
                stats["total_children"] += 1
                
                # Count training images
                training_dir = child_dir / "training_images"
                if training_dir.exists():
                    training_images = len(list(training_dir.glob("*")))
                    stats["total_training_images"] += training_images
                
                # Count LoRA models
                models_dir = child_dir / "lora_models"
                if models_dir.exists():
                    lora_models = len(list(models_dir.glob("*.safetensors")))
                    stats["total_lora_models"] += lora_models
                
                # Count generated images
                gen_dir = child_dir / "generated_images"
                if gen_dir.exists():
                    gen_images = len(list(gen_dir.glob("*.png")))
                    stats["total_generated_images"] += gen_images
                
                # Calculate size
                for root, dirs, files in os.walk(child_dir):
                    for file in files:
                        file_path = Path(root) / file
                        try:
                            stats["storage_size_bytes"] += file_path.stat().st_size
                        except:
                            pass
        
        return stats


# Global instance
simple_storage = SimpleChildStorage()