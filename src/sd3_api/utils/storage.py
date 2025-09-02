"""Storage utilities for managing training data and models."""

import os
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Base data directory
DATA_DIR = Path("data")
TRAINING_IMAGES_DIR = DATA_DIR / "training_images"
LORA_MODELS_DIR = DATA_DIR / "lora_models"
GENERATED_IMAGES_DIR = DATA_DIR / "generated_images"
TEMP_DIR = DATA_DIR / "temp"

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".mpo"}


class StorageManager:
    """Manages file storage for LoRA training system."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or DATA_DIR
        self.training_images_dir = self.base_dir / "training_images"
        self.lora_models_dir = self.base_dir / "lora_models"
        self.generated_images_dir = self.base_dir / "generated_images"
        self.temp_dir = self.base_dir / "temp"
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.base_dir,
            self.training_images_dir,
            self.lora_models_dir,
            self.generated_images_dir,
            self.temp_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def get_child_training_dir(self, child_id: str) -> Path:
        """Get training images directory for a child."""
        child_dir = self.training_images_dir / child_id
        child_dir.mkdir(exist_ok=True)
        return child_dir
    
    def get_child_model_dir(self, child_id: str) -> Path:
        """Get LoRA model directory for a child."""
        model_dir = self.lora_models_dir / child_id
        model_dir.mkdir(exist_ok=True)
        return model_dir
    
    def save_training_image(self, child_id: str, filename: str, image_data: bytes) -> str:
        """
        Save training image for a child.
        
        Args:
            child_id: Child identifier
            filename: Original filename
            image_data: Image file bytes
            
        Returns:
            Path to saved file
        """
        # Validate file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format: {file_ext}")
        
        # Create child directory
        child_dir = self.get_child_training_dir(child_id)
        
        # Generate unique filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(filename).stem
        safe_filename = f"{timestamp}_{base_name}{file_ext}"
        file_path = child_dir / safe_filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(image_data)
        
        logger.info(f"Saved training image: {file_path}")
        return str(file_path)
    
    def get_training_images(self, child_id: str) -> List[Path]:
        """Get all training images for a child."""
        child_dir = self.get_child_training_dir(child_id)
        if not child_dir.exists():
            return []
        
        images = []
        for file_path in child_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                images.append(file_path)
        
        return sorted(images)
    
    def get_lora_model_path(self, child_id: str, version: int = 1) -> Path:
        """Get LoRA model file path for a child."""
        model_dir = self.get_child_model_dir(child_id)
        return model_dir / f"lora_v{version}.safetensors"
    
    def save_generated_image(self, image_data: bytes, prompt: str, 
                           children_ids: Optional[List[str]] = None) -> str:
        """
        Save generated image.
        
        Args:
            image_data: Image bytes
            prompt: Generation prompt
            children_ids: List of child IDs used in generation
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Create filename with child IDs if provided
        if children_ids:
            child_part = "_".join(children_ids)
            filename = f"{timestamp}_{child_part}.png"
        else:
            filename = f"{timestamp}.png"
        
        file_path = self.generated_images_dir / filename
        
        with open(file_path, "wb") as f:
            f.write(image_data)
        
        logger.info(f"Saved generated image: {file_path}")
        return str(file_path)
    
    def cleanup_child_data(self, child_id: str) -> None:
        """Remove all data for a child."""
        # Remove training images
        training_dir = self.training_images_dir / child_id
        if training_dir.exists():
            shutil.rmtree(training_dir)
            logger.info(f"Removed training directory: {training_dir}")
        
        # Remove LoRA models
        model_dir = self.lora_models_dir / child_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info(f"Removed model directory: {model_dir}")
    
    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        def get_dir_size(directory: Path) -> int:
            """Get total size of directory in bytes."""
            total = 0
            try:
                for entry in directory.rglob('*'):
                    if entry.is_file():
                        total += entry.stat().st_size
            except (OSError, PermissionError):
                pass
            return total
        
        def get_file_count(directory: Path) -> int:
            """Get total number of files in directory."""
            count = 0
            try:
                for entry in directory.rglob('*'):
                    if entry.is_file():
                        count += 1
            except (OSError, PermissionError):
                pass
            return count
        
        stats = {
            "training_images": {
                "size_bytes": get_dir_size(self.training_images_dir),
                "file_count": get_file_count(self.training_images_dir),
                "children_count": len(list(self.training_images_dir.iterdir())) if self.training_images_dir.exists() else 0
            },
            "lora_models": {
                "size_bytes": get_dir_size(self.lora_models_dir),
                "file_count": get_file_count(self.lora_models_dir),
                "models_count": len(list(self.lora_models_dir.iterdir())) if self.lora_models_dir.exists() else 0
            },
            "generated_images": {
                "size_bytes": get_dir_size(self.generated_images_dir),
                "file_count": get_file_count(self.generated_images_dir)
            }
        }
        
        return stats
    
    def validate_image_file(self, file_path: str) -> dict:
        """
        Validate an image file and return metadata.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dictionary with validation results and metadata
        """
        from PIL import Image
        
        try:
            file_path_obj = Path(file_path)
            
            # Check file exists
            if not file_path_obj.exists():
                return {"valid": False, "error": "File does not exist"}
            
            # Check file extension
            file_ext = file_path_obj.suffix.lower()
            if file_ext not in SUPPORTED_IMAGE_FORMATS:
                return {"valid": False, "error": f"Unsupported format: {file_ext}"}
            
            # Try to open and validate image
            with Image.open(file_path) as img:
                width, height = img.size
                format_name = img.format
                mode = img.mode
            
            # Get file size
            file_size = file_path_obj.stat().st_size
            
            return {
                "valid": True,
                "width": width,
                "height": height,
                "format": format_name,
                "mode": mode,
                "file_size": file_size,
                "file_extension": file_ext
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}


# Global storage manager instance
storage_manager = StorageManager()