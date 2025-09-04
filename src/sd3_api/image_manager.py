"""Image management for LoRA training."""

import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class ImageManager:
    """Manages training images for LoRA training."""
    
    def __init__(self, images_dir: str = "training_images"):
        from .config import WORKSPACE_DIR, ensure_workspace_dirs
        
        self.images_base_dir = Path(WORKSPACE_DIR) / images_dir
        self.images_base_dir.mkdir(parents=True, exist_ok=True)
        ensure_workspace_dirs()
    
    def upload_images(self, person_id: str, images: List[Image.Image]) -> dict:
        """
        Upload and save images for a person.
        
        Args:
            person_id: Unique identifier for the person
            images: List of PIL Images to save
            
        Returns:
            Dictionary with upload status
        """
        logger.info(f"Uploading {len(images)} images for person_id: {person_id}")
        
        # Create person directory
        person_dir = self.images_base_dir / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        
        # Get existing image count
        existing_images = list(person_dir.glob("*.jpg"))
        start_index = len(existing_images)
        
        # Save new images
        saved_count = 0
        for i, image in enumerate(images):
            image_path = person_dir / f"image_{start_index + i + 1:03d}.jpg"
            
            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])
                image = rgb_image
            
            # Resize if too large (optional optimization)
            if image.width > 1024 or image.height > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            image.save(image_path, "JPEG", quality=95)
            saved_count += 1
        
        # Update metadata
        total_images = start_index + saved_count
        self._save_images_metadata(person_id, total_images)
        
        logger.info(f"Saved {saved_count} images for {person_id}, total: {total_images}")
        
        return {
            "person_id": person_id,
            "num_images": saved_count,
            "total_images": total_images,
            "status": "uploaded"
        }
    
    def get_images(self, person_id: str) -> List[Image.Image]:
        """
        Load all images for a person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            List of PIL Images
        """
        person_dir = self.images_base_dir / person_id
        if not person_dir.exists():
            return []
        
        images = []
        image_files = sorted(person_dir.glob("*.jpg"))
        
        for image_path in image_files:
            try:
                image = Image.open(image_path)
                images.append(image.copy())  # Copy to avoid file handle issues
                image.close()
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
        
        logger.info(f"Loaded {len(images)} images for {person_id}")
        return images
    
    def copy_images(self, source_person_id: str, target_person_id: str) -> dict:
        """
        Copy images from one person to another.
        
        Args:
            source_person_id: Source person ID
            target_person_id: Target person ID
            
        Returns:
            Dictionary with copy status
        """
        source_dir = self.images_base_dir / source_person_id
        target_dir = self.images_base_dir / target_person_id
        
        if not source_dir.exists():
            raise ValueError(f"No images found for source person_id: {source_person_id}")
        
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all images
        copied_count = 0
        for image_file in source_dir.glob("*.jpg"):
            target_file = target_dir / image_file.name
            shutil.copy2(image_file, target_file)
            copied_count += 1
        
        # Update metadata
        self._save_images_metadata(target_person_id, copied_count)
        
        logger.info(f"Copied {copied_count} images from {source_person_id} to {target_person_id}")
        
        return {
            "source_person_id": source_person_id,
            "target_person_id": target_person_id,
            "num_images": copied_count,
            "status": "copied"
        }
    
    def get_images_status(self, person_id: str) -> dict:
        """
        Get status of images for a person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            Dictionary with image status
        """
        person_dir = self.images_base_dir / person_id
        
        if not person_dir.exists():
            return {
                "person_id": person_id,
                "num_images": 0,
                "images_ready": False
            }
        
        num_images = len(list(person_dir.glob("*.jpg")))
        
        return {
            "person_id": person_id,
            "num_images": num_images,
            "images_ready": num_images >= 5  # Minimum 5 images for training
        }
    
    def list_available_image_sets(self) -> List[dict]:
        """
        List all available image sets.
        
        Returns:
            List of image set information
        """
        image_sets = []
        
        for person_dir in self.images_base_dir.iterdir():
            if person_dir.is_dir():
                status = self.get_images_status(person_dir.name)
                image_sets.append(status)
        
        return sorted(image_sets, key=lambda x: x["person_id"])
    
    def delete_images(self, person_id: str) -> bool:
        """
        Delete all images for a person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            True if deleted successfully
        """
        person_dir = self.images_base_dir / person_id
        metadata_path = self.images_base_dir / f"{person_id}_images_metadata.json"
        
        try:
            if person_dir.exists():
                shutil.rmtree(person_dir)
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"Deleted images for {person_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete images for {person_id}: {e}")
            return False
    
    def _save_images_metadata(self, person_id: str, num_images: int) -> None:
        """Save metadata for uploaded images."""
        import datetime
        
        metadata = {
            "person_id": person_id,
            "num_images": num_images,
            "created_at": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        metadata_path = self.images_base_dir / f"{person_id}_images_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)