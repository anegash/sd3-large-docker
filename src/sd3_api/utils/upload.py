"""File upload utilities for training images."""

import os
import io
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import UploadFile, HTTPException
from PIL import Image
import aiofiles

from .storage import storage_manager, SUPPORTED_IMAGE_FORMATS

logger = logging.getLogger(__name__)

# Upload constraints
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB (increased for high-res iPhone photos)
MAX_FILES_PER_CHILD = 50
MIN_IMAGE_SIZE = 512  # Minimum width/height
MAX_IMAGE_SIZE = 6000  # Maximum width/height (increased for iPhone photos)


class ImageUploadValidator:
    """Validates uploaded images for training."""
    
    @staticmethod
    def validate_file_size(file_size: int) -> None:
        """Validate file size."""
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
    
    @staticmethod
    def validate_file_format(filename: str) -> None:
        """Validate file format."""
        file_ext = Path(filename).suffix.lower()
        if file_ext not in SUPPORTED_IMAGE_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {file_ext}. Supported: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
            )
    
    @staticmethod
    async def validate_image_content(file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Validate image content and return metadata.
        
        Args:
            file_content: Image file bytes
            filename: Original filename
            
        Returns:
            Dictionary with image metadata
            
        Raises:
            HTTPException: If image is invalid
        """
        try:
            # Try to open the image
            with Image.open(io.BytesIO(file_content)) as img:
                width, height = img.size
                format_name = img.format
                mode = img.mode
                
                # Validate dimensions
                if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Image too small. Minimum size: {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}px"
                    )
                
                if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Image too large. Maximum size: {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE}px"
                    )
                
                # Check if image is corrupted or has issues
                img.verify()
                
                return {
                    "width": width,
                    "height": height,
                    "format": format_name,
                    "mode": mode,
                    "filename": filename,
                    "size_bytes": len(file_content)
                }
                
        except Exception as e:
            logger.error(f"Image validation failed for {filename}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {filename}. {str(e)}"
            )


class ImageUploadHandler:
    """Handles image uploads for training."""
    
    def __init__(self):
        self.validator = ImageUploadValidator()
    
    async def upload_training_images(
        self, 
        child_id: str, 
        files: List[UploadFile],
        descriptions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Upload multiple training images for a child.
        
        Args:
            child_id: Child identifier
            files: List of uploaded files
            descriptions: Optional descriptions for each image
            
        Returns:
            List of upload results
        """
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > MAX_FILES_PER_CHILD:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files. Maximum: {MAX_FILES_PER_CHILD}"
            )
        
        # Ensure descriptions list has same length as files
        if descriptions is None:
            descriptions = [None] * len(files)
        elif len(descriptions) != len(files):
            raise HTTPException(
                status_code=400,
                detail="Number of descriptions must match number of files"
            )
        
        results = []
        uploaded_files = []  # Track uploaded files for cleanup on failure
        
        try:
            for idx, (file, description) in enumerate(zip(files, descriptions)):
                logger.info(f"Processing file {idx + 1}/{len(files)}: {file.filename}")
                
                # Validate file
                if not file.filename:
                    raise HTTPException(status_code=400, detail="Invalid filename")
                
                self.validator.validate_file_format(file.filename)
                
                # Read file content
                file_content = await file.read()
                self.validator.validate_file_size(len(file_content))
                
                # Validate image content
                image_metadata = await self.validator.validate_image_content(
                    file_content, file.filename
                )
                
                # Save file
                file_path = storage_manager.save_training_image(
                    child_id, file.filename, file_content
                )
                uploaded_files.append(file_path)
                
                # Add to results
                result = {
                    "filename": file.filename,
                    "file_path": file_path,
                    "description": description,
                    "metadata": image_metadata,
                    "status": "success"
                }
                results.append(result)
                
                logger.info(f"Successfully uploaded {file.filename} to {file_path}")
            
            return results
            
        except Exception as e:
            # Cleanup uploaded files on failure
            for file_path in uploaded_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Cleaned up {file_path} after upload failure")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup {file_path}: {cleanup_error}")
            
            # Re-raise the original exception
            raise
    
    async def upload_single_image(
        self, 
        child_id: str, 
        file: UploadFile,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload a single training image.
        
        Args:
            child_id: Child identifier
            file: Uploaded file
            description: Optional description
            
        Returns:
            Upload result
        """
        results = await self.upload_training_images(
            child_id, [file], [description] if description else None
        )
        return results[0]
    
    def get_upload_stats(self, child_id: str) -> Dict[str, Any]:
        """
        Get upload statistics for a child.
        
        Args:
            child_id: Child identifier
            
        Returns:
            Statistics dictionary
        """
        # Get training images from storage
        image_paths = storage_manager.get_training_images(child_id)
        
        if not image_paths:
            return {
                "total_images": 0,
                "total_size_bytes": 0,
                "formats": {},
                "size_distribution": {}
            }
        
        total_size = 0
        formats = {}
        size_distribution = {"small": 0, "medium": 0, "large": 0}
        
        for image_path in image_paths:
            try:
                # Get file size
                file_size = image_path.stat().st_size
                total_size += file_size
                
                # Get format
                file_ext = image_path.suffix.lower()
                formats[file_ext] = formats.get(file_ext, 0) + 1
                
                # Get image dimensions for size distribution
                with Image.open(image_path) as img:
                    width, height = img.size
                    max_dimension = max(width, height)
                    
                    if max_dimension < 800:
                        size_distribution["small"] += 1
                    elif max_dimension < 1200:
                        size_distribution["medium"] += 1
                    else:
                        size_distribution["large"] += 1
                        
            except Exception as e:
                logger.warning(f"Failed to get stats for {image_path}: {e}")
        
        return {
            "total_images": len(image_paths),
            "total_size_bytes": total_size,
            "formats": formats,
            "size_distribution": size_distribution,
            "can_add_more": len(image_paths) < MAX_FILES_PER_CHILD
        }


# Global upload handler instance
upload_handler = ImageUploadHandler()