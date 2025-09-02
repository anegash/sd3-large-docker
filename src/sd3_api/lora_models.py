"""Pydantic models for LoRA API endpoints."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field

from .database.models import TrainingStatus


class ChildCreateRequest(BaseModel):
    """Request model for creating a new child."""
    id: str = Field(..., description="Unique child identifier (e.g., 'child_001')")
    name: str = Field(..., description="Child's name")
    description: Optional[str] = Field(None, description="Optional description")


class ChildResponse(BaseModel):
    """Response model for child information."""
    id: str
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    training_image_count: int = Field(0, description="Number of training images")
    has_active_model: bool = Field(False, description="Whether child has an active LoRA model")


class TrainingImageResponse(BaseModel):
    """Response model for training image information."""
    id: int
    filename: str
    description: Optional[str]
    width: Optional[int]
    height: Optional[int]
    file_size: Optional[int]
    uploaded_at: datetime


class UploadImageResponse(BaseModel):
    """Response model for image upload."""
    filename: str
    file_path: str
    description: Optional[str]
    metadata: Dict[str, Any]
    status: str


class BatchUploadResponse(BaseModel):
    """Response model for batch image upload."""
    uploaded_images: List[UploadImageResponse]
    total_uploaded: int
    child_id: str


class TrainingConfigRequest(BaseModel):
    """Request model for training configuration."""
    lora_rank: int = Field(64, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(32, ge=1, le=512, description="LoRA alpha")
    lora_dropout: float = Field(0.1, ge=0.0, le=1.0, description="LoRA dropout")
    learning_rate: float = Field(1e-4, ge=1e-6, le=1e-2, description="Learning rate")
    training_steps: int = Field(1000, ge=100, le=5000, description="Number of training steps")
    batch_size: int = Field(1, ge=1, le=8, description="Batch size")
    gradient_accumulation_steps: int = Field(4, ge=1, le=16, description="Gradient accumulation steps")
    max_grad_norm: float = Field(1.0, ge=0.1, le=10.0, description="Max gradient norm")
    use_8bit_adam: bool = Field(False, description="Use 8-bit Adam optimizer")
    mixed_precision: str = Field("fp16", description="Mixed precision mode")
    save_steps: int = Field(250, ge=50, le=1000, description="Save checkpoint every N steps")
    validation_steps: int = Field(100, ge=50, le=500, description="Validation every N steps")
    seed: int = Field(42, ge=0, description="Random seed")


class StartTrainingRequest(BaseModel):
    """Request model for starting training."""
    training_config: Optional[TrainingConfigRequest] = Field(None, description="Optional custom training configuration")


class LoRAModelResponse(BaseModel):
    """Response model for LoRA model information."""
    id: int
    child_id: str
    version: int
    training_status: str
    training_progress: float
    training_loss: Optional[float]
    error_message: Optional[str]
    created_at: datetime
    training_started_at: Optional[datetime]
    training_completed_at: Optional[datetime]
    is_active: bool


class TrainingStatusResponse(BaseModel):
    """Response model for training status."""
    model_id: int
    child_id: str
    status: str
    progress: float = Field(ge=0.0, le=1.0, description="Training progress (0.0 to 1.0)")
    message: str
    loss: Optional[float] = Field(None, description="Current training loss")
    error: Optional[str] = Field(None, description="Error message if failed")
    task_id: Optional[str] = Field(None, description="Celery task ID")
    started_at: Optional[datetime]
    estimated_completion: Optional[datetime]


class GenerateWithChildrenRequest(BaseModel):
    """Request model for generating images with specific children."""
    prompt: str = Field(..., description="Text prompt (can include {child_id} tokens)")
    children_ids: Optional[List[str]] = Field(None, description="Explicit list of child IDs to use")
    steps: int = Field(15, ge=1, le=150, description="Number of inference steps")
    guidance: float = Field(7.5, ge=1.0, le=15.0, description="Guidance scale")
    width: int = Field(1024, ge=512, le=2048, description="Image width")
    height: int = Field(1024, ge=512, le=2048, description="Image height")
    seed: Optional[int] = Field(None, description="Random seed for generation")


class GenerateWithChildrenResponse(BaseModel):
    """Response model for image generation with children."""
    image: str = Field(..., description="Base64-encoded PNG image")
    children_used: List[str] = Field(..., description="List of child IDs used in generation")
    original_prompt: str = Field(..., description="Original prompt")
    modified_prompt: str = Field(..., description="Modified prompt with token replacements")
    seed: Optional[int] = Field(None, description="Seed used for generation")


class ChildStatisticsResponse(BaseModel):
    """Response model for child statistics."""
    child_id: str
    training_images: Dict[str, Any] = Field(..., description="Training image statistics")
    model_info: Optional[Dict[str, Any]] = Field(None, description="LoRA model information")
    generation_history: Dict[str, Any] = Field(..., description="Generation history statistics")


class SystemStatsResponse(BaseModel):
    """Response model for system statistics."""
    total_children: int
    total_training_images: int
    total_lora_models: int
    active_models: int
    storage_stats: Dict[str, Any]
    training_queue_size: int
    loaded_loras: List[str]