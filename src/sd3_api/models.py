"""Pydantic models for API request/response."""

from typing import List, Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request model for image generation."""
    
    prompt: str = Field(..., description="Text prompt for image generation")
    steps: int = Field(15, ge=1, le=150, description="Number of inference steps")
    guidance: float = Field(7.5, ge=1.0, le=15.0, description="Guidance scale")
    person_id: Optional[str] = Field(None, description="Person ID for LoRA weights")


class GenerateResponse(BaseModel):
    """Response model for successful image generation."""
    
    image: str = Field(..., description="Base64-encoded PNG image")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(..., description="Error message")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    message: str = Field(..., description="Health status message")
    device: str = Field(..., description="Device being used for inference")


class TrainLoRARequest(BaseModel):
    """Request model for LoRA training."""
    
    person_id: str = Field(..., description="Unique identifier for the person")
    num_train_epochs: int = Field(100, ge=10, le=500, description="Number of training epochs")
    learning_rate: float = Field(1e-4, gt=0, le=1e-2, description="Learning rate for training")


class TrainLoRAResponse(BaseModel):
    """Response model for LoRA training."""
    
    message: str = Field(..., description="Training status message")
    person_id: str = Field(..., description="Person ID being trained")


class LoRAListResponse(BaseModel):
    """Response model for listing available LoRA models."""
    
    person_ids: List[str] = Field(..., description="List of available person IDs")