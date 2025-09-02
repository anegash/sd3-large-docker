"""Pydantic models for API request/response."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request model for image generation."""
    
    prompt: str = Field(..., description="Text prompt for image generation")
    steps: int = Field(15, ge=1, le=150, description="Number of inference steps")
    guidance: float = Field(7.5, ge=1.0, le=15.0, description="Guidance scale")


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