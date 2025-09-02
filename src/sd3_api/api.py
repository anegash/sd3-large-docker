"""FastAPI application for SD3 image generation."""

import asyncio
import base64
import io
import logging
from contextlib import asynccontextmanager
from typing import Optional, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from .models import GenerateRequest, GenerateResponse, ErrorResponse, HealthResponse
from .pipeline import SD3Pipeline

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Server starting up - beginning model load...")
    asyncio.create_task(load_pipeline_async())
    yield
    # Shutdown (nothing to do for now)

# Initialize FastAPI app
app = FastAPI(
    title="SD3 Large API",
    description="Stable Diffusion 3.5 Large image generation service",
    version="0.1.0",
    lifespan=lifespan,
)

# Global pipeline instance 
pipeline: Optional[SD3Pipeline] = None

def get_pipeline() -> SD3Pipeline:
    """Get the global pipeline instance."""
    global pipeline
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return pipeline

async def load_pipeline_async():
    """Load the pipeline in a background task."""
    global pipeline
    try:
        logger.info("Starting model loading in background...")
        # Create pipeline without eager loading first
        pipeline = SD3Pipeline(eager_load=False)
        # Then initialize it
        await asyncio.get_event_loop().run_in_executor(None, pipeline.initialize_async)
        logger.info("Model loading completed successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Keep the pipeline object so we can show the error status
        if pipeline is None:
            pipeline = SD3Pipeline(eager_load=False)
            pipeline.load_error = str(e)


@app.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    global pipeline
    
    if pipeline is None:
        status = "Initializing..."
    else:
        status = pipeline.status
    
    return HealthResponse(
        message="Stable Diffusion 3.5 API is running!",
        device=status
    )


@app.get("/generate", response_model=Union[GenerateResponse, ErrorResponse])
async def generate_image(
    prompt: str = Query(..., description="Text prompt for image generation"),
    steps: int = Query(15, ge=1, le=150, description="Number of inference steps"),
    guidance: float = Query(7.5, ge=1.0, le=15.0, description="Guidance scale")
) -> Union[GenerateResponse, ErrorResponse]:
    """Generate an image from a text prompt."""
    
    try:
        pip = get_pipeline()
        
        # Check if pipeline is still loading
        if pip.is_loading:
            raise HTTPException(status_code=503, detail="Model is still loading, please wait...")
        
        if pip.load_error:
            raise HTTPException(status_code=500, detail=f"Model failed to load: {pip.load_error}")
            
        if not pip.is_ready:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        # Generate the image
        image = pip.generate_image(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance
        )
        
        # Convert image to base64
        img_io = io.BytesIO()
        image.save(img_io, format="PNG")
        img_io.seek(0)
        base64_img = base64.b64encode(img_io.read()).decode("utf-8")
        
        return GenerateResponse(image=base64_img)
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error=str(e)).model_dump()
        )


@app.post("/generate", response_model=Union[GenerateResponse, ErrorResponse])
async def generate_image_post(
    request: GenerateRequest
) -> Union[GenerateResponse, ErrorResponse]:
    """Generate an image from a text prompt using POST method."""
    
    return await generate_image(
        prompt=request.prompt,
        steps=request.steps,
        guidance=request.guidance
    )