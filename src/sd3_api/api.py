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
from .lora.pipeline import LoRASD3Pipeline
from .lora_api import lora_router
from .lora_models import GenerateWithChildrenRequest, GenerateWithChildrenResponse
from .database.connection import create_tables

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Server starting up - beginning model load and database setup...")
    
    # Create database tables
    create_tables()
    logger.info("Database tables created/verified")
    
    # Load pipeline
    asyncio.create_task(load_pipeline_async())
    yield
    # Shutdown (nothing to do for now)

# Initialize FastAPI app
app = FastAPI(
    title="SD3 Large LoRA API",
    description="Stable Diffusion 3.5 Large image generation service with LoRA training",
    version="0.6.0",
    lifespan=lifespan,
)

# Include LoRA router
app.include_router(lora_router)

# Global pipeline instance (now using LoRA-enabled pipeline)
pipeline: Optional[LoRASD3Pipeline] = None

def get_pipeline() -> LoRASD3Pipeline:
    """Get the global pipeline instance."""
    global pipeline
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return pipeline

async def load_pipeline_async():
    """Load the pipeline in a background task."""
    global pipeline
    try:
        logger.info("Starting LoRA-enabled model loading in background...")
        # Create LoRA-enabled pipeline without eager loading first
        pipeline = LoRASD3Pipeline(eager_load=False)
        # Then initialize it
        await asyncio.get_event_loop().run_in_executor(None, pipeline.initialize_async)
        logger.info("LoRA-enabled model loading completed successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Keep the pipeline object so we can show the error status
        if pipeline is None:
            pipeline = LoRASD3Pipeline(eager_load=False)
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
        device=status,
        version="0.6.0"
    )


@app.get("/version")
async def get_version():
    """Get API version and build information."""
    import os
    import datetime
    
    return {
        "version": "0.6.0",
        "api_title": "SD3 Large LoRA API",
        "build": "2025-09-04-import-diagnosis",
        "features": [
            "SD3.5 Large image generation",
            "LoRA training system", 
            "Multi-child generation",
            "Celery background tasks",
            "Redis job queue"
        ],
        "changes": [
            "Added import failure diagnosis for training tasks",
            "Created test_training_imports task to isolate dependency issues", 
            "Enhanced debugging to catch pre-execution failures",
            "Added debug endpoint to test specific training imports"
        ],
        "build_date": datetime.datetime.now().isoformat(),
        "python_path": os.environ.get("PYTHONPATH", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "container_info": "RunPod PyTorch 2.2.1 CUDA 12.1.1"
    }


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


@app.post("/generate/with-children", response_model=Union[GenerateWithChildrenResponse, ErrorResponse])
async def generate_image_with_children(
    request: GenerateWithChildrenRequest
) -> Union[GenerateWithChildrenResponse, ErrorResponse]:
    """Generate image with specific children using LoRA adapters."""
    
    try:
        pip = get_pipeline()
        
        # Check if pipeline is still loading
        if pip.is_loading:
            raise HTTPException(status_code=503, detail="Model is still loading, please wait...")
        
        if pip.load_error:
            raise HTTPException(status_code=500, detail=f"Model failed to load: {pip.load_error}")
            
        if not pip.is_ready:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        # Generate the image with LoRA support
        image, used_children = pip.generate_image_with_lora(
            prompt=request.prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance,
            child_ids=request.children_ids
        )
        
        # Convert image to base64
        img_io = io.BytesIO()
        image.save(img_io, format="PNG")
        img_io.seek(0)
        base64_img = base64.b64encode(img_io.read()).decode("utf-8")
        
        # Create modified prompt for response
        modified_prompt = pip._replace_child_tokens_in_prompt(request.prompt, used_children)
        
        # Save to generation history
        from .utils.storage import storage_manager
        from .database.manager import db_manager
        
        # Save generated image
        image_path = storage_manager.save_generated_image(
            base64.b64decode(base64_img),
            request.prompt,
            used_children
        )
        
        # Add to generation history
        db_manager.add_generation_history(
            prompt=request.prompt,
            children_ids=used_children,
            image_path=image_path,
            steps=request.steps,
            guidance_scale=request.guidance,
            seed=request.seed
        )
        
        return GenerateWithChildrenResponse(
            image=base64_img,
            children_used=used_children,
            original_prompt=request.prompt,
            modified_prompt=modified_prompt,
            seed=request.seed
        )
        
    except Exception as e:
        logger.error(f"Image generation with children failed: {e}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error=str(e)).model_dump()
        )