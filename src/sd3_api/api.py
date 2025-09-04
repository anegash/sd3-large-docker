"""FastAPI application for SD3 image generation."""

import asyncio
import base64
import io
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image

from .models import (
    GenerateRequest, GenerateResponse, ErrorResponse, HealthResponse,
    TrainLoRARequest, TrainLoRAResponse, LoRAListResponse
)
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
    guidance: float = Query(7.5, ge=1.0, le=15.0, description="Guidance scale"),
    person_id: Optional[str] = Query(None, description="Person ID for LoRA weights")
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
            guidance_scale=guidance,
            person_id=person_id
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
        guidance=request.guidance,
        person_id=request.person_id
    )


@app.post("/train-lora", response_model=Union[TrainLoRAResponse, ErrorResponse])
async def train_lora(
    person_id: str = Form(..., description="Unique identifier for the person"),
    num_train_epochs: int = Form(100, description="Number of training epochs"),
    learning_rate: float = Form(1e-4, description="Learning rate for training"),
    files: List[UploadFile] = File(..., description="Training images (5-20 images)")
) -> Union[TrainLoRAResponse, ErrorResponse]:
    """Train LoRA weights for a specific person using uploaded images."""
    
    try:
        # Validate number of images
        if len(files) < 5 or len(files) > 20:
            raise HTTPException(
                status_code=400, 
                detail="Please upload between 5 and 20 images for training"
            )
        
        pip = get_pipeline()
        
        if not pip.is_ready:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        # Load and validate images
        images = []
        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not a valid image"
                )
            
            # Read and convert to PIL Image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            images.append(image)
        
        logger.info(f"Starting LoRA training for {person_id} with {len(images)} images")
        
        # Train LoRA (this is synchronous and may take a while)
        pip.lora_trainer.train_lora(
            person_id=person_id,
            images=images,
            pipeline=pip.pipeline,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate
        )
        
        return TrainLoRAResponse(
            message=f"LoRA training completed successfully for {person_id}",
            person_id=person_id
        )
        
    except Exception as e:
        logger.error(f"LoRA training failed: {e}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error=str(e)).model_dump()
        )


@app.get("/lora", response_model=LoRAListResponse)
async def list_lora_models() -> LoRAListResponse:
    """List all available LoRA person IDs."""
    
    try:
        pip = get_pipeline()
        person_ids = pip.get_available_loras()
        
        return LoRAListResponse(person_ids=person_ids)
        
    except Exception as e:
        logger.error(f"Failed to list LoRA models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/lora/{person_id}")
async def delete_lora_model(person_id: str):
    """Delete LoRA weights for a specific person."""
    
    try:
        pip = get_pipeline()
        lora_path = pip.lora_trainer.get_lora_path(person_id)
        
        if lora_path is None:
            raise HTTPException(status_code=404, detail=f"LoRA model for {person_id} not found")
        
        # Remove the directory
        import shutil
        shutil.rmtree(lora_path)
        
        # Remove metadata file
        metadata_path = pip.lora_trainer.lora_weights_dir / f"{person_id}_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Unload if currently loaded
        if pip.current_lora_id == person_id:
            pip.unload_lora_weights()
        
        return {"message": f"Successfully deleted LoRA model for {person_id}"}
        
    except Exception as e:
        logger.error(f"Failed to delete LoRA model: {e}")
        raise HTTPException(status_code=500, detail=str(e))