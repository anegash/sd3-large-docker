"""FastAPI application for SDXL image generation."""

import asyncio
import base64
import io
import logging
import subprocess
from contextlib import asynccontextmanager
from typing import List, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from .models import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ImagesListResponse,
    ImagesStatusResponse,
    LoRAListResponse,
    TrainLoRAResponse,
    UploadImagesResponse,
)
from .pipeline import SDXLPipeline

logger = logging.getLogger(__name__)


def get_version_info() -> dict:
    """Get version information from git."""
    try:
        # Get current branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
            cwd=".", 
            universal_newlines=True
        ).strip()
        
        # Get current commit hash
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], 
            cwd=".", 
            universal_newlines=True
        ).strip()
        
        return {
            "version": "1.0.9",  # API version - increment for each push
            "branch": branch,
            "commit": commit
        }
    except Exception as e:
        logger.warning(f"Failed to get git info: {e}")
        return {
            "version": "1.0.9",
            "branch": "unknown", 
            "commit": "unknown"
        }


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
    title="SDXL API",
    description="Stable Diffusion XL image generation service with LoRA training",
    version="0.1.0",
    lifespan=lifespan,
)

# Global pipeline instance
pipeline: Optional[SDXLPipeline] = None


def get_pipeline() -> SDXLPipeline:
    """Get the global SDXL pipeline instance."""
    global pipeline
    if pipeline is None:
        raise RuntimeError("SDXL pipeline not initialized")
    return pipeline


async def load_pipeline_async():
    """Load the SDXL pipeline in a background task."""
    global pipeline
    try:
        logger.info("Starting SDXL model loading in background...")
        # Create pipeline without eager loading first
        pipeline = SDXLPipeline(eager_load=False)
        # Then initialize it
        await asyncio.get_event_loop().run_in_executor(None, pipeline.initialize_async)
        logger.info("SDXL model loading completed successfully")
    except Exception as e:
        logger.error(f"Failed to load SDXL model: {e}")
        # Keep the pipeline object so we can show the error status
        if pipeline is None:
            pipeline = SDXLPipeline(eager_load=False)
            pipeline.load_error = str(e)


@app.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    global pipeline

    if pipeline is None:
        status = "Initializing..."
    else:
        status = pipeline.status

    # Get version information
    version_info = get_version_info()

    return HealthResponse(
        message="Stable Diffusion XL API is running!", 
        device=status,
        version=version_info["version"],
        branch=version_info["branch"], 
        commit=version_info["commit"]
    )


@app.get("/generate", response_model=Union[GenerateResponse, ErrorResponse])
async def generate_image(
    prompt: str = Query(..., description="Text prompt for image generation"),
    steps: int = Query(20, ge=1, le=150, description="Number of inference steps"),
    guidance: float = Query(7.5, ge=1.0, le=15.0, description="Guidance scale"),
    width: int = Query(
        1024, ge=512, le=2048, description="Image width (divisible by 8)"
    ),
    height: int = Query(
        1024, ge=512, le=2048, description="Image height (divisible by 8)"
    ),
    person_id: Optional[str] = Query(None, description="Person ID for LoRA weights"),
) -> Union[GenerateResponse, ErrorResponse]:
    """Generate an image from a text prompt."""

    try:
        pip = get_pipeline()

        # Check if pipeline is still loading
        if pip.is_loading:
            raise HTTPException(
                status_code=503, detail="Model is still loading, please wait..."
            )

        if pip.load_error:
            raise HTTPException(
                status_code=500, detail=f"Model failed to load: {pip.load_error}"
            )

        if not pip.is_ready:
            raise HTTPException(status_code=503, detail="Model not ready")

        # Generate the image
        image = pip.generate_image(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            person_id=person_id,
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
            status_code=500, content=ErrorResponse(error=str(e)).model_dump()
        )


@app.post("/generate", response_model=Union[GenerateResponse, ErrorResponse])
async def generate_image_post(
    request: GenerateRequest,
) -> Union[GenerateResponse, ErrorResponse]:
    """Generate an image from a text prompt using POST method."""

    return await generate_image(
        prompt=request.prompt,
        steps=request.steps,
        guidance=request.guidance,
        width=request.width,
        height=request.height,
        person_id=request.person_id,
    )


@app.post("/upload-images", response_model=Union[UploadImagesResponse, ErrorResponse])
async def upload_images(
    person_id: str = Form(..., description="Unique identifier for the person"),
    files: List[UploadFile] = File(..., description="Training images to upload"),
) -> Union[UploadImagesResponse, ErrorResponse]:
    """Upload training images for a specific person."""

    try:
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No images provided")

        # Load and validate images
        images = []
        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400, detail=f"File {file.filename} is not a valid image"
                )

            # Read and convert to PIL Image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            images.append(image)

        logger.info(f"Uploading {len(images)} images for {person_id}")

        # Use image manager to save images
        from .image_manager import ImageManager

        image_manager = ImageManager()

        result = image_manager.upload_images(person_id, images)

        return UploadImagesResponse(
            message=f"Successfully uploaded {result['num_images']} images for {person_id}",
            person_id=person_id,
            num_images=result["num_images"],
            total_images=result["total_images"],
        )

    except Exception as e:
        logger.error(f"Image upload failed: {e}")
        return JSONResponse(
            status_code=500, content=ErrorResponse(error=str(e)).model_dump()
        )


@app.post("/train-lora", response_model=Union[TrainLoRAResponse, ErrorResponse])
async def train_lora(
    person_id: str = Form(..., description="Unique identifier for the person"),
    num_train_epochs: int = Form(100, description="Number of training epochs"),
    learning_rate: float = Form(1e-4, description="Learning rate for training"),
    source_person_id: Optional[str] = Form(
        None, description="Copy images from existing person_id"
    ),
) -> Union[TrainLoRAResponse, ErrorResponse]:
    """Train LoRA weights for a specific person."""

    try:
        pip = get_pipeline()

        if not pip.is_ready:
            raise HTTPException(status_code=503, detail="Model not ready")

        # Use stored images for training
        logger.info(f"Starting LoRA training for {person_id} using stored images")

        pip.lora_trainer.train_lora_from_images(
            person_id=person_id,
            pipeline=pip.pipeline,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            source_person_id=source_person_id,
        )

        return TrainLoRAResponse(
            message=f"LoRA training completed successfully for {person_id}",
            person_id=person_id,
        )

    except Exception as e:
        logger.error(f"LoRA training failed: {e}")
        return JSONResponse(
            status_code=500, content=ErrorResponse(error=str(e)).model_dump()
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


@app.get("/images", response_model=ImagesListResponse)
async def list_image_sets() -> ImagesListResponse:
    """List all available image sets."""

    try:
        from .image_manager import ImageManager

        image_manager = ImageManager()

        image_sets_data = image_manager.list_available_image_sets()
        image_sets = [ImagesStatusResponse(**data) for data in image_sets_data]

        return ImagesListResponse(image_sets=image_sets)

    except Exception as e:
        logger.error(f"Failed to list image sets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images/{person_id}", response_model=ImagesStatusResponse)
async def get_images_status(person_id: str) -> ImagesStatusResponse:
    """Get image status for a specific person."""

    try:
        from .image_manager import ImageManager

        image_manager = ImageManager()

        status_data = image_manager.get_images_status(person_id)

        return ImagesStatusResponse(**status_data)

    except Exception as e:
        logger.error(f"Failed to get images status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/images/{person_id}")
async def delete_images(person_id: str):
    """Delete all images for a specific person."""

    try:
        from .image_manager import ImageManager

        image_manager = ImageManager()

        success = image_manager.delete_images(person_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete images")

        return {"message": f"Successfully deleted images for {person_id}"}

    except Exception as e:
        logger.error(f"Failed to delete images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/lora/{person_id}")
async def delete_lora_model(person_id: str):
    """Delete LoRA weights for a specific person."""

    try:
        pip = get_pipeline()
        lora_path = pip.lora_trainer.get_lora_path(person_id)

        if lora_path is None:
            raise HTTPException(
                status_code=404, detail=f"LoRA model for {person_id} not found"
            )

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
