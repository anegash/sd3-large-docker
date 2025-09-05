"""
RunPod serverless handler for Stable Diffusion XL image generation.
"""
import runpod
import torch
from PIL import Image
import io
import base64
from src.sd3_api.pipeline import SDXLPipeline
from src.sd3_api.models import GenerateRequest
from src.sd3_api.config import DEFAULT_STEPS, DEFAULT_GUIDANCE


pipeline = None


def initialize():
    """Initialize the SDXL pipeline on cold start."""
    global pipeline
    if pipeline is None:
        pipeline = SDXLPipeline()
        pipeline._initialize_pipeline()
    return pipeline


def generate_image(job):
    """
    RunPod handler function for image generation.
    
    Expected input format:
    {
        "prompt": "a beautiful sunset",
        "steps": 20,
        "guidance": 7.5,
        "width": 1024,
        "height": 1024,
        "seed": null
    }
    """
    try:
        # Initialize pipeline if needed
        pipe = initialize()
        
        # Get job input
        job_input = job["input"]
        
        # Validate and create request
        request = GenerateRequest(
            prompt=job_input.get("prompt", ""),
            steps=job_input.get("steps", DEFAULT_STEPS),
            guidance=job_input.get("guidance", DEFAULT_GUIDANCE),
            width=job_input.get("width", 1024),
            height=job_input.get("height", 1024)
        )
        
        # Generate image using SDXL
        image = pipe.generate_image(
            prompt=request.prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance,
            width=request.width,
            height=request.height
        )
        
        # Convert PIL image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {
            "image": image_base64,
            "format": "png",
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "guidance": request.guidance
        }
        
    except Exception as e:
        return {"error": str(e)}


# Start the RunPod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": generate_image})