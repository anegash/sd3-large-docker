import torch
from fastapi import FastAPI, Query
from diffusers import StableDiffusion3Pipeline
from PIL import Image
import io
import base64

# Initialize FastAPI app
app = FastAPI()

# Load Stable Diffusion 3.5 Model
model_id = "stabilityai/stable-diffusion-3.5-large"
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, variant="fp16"
)

# Move to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

# API Endpoint
@app.get("/generate")
def generate_image(
    prompt: str = Query(..., description="Text prompt for image generation"),
    steps: int = Query(15, le=150, description="Number of inference steps"),
    guidance: float = Query(7.5, le=15.0, description="Guidance scale")
):
    try:
        # Generate the image
        image = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
        ).images[0]

        # Convert image to Base64
        img_io = io.BytesIO()
        image.save(img_io, format="PNG")
        img_io.seek(0)
        base64_img = base64.b64encode(img_io.read()).decode("utf-8")

        return {"image": base64_img}

    except Exception as e:
        return {"error": str(e)}

# Root endpoint
@app.get("/")
def home():
    return {"message": "Stable Diffusion 3.5 API is running!"}