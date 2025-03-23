import os
import io
import uuid
import boto3
import torch
import runpod
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import snapshot_download

# Load model once when the container starts
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HuggingFace token not found! Set HF_TOKEN environment variable.")

print("Downloading model...")
model_path = snapshot_download(
    repo_id="stabilityai/stable-diffusion-3.5-large",
    local_dir="/runpod-volume/models",
    token=HF_TOKEN,
    local_dir_use_symlinks=False,
    resume_download=True
)

print("Loading pipeline...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, use_safetensors=True
)
pipe.enable_xformers_memory_efficient_attention()
pipe.to("cuda")

print("Model loaded and ready!")


# Handler function for serverless
def handler(job):
    input_data = job["input"]
    prompt = input_data.get("prompt", "")
    negative_prompt = input_data.get("negative_prompt", "")
    num_inference_steps = input_data.get("num_inference_steps", 30)
    guidance_scale = input_data.get("guidance_scale", 7.5)

    # Run inference
    images = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images

    img = images[0]

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET")
    )

    bucket_name = "my-sd-output-bucket"
    img_key = f"outputs/{uuid.uuid4()}.png"
    s3.upload_fileobj(buffered, bucket_name, img_key, ExtraArgs={"ContentType": "image/png"})

    url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket_name, "Key": img_key}, ExpiresIn=3600
    )

    return {"image_url": url}


# Start serverless worker
runpod.serverless.start({"handler": handler})