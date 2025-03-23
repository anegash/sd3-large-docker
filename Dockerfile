# Use a base image with PyTorch and CUDA
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt update && apt install -y git ffmpeg libsm6 libxext6

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install runpod torch torchvision torchaudio diffusers transformers pillow boto3 huggingface_hub

# Copy your serverless handler script and other files
COPY . /app

# Set the default command to run your handler
CMD ["python3", "-u", "rp_handler.py"]