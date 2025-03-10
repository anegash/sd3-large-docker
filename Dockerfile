# Use a base image with PyTorch and CUDA
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt update && apt install -y git ffmpeg libsm6 libxext6

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install fastapi uvicorn torch torchvision torchaudio diffusers transformers pillow

# Copy application files
COPY . /app

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]