# Use a base image with PyTorch and CUDA
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt update && apt install -y git ffmpeg libsm6 libxext6

# Install Poetry
RUN pip install --upgrade pip && \
    pip install poetry

# Configure Poetry
RUN poetry config virtualenvs.create false

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* /app/

# Install dependencies
RUN poetry install --only main --no-interaction --no-ansi

# Copy application files
COPY . /app

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI application
CMD ["python", "main.py"]