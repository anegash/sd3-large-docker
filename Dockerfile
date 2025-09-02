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

# Install all dependencies including LoRA training (dev includes training deps)
RUN poetry install --with dev --no-interaction --no-ansi && \
    pip install runpod

# Copy application files
COPY . /app

# Install Redis for Celery (needed for LoRA training)
RUN apt update && apt install -y redis-server

# Create directories for persistent data
RUN mkdir -p /app/data /app/logs

# Expose ports for API and Celery monitoring
EXPOSE 8000 5555

# Make startup script executable
RUN chmod +x /app/start_pod.sh 2>/dev/null || true

# Default to Pod startup (includes API + Celery)
# For serverless: CMD ["python", "handler.py"]
# For local dev: CMD ["python", "main.py"]
CMD ["/app/start_pod.sh"]