"""Celery application configuration."""

import os
from celery import Celery

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "sd3_lora_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["sd3_api.tasks.training_tasks"]
)

# Configure Celery
celery_app.conf.update(
    # Task serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task routing
    task_routes={
        "sd3_api.tasks.training_tasks.train_lora": {"queue": "training"},
        "sd3_api.tasks.training_tasks.test_celery": {"queue": "training"},
        "sd3_api.tasks.training_tasks.test_training_imports": {"queue": "training"},
        "sd3_api.tasks.training_tasks.cleanup_training_data": {"queue": "cleanup"},
    },
    
    # Worker configuration
    worker_prefetch_multiplier=1,  # Only fetch one task at a time for training
    task_acks_late=True,  # Acknowledge task only after completion
    worker_disable_rate_limits=True,
    
    # Task time limits
    task_soft_time_limit=3600,  # 1 hour soft limit
    task_time_limit=7200,       # 2 hour hard limit
    
    # Result backend settings
    result_expires=86400,  # Results expire after 24 hours
    
    # Retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
)