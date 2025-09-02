"""SQLAlchemy database models for LoRA training."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator, VARCHAR

Base = declarative_base()


class TrainingStatus(str, Enum):
    """Training status enumeration."""
    PENDING = "pending"
    PREPARING = "preparing"
    TRAINING = "training" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EnumType(TypeDecorator):
    """Custom enum type for SQLAlchemy."""
    impl = VARCHAR
    
    def __init__(self, enum_class, *args, **kwargs):
        self.enum_class = enum_class
        super().__init__(*args, **kwargs)
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        return value.value if isinstance(value, Enum) else value
    
    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return self.enum_class(value)


class Child(Base):
    """Child model for managing training subjects."""
    __tablename__ = "children"
    
    id = Column(String(50), primary_key=True)  # e.g., "child_001"
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    training_images = relationship("TrainingImage", back_populates="child", cascade="all, delete-orphan")
    lora_models = relationship("LoRAModel", back_populates="child", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Child(id='{self.id}', name='{self.name}')>"


class TrainingImage(Base):
    """Training image model."""
    __tablename__ = "training_images"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    child_id = Column(String(50), ForeignKey("children.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    description = Column(Text, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)  # bytes
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    child = relationship("Child", back_populates="training_images")
    
    def __repr__(self):
        return f"<TrainingImage(id={self.id}, child_id='{self.child_id}', filename='{self.filename}')>"


class LoRAModel(Base):
    """LoRA model information."""
    __tablename__ = "lora_models"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    child_id = Column(String(50), ForeignKey("children.id"), nullable=False)
    model_path = Column(String(512), nullable=False)
    version = Column(Integer, default=1)
    
    # Training configuration
    training_steps = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    lora_rank = Column(Integer, nullable=False, default=64)
    lora_alpha = Column(Integer, nullable=False, default=32)
    lora_dropout = Column(Float, nullable=False, default=0.1)
    
    # Training metadata
    training_status = Column(EnumType(TrainingStatus), default=TrainingStatus.PENDING)
    training_started_at = Column(DateTime, nullable=True)
    training_completed_at = Column(DateTime, nullable=True)
    training_progress = Column(Float, default=0.0)  # 0.0 to 1.0
    training_loss = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Model metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(String(10), default="true")  # "true" or "false" for active model
    
    # Relationships
    child = relationship("Child", back_populates="lora_models")
    
    def __repr__(self):
        return f"<LoRAModel(id={self.id}, child_id='{self.child_id}', status='{self.training_status}')>"


class GenerationHistory(Base):
    """History of generated images."""
    __tablename__ = "generation_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(Text, nullable=False)
    children_ids = Column(String(500), nullable=True)  # Comma-separated child IDs used
    image_path = Column(String(512), nullable=False)
    steps = Column(Integer, nullable=False)
    guidance_scale = Column(Float, nullable=False)
    seed = Column(Integer, nullable=True)
    generation_time = Column(Float, nullable=True)  # seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<GenerationHistory(id={self.id}, prompt='{self.prompt[:50]}...')>"