"""Validation utilities for LoRA API."""

import re
from typing import Dict, Any, List
from pathlib import Path

from fastapi import HTTPException


class ChildIDValidator:
    """Validator for child IDs."""
    
    # Child ID pattern: letters, numbers, underscores, 3-50 chars
    CHILD_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_]{3,50}$')
    
    @classmethod
    def validate_child_id(cls, child_id: str) -> str:
        """
        Validate child ID format.
        
        Args:
            child_id: Child identifier
            
        Returns:
            Validated child ID
            
        Raises:
            HTTPException: If child ID is invalid
        """
        if not child_id:
            raise HTTPException(status_code=400, detail="Child ID cannot be empty")
        
        if not cls.CHILD_ID_PATTERN.match(child_id):
            raise HTTPException(
                status_code=400,
                detail="Child ID must be 3-50 characters long and contain only letters, numbers, and underscores"
            )
        
        return child_id


class TrainingConfigValidator:
    """Validator for training configuration."""
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize training configuration.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            Validated configuration
            
        Raises:
            HTTPException: If configuration is invalid
        """
        # Default values
        defaults = {
            "lora_rank": 64,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "learning_rate": 1e-4,
            "training_steps": 1000,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "use_8bit_adam": False,
            "mixed_precision": "fp16",
            "save_steps": 250,
            "validation_steps": 100,
            "seed": 42
        }
        
        # Apply defaults
        validated_config = {**defaults, **config}
        
        # Validation rules
        validations = [
            ("lora_rank", lambda x: 1 <= x <= 256, "LoRA rank must be between 1 and 256"),
            ("lora_alpha", lambda x: 1 <= x <= 512, "LoRA alpha must be between 1 and 512"),
            ("lora_dropout", lambda x: 0.0 <= x <= 1.0, "LoRA dropout must be between 0.0 and 1.0"),
            ("learning_rate", lambda x: 1e-6 <= x <= 1e-2, "Learning rate must be between 1e-6 and 1e-2"),
            ("training_steps", lambda x: 100 <= x <= 5000, "Training steps must be between 100 and 5000"),
            ("batch_size", lambda x: 1 <= x <= 8, "Batch size must be between 1 and 8"),
            ("gradient_accumulation_steps", lambda x: 1 <= x <= 16, "Gradient accumulation steps must be between 1 and 16"),
            ("max_grad_norm", lambda x: 0.1 <= x <= 10.0, "Max gradient norm must be between 0.1 and 10.0"),
            ("save_steps", lambda x: 50 <= x <= 1000, "Save steps must be between 50 and 1000"),
            ("validation_steps", lambda x: 50 <= x <= 500, "Validation steps must be between 50 and 500"),
            ("seed", lambda x: x >= 0, "Seed must be non-negative"),
        ]
        
        for field, validator, error_msg in validations:
            if field in validated_config:
                if not validator(validated_config[field]):
                    raise HTTPException(status_code=400, detail=error_msg)
        
        # Additional logical validations
        if validated_config["save_steps"] > validated_config["training_steps"]:
            raise HTTPException(
                status_code=400,
                detail="Save steps cannot be greater than total training steps"
            )
        
        if validated_config["validation_steps"] > validated_config["training_steps"]:
            raise HTTPException(
                status_code=400,
                detail="Validation steps cannot be greater than total training steps"
            )
        
        return validated_config


class PromptValidator:
    """Validator for generation prompts."""
    
    MAX_PROMPT_LENGTH = 1000
    CHILD_TOKEN_PATTERN = re.compile(r'\{([^}]+)\}')
    
    @classmethod
    def validate_prompt(cls, prompt: str) -> Dict[str, Any]:
        """
        Validate generation prompt.
        
        Args:
            prompt: Text prompt
            
        Returns:
            Dictionary with validation results
            
        Raises:
            HTTPException: If prompt is invalid
        """
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if len(prompt) > cls.MAX_PROMPT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt too long. Maximum length: {cls.MAX_PROMPT_LENGTH} characters"
            )
        
        # Extract child tokens
        child_tokens = cls.CHILD_TOKEN_PATTERN.findall(prompt)
        
        # Validate child tokens
        for token in child_tokens:
            try:
                ChildIDValidator.validate_child_id(token)
            except HTTPException:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid child ID in prompt: '{token}'"
                )
        
        return {
            "prompt": prompt.strip(),
            "child_tokens": list(set(child_tokens)),  # Remove duplicates
            "has_child_tokens": len(child_tokens) > 0
        }


class FileValidator:
    """Validator for uploaded files."""
    
    @staticmethod
    def validate_file_list(files: List[Any], max_files: int = 50) -> None:
        """
        Validate list of uploaded files.
        
        Args:
            files: List of uploaded files
            max_files: Maximum number of files allowed
            
        Raises:
            HTTPException: If file list is invalid
        """
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > max_files:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files. Maximum allowed: {max_files}"
            )
        
        # Check for duplicate filenames
        filenames = [file.filename for file in files if file.filename]
        if len(filenames) != len(set(filenames)):
            raise HTTPException(
                status_code=400,
                detail="Duplicate filenames detected"
            )
    
    @staticmethod
    def validate_description_list(descriptions: List[str], file_count: int) -> List[str]:
        """
        Validate and normalize description list.
        
        Args:
            descriptions: List of descriptions
            file_count: Number of files
            
        Returns:
            Validated description list
            
        Raises:
            HTTPException: If descriptions are invalid
        """
        if descriptions is None:
            return [None] * file_count
        
        if len(descriptions) != file_count:
            raise HTTPException(
                status_code=400,
                detail="Number of descriptions must match number of files"
            )
        
        # Validate each description
        validated_descriptions = []
        for desc in descriptions:
            if desc:
                desc = desc.strip()
                if len(desc) > 500:  # Max description length
                    raise HTTPException(
                        status_code=400,
                        detail="Description too long. Maximum length: 500 characters"
                    )
                validated_descriptions.append(desc if desc else None)
            else:
                validated_descriptions.append(None)
        
        return validated_descriptions


# Validation functions for use in API endpoints

def validate_child_id_param(child_id: str) -> str:
    """Validate child ID parameter."""
    return ChildIDValidator.validate_child_id(child_id)

def validate_training_config_request(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate training configuration request."""
    return TrainingConfigValidator.validate_training_config(config)

def validate_generation_prompt(prompt: str) -> Dict[str, Any]:
    """Validate generation prompt."""
    return PromptValidator.validate_prompt(prompt)

def validate_file_uploads(files: List[Any], descriptions: List[str] = None) -> tuple:
    """Validate file uploads and descriptions."""
    FileValidator.validate_file_list(files)
    validated_descriptions = FileValidator.validate_description_list(descriptions, len(files))
    return files, validated_descriptions