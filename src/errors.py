#!/usr/bin/env python3
"""
Custom exception classes for the Blender YOLO Dataset Generator.
Provides specific error types with helpful messages.
"""

from typing import Optional, List


class BlenderYOLOError(Exception):
    """Base exception for all Blender YOLO errors."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.get_full_message())
    
    def get_full_message(self) -> str:
        """Get formatted error message with details."""
        if self.details:
            return f"{self.message}\n\nDetails:\n{self.details}"
        return self.message


class ConfigurationError(BlenderYOLOError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 suggestion: Optional[str] = None):
        self.config_key = config_key
        self.suggestion = suggestion
        
        details = []
        if config_key:
            details.append(f"Configuration key: {config_key}")
        if suggestion:
            details.append(f"Suggestion: {suggestion}")
        
        details_str = "\n".join(details) if details else None
        super().__init__(message, details_str)


class ModelLoadError(BlenderYOLOError):
    """3D model loading errors."""
    
    def __init__(self, message: str, model_path: Optional[str] = None,
                 class_name: Optional[str] = None):
        self.model_path = model_path
        self.class_name = class_name
        
        details = []
        if model_path:
            details.append(f"Model path: {model_path}")
        if class_name:
            details.append(f"Class name: {class_name}")
        
        details_str = "\n".join(details) if details else None
        super().__init__(message, details_str)


class SceneGenerationError(BlenderYOLOError):
    """Scene generation errors."""
    
    def __init__(self, message: str, scene_id: Optional[int] = None,
                 recoverable: bool = True):
        self.scene_id = scene_id
        self.recoverable = recoverable
        
        details = []
        if scene_id is not None:
            details.append(f"Scene ID: {scene_id}")
        details.append(f"Recoverable: {recoverable}")
        
        details_str = "\n".join(details) if details else None
        super().__init__(message, details_str)


class AnnotationError(BlenderYOLOError):
    """Annotation extraction/conversion errors."""
    
    def __init__(self, message: str, image_name: Optional[str] = None,
                 format_type: Optional[str] = None):
        self.image_name = image_name
        self.format_type = format_type
        
        details = []
        if image_name:
            details.append(f"Image: {image_name}")
        if format_type:
            details.append(f"Format: {format_type}")
        
        details_str = "\n".join(details) if details else None
        super().__init__(message, details_str)


class ValidationError(BlenderYOLOError):
    """Dataset validation errors."""
    
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        self.validation_errors = errors or []
        
        if self.validation_errors:
            details = "Validation errors:\n" + "\n".join(
                f"  - {err}" for err in self.validation_errors
            )
        else:
            details = None
        
        super().__init__(message, details)


class DependencyError(BlenderYOLOError):
    """Missing or incompatible dependency errors."""
    
    def __init__(self, message: str, dependency: Optional[str] = None,
                 install_command: Optional[str] = None):
        self.dependency = dependency
        self.install_command = install_command
        
        details = []
        if dependency:
            details.append(f"Missing dependency: {dependency}")
        if install_command:
            details.append(f"Install with: {install_command}")
        
        details_str = "\n".join(details) if details else None
        super().__init__(message, details_str)


class ResourceError(BlenderYOLOError):
    """Resource-related errors (memory, disk, GPU, etc.)."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None,
                 required: Optional[str] = None, available: Optional[str] = None):
        self.resource_type = resource_type
        self.required = required
        self.available = available
        
        details = []
        if resource_type:
            details.append(f"Resource: {resource_type}")
        if required:
            details.append(f"Required: {required}")
        if available:
            details.append(f"Available: {available}")
        
        details_str = "\n".join(details) if details else None
        super().__init__(message, details_str)


def format_user_error(error: Exception, show_traceback: bool = False) -> str:
    """
    Format error message for user display.
    
    Args:
        error: Exception to format
        show_traceback: Whether to include traceback
        
    Returns:
        Formatted error message
    """
    import traceback
    
    if isinstance(error, BlenderYOLOError):
        msg = f"\n❌ ERROR: {error.get_full_message()}\n"
    else:
        msg = f"\n❌ ERROR: {str(error)}\n"
    
    if show_traceback:
        tb = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        msg += f"\nTraceback:\n{tb}"
    
    return msg


def suggest_fix(error: Exception) -> Optional[str]:
    """
    Suggest a fix for common errors.
    
    Args:
        error: Exception to analyze
        
    Returns:
        Suggestion string or None
    """
    error_str = str(error).lower()
    
    # Common error patterns and suggestions
    suggestions = {
        'no module named': "Install missing dependencies with: pip install -r requirements.txt",
        'cuda out of memory': "Try reducing batch size or image resolution in config",
        'permission denied': "Check file permissions or run with appropriate privileges",
        'no such file or directory': "Verify all file paths in configuration exist",
        'invalid configuration': "Check configuration against schema with: python src/config_parser.py config.json",
        'no object classes found': "Ensure models/ directory contains subdirectories with .obj files",
        'splits must sum to 1.0': "Adjust train_split, val_split, test_split to sum exactly to 1.0",
    }
    
    for pattern, suggestion in suggestions.items():
        if pattern in error_str:
            return f"💡 Suggestion: {suggestion}"
    
    return None


if __name__ == '__main__':
    # Test error types
    try:
        raise ConfigurationError(
            "Invalid YOLO format specified",
            config_key="output.yolo_format",
            suggestion="Use one of: yolov11, yolov26, yolov11-obb, yolov26-obb"
        )
    except Exception as e:
        print(format_user_error(e))
        suggestion = suggest_fix(e)
        if suggestion:
            print(suggestion)
    
    print("\n" + "="*60 + "\n")
    
    try:
        raise SceneGenerationError(
            "Failed to place object in scene",
            scene_id=42,
            recoverable=True
        )
    except Exception as e:
        print(format_user_error(e))
