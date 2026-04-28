"""
Configuration parser and validator for Blender YOLO Dataset Generator.

This module handles loading, parsing, and validating configuration files
for the dataset generation pipeline.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import jsonschema
from jsonschema import validate, ValidationError


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


class Config:
    """Configuration container with dot notation access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict
        self._apply_defaults()
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def _apply_defaults(self):
        """Apply default values for optional fields."""
        defaults = {
            'seed': {'numpy': None, 'blenderproc': None},
            'camera': {
                'randomize_params_percent': 0.0
            },
            'rendering': {
                'max_samples': 32,
                'denoiser': 'OPTIX'
            },
            'scene': {
                'room_size_multiplier_min': 5.0,
                'room_size_multiplier_max': 10.0,
                'simulate_physics': False,
                'max_textures': 50,
                'distractors': {
                    'min_count': 20,
                    'max_count': 50,
                    'min_size_rel_scene': 0.05,
                    'max_size_rel_scene': 0.1,
                    'custom_distractors_path': None,
                    'custom_distractor_prob': 0.5,
                    'displacement_max': 0.0,
                    'pbr_noise': 0.5,
                    'emissive_prob': 0.0,
                    'emissive_strength_min': 2.0,
                    'emissive_strength_max': 5.0
                },
                'lights': {
                    'min_count': 3,
                    'max_count': 6,
                    'min_intensity': 50,
                    'max_intensity': 200
                },
                'objects': {
                    'min_count': 1,
                    'max_count': 5,
                    'multiple_occurrences': True,
                    'scale_noise': 0.2,
                    'displacement_max': 0.0,
                    'pbr_noise': 0.3,
                    'cam_min_dist_rel': 1.0,
                    'cam_max_dist_rel': 3.0
                }
            },
            'dataset': {
                'scenes_per_run': 1,
                'empty_images_per_scene': 0,
                'train_split': 0.7,
                'val_split': 0.2,
                'test_split': 0.1
            },
            'output': {
                'save_depth': False,
                'save_normals': False,
                'save_segmentation': False,
                'save_pose': False,
                'detection_params': {
                    'min_bbox_side_px': 10,
                    'min_visibility': 0.3,
                    'occlusion_samples': 100
                }
            }
        }
        
        # Recursively apply defaults
        self._config = self._merge_dicts(defaults, self._config)
    
    @staticmethod
    def _merge_dicts(default: Dict, override: Dict) -> Dict:
        """Recursively merge dictionaries, with override taking precedence."""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result


class ConfigParser:
    """Parser for configuration files with validation."""
    
    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize configuration parser.
        
        Args:
            schema_path: Path to JSON schema file for validation
        """
        self.schema = None
        if schema_path:
            self.load_schema(schema_path)
        elif os.path.exists('configs/config_schema.json'):
            self.load_schema('configs/config_schema.json')
    
    def load_schema(self, schema_path: str):
        """Load JSON schema for validation."""
        try:
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
        except Exception as e:
            raise ConfigError(f"Failed to load schema from {schema_path}: {e}")
    
    def load(self, config_path: str) -> Config:
        """
        Load and parse configuration file.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Config object
            
        Raises:
            ConfigError: If configuration is invalid
        """
        if not os.path.exists(config_path):
            raise ConfigError(f"Configuration file not found: {config_path}")
        
        # Load file based on extension
        config_dict = self._load_file(config_path)
        
        # Validate against schema
        if self.schema:
            self._validate(config_dict)
        
        # Create Config object (applies defaults)
        config_obj = Config(config_dict)
        
        # Additional custom validation on config with defaults applied
        self._validate_custom(config_obj._config)
        
        # Return Config object
        return config_obj
    
    def _load_file(self, path: str) -> Dict[str, Any]:
        """Load configuration from JSON or YAML file."""
        ext = Path(path).suffix.lower()
        
        try:
            with open(path, 'r') as f:
                if ext in ['.json', '.jsonc']:
                    # Strip comments for JSON files
                    content = self._strip_json_comments(f.read())
                    return json.loads(content)
                elif ext in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                else:
                    raise ConfigError(f"Unsupported file format: {ext}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in {path}: {e}")
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {path}: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load {path}: {e}")
    
    @staticmethod
    def _strip_json_comments(content: str) -> str:
        """Strip lines starting with // from JSON content."""
        lines = content.split('\n')
        filtered_lines = [
            line for line in lines 
            if not line.strip().startswith('//')
        ]
        return '\n'.join(filtered_lines)
    
    def _validate(self, config_dict: Dict[str, Any]):
        """Validate configuration against JSON schema."""
        try:
            validate(instance=config_dict, schema=self.schema)
        except ValidationError as e:
            # Create user-friendly error message
            error_path = ' -> '.join(str(p) for p in e.path)
            raise ConfigError(
                f"Configuration validation error at '{error_path}': {e.message}"
            )
    
    def _validate_custom(self, config: Dict[str, Any]):
        """Perform custom validation beyond schema."""
        errors = []
        
        # Validate paths exist
        if 'models_path' in config:
            if not os.path.exists(config['models_path']):
                errors.append(f"models_path does not exist: {config['models_path']}")
        
        if 'cc_textures_path' in config:
            if config['cc_textures_path'] and not os.path.exists(config['cc_textures_path']):
                errors.append(f"cc_textures_path does not exist: {config['cc_textures_path']}")
        
        # Validate splits sum to 1.0
        if 'dataset' in config:
            splits = config['dataset']
            total = splits.get('train_split', 0) + splits.get('val_split', 0) + splits.get('test_split', 0)
            if abs(total - 1.0) > 0.001:
                errors.append(f"Dataset splits must sum to 1.0, got {total}")
        
        # Validate min < max constraints
        if 'scene' in config:
            scene = config['scene']
            
            # Room size
            if scene.get('room_size_multiplier_min', 0) >= scene.get('room_size_multiplier_max', 1):
                errors.append("room_size_multiplier_min must be < room_size_multiplier_max")
            
            # Distractors
            if 'distractors' in scene:
                d = scene['distractors']
                if d.get('min_count', 0) > d.get('max_count', 1):
                    errors.append("distractors.min_count must be <= max_count")
                if d.get('min_size_rel_scene', 0) > d.get('max_size_rel_scene', 1):
                    errors.append("distractors.min_size_rel_scene must be <= max_size_rel_scene")
                if d.get('emissive_strength_min', 0) > d.get('emissive_strength_max', 1):
                    errors.append("distractors.emissive_strength_min must be <= max")
            
            # Lights
            if 'lights' in scene:
                l = scene['lights']
                if l.get('min_count', 0) > l.get('max_count', 1):
                    errors.append("lights.min_count must be <= max_count")
                if l.get('min_intensity', 0) > l.get('max_intensity', 1):
                    errors.append("lights.min_intensity must be <= max_intensity")
            
            # Objects
            if 'objects' in scene:
                o = scene['objects']
                if o.get('min_count', 1) > o.get('max_count', 1):
                    errors.append("objects.min_count must be <= max_count")
                if o.get('min_count', 1) < 1:
                    errors.append("objects.min_count must be at least 1")
                if o.get('cam_min_dist_rel', 0) > o.get('cam_max_dist_rel', 1):
                    errors.append("objects.cam_min_dist_rel must be <= cam_max_dist_rel")
        
        # Validate YOLO format
        if 'output' in config:
            valid_formats = ['yolov11', 'yolov26', 'yolov11-obb', 'yolov26-obb']
            yolo_format = config['output'].get('yolo_format', '')
            if yolo_format not in valid_formats:
                errors.append(f"Invalid yolo_format: {yolo_format}. Must be one of {valid_formats}")
        
        if errors:
            raise ConfigError("Configuration validation failed:\n  - " + "\n  - ".join(errors))
    
    def save(self, config: Union[Config, Dict[str, Any]], path: str):
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            path: Output file path
        """
        if isinstance(config, Config):
            config_dict = config.to_dict()
        else:
            config_dict = config
        
        ext = Path(path).suffix.lower()
        
        try:
            with open(path, 'w') as f:
                if ext in ['.json']:
                    json.dump(config_dict, f, indent=2)
                elif ext in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False)
                else:
                    raise ConfigError(f"Unsupported output format: {ext}")
        except Exception as e:
            raise ConfigError(f"Failed to save configuration to {path}: {e}")


def load_config(config_path: str, schema_path: Optional[str] = None) -> Config:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        schema_path: Optional path to schema file
        
    Returns:
        Config object
    """
    parser = ConfigParser(schema_path)
    return parser.load(config_path)


if __name__ == '__main__':
    # Test configuration loading
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python config_parser.py <config_file>")
        sys.exit(1)
    
    try:
        config = load_config(sys.argv[1])
        print("Configuration loaded successfully!")
        print(f"Models path: {config['models_path']}")
        print(f"Output format: {config['output']['yolo_format']}")
        print(f"Number of scenes: {config['dataset']['num_scenes']}")
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
