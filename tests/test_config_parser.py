"""
Unit tests for configuration parser.

Tests configuration loading, validation, and error handling.
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config_parser import ConfigParser, Config, ConfigError


class TestConfigParser:
    """Test suite for ConfigParser class."""
    
    @pytest.fixture
    def valid_config_dict(self):
        """Return a valid configuration dictionary."""
        return {
            "models_path": "./models",
            "camera": {
                "px": 600,
                "py": 600,
                "u0": 320,
                "v0": 240,
                "width": 640,
                "height": 480
            },
            "dataset": {
                "num_scenes": 10,
                "images_per_scene": 5,
                "train_split": 0.7,
                "val_split": 0.2,
                "test_split": 0.1
            },
            "output": {
                "save_path": "./output/test",
                "yolo_format": "yolov11"
            }
        }
    
    @pytest.fixture
    def temp_config_file(self, valid_config_dict, tmp_path):
        """Create a temporary config file."""
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(valid_config_dict, f)
        return config_file
    
    @pytest.fixture
    def temp_models_dir(self, tmp_path):
        """Create temporary models directory."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        return models_dir
    
    def test_load_valid_config(self, temp_config_file, temp_models_dir):
        """Test loading a valid configuration."""
        # Update config to use temp models dir
        with open(temp_config_file, 'r') as f:
            config_dict = json.load(f)
        config_dict['models_path'] = str(temp_models_dir)
        with open(temp_config_file, 'w') as f:
            json.dump(config_dict, f)
        
        parser = ConfigParser()
        config = parser.load(str(temp_config_file))
        
        assert isinstance(config, Config)
        assert config['models_path'] == str(temp_models_dir)
        assert config['camera']['width'] == 640
    
    def test_config_defaults(self, temp_config_file, temp_models_dir):
        """Test that default values are applied."""
        # Minimal config
        minimal_config = {
            "models_path": str(temp_models_dir),
            "camera": {"px": 600, "py": 600, "u0": 320, "v0": 240, "width": 640, "height": 480},
            "dataset": {"num_scenes": 10, "images_per_scene": 5},
            "output": {"save_path": "./output", "yolo_format": "yolov11"}
        }
        
        config_file = temp_config_file.parent / "minimal_config.json"
        with open(config_file, 'w') as f:
            json.dump(minimal_config, f)
        
        parser = ConfigParser()
        config = parser.load(str(config_file))
        
        # Check defaults are applied
        assert config.get('dataset.train_split') == 0.7
        assert config.get('dataset.val_split') == 0.2
        assert config.get('dataset.test_split') == 0.1
    
    def test_invalid_json(self, tmp_path):
        """Test handling of invalid JSON."""
        config_file = tmp_path / "invalid.json"
        with open(config_file, 'w') as f:
            f.write("{ invalid json }")
        
        parser = ConfigParser()
        with pytest.raises(ConfigError, match="Invalid JSON"):
            parser.load(str(config_file))
    
    def test_missing_required_field(self, tmp_path):
        """Test validation of required fields."""
        config_file = tmp_path / "missing_field.json"
        config_dict = {
            "camera": {"px": 600, "py": 600, "u0": 320, "v0": 240, "width": 640, "height": 480},
            "output": {"save_path": "./output", "yolo_format": "yolov11"}
            # Missing models_path and dataset
        }
        with open(config_file, 'w') as f:
            json.dump(config_dict, f)
        
        parser = ConfigParser()
        with pytest.raises(ConfigError):
            parser.load(str(config_file))
    
    def test_invalid_split_ratios(self, temp_config_file, temp_models_dir):
        """Test validation of dataset split ratios."""
        with open(temp_config_file, 'r') as f:
            config_dict = json.load(f)
        
        config_dict['models_path'] = str(temp_models_dir)
        config_dict['dataset']['train_split'] = 0.5
        config_dict['dataset']['val_split'] = 0.3
        config_dict['dataset']['test_split'] = 0.3  # Sum > 1.0
        
        config_file = temp_config_file.parent / "invalid_splits.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f)
        
        parser = ConfigParser()
        with pytest.raises(ConfigError, match="splits must sum to 1.0"):
            parser.load(str(config_file))
    
    def test_invalid_yolo_format(self, temp_config_file, temp_models_dir):
        """Test validation of YOLO format."""
        with open(temp_config_file, 'r') as f:
            config_dict = json.load(f)
        
        config_dict['models_path'] = str(temp_models_dir)
        config_dict['output']['yolo_format'] = 'invalid_format'
        
        config_file = temp_config_file.parent / "invalid_yolo.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f)
        
        parser = ConfigParser()
        with pytest.raises(ConfigError, match="yolo_format"):
            parser.load(str(config_file))
    
    def test_min_max_validation(self, temp_config_file, temp_models_dir):
        """Test min < max constraints."""
        with open(temp_config_file, 'r') as f:
            config_dict = json.load(f)
        
        config_dict['models_path'] = str(temp_models_dir)
        config_dict['scene'] = {
            'room_size_multiplier_min': 10.0,
            'room_size_multiplier_max': 5.0  # min > max (invalid)
        }
        
        config_file = temp_config_file.parent / "invalid_minmax.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f)
        
        parser = ConfigParser()
        with pytest.raises(ConfigError, match="room_size_multiplier_min must be"):
            parser.load(str(config_file))
    
    def test_config_get_method(self, valid_config_dict):
        """Test Config.get() method with nested keys."""
        config = Config(valid_config_dict)
        
        assert config.get('camera.width') == 640
        assert config.get('dataset.train_split') == 0.7
        assert config.get('nonexistent.key', 'default') == 'default'
    
    def test_json_comments(self, tmp_path, temp_models_dir):
        """Test handling of JSON files with comments."""
        config_file = tmp_path / "commented.json"
        with open(config_file, 'w') as f:
            f.write("""
{
  // This is a comment
  "models_path": "%s",
  // Another comment
  "camera": {
    "px": 600,
    "py": 600,
    "u0": 320,
    "v0": 240,
    "width": 640,
    "height": 480
  },
  "dataset": {
    "num_scenes": 10,
    "images_per_scene": 5
  },
  "output": {
    "save_path": "./output",
    "yolo_format": "yolov11"
  }
}
            """ % str(temp_models_dir))
        
        parser = ConfigParser()
        config = parser.load(str(config_file))
        assert config['camera']['width'] == 640


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
