#!/usr/bin/env python3
"""
Integration tests for the complete dataset generation pipeline.
Tests end-to-end workflow from configuration to YOLO dataset output.
"""

import os
import sys
import json
import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_parser import ConfigParser, Config
from object_loader import ObjectLoader
from yolo_converter import YOLOConverter, YOLODatasetFormatter
from dataset_manager import DatasetSplitter, DatasetOrganizer, DatasetValidator


class TestIntegrationBasic:
    """Test basic integration without BlenderProc."""
    
    @pytest.fixture
    def test_env(self, tmp_path):
        """Create test environment with sample models and config."""
        # Create directory structure
        models_dir = tmp_path / "models"
        output_dir = tmp_path / "output"
        
        # Create sample model directories
        for class_name in ['cube', 'sphere', 'cylinder']:
            class_dir = models_dir / class_name
            class_dir.mkdir(parents=True)
            
            # Create minimal OBJ file
            obj_file = class_dir / "model.obj"
            obj_file.write_text(f"# {class_name}\nv 0 0 0\n")
        
        # Create config
        config = {
            "models_path": str(models_dir),
            "camera": {
                "px": 600, "py": 600,
                "u0": 320, "v0": 240,
                "width": 640, "height": 480
            },
            "dataset": {
                "num_scenes": 5,
                "images_per_scene": 3,
                "train_split": 0.6,
                "val_split": 0.2,
                "test_split": 0.2
            },
            "output": {
                "save_path": str(output_dir),
                "yolo_format": "yolov11"
            }
        }
        
        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return {
            'config_file': config_file,
            'models_dir': models_dir,
            'output_dir': output_dir,
            'tmp_path': tmp_path
        }
    
    def test_config_loading(self, test_env):
        """Test configuration loading and validation."""
        parser = ConfigParser()
        config = parser.load(str(test_env['config_file']))
        
        assert isinstance(config, Config)
        assert config['models_path'] == str(test_env['models_dir'])
        assert config.get('output.yolo_format') == 'yolov11'
        assert config.get('dataset.num_scenes') == 5
        assert config.get('dataset.images_per_scene') == 3
    
    def test_object_loader_integration(self, test_env):
        """Test object loader with sample models."""
        loader = ObjectLoader(str(test_env['models_dir']))
        
        # Should find 3 classes
        assert loader.get_num_classes() == 3
        
        # Check alphabetical sorting
        class_names = loader.get_class_names()
        assert class_names == ['cube', 'cylinder', 'sphere']
        
        # Check IDs
        cube = loader.get_class_by_name('cube')
        assert cube.class_id == 0
        
        cylinder = loader.get_class_by_name('cylinder')
        assert cylinder.class_id == 1
        
        sphere = loader.get_class_by_name('sphere')
        assert sphere.class_id == 2
    
    def test_yolo_formatter_integration(self, test_env):
        """Test YOLO dataset formatter."""
        loader = ObjectLoader(str(test_env['models_dir']))
        class_names = loader.get_class_names()
        formatter = YOLODatasetFormatter(str(test_env['output_dir']), class_names, 'yolov11')
        
        # Check directories exist (created in __init__)
        for split in ['train', 'val', 'test']:
            assert (test_env['output_dir'] / 'images' / split).exists()
            assert (test_env['output_dir'] / 'labels' / split).exists()
        
        # Create data.yaml
        formatter.create_data_yaml()
        
        data_yaml = test_env['output_dir'] / 'data.yaml'
        assert data_yaml.exists()
        
        # Verify content
        import yaml
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        assert data['nc'] == 3
        # Names is a dict mapping ID to name
        assert data['names'][0] == 'cube'
        assert data['names'][1] == 'cylinder'
        assert data['names'][2] == 'sphere'
        assert 'train' in data
        assert 'val' in data
        assert 'test' in data
    
    def test_dataset_manager_integration(self, test_env):
        """Test dataset splitter with file organization."""
        # Split 15 samples
        splitter = DatasetSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42)
        splits = splitter.split_indices(num_samples=15)
        
        assert len(splits['train']) == 9
        assert len(splits['val']) == 3
        assert len(splits['test']) == 3
        
        # Check no overlap
        all_indices = set(splits['train']) | set(splits['val']) | set(splits['test'])
        assert len(all_indices) == 15
    
    def test_yolo_conversion_pipeline(self, test_env):
        """Test complete YOLO conversion pipeline."""
        from bbox_extractor import BoundingBox
        
        # Create sample bounding boxes
        bboxes = [
            BoundingBox(
                class_id=0, class_name='cube',
                bbox_2d=np.array([100, 100, 200, 200]),
                visibility=1.0, area_px=10000
            ),
            BoundingBox(
                class_id=1, class_name='cylinder',
                bbox_2d=np.array([300, 200, 400, 350]),
                visibility=1.0, area_px=15000
            )
        ]
        
        # Convert to YOLO format
        converter = YOLOConverter('yolov11', image_width=640, image_height=480)
        
        for bbox in bboxes:
            yolo_line = converter.convert_bbox(bbox)
            assert yolo_line is not None
            
            parts = yolo_line.split()
            assert len(parts) == 5  # class x y w h
            
            # Check all values are in [0, 1]
            for val in parts[1:]:
                assert 0 <= float(val) <= 1


class TestOBBIntegration:
    """Test OBB-specific integration."""
    
    @pytest.fixture
    def obb_config(self, tmp_path):
        """Create OBB configuration."""
        models_dir = tmp_path / "models"
        output_dir = tmp_path / "output"
        
        # Create sample model
        class_dir = models_dir / "rotated_obj"
        class_dir.mkdir(parents=True)
        obj_file = class_dir / "model.obj"
        obj_file.write_text("# rotated object\nv 0 0 0\n")
        
        config = {
            "models_path": str(models_dir),
            "camera": {
                "px": 600, "py": 600,
                "u0": 320, "v0": 240,
                "width": 640, "height": 480
            },
            "dataset": {
                "num_scenes": 2,
                "images_per_scene": 2
            },
            "output": {
                "save_path": str(output_dir),
                "yolo_format": "yolov11-obb"
            }
        }
        
        config_file = tmp_path / "obb_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return {
            'config_file': config_file,
            'models_dir': models_dir,
            'output_dir': output_dir
        }
    
    def test_obb_config_loading(self, obb_config):
        """Test OBB configuration."""
        parser = ConfigParser()
        config = parser.load(str(obb_config['config_file']))
        
        assert config.get('output.yolo_format') == 'yolov11-obb'
    
    def test_obb_yolo_format(self, obb_config):
        """Test OBB YOLO format conversion."""
        from bbox_extractor import BoundingBox
        import math
        
        # Create OBB bbox with 4 corners (rotated rectangle)
        # Define corners of a 100x100 box rotated 45 degrees around center (150, 150)
        angle = math.pi / 4
        cx, cy = 150, 150
        w, h = 100, 100
        
        # Calculate rotated corners
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        corners = np.array([
            [cx - w/2*cos_a + h/2*sin_a, cy - w/2*sin_a - h/2*cos_a],
            [cx + w/2*cos_a + h/2*sin_a, cy + w/2*sin_a - h/2*cos_a],
            [cx + w/2*cos_a - h/2*sin_a, cy + w/2*sin_a + h/2*cos_a],
            [cx - w/2*cos_a - h/2*sin_a, cy - w/2*sin_a + h/2*cos_a]
        ])
        
        bbox = BoundingBox(
            class_id=0, class_name='rotated_obj',
            bbox_2d=corners,  # 4x2 array of corners
            angle=angle,  # 45 degrees
            visibility=1.0, area_px=10000
        )
        
        converter = YOLOConverter('yolov11-obb', image_width=640, image_height=480)
        yolo_line = converter.convert_bbox(bbox)
        
        assert yolo_line is not None
        parts = yolo_line.split()
        assert len(parts) == 6  # class x y w h angle
        
        # Check angle is present and in valid range
        angle_out = float(parts[5])
        assert -math.pi/2 <= angle_out <= math.pi/2
    
    def test_obb_data_yaml(self, obb_config):
        """Test OBB data.yaml creation."""
        from object_loader import ObjectLoader, ObjectClass
        
        loader = ObjectLoader(str(obb_config['models_dir']))
        class_names = loader.get_class_names()
        formatter = YOLODatasetFormatter(str(obb_config['output_dir']), class_names, 'yolov11-obb')
        
        # Structure created in __init__, just create data.yaml
        formatter.create_data_yaml()
        
        data_yaml = obb_config['output_dir'] / 'data.yaml'
        assert data_yaml.exists()


class TestErrorHandling:
    """Test error handling in integration scenarios."""
    
    def test_missing_models_directory(self, tmp_path):
        """Test handling of missing models directory."""
        config = {
            "models_path": str(tmp_path / "nonexistent"),
            "camera": {"px": 600, "py": 600, "u0": 320, "v0": 240, "width": 640, "height": 480},
            "dataset": {"num_scenes": 1, "images_per_scene": 1},
            "output": {"save_path": str(tmp_path / "output"), "yolo_format": "yolov11"}
        }
        
        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        parser = ConfigParser()
        with pytest.raises(Exception):  # Should raise error for missing models
            parser.load(str(config_file))
    
    def test_invalid_split_ratios(self, tmp_path):
        """Test invalid split ratios."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        config = {
            "models_path": str(models_dir),
            "camera": {"px": 600, "py": 600, "u0": 320, "v0": 240, "width": 640, "height": 480},
            "dataset": {
                "num_scenes": 1,
                "images_per_scene": 1,
                "train_split": 0.5,
                "val_split": 0.3,
                "test_split": 0.3  # Sum > 1.0
            },
            "output": {"save_path": str(tmp_path / "output"), "yolo_format": "yolov11"}
        }
        
        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        parser = ConfigParser()
        with pytest.raises(Exception):  # Should raise error for invalid splits
            parser.load(str(config_file))
    
    def test_empty_models_directory(self, tmp_path):
        """Test handling of empty models directory."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        from object_loader import ObjectLoader
        
        with pytest.raises(ValueError, match="No object class directories"):
            ObjectLoader(str(models_dir))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
