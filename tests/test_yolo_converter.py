"""
Unit tests for YOLO converter.

Tests coordinate conversion, format generation, and normalization.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from bbox_extractor import BoundingBox
from yolo_converter import YOLOConverter, YOLODatasetFormatter


class TestYOLOConverter:
    """Test suite for YOLOConverter class."""
    
    @pytest.fixture
    def image_size(self):
        """Standard image size for tests."""
        return (640, 480)  # width, height
    
    @pytest.fixture
    def bbox_aabb(self):
        """Create a standard axis-aligned bounding box."""
        return BoundingBox(
            class_id=0,
            class_name="test_object",
            bbox_2d=np.array([100, 150, 300, 350]),  # x_min, y_min, x_max, y_max
            visibility=0.9
        )
    
    @pytest.fixture
    def bbox_obb(self):
        """Create an oriented bounding box."""
        # Rotated rectangle corners
        corners = np.array([
            [120, 140],
            [320, 160],
            [300, 360],
            [100, 340]
        ])
        return BoundingBox(
            class_id=1,
            class_name="rotated_object",
            bbox_2d=corners,
            angle=np.pi / 6,  # 30 degrees
            visibility=0.85
        )
    
    def test_standard_yolo_from_aabb(self, bbox_aabb, image_size):
        """Test conversion of AABB to standard YOLO format."""
        converter = YOLOConverter('yolov11', image_size[0], image_size[1])
        result = converter.convert_bbox(bbox_aabb)
        
        # Parse result
        parts = result.split()
        assert len(parts) == 5
        
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # Verify class ID
        assert class_id == 0
        
        # Verify normalization (should be in [0, 1])
        assert 0 <= x_center <= 1
        assert 0 <= y_center <= 1
        assert 0 <= width <= 1
        assert 0 <= height <= 1
        
        # Verify values (bbox is [100, 150, 300, 350] on 640x480)
        # Center: (200, 250), Size: (200, 200)
        # Normalized: center (200/640, 250/480), size (200/640, 200/480)
        assert abs(x_center - 200/640) < 0.001
        assert abs(y_center - 250/480) < 0.001
        assert abs(width - 200/640) < 0.001
        assert abs(height - 200/480) < 0.001
    
    def test_obb_yolo_format(self, bbox_obb, image_size):
        """Test conversion to OBB YOLO format."""
        converter = YOLOConverter('yolov11-obb', image_size[0], image_size[1])
        result = converter.convert_bbox(bbox_obb)
        
        # Parse result
        parts = result.split()
        assert len(parts) == 6  # class, x, y, w, h, angle
        
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        angle = float(parts[5])
        
        # Verify class ID
        assert class_id == 1
        
        # Verify normalization
        assert 0 <= x_center <= 1
        assert 0 <= y_center <= 1
        assert 0 <= width <= 1
        assert 0 <= height <= 1
        
        # Verify angle is in radians
        assert -np.pi <= angle <= np.pi
    
    def test_aabb_to_obb_conversion(self, bbox_aabb, image_size):
        """Test converting AABB to OBB format (should have angle=0)."""
        converter = YOLOConverter('yolov11-obb', image_size[0], image_size[1])
        result = converter.convert_bbox(bbox_aabb)
        
        parts = result.split()
        angle = float(parts[5])
        
        # AABB converted to OBB should have angle near 0
        assert abs(angle) < 0.001
    
    def test_coordinate_clamping(self, image_size):
        """Test that coordinates are clamped to valid range."""
        # Create bbox that extends beyond image
        bbox = BoundingBox(
            class_id=0,
            class_name="test",
            bbox_2d=np.array([-50, -50, 700, 500]),  # Extends beyond 640x480
            visibility=0.5
        )
        
        converter = YOLOConverter('yolov11', image_size[0], image_size[1])
        result = converter.convert_bbox(bbox)
        
        parts = result.split()
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # All values should be in [0, 1]
        assert 0 <= x_center <= 1
        assert 0 <= y_center <= 1
        assert 0 <= width <= 1
        assert 0 <= height <= 1
    
    def test_multiple_yolo_formats(self, bbox_aabb, image_size):
        """Test all supported YOLO formats."""
        formats = ['yolov11', 'yolov26', 'yolov11-obb', 'yolov26-obb']
        
        for fmt in formats:
            converter = YOLOConverter(fmt, image_size[0], image_size[1])
            result = converter.convert_bbox(bbox_aabb)
            
            parts = result.split()
            # Standard formats have 5 parts, OBB has 6
            if 'obb' in fmt:
                assert len(parts) == 6
            else:
                assert len(parts) == 5
    
    def test_invalid_format(self, image_size):
        """Test error on invalid YOLO format."""
        with pytest.raises(ValueError, match="Invalid YOLO format"):
            YOLOConverter('invalid_format', image_size[0], image_size[1])
    
    def test_convert_multiple_bboxes(self, bbox_aabb, bbox_obb, image_size):
        """Test converting multiple bounding boxes."""
        converter = YOLOConverter('yolov11', image_size[0], image_size[1])
        result = converter.convert_bboxes([bbox_aabb, bbox_obb])
        
        lines = result.strip().split('\n')
        assert len(lines) == 2
        
        # Each line should have 5 values
        for line in lines:
            parts = line.split()
            assert len(parts) == 5
    
    def test_angle_normalization(self, image_size):
        """Test that OBB angles are normalized correctly."""
        # Test various angles
        test_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -np.pi/4]
        
        for angle in test_angles:
            corners = np.array([[10, 10], [110, 20], [100, 120], [0, 110]])
            bbox = BoundingBox(
                class_id=0,
                class_name="test",
                bbox_2d=corners,
                angle=angle,
                visibility=1.0
            )
            
            converter = YOLOConverter('yolov11-obb', image_size[0], image_size[1])
            result = converter.convert_bbox(bbox)
            
            parts = result.split()
            normalized_angle = float(parts[5])
            
            # Angle should be in [-pi/2, pi/2] range
            assert -np.pi/2 <= normalized_angle <= np.pi/2


class TestYOLODatasetFormatter:
    """Test suite for YOLODatasetFormatter class."""
    
    def test_create_structure(self, tmp_path):
        """Test dataset directory structure creation."""
        formatter = YOLODatasetFormatter(
            str(tmp_path / "dataset"),
            ['class_a', 'class_b'],
            'yolov11'
        )
        
        # Check directories exist
        dataset_path = tmp_path / "dataset"
        for split in ['train', 'val', 'test']:
            assert (dataset_path / 'images' / split).exists()
            assert (dataset_path / 'labels' / split).exists()
    
    def test_create_data_yaml(self, tmp_path):
        """Test data.yaml creation."""
        formatter = YOLODatasetFormatter(
            str(tmp_path / "dataset"),
            ['cube', 'sphere', 'cylinder'],
            'yolov11'
        )
        formatter.create_data_yaml()
        
        yaml_file = tmp_path / "dataset" / "data.yaml"
        assert yaml_file.exists()
        
        with open(yaml_file, 'r') as f:
            content = f.read()
        
        # Check content
        assert 'nc: 3' in content
        assert 'cube' in content
        assert 'sphere' in content
        assert 'cylinder' in content
    
    def test_create_data_yaml_obb(self, tmp_path):
        """Test data.yaml creation for OBB format."""
        formatter = YOLODatasetFormatter(
            str(tmp_path / "dataset"),
            ['class_a'],
            'yolov11-obb'
        )
        formatter.create_data_yaml()
        
        yaml_file = tmp_path / "dataset" / "data.yaml"
        with open(yaml_file, 'r') as f:
            content = f.read()
        
        # Should include OBB task specification
        assert 'task: obb' in content
    
    def test_create_classes_file(self, tmp_path):
        """Test classes.txt creation."""
        class_names = ['dog', 'cat', 'bird']
        formatter = YOLODatasetFormatter(
            str(tmp_path / "dataset"),
            class_names,
            'yolov11'
        )
        formatter.create_classes_file()
        
        classes_file = tmp_path / "dataset" / "classes.txt"
        assert classes_file.exists()
        
        with open(classes_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        assert lines == class_names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
