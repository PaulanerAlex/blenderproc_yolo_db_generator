"""
YOLO Format Converter.

This module converts bounding box annotations to YOLO formats:
- YOLOv11/v26: Standard axis-aligned bounding boxes
- YOLOv11-OBB/v26-OBB: Oriented bounding boxes with rotation
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
from bbox_extractor import BoundingBox


class YOLOConverter:
    """Converts bounding boxes to YOLO annotation format."""
    
    def __init__(self, yolo_format: str, image_width: int, image_height: int):
        """
        Initialize YOLO converter.
        
        Args:
            yolo_format: One of 'yolov11', 'yolov26', 'yolov11-obb', 'yolov26-obb'
            image_width: Image width in pixels
            image_height: Image height in pixels
        """
        self.yolo_format = yolo_format.lower()
        self.image_width = image_width
        self.image_height = image_height
        
        # Validate format
        valid_formats = ['yolov11', 'yolov26', 'yolov11-obb', 'yolov26-obb']
        if self.yolo_format not in valid_formats:
            raise ValueError(f"Invalid YOLO format: {yolo_format}. Must be one of {valid_formats}")
        
        self.is_obb = 'obb' in self.yolo_format
    
    def convert_bbox(self, bbox: BoundingBox) -> str:
        """
        Convert bounding box to YOLO format string.
        
        Args:
            bbox: BoundingBox object
            
        Returns:
            YOLO format annotation line
        """
        if self.is_obb:
            return self._convert_to_obb(bbox)
        else:
            return self._convert_to_standard(bbox)
    
    def _convert_to_standard(self, bbox: BoundingBox) -> str:
        """
        Convert to standard YOLO format (YOLOv11/v26).
        Format: class_id x_center y_center width height (normalized 0-1)
        
        Args:
            bbox: BoundingBox object
            
        Returns:
            YOLO annotation line
        """
        # Get AABB from bbox
        if bbox.angle is None:
            # Already AABB
            x_min, y_min, x_max, y_max = bbox.bbox_2d
        else:
            # Convert OBB to AABB
            x_min = bbox.bbox_2d[:, 0].min()
            y_min = bbox.bbox_2d[:, 1].min()
            x_max = bbox.bbox_2d[:, 0].max()
            y_max = bbox.bbox_2d[:, 1].max()
        
        # Calculate center and size
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        
        # Normalize to 0-1
        x_center_norm = x_center / self.image_width
        y_center_norm = y_center / self.image_height
        width_norm = width / self.image_width
        height_norm = height / self.image_height
        
        # Clamp to valid range
        x_center_norm = np.clip(x_center_norm, 0, 1)
        y_center_norm = np.clip(y_center_norm, 0, 1)
        width_norm = np.clip(width_norm, 0, 1)
        height_norm = np.clip(height_norm, 0, 1)
        
        return f"{bbox.class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
    
    def _convert_to_obb(self, bbox: BoundingBox) -> str:
        """
        Convert to OBB YOLO format (YOLOv11-OBB/v26-OBB).
        Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized 0-1)
        
        Args:
            bbox: BoundingBox object
            
        Returns:
            YOLO OBB annotation line
        """
        if bbox.angle is None:
            # Convert AABB to 4 corners
            x_min, y_min, x_max, y_max = bbox.bbox_2d
            corners = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ])
        else:
            # Use OBB corners
            corners = bbox.bbox_2d
        
        # Normalize and flatten corners
        normalized_coords = []
        for point in corners:
            x_norm = np.clip(point[0] / self.image_width, 0, 1)
            y_norm = np.clip(point[1] / self.image_height, 0, 1)
            normalized_coords.extend([x_norm, y_norm])
        
        # Format as string: class_id x1 y1 x2 y2 x3 y3 x4 y4
        coords_str = " ".join([f"{c:.6f}" for c in normalized_coords])
        return f"{bbox.class_id} {coords_str}"
    
    def convert_bboxes(self, bboxes: List[BoundingBox]) -> str:
        """
        Convert list of bounding boxes to YOLO format.
        
        Args:
            bboxes: List of BoundingBox objects
            
        Returns:
            Multi-line string with YOLO annotations
        """
        lines = []
        for bbox in bboxes:
            line = self.convert_bbox(bbox)
            lines.append(line)
        return '\n'.join(lines)
    
    def save_annotations(self, 
                        bboxes: List[BoundingBox],
                        output_path: str):
        """
        Save annotations to file.
        
        Args:
            bboxes: List of BoundingBox objects
            output_path: Path to save annotations
        """
        annotations = self.convert_bboxes(bboxes)
        
        # Create parent directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if annotations:
                f.write(annotations)
                if not annotations.endswith('\n'):
                    f.write('\n')


class YOLODatasetFormatter:
    """Formats complete YOLO dataset with proper structure."""
    
    def __init__(self, 
                 output_path: str,
                 class_names: List[str],
                 yolo_format: str):
        """
        Initialize dataset formatter.
        
        Args:
            output_path: Root path for dataset
            class_names: List of class names
            yolo_format: YOLO format (e.g., 'yolov11', 'yolov11-obb')
        """
        self.output_path = Path(output_path)
        self.class_names = class_names
        self.yolo_format = yolo_format
        
        # Create directory structure
        self._create_structure()
    
    def _create_structure(self):
        """Create YOLO dataset directory structure."""
        # Create main directories
        for split in ['train', 'val', 'test']:
            (self.output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    def create_data_yaml(self):
        """Create data.yaml configuration file for YOLO."""
        yaml_content = f"""# YOLO Dataset Configuration
# Generated by Blender YOLO Dataset Generator
# Format: {self.yolo_format}

path: {self.output_path.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(self.class_names)}
names:
"""
        
        # Add class names
        for i, name in enumerate(self.class_names):
            yaml_content += f"  {i}: {name}\n"
        
        # Add OBB flag if applicable
        if 'obb' in self.yolo_format.lower():
            yaml_content += "\n# Oriented Bounding Box (OBB) format\n"
            yaml_content += "task: obb\n"
        
        # Save
        yaml_path = self.output_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created data.yaml at {yaml_path}")
    
    def create_classes_file(self):
        """Create classes.txt file with class names."""
        classes_path = self.output_path / 'classes.txt'
        with open(classes_path, 'w') as f:
            for name in self.class_names:
                f.write(f"{name}\n")
        
        print(f"Created classes.txt at {classes_path}")
    
    def get_image_path(self, split: str, image_name: str) -> Path:
        """Get path for image file."""
        return self.output_path / 'images' / split / image_name
    
    def get_label_path(self, split: str, label_name: str) -> Path:
        """Get path for label file."""
        return self.output_path / 'labels' / split / label_name


def test_yolo_converter():
    """Test YOLO converter with sample data."""
    print("Testing YOLO Converter...")
    
    # Create sample bounding boxes
    bbox_aabb = BoundingBox(
        class_id=0,
        class_name="object_a",
        bbox_2d=np.array([100, 150, 300, 350]),  # AABB
        visibility=0.9
    )
    
    bbox_obb = BoundingBox(
        class_id=1,
        class_name="object_b",
        bbox_2d=np.array([[120, 140], [320, 160], [300, 360], [100, 340]]),  # OBB corners
        angle=np.pi / 6,  # 30 degrees
        visibility=0.85
    )
    
    # Test standard format
    print("\n--- Standard YOLO (v11) ---")
    converter_std = YOLOConverter('yolov11', 640, 480)
    print(converter_std.convert_bbox(bbox_aabb))
    print(converter_std.convert_bbox(bbox_obb))
    
    # Test OBB format
    print("\n--- OBB YOLO (v11-obb) ---")
    converter_obb = YOLOConverter('yolov11-obb', 640, 480)
    print(converter_obb.convert_bbox(bbox_aabb))
    print(converter_obb.convert_bbox(bbox_obb))
    
    print("\n✓ YOLO converter test completed")


if __name__ == '__main__':
    test_yolo_converter()
