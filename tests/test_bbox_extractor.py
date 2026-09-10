"""
Unit tests for BBoxExtractor.
Tests bounding box extraction, unclipped occlusion handling, and visibility calculation.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bbox_extractor import BBoxExtractor, BoundingBox


class TestBBoxExtractor:
    """Test suite for BBoxExtractor."""

    def test_intercepted_object_multiple_contours(self):
        extractor = BBoxExtractor({
            'output': {
                'yolo_format': 'yolov11',
                'detection_params': {
                    'min_bbox_side_px': 5,
                    'min_visibility': 0.0
                }
            }
        })
        
        image_shape = (100, 100, 3)
        
        # Left piece at x in [10, 30], y in [20, 80]
        # Right piece at x in [70, 90], y in [20, 80]
        # Middle (x in [30, 70]) is occluded by an intercepting distractor
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 10:30] = 1
        mask[20:80, 70:90] = 1
        
        bbox = extractor._extract_bbox_from_mask(mask, class_id=0, class_name='gate', image_shape=image_shape)
        
        # Bounding box should span from x=10 to x=90 (width 80), not just one half (width 20)!
        x_min, y_min, x_max, y_max = bbox.bbox_2d
        assert x_min <= 10
        assert x_max >= 90
        assert y_min <= 20
        assert y_max >= 80

    def test_hollow_frame_visibility_not_penalized(self):
        extractor = BBoxExtractor({
            'output': {
                'yolo_format': 'yolov11',
                'detection_params': {
                    'min_bbox_side_px': 5,
                    'min_visibility': 0.3
                }
            }
        })
        
        image_shape = (200, 200, 3)
        mask = np.zeros((200, 200), dtype=np.uint8)
        
        # Draw a thin hollow square frame: outer [20, 180], inner [40, 160]
        mask[20:180, 20:40] = 1   # left
        mask[20:180, 160:180] = 1 # right
        mask[20:40, 20:180] = 1   # top
        mask[160:180, 20:180] = 1 # bottom
        
        bbox = extractor._extract_bbox_from_mask(mask, class_id=0, class_name='gate', image_shape=image_shape)
        
        # Should pass the filter
        assert extractor._should_keep_bbox(bbox) is True
        assert bbox.visibility > 0.3

    def test_filter_heavily_occluded_object(self):
        extractor = BBoxExtractor({
            'output': {
                'yolo_format': 'yolov11',
                'detection_params': {
                    'min_bbox_side_px': 5,
                    'min_visibility': 0.5
                }
            }
        })
        
        low_vis_bbox = BoundingBox(
            class_id=0,
            class_name='gate',
            bbox_2d=np.array([10, 10, 90, 90]),
            visibility=0.10,
            area_px=50
        )
        
        high_vis_bbox = BoundingBox(
            class_id=0,
            class_name='gate',
            bbox_2d=np.array([10, 10, 90, 90]),
            visibility=0.80,
            area_px=400
        )
        
        assert extractor._should_keep_bbox(low_vis_bbox) is False
        assert extractor._should_keep_bbox(high_vis_bbox) is True

    def test_extract_full_bbox_depth_occlusion(self):
        extractor = BBoxExtractor({
            'output': {
                'yolo_format': 'yolov11',
                'detection_params': {
                    'min_bbox_side_px': 5,
                    'min_visibility': 0.8
                }
            }
        })
        
        # Test fallback when blenderproc is mocked/none
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        
        bbox = extractor._extract_full_bbox(
            obj=None,
            mask=mask,
            class_id=0,
            class_name='gate',
            image_shape=(100, 100, 3)
        )
        assert bbox is not None
        assert bbox.visibility == 1.0
        assert bbox.bbox_2d[0] == 20
        assert bbox.bbox_2d[2] == 79

    def test_project_points_exact_fallback(self):
        from bbox_extractor import project_points_exact
        # Empty input
        empty_res = project_points_exact([])
        assert len(empty_res) == 0
        
        # When bpy is not available (running in pure python), returns empty array gracefully
        res = project_points_exact(np.array([[1.0, 2.0, 3.0]]))
        assert isinstance(res, np.ndarray)

    def test_standard_yolo_returns_aabb(self):
        extractor = BBoxExtractor({
            'output': {
                'yolo_format': 'yolov11',
                'detection_params': {
                    'min_bbox_side_px': 5,
                    'min_visibility': 0.1
                }
            }
        })
        # Triangle points that would have triggered minAreaRect in the past
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1
        bbox = extractor._extract_full_bbox(
            obj=None,
            mask=mask,
            class_id=0,
            class_name='gate',
            image_shape=(100, 100, 3)
        )
        assert bbox is not None
        assert bbox.angle is None
        assert len(bbox.bbox_2d) == 4

    def test_visibility_respects_force_fully_visible_flag(self):
        # When force_fully_visible is False:
        extractor_false = BBoxExtractor({
            'output': {
                'yolo_format': 'yolov11',
                'detection_params': {
                    'min_bbox_side_px': 5,
                    'min_visibility': 0.8,
                    'force_fully_visible': False
                }
            }
        })
        # When force_fully_visible is True:
        extractor_true = BBoxExtractor({
            'output': {
                'yolo_format': 'yolov11',
                'detection_params': {
                    'min_bbox_side_px': 5,
                    'min_visibility': 0.8,
                    'force_fully_visible': True
                }
            }
        })
        assert extractor_false.force_fully_visible is False
        assert extractor_true.force_fully_visible is True


