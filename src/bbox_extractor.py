"""
Bounding Box Extraction from BlenderProc Output.

This module extracts both standard axis-aligned bounding boxes (AABB)
and oriented bounding boxes (OBB) from BlenderProc HDF5 output files.
"""

import numpy as np
import h5py
import blenderproc as bproc
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import cv2


class BoundingBox:
    """Represents a bounding box annotation."""
    
    def __init__(self, 
                 class_id: int,
                 class_name: str,
                 bbox_2d: np.ndarray,
                 angle: Optional[float] = None,
                 visibility: float = 1.0,
                 area_px: float = 0.0):
        """
        Initialize bounding box.
        
        Args:
            class_id: Class ID (0-indexed)
            class_name: Class name
            bbox_2d: 2D bounding box [x_min, y_min, x_max, y_max] or OBB corners
            angle: Rotation angle in radians (for OBB, optional)
            visibility: Visibility ratio (0-1)
            area_px: Area in pixels
        """
        self.class_id = class_id
        self.class_name = class_name
        self.bbox_2d = bbox_2d
        self.angle = angle
        self.visibility = visibility
        self.area_px = area_px
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'bbox_2d': self.bbox_2d.tolist(),
            'angle': self.angle,
            'visibility': self.visibility,
            'area_px': self.area_px
        }


class BBoxExtractor:
    """Extracts bounding boxes from BlenderProc output."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize bbox extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.detection_params = config.get('output', {}).get('detection_params', {})
        self.min_bbox_side = self.detection_params.get('min_bbox_side_px', 10)
        self.min_visibility = self.detection_params.get('min_visibility', 0.3)
        self.force_fully_visible = self.detection_params.get('force_fully_visible', False)
        self.occlusion_samples = self.detection_params.get('occlusion_samples', 100)
    
    def extract_from_dict(self,
                          data: Dict[str, Any],
                          class_mapping: Dict[int, str],
                          target_objects: List[bproc.types.MeshObject],
                          image_idx: int = 0) -> List[BoundingBox]:
        """
        Extract bounding boxes from dictionary (output of bproc.renderer.render()).
        
        Args:
            data: Data dictionary from bproc.renderer.render()
            class_mapping: Mapping from class ID to class name
            target_objects: List of target MeshObjects in scene
            image_idx: Index of image in data arrays
            
        Returns:
            List of bounding boxes
        """
        # Load image (for shape)
        image_shape = data['colors'][image_idx].shape
        height, width = image_shape[:2]
        
        # Load instance segmentation
        if 'instance_segmaps' in data:
            instance_segmap = data['instance_segmaps'][image_idx]
        elif 'class_segmaps' in data:
            instance_segmap = data['class_segmaps'][image_idx]
        else:
            return []
            
        bboxes = []
        
        # Create map from pass_index to MeshObject
        idx_to_obj = {obj.blender_obj.pass_index: obj for obj in target_objects}
        
        # Get unique instance IDs
        unique_instances = np.unique(instance_segmap)
        
        for instance_id in unique_instances:
            if instance_id == 0:  # Skip background
                continue
            
            # Find the corresponding object
            obj = idx_to_obj.get(instance_id)
            if not obj:
                continue
                
            # Get mask for this instance
            mask = (instance_segmap == instance_id).astype(np.uint8)
            visible_pixels = np.sum(mask > 0)
            
            if visible_pixels == 0:
                continue

            # 1. Check if clipped by image boundaries using 3D projection
            bbox_3d_corners = obj.get_bound_box(local_coords=False)
            
            # Use BlenderProc's built-in projection
            try:
                proj_2d = bproc.camera.project_points(bbox_3d_corners, frame=image_idx)
            except Exception:
                continue
            
            if proj_2d is None:
                continue
                
            # Check if any projected corner is outside the frame
            is_clipped = (np.any(proj_2d[:, 0] < -0.5) or np.any(proj_2d[:, 0] >= width - 0.5) or
                          np.any(proj_2d[:, 1] < -0.5) or np.any(proj_2d[:, 1] >= height - 0.5))
            
            if self.force_fully_visible and is_clipped:
                continue

            # 2. Calculate "True" Visibility
            # Visibility = (actual visible pixels) / (total pixels if unoccluded)
            # Area that *should* be in frame (AABB of projected 2D corners, clamped)
            in_frame_x_min = np.clip(np.min(proj_2d[:, 0]), 0, width)
            in_frame_x_max = np.clip(np.max(proj_2d[:, 0]), 0, width)
            in_frame_y_min = np.clip(np.min(proj_2d[:, 1]), 0, height)
            in_frame_y_max = np.clip(np.max(proj_2d[:, 1]), 0, height)
            
            expected_area_in_frame = (in_frame_x_max - in_frame_x_min) * (in_frame_y_max - in_frame_y_min)
            
            # Visibility relative to what should be seen in this frame
            true_visibility = visible_pixels / expected_area_in_frame if expected_area_in_frame > 0 else 0.0
            
            # Get class ID
            class_id = obj.blender_obj.get("category_id")
            if class_id is None:
                continue
            
            # Extract bounding box from visible pixels
            bbox = self._extract_bbox_from_mask(mask, class_id, class_mapping[class_id], image_shape)
            
            # Override visibility with our "true" calculation
            bbox.visibility = true_visibility
            
            # Filter based on criteria
            if self._should_keep_bbox(bbox):
                bboxes.append(bbox)
        
        return bboxes

    def extract_from_hdf5(self,
                         hdf5_path: str,
                         class_mapping: Dict[int, str],
                         image_idx: int = 0) -> Tuple[np.ndarray, List[BoundingBox]]:
        """
        Extract bounding boxes from HDF5 file.
        
        Args:
            hdf5_path: Path to HDF5 file
            class_mapping: Mapping from class ID to class name
            image_idx: Index of image in HDF5 file
            
        Returns:
            Tuple of (image, list of bounding boxes)
        """
        with h5py.File(hdf5_path, 'r') as f:
            # Load image
            colors = f['colors']
            image = np.array(colors[image_idx])
            
            # Load instance segmentation if available
            if 'instance_segmaps' in f:
                instance_segmap = np.array(f['instance_segmaps'][image_idx])
            else:
                # Fallback: use class segmentation
                instance_segmap = np.array(f['class_segmaps'][image_idx]) if 'class_segmaps' in f else None
            
            # Load instance attribute maps (contains class IDs)
            if 'instance_attribute_maps' in f:
                instance_attrs = f['instance_attribute_maps'][image_idx]
            else:
                instance_attrs = None
            
            bboxes = []
            
            if instance_segmap is not None:
                # Get unique instance IDs
                unique_instances = np.unique(instance_segmap)
                
                for instance_id in unique_instances:
                    if instance_id == 0:  # Skip background
                        continue
                    
                    # Get mask for this instance
                    mask = (instance_segmap == instance_id).astype(np.uint8)
                    
                    # Get class ID from instance attributes
                    if instance_attrs is not None and 'class_id' in instance_attrs:
                        class_id = int(instance_attrs['class_id'][instance_id - 1])
                    else:
                        # Fallback: assume instance_id is class_id
                        class_id = int(instance_id) - 1
                    
                    # Skip if class not in mapping
                    if class_id not in class_mapping:
                        continue
                    
                    # Extract bounding box
                    bbox = self._extract_bbox_from_mask(mask, class_id, class_mapping[class_id], image.shape)
                    
                    # Filter based on criteria
                    if self._should_keep_bbox(bbox):
                        bboxes.append(bbox)
        
        return image, bboxes
    
    def _extract_bbox_from_mask(self,
                                mask: np.ndarray,
                                class_id: int,
                                class_name: str,
                                image_shape: Tuple[int, ...]) -> BoundingBox:
        """
        Extract bounding box from segmentation mask.
        
        Args:
            mask: Binary segmentation mask
            class_id: Class ID
            class_name: Class name
            image_shape: Shape of the image
            
        Returns:
            BoundingBox object
        """
        # Calculate visibility
        total_pixels = np.sum(mask > 0)
        area_px = float(total_pixels)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Return empty bbox
            return BoundingBox(class_id, class_name, np.array([0, 0, 0, 0]), visibility=0.0)
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Get axis-aligned bounding box
        x, y, w, h = cv2.boundingRect(contour)
        bbox_aabb = np.array([x, y, x + w, y + h])
        
        # Calculate visibility (ratio of actual pixels to bbox area)
        # The visibility calculated here is the "simple" ratio. 
        # We will override this in extract_from_dict with true_visibility.
        bbox_area = w * h
        visibility = total_pixels / bbox_area if bbox_area > 0 else 0.0
        
        # Get oriented bounding box
        if len(contour) >= 5:  # Need at least 5 points for minAreaRect
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Get angle (in radians)
            angle = np.deg2rad(rect[2])
            
            # Use OBB if it's significantly better than AABB
            obb_area = rect[1][0] * rect[1][1]
            if obb_area < bbox_area * 0.9:  # OBB is at least 10% smaller
                return BoundingBox(class_id, class_name, box, angle=angle, 
                                 visibility=visibility, area_px=area_px)
        
        # Return AABB
        return BoundingBox(class_id, class_name, bbox_aabb, 
                         visibility=visibility, area_px=area_px)
    
    def _should_keep_bbox(self, bbox: BoundingBox) -> bool:
        """
        Determine if bbox should be kept based on filtering criteria.
        
        Args:
            bbox: BoundingBox to check
            
        Returns:
            True if bbox should be kept
        """
        # Check visibility
        if bbox.visibility < self.min_visibility:
            return False
        
        # Check minimum size
        if bbox.angle is None:  # AABB
            x_min, y_min, x_max, y_max = bbox.bbox_2d
            width = x_max - x_min
            height = y_max - y_min
            if width < self.min_bbox_side or height < self.min_bbox_side:
                return False
        else:  # OBB
            # Check area
            rect_area = cv2.contourArea(bbox.bbox_2d)
            if rect_area < self.min_bbox_side ** 2:
                return False
        
        return True
    
    def compute_occlusion(self,
                         bbox: BoundingBox,
                         depth_map: np.ndarray,
                         obj_depth: float,
                         num_samples: int = 100) -> float:
        """
        Compute occlusion percentage for bounding box.
        
        Args:
            bbox: Bounding box
            depth_map: Depth map from rendering
            obj_depth: Expected depth of object
            num_samples: Number of points to sample
            
        Returns:
            Occlusion ratio (0 = fully visible, 1 = fully occluded)
        """
        if bbox.angle is None:  # AABB
            x_min, y_min, x_max, y_max = bbox.bbox_2d.astype(int)
            # Sample points within bbox
            xs = np.random.randint(x_min, x_max + 1, num_samples)
            ys = np.random.randint(y_min, y_max + 1, num_samples)
        else:  # OBB
            # Sample points within OBB polygon
            x_min, y_min = bbox.bbox_2d.min(axis=0)
            x_max, y_max = bbox.bbox_2d.max(axis=0)
            
            # Generate candidate points
            xs = np.random.randint(int(x_min), int(x_max) + 1, num_samples * 2)
            ys = np.random.randint(int(y_min), int(y_max) + 1, num_samples * 2)
            
            # Filter points inside polygon
            points = np.column_stack([xs, ys])
            mask = cv2.pointPolygonTest(bbox.bbox_2d.astype(np.float32), tuple(points[0]), False) >= 0
            
            # Take first num_samples points
            xs = xs[mask][:num_samples]
            ys = ys[mask][:num_samples]
        
        if len(xs) == 0:
            return 1.0  # Fully occluded if no valid samples
        
        # Check depth at sampled points
        depths = depth_map[ys, xs]
        
        # Count occluded points (depth significantly less than object depth)
        occluded = np.sum(depths < obj_depth - 0.1)
        
        return occluded / len(xs)


if __name__ == '__main__':
    print("Bounding box extraction module")
    print("This module extracts bounding boxes from BlenderProc HDF5 output")
