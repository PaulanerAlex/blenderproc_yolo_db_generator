"""
Bounding Box Extraction from BlenderProc Output.

This module extracts both standard axis-aligned bounding boxes (AABB)
and oriented bounding boxes (OBB) from BlenderProc HDF5 output files.
"""

import numpy as np
try:
    import h5py
except ImportError:
    h5py = None

try:
    import blenderproc as bproc
except ImportError:
    bproc = None

try:
    import bpy
except ImportError:
    bpy = None

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import cv2


def project_points_exact(points: np.ndarray, frame: Optional[int] = None) -> np.ndarray:
    """
    Project 3D world points into 2D camera image pixel coordinates.
    Accurately accounts for camera location, rotation, focal length,
    sensor fit, resolution, and camera shifts (shift_x, shift_y),
    matching Blender's Cycles and Eevee renderers to the subpixel.
    """
    if points is None or len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    try:
        import bpy
        scene = bpy.context.scene
        cam_ob = scene.camera

        if frame is not None and bproc is not None and hasattr(bproc, 'camera') and hasattr(bproc.camera, 'get_camera_pose'):
            cam2world = bproc.camera.get_camera_pose(frame)
        else:
            if frame is not None:
                scene.frame_set(frame)
            cam2world = np.array(cam_ob.matrix_world)

        world2cam = np.linalg.inv(cam2world)
        pts_homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        pts_cam = (world2cam @ pts_homo.T).T

        xc = pts_cam[:, 0]
        yc = pts_cam[:, 1]
        zc = -pts_cam[:, 2]  # depth along optical axis

        cam = cam_ob.data
        view_frame = cam.view_frame(scene=scene)
        min_x, max_x = view_frame[2].x, view_frame[1].x
        min_y, max_y = view_frame[1].y, view_frame[0].y

        zc_safe = np.where(np.abs(zc) < 1e-6, 1e-6, zc)
        scale = zc_safe / (-view_frame[0].z)

        fx_min = min_x * scale
        fx_max = max_x * scale
        fy_min = min_y * scale
        fy_max = max_y * scale

        x_ndc = (xc - fx_min) / (fx_max - fx_min)
        y_ndc = (yc - fy_min) / (fy_max - fy_min)

        px = x_ndc * scene.render.resolution_x
        py = (1.0 - y_ndc) * scene.render.resolution_y

        res = np.column_stack([px, py])
        res[zc <= 0] = np.nan
        return res
    except Exception:
        # Fallback if bpy not available
        return np.empty((0, 2), dtype=np.float64)



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
        self.min_visibility = float(self.detection_params.get('min_visibility', 0.3))
        if self.min_visibility > 1.0:
            self.min_visibility /= 100.0
        self.force_fully_visible = self.detection_params.get('force_fully_visible', False)
        self.occlusion_samples = self.detection_params.get('occlusion_samples', 100)
    
    def extract_from_dict(self,
                          data: Dict[str, Any],
                          class_mapping: Dict[int, str],
                          target_objects: List[Any],
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
            
        # Check if depth map is available
        depth_map = None
        if 'depth' in data and len(data['depth']) > image_idx:
            depth_map = data['depth'][image_idx]
        elif 'distance' in data and len(data['distance']) > image_idx:
            depth_map = data['distance'][image_idx]

        bboxes = []
        
        # Create map from pass_index to MeshObject
        idx_to_obj = {obj.blender_obj.pass_index: obj for obj in target_objects if hasattr(obj, 'blender_obj') and obj.blender_obj is not None}
        
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

            # Get class ID
            class_id = obj.blender_obj.get("category_id")
            if class_id is None:
                continue
            
            # Extract full unclipped bounding box using 3D object geometry + mask
            bbox = self._extract_full_bbox(
                obj,
                mask,
                class_id,
                class_mapping.get(class_id, f"class_{class_id}"),
                image_shape,
                image_idx=image_idx,
                depth_map=depth_map
            )
            
            if bbox is None:
                continue

            # Check if clipped by image boundaries (if force_fully_visible is requested)
            if self.force_fully_visible:
                if bbox.angle is None:
                    x_min, y_min, x_max, y_max = bbox.bbox_2d
                    if x_min < 0 or y_min < 0 or x_max > image_shape[1] or y_max > image_shape[0]:
                        continue
                else:
                    corners = bbox.bbox_2d
                    if np.any(corners[:, 0] < 0) or np.any(corners[:, 0] > image_shape[1]) or \
                       np.any(corners[:, 1] < 0) or np.any(corners[:, 1] > image_shape[0]):
                        continue

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
                    class_id = None
                    if instance_attrs is not None:
                         # Mapping varies depending on bproc version/settings
                         if isinstance(instance_attrs, list) and instance_id - 1 < len(instance_attrs):
                             attr = instance_attrs[instance_id - 1]
                             if isinstance(attr, dict):
                                 class_id = attr.get('category_id') or attr.get('class_id')
                    
                    if class_id is None:
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
    
    def _extract_full_bbox(self,
                           obj: Any,
                           mask: np.ndarray,
                           class_id: int,
                           class_name: str,
                           image_shape: Tuple[int, ...],
                           image_idx: int = 0,
                           depth_map: Optional[np.ndarray] = None) -> Optional[BoundingBox]:
        """
        Extract full (amodal) bounding box of object, projecting its 3D geometry
        so it is not clipped when occluded/intercepted by other objects.
        """
        img_h, img_w = image_shape[:2]
        pts_2d = None
        sample_pts_2d = None
        sample_depths = None

        # 1. Project 3D geometry if blenderproc and blender_obj are available
        if bproc is not None and hasattr(obj, 'blender_obj') and obj.blender_obj is not None:
            try:
                b_obj = obj.blender_obj
                mat_world = np.array(b_obj.matrix_world)
                
                pts_3d_list = []
                
                # Use true physical mesh vertices (avoid 3D cuboid corners that project into empty air)
                if hasattr(b_obj, 'data') and hasattr(b_obj.data, 'vertices'):
                    mesh_verts = [v.co for v in b_obj.data.vertices]
                    for v in mesh_verts:
                        pts_3d_list.append((mat_world @ np.array([*v, 1.0]))[:3])
                    
                    # Edge sample points for accurate surface sampling
                    if hasattr(b_obj.data, 'edges'):
                        for edge in b_obj.data.edges:
                            v1 = (mat_world @ np.array([*mesh_verts[edge.vertices[0]], 1.0]))[:3]
                            v2 = (mat_world @ np.array([*mesh_verts[edge.vertices[1]], 1.0]))[:3]
                            for t in [0.25, 0.5, 0.75]:
                                pts_3d_list.append(v1 * (1.0 - t) + v2 * t)
                elif hasattr(obj, 'get_bound_box'):
                    pts_3d_list.extend(obj.get_bound_box())
                elif hasattr(b_obj, 'bound_box'):
                    for corner in b_obj.bound_box:
                        pts_3d_list.append((mat_world @ np.array([*corner, 1.0]))[:3])

                if pts_3d_list:
                    pts_3d = np.array(pts_3d_list, dtype=np.float64)
                    
                    # Transform to camera space to check depth and clip points behind camera
                    try:
                        import bpy
                        cam2world = bproc.camera.get_camera_pose(image_idx) if bproc is not None else np.array(bpy.context.scene.camera.matrix_world)
                    except Exception:
                        cam2world = np.eye(4)

                    world2cam = np.linalg.inv(cam2world)
                    pts_homo = np.concatenate([pts_3d, np.ones((len(pts_3d), 1))], axis=1)
                    pts_cam = (world2cam @ pts_homo.T).T
                    depths = -pts_cam[:, 2]  # Depth along optical axis in Blender camera frame
                    
                    in_front = depths > 0.05
                    if np.any(in_front):
                        valid_3d = pts_3d[in_front]
                        sample_depths = depths[in_front]
                        sample_pts_2d = project_points_exact(valid_3d, frame=image_idx)
                        pts_2d = sample_pts_2d.copy()
            except Exception as e:
                pts_2d = None

        # 2. Combine with visible mask points
        mask_y, mask_x = np.where(mask > 0)
        area_px = float(len(mask_x))
        if len(mask_x) == 0 and pts_2d is None:
            return None

        if len(mask_x) > 0:
            mask_pts = np.column_stack([mask_x, mask_y])
            if pts_2d is not None:
                all_pts = np.vstack([pts_2d, mask_pts])
            else:
                all_pts = mask_pts
        else:
            all_pts = pts_2d

        if all_pts is None or len(all_pts) == 0:
            return None

        # 3. Calculate true visibility ratio
        # A sample point is occluded IF AND ONLY IF an obstacle in front is closer to the camera.
        # Background seen through holes/empty centers has buffer_depth > sample_depth and is NOT occlusion.
        if sample_pts_2d is not None and len(sample_pts_2d) > 0:
            px = sample_pts_2d[:, 0]
            py = sample_pts_2d[:, 1]
            in_bounds = (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
            
            if np.sum(in_bounds) == 0:
                visibility = 0.0
            else:
                dilated_mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
                int_x = np.clip(np.round(px[in_bounds]).astype(int), 0, img_w - 1)
                int_y = np.clip(np.round(py[in_bounds]).astype(int), 0, img_h - 1)
                
                mask_vis = dilated_mask[int_y, int_x] > 0
                
                if depth_map is not None and sample_depths is not None:
                    pts_depth = sample_depths[in_bounds]
                    buffer_depth = depth_map[int_y, int_x]
                    valid_depth = ~np.isnan(buffer_depth) & (buffer_depth > 0)
                    # Point is occluded only if another surface is >= 5cm closer to camera AND not on object mask
                    is_occluded = valid_depth & (buffer_depth < (pts_depth - 0.05)) & (~mask_vis)
                    unoccluded_count = np.sum(~is_occluded)
                else:
                    unoccluded_count = np.sum(mask_vis)
                    
                total_samples = len(sample_pts_2d) if self.force_fully_visible else np.sum(in_bounds)
                visibility = float(unoccluded_count) / float(total_samples) if total_samples > 0 else 0.0
        else:
            # Fallback for mask-only: ratio of mask pixels to convex hull area
            hull = cv2.convexHull(all_pts.astype(np.float32))
            hull_area = cv2.contourArea(hull)
            visibility = min(1.0, area_px / hull_area) if hull_area > 0 else 1.0

        # 4. Construct Bounding Box
        yolo_format = self.config.get('output', {}).get('yolo_format', '').lower()
        is_obb = 'obb' in yolo_format

        x_min = float(np.min(all_pts[:, 0]))
        x_max = float(np.max(all_pts[:, 0]))
        y_min = float(np.min(all_pts[:, 1]))
        y_max = float(np.max(all_pts[:, 1]))
        bbox_aabb = np.array([x_min, y_min, x_max, y_max])

        if is_obb and len(all_pts) >= 5:
            rect = cv2.minAreaRect(all_pts.astype(np.float32))
            box = cv2.boxPoints(rect)
            angle = np.deg2rad(rect[2])
            return BoundingBox(class_id, class_name, box, angle=angle, visibility=visibility, area_px=area_px)

        return BoundingBox(class_id, class_name, bbox_aabb, visibility=visibility, area_px=area_px)

    def _extract_bbox_from_mask(self,
                                mask: np.ndarray,
                                class_id: int,
                                class_name: str,
                                image_shape: Tuple[int, ...]) -> BoundingBox:
        """
        Extract bounding box from segmentation mask (fallback method).
        
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
        
        # Combine all contours so disconnected fragments of the same object are not discarded
        all_contour_pts = np.vstack(contours)
        
        # Get axis-aligned bounding box
        x, y, w, h = cv2.boundingRect(all_contour_pts)
        bbox_aabb = np.array([x, y, x + w, y + h])
        
        # Visibility: ratio of actual pixels to convex hull area (better than raw bbox for hollow shapes)
        hull = cv2.convexHull(all_contour_pts)
        hull_area = cv2.contourArea(hull)
        visibility = min(1.0, area_px / hull_area) if hull_area > 0 else 1.0
        
        # Get oriented bounding box
        yolo_format = self.config.get('output', {}).get('yolo_format', '').lower()
        if len(all_contour_pts) >= 5:
            rect = cv2.minAreaRect(all_contour_pts)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            angle = np.deg2rad(rect[2])
            
            obb_area = rect[1][0] * rect[1][1]
            bbox_area = max(1.0, w * h)
            if 'obb' in yolo_format or obb_area < bbox_area * 0.9:
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
