"""
Randomization utilities for scene generation.

This module provides utilities for randomizing various aspects of the scene
including object properties, materials, and placement.
"""

import numpy as np
import blenderproc as bproc
import numbers
from typing import List, Dict, Any, Optional
from object_loader import ObjectClass


class ObjectRandomizer:
    """Handles randomization of object properties."""
    
    def __init__(self, config: Dict[str, Any], rng: Optional[np.random.Generator] = None):
        """
        Initialize object randomizer.
        
        Args:
            config: Configuration dictionary
            rng: NumPy random generator (optional)
        """
        self.config = config.get('scene', {}).get('objects', {})
        self.rng = rng if rng is not None else np.random.default_rng()

    def _add_material_noise(self, obj: bproc.types.MeshObject, noise_amount: float):
        """Add random noise to material PBR properties."""
        import numbers
        for mat in obj.get_materials():
            # Randomize roughness
            current_roughness = mat.get_principled_shader_value("Roughness")
            if current_roughness is not None and isinstance(current_roughness, numbers.Number):
                new_roughness = np.clip(
                    current_roughness + self.rng.uniform(-noise_amount, noise_amount),
                    0, 1
                )
                mat.set_principled_shader_value("Roughness", new_roughness)
            
            # Randomize metallic
            current_metallic = mat.get_principled_shader_value("Metallic")
            if current_metallic is not None and isinstance(current_metallic, numbers.Number):
                new_metallic = np.clip(
                    current_metallic + self.rng.uniform(-noise_amount/2, noise_amount/2),
                    0, 1
                )
                mat.set_principled_shader_value("Metallic", new_metallic)
    
    def randomize_scale(self, obj: bproc.types.MeshObject, base_scale: float = 1.0):
        """
        Apply random scale variation to object.
        
        Args:
            obj: Blender object
            base_scale: Base scale factor
        """
        scale_noise = self.config.get('scale_noise', 0.2)
        
        if scale_noise > 0:
            # Apply uniform scaling with noise
            scale_factor = base_scale * (1.0 + self.rng.uniform(-scale_noise, scale_noise))
            current_scale = obj.get_scale()
            obj.set_scale(current_scale * scale_factor)
    
    def randomize_material(self, obj: bproc.types.MeshObject, cc_materials: List = None):
        """
        Apply random material variations.
        
        Args:
            obj: Blender object
            cc_materials: List of CC0 materials (optional)
        """
        pbr_noise = self.config.get('pbr_noise', 0.3)
        
        # If CC materials available, optionally replace material
        if cc_materials and self.rng.random() < 0.8:
            material = self.rng.choice(cc_materials)
            obj.replace_materials(material)
        
        # Add PBR noise to existing materials
        if pbr_noise > 0:
            for mat in obj.get_materials():
                # Randomize roughness
                current_roughness = mat.get_principled_shader_value("Roughness")
                if current_roughness is not None and isinstance(current_roughness, numbers.Number):
                    new_roughness = np.clip(
                        current_roughness + self.rng.uniform(-pbr_noise, pbr_noise),
                        0, 1
                    )
                    mat.set_principled_shader_value("Roughness", new_roughness)
                
                # Randomize metallic
                current_metallic = mat.get_principled_shader_value("Metallic")
                if current_metallic is not None and isinstance(current_metallic, numbers.Number):
                    new_metallic = np.clip(
                        current_metallic + self.rng.uniform(-pbr_noise/2, pbr_noise/2),
                        0, 1
                    )
                    mat.set_principled_shader_value("Metallic", new_metallic)
                
                # Randomize base color slightly (only if it's a color, not a texture)
                base_color = mat.get_principled_shader_value("Base Color")
                if base_color is not None and isinstance(base_color, (list, tuple, np.ndarray)):
                    color_variation = self.rng.uniform(-0.1, 0.1, size=3)
                    new_color = np.clip(np.array(base_color[:3]) + color_variation, 0, 1)
                    mat.set_principled_shader_value("Base Color", list(new_color) + [1.0])
    
    def randomize_rotation(self, obj: bproc.types.MeshObject):
        """Apply random rotation to object."""
        rotation = [
            self.rng.uniform(0, 2 * np.pi),
            self.rng.uniform(0, 2 * np.pi),
            self.rng.uniform(0, 2 * np.pi)
        ]
        obj.set_rotation_euler(rotation)


class CameraSampler:
    """Samples camera poses to view objects."""
    
    def __init__(self, config: Dict[str, Any], rng: Optional[np.random.Generator] = None):
        """
        Initialize camera sampler.
        
        Args:
            config: Configuration dictionary
            rng: NumPy random generator (optional)
        """
        self.config = config.get('scene', {}).get('objects', {})
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def sample_camera_pose(self, 
                          target_obj: bproc.types.MeshObject,
                          room_size: float) -> bool:
        """
        Sample a camera pose looking at target object.
        
        Args:
            target_obj: Object to look at
            room_size: Size of the room
            
        Returns:
            True if valid pose found, False otherwise
        """
        # Get object bounding box
        bbox = target_obj.get_bound_box()
        obj_size = np.max(np.ptp(bbox, axis=0))
        
        # Calculate camera distance range
        min_dist_rel = self.config.get('cam_min_dist_rel', 2.0)
        max_dist_rel = self.config.get('cam_max_dist_rel', 5.0)
        
        # Enforce minimum absolute distance bounds to ensure object takes up ~1/4 to 1/2 screen
        # Not just relative, because small objects map extremely close and hit clipping planes
        min_dist = max(obj_size * min_dist_rel, 0.2)
        max_dist = max(obj_size * max_dist_rel, 0.5)
        
        # Sample distance
        distance = self.rng.uniform(min_dist, max_dist)
        
        # Sample point of interest on object (use bounding box center, not origin)
        bbox = target_obj.get_bound_box()
        obj_center = np.mean(bbox, axis=0)
        
        poi = obj_center + self.rng.uniform(-obj_size/4, obj_size/4, size=3)
        
        # Sample camera location on sphere around POI
        theta = self.rng.uniform(0, 2 * np.pi)  # Azimuth
        phi = self.rng.uniform(0, np.pi/2)  # Elevation (0 to 90 degrees)
        
        cam_location = poi + distance * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        
        # Check if camera is inside room
        if np.max(np.abs(cam_location[:2])) > room_size/2 * 0.9:
            return False
        if cam_location[2] < 0.1 or cam_location[2] > room_size * 0.9:
            return False
        
        # Set camera pose
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            poi - cam_location,
            inplane_rot=self.rng.uniform(0, 2*np.pi)
        )
        
        cam2world_matrix = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
        
        return True


class PhysicsSimulator:
    """Handles physics simulation for object placement."""
    
    @staticmethod
    def simulate(duration: float = 2.0, substeps: int = 10):
        """
        Run physics simulation.
        
        Args:
            duration: Simulation duration in seconds
            substeps: Number of substeps per frame
        """
        # Enable physics for all objects that should participate
        for obj in bproc.object.get_all_mesh_objects():
            if not obj.get_name().startswith("Wall") and \
               not obj.get_name().startswith("Floor") and \
               not obj.get_name().startswith("Ceiling"):
                obj.enable_rigidbody(
                    active=True,
                    collision_shape='CONVEX_HULL'
                )
            else:
                # Walls, floor, ceiling are passive
                obj.enable_rigidbody(
                    active=False,
                    collision_shape='BOX'
                )
        
        # Run simulation
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=duration,
            max_simulation_time=duration * 2,  # Must be greater than min
            check_object_interval=0.1,
            substeps_per_frame=substeps
        )


class SceneRandomizer:
    """Main randomization coordinator for scene generation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scene randomizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.rng = np.random.default_rng(config.get('seed', {}).get('numpy'))
        
        self.obj_randomizer = ObjectRandomizer(config, self.rng)
        self.camera_sampler = CameraSampler(config, self.rng)
    
    def randomize_material(self, obj: bproc.types.MeshObject, cc_materials: List[bproc.types.Material], target_mat: Optional[bproc.types.Material] = None):
        """
        Randomize object material.

        Args:
            obj: The MeshObject to randomize
            cc_materials: List of available materials
            target_mat: Optional specific material to apply
        """
        # Apply material
        if target_mat:
            print(f"    DEBUG: Applying target material: {target_mat.get_name()} to {obj.get_name()}")
            obj.replace_materials(target_mat)
        elif cc_materials:
            # Use random CC material with 80% probability
            if self.rng.random() < 0.8:
                material = self.rng.choice(cc_materials)
                print(f"    DEBUG: Applying random material: {material.get_name()} to {obj.get_name()}")
                obj.replace_materials(material)
            else:
                print(f"    DEBUG: No material applied to {obj.get_name()} (random skip)")
        else:
            print(f"    DEBUG: No materials available for {obj.get_name()}")

        # Randomize PBR properties (e.g., roughness/metallic)
        self.obj_randomizer._add_material_noise(obj, 0.1)

    def randomize_object(self,
                         obj: bproc.types.MeshObject,
                         cc_materials: List = None,
                         apply_scale: bool = True,
                         apply_material: bool = True,
                         apply_rotation: bool = True,
                         target_mat: Optional[bproc.types.Material] = None):
        """
        Apply all randomizations to an object.

        Args:
            obj: Object to randomize
            cc_materials: Available CC0 materials
            apply_scale: Whether to randomize scale
            apply_material: Whether to randomize material
            apply_rotation: Whether to randomize rotation
            target_mat: Optional specific material to apply
        """
        if apply_scale:
            self.obj_randomizer.randomize_scale(obj)

        if apply_material:
            self.randomize_material(obj, cc_materials, target_mat)

        if apply_rotation:
            self.obj_randomizer.randomize_rotation(obj)
    def sample_cameras(self,
                      target_objects: List[bproc.types.MeshObject],
                      room_size: float,
                      num_samples: int,
                      max_attempts_per_sample: int = 10) -> int:
        """
        Sample camera poses for scene.
        
        Args:
            target_objects: Objects to focus on
            room_size: Size of the room
            num_samples: Number of camera poses to sample
            max_attempts_per_sample: Maximum attempts per pose
            
        Returns:
            Number of successful camera poses added
        """
        successful = 0
        
        for i in range(num_samples):
            # Pick random target object
            target = self.rng.choice(target_objects)
            
            # Try to find valid camera pose
            for attempt in range(max_attempts_per_sample):
                if self.camera_sampler.sample_camera_pose(target, room_size):
                    successful += 1
                    break
        
        return successful
    
    def apply_physics(self, enable: bool = True):
        """
        Apply physics simulation if enabled.
        
        Args:
            enable: Whether physics simulation is enabled in config
        """
        if enable and self.config.get('scene', {}).get('simulate_physics', False):
            PhysicsSimulator.simulate()
            
            # Remove objects that fell out of scene
            for obj in bproc.object.get_all_mesh_objects():
                location = obj.get_location()
                # If object fell too far, delete it
                if location[2] < -5.0:
                    obj.delete()


if __name__ == '__main__':
    print("Randomization utilities module")
    print("This module is meant to be imported, not run directly")
