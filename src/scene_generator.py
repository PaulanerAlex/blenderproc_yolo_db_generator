"""
BlenderProc Scene Generator for YOLO Dataset Generation.

This module handles scene generation using BlenderProc, including:
- Room/environment creation
- Object placement
- Lighting setup
- Camera positioning
- Rendering
"""

import blenderproc as bproc
import numpy as np
import bpy
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import mathutils


class SceneGenerator:
    """Generates synthetic scenes using BlenderProc."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scene generator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scene_config = config.get('scene', {})
        self.camera_config = config.get('camera', {})
        self.rendering_config = config.get('rendering', {})
        
        # Initialize BlenderProc
        bproc.init()
        
        # Set random seeds if provided
        if 'seed' in config:
            if config['seed'].get('numpy'):
                np.random.seed(config['seed']['numpy'])
            if config['seed'].get('blenderproc'):
                # Try to set blenderproc seed (function may not exist in all versions)
                try:
                    bproc.utility.set_random_seed(int(config['seed']['blenderproc']))
                except AttributeError:
                    # Function not available in this BlenderProc version, skip
                    print("Warning: bproc.utility.set_random_seed() not available in this BlenderProc version, skipping")
        
        # Load CC0 textures if available
        self.cc_materials = []
        if config.get('cc_textures_path'):
            self._load_cc_materials(config['cc_textures_path'])
    
    def load_cc_materials(self):
        """Load CC0 texture materials from configured path."""
        textures_path = self.config.get('cc_textures_path')
        if textures_path:
            self._load_cc_materials(textures_path)

    def _load_cc_materials(self, textures_path: str):
        """Load CC0 texture materials from directory structure."""
        import os
        from blenderproc.python.material import MaterialLoaderUtility
        
        if not os.path.exists(textures_path):
            print(f"Warning: CC0 textures path does not exist: {textures_path}")
            return

        print(f"Loading CC0 textures from {textures_path}...")
        
        # Limit number of textures to load to avoid memory issues
        max_textures = self.scene_config.get('max_textures', 50)
        loaded_count = 0
        
        # Initialize materials dict
        self.cc_materials_dict = {}
        
        # Get global blacklist from scene config
        self.texture_blacklist = self.scene_config.get('texture_blacklist', [])
        
        dir_list = os.listdir(textures_path)
        
        for asset in dir_list:
            if asset in self.texture_blacklist:
                continue

            # Check for color image
            asset_path = os.path.join(textures_path, asset)
            if not os.path.isdir(asset_path):
                continue
                
            color_images = [f for f in os.listdir(asset_path) if f.endswith('Color.jpg')]
            if not color_images:
                continue
            
            if loaded_count >= max_textures:
                break
                
            base_image_name = color_images[0]
            base_image_path = os.path.join(asset_path, base_image_name)
            
            # Create a new material
            # Prefix with CC_ to easily identify them
            mat_name = f"CC_{asset}"
            
            # Check if material already exists to avoid duplicates
            if mat_name in bpy.data.materials:
                mat = bproc.types.Material(bpy.data.materials[mat_name])
                self.cc_materials.append(mat)
                self.cc_materials_dict[asset] = mat
                continue

            try:
                # Use bproc to create a basic material wrapper
                mat = bproc.material.create(mat_name)
                nodes = mat.blender_obj.node_tree.nodes
                links = mat.blender_obj.node_tree.links
                
                # Find the principled BSDF and output nodes
                principled_bsdf = None
                output_node = None
                for node in nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        principled_bsdf = node
                    elif node.type == 'OUTPUT_MATERIAL':
                        output_node = node
                
                if not principled_bsdf:
                    continue

                # Collect texture nodes to connect UV maps later
                tex_nodes = []
                
                # Add base color
                base_color_node = MaterialLoaderUtility.add_base_color(
                    nodes, links, base_image_path, principled_bsdf
                )
                if base_color_node:
                    tex_nodes.append(base_color_node)
                
                # Look for other maps with similar naming
                prefix = base_image_name.replace('Color.jpg', '')
                
                # Normal maps: prefer GL for Blender
                normal_gl_path = os.path.join(asset_path, prefix + 'NormalGL.jpg')
                if os.path.exists(normal_gl_path):
                    n_node = MaterialLoaderUtility.add_normal(
                        nodes, links, normal_gl_path, principled_bsdf, invert_y_channel=True
                    )
                    if n_node:
                        tex_nodes.append(n_node)
                else:
                    normal_dx_path = os.path.join(asset_path, prefix + 'NormalDX.jpg')
                    if os.path.exists(normal_dx_path):
                        n_node = MaterialLoaderUtility.add_normal(
                            nodes, links, normal_dx_path, principled_bsdf, invert_y_channel=False
                        )
                        if n_node:
                            tex_nodes.append(n_node)

                # Roughness
                roughness_path = os.path.join(asset_path, prefix + 'Roughness.jpg')
                if os.path.exists(roughness_path):
                    r_node = MaterialLoaderUtility.add_roughness(
                        nodes, links, roughness_path, principled_bsdf
                    )
                    if r_node:
                        tex_nodes.append(r_node)

                # Metallic
                metallic_path = os.path.join(asset_path, prefix + 'Metalness.jpg')
                if not os.path.exists(metallic_path):
                    metallic_path = os.path.join(asset_path, prefix + 'Metallic.jpg')
                
                if os.path.exists(metallic_path):
                    m_node = MaterialLoaderUtility.add_metal(
                        nodes, links, metallic_path, principled_bsdf
                    )
                    if m_node:
                        tex_nodes.append(m_node)

                # Displacement
                displacement_path = os.path.join(asset_path, prefix + 'Displacement.jpg')
                if os.path.exists(displacement_path) and output_node:
                    d_node = MaterialLoaderUtility.add_displacement(
                        nodes, links, displacement_path, output_node
                    )
                    if d_node:
                        tex_nodes.append(d_node)

                # Connect UVs
                MaterialLoaderUtility.connect_uv_maps(nodes, links, tex_nodes)
                
                self.cc_materials.append(mat)
                self.cc_materials_dict[asset] = mat
                loaded_count += 1
                
            except Exception as e:
                print(f"Error loading material {asset}: {e}")
        
        print(f"Successfully loaded {len(self.cc_materials)} CC0 materials")
    
    def create_room(self, room_size: float) -> List[bproc.types.MeshObject]:
        """
        Create a simple cubic room.
        
        Args:
            room_size: Size of the room (cubic)
            
        Returns:
            List of room mesh objects (floor, walls, ceiling)
        """
        room_objects = []
        
        # Create floor
        floor = bproc.object.create_primitive('PLANE', scale=[room_size/2, room_size/2, 1])
        floor.set_name("Floor")
        floor.set_location([0, 0, 0])
        room_objects.append(floor)
        
        # Create walls (4 sides)
        wall_height = room_size
        wall_thickness = 0.1
        
        # Front wall
        wall_front = bproc.object.create_primitive(
            'CUBE', 
            scale=[room_size/2, wall_thickness/2, wall_height/2]
        )
        wall_front.set_location([0, room_size/2, wall_height/2])
        wall_front.set_name("Wall_Front")
        room_objects.append(wall_front)
        
        # Back wall
        wall_back = bproc.object.create_primitive(
            'CUBE',
            scale=[room_size/2, wall_thickness/2, wall_height/2]
        )
        wall_back.set_location([0, -room_size/2, wall_height/2])
        wall_back.set_name("Wall_Back")
        room_objects.append(wall_back)
        
        # Left wall
        wall_left = bproc.object.create_primitive(
            'CUBE',
            scale=[wall_thickness/2, room_size/2, wall_height/2]
        )
        wall_left.set_location([-room_size/2, 0, wall_height/2])
        wall_left.set_name("Wall_Left")
        room_objects.append(wall_left)
        
        # Right wall
        wall_right = bproc.object.create_primitive(
            'CUBE',
            scale=[wall_thickness/2, room_size/2, wall_height/2]
        )
        wall_right.set_location([room_size/2, 0, wall_height/2])
        wall_right.set_name("Wall_Right")
        room_objects.append(wall_right)
        
        # Ceiling
        ceiling = bproc.object.create_primitive('PLANE', scale=[room_size/2, room_size/2, 1])
        ceiling.set_name("Ceiling")
        ceiling.set_location([0, 0, wall_height])
        ceiling.set_rotation_euler([np.pi, 0, 0])  # Flip upside down
        room_objects.append(ceiling)
        
        # Assign materials to room surfaces
        if self.cc_materials:
            # Use global blacklist
            available_materials = [m for m in self.cc_materials if m.get_name().replace("CC_", "") not in self.texture_blacklist]
            
            for obj in room_objects:
                if available_materials:
                    material = np.random.choice(available_materials)
                    obj.replace_materials(material)
                else:
                    self.apply_neutral_color(obj)
        else:
            # If no CC0 materials, apply neutral colors to room surfaces
            for obj in room_objects:
                self.apply_neutral_color(obj)
        
        return room_objects
    
    def add_distractors(self, room_size: float, num_distractors: int) -> List[bproc.types.MeshObject]:
        """
        Add distractor objects to the scene.
        
        Args:
            room_size: Size of the room
            num_distractors: Number of distractors to add
            
        Returns:
            List of distractor objects
        """
        distractors = []
        distractor_config = self.scene_config.get('distractors', {})
        
        # Use global blacklist
        available_materials = [m for m in self.cc_materials if m.get_name().replace("CC_", "") not in self.texture_blacklist]
        
        shapes = ['CUBE', 'SPHERE', 'CYLINDER', 'MONKEY']
        
        for i in range(num_distractors):
            # Random shape
            shape = np.random.choice(shapes)
            
            # Random size (relative to room)
            min_size = distractor_config.get('min_size_rel_scene', 0.05) * room_size
            max_size = distractor_config.get('max_size_rel_scene', 0.1) * room_size
            size = np.random.uniform(min_size, max_size)
            
            # Create distractor
            distractor = bproc.object.create_primitive(shape, scale=[size, size, size])
            distractor.set_name(f"Distractor_{i}")
            
            # Random position within room
            x = np.random.uniform(-room_size/3, room_size/3)
            y = np.random.uniform(-room_size/3, room_size/3)
            z = np.random.uniform(size, room_size - size)
            distractor.set_location([x, y, z])
            
            # Random rotation
            distractor.set_rotation_euler([
                np.random.uniform(0, 2*np.pi),
                np.random.uniform(0, 2*np.pi),
                np.random.uniform(0, 2*np.pi)
            ])
            
            # Assign random material
            if available_materials:
                material = np.random.choice(available_materials)
                distractor.replace_materials(material)
                
                # Add PBR noise
                pbr_noise = distractor_config.get('pbr_noise', 0.5)
                if pbr_noise > 0:
                    self._add_material_noise(distractor, pbr_noise)
            else:
                self.apply_base_color(distractor)
            
            # Make emissive with some probability
            emissive_prob = distractor_config.get('emissive_prob', 0.0)
            if np.random.random() < emissive_prob:
                strength_min = distractor_config.get('emissive_strength_min', 2.0)
                strength_max = distractor_config.get('emissive_strength_max', 5.0)
                strength = np.random.uniform(strength_min, strength_max)
                
                for mat in distractor.get_materials():
                    # emission_color must be RGBA (4 values), not RGB (3 values)
                    emission_color = np.random.uniform([0.5, 0.5, 0.5, 1.0], [1, 1, 1, 1])
                    mat.make_emissive(
                        emission_strength=strength,
                        emission_color=emission_color
                    )
            
            distractors.append(distractor)
        
        return distractors
    
    def _add_material_noise(self, obj: bproc.types.MeshObject, noise_amount: float):
        """Add random noise to material PBR properties."""
        import numbers
        for mat in obj.get_materials():
            # Randomize roughness
            current_roughness = mat.get_principled_shader_value("Roughness")
            if current_roughness is not None and isinstance(current_roughness, numbers.Number):
                new_roughness = np.clip(
                    current_roughness + np.random.uniform(-noise_amount, noise_amount),
                    0, 1
                )
                mat.set_principled_shader_value("Roughness", new_roughness)
            
            # Randomize metallic
            current_metallic = mat.get_principled_shader_value("Metallic")
            if current_metallic is not None and isinstance(current_metallic, numbers.Number):
                new_metallic = np.clip(
                    current_metallic + np.random.uniform(-noise_amount/2, noise_amount/2),
                    0, 1
                )
                mat.set_principled_shader_value("Metallic", new_metallic)
    
    def add_lights(self, room_size: float, num_lights: int):
        """
        Add lights to the scene.
        
        Args:
            room_size: Size of the room
            num_lights: Number of lights to add
        """
        light_config = self.scene_config.get('lights', {})
        min_intensity = light_config.get('min_intensity', 50)
        max_intensity = light_config.get('max_intensity', 200)
        
        for i in range(num_lights):
            # Random light type
            light_type = np.random.choice(['POINT', 'SPOT'])
            
            # Random position (upper part of room)
            x = np.random.uniform(-room_size/3, room_size/3)
            y = np.random.uniform(-room_size/3, room_size/3)
            z = np.random.uniform(room_size * 0.6, room_size * 0.9)
            
            # Create light
            light = bproc.types.Light()
            light.set_type(light_type)
            light.set_location([x, y, z])
            light.set_energy(np.random.uniform(min_intensity, max_intensity))
            
            # Random color (slight variation from white)
            color = np.random.uniform([0.9, 0.9, 0.9], [1.0, 1.0, 1.0])
            light.set_color(color)
    
    def setup_camera(self):
        """Setup camera with configured intrinsics."""
        # Get camera parameters
        px = self.camera_config.get('px', 600)
        py = self.camera_config.get('py', 600)
        u0 = self.camera_config.get('u0', 320)
        v0 = self.camera_config.get('v0', 240)
        width = self.camera_config.get('width', 640)
        height = self.camera_config.get('height', 480)
        
        # Apply randomization if configured
        randomize_percent = self.camera_config.get('randomize_params_percent', 0) / 100.0
        if randomize_percent > 0:
            px *= np.random.uniform(1 - randomize_percent, 1 + randomize_percent)
            py *= np.random.uniform(1 - randomize_percent, 1 + randomize_percent)
            u0 *= np.random.uniform(1 - randomize_percent, 1 + randomize_percent)
            v0 *= np.random.uniform(1 - randomize_percent, 1 + randomize_percent)
        
        # Set camera intrinsics
        bproc.camera.set_resolution(width, height)
        
        # Convert to Blender camera parameters
        K = np.array([
            [px, 0, u0],
            [0, py, v0],
            [0, 0, 1]
        ])
        bproc.camera.set_intrinsics_from_K_matrix(K, width, height)
        
        # Adjust clipping planes for small/close objects
        import bpy
        bpy.context.scene.camera.data.clip_start = 0.001
        bpy.context.scene.camera.data.clip_end = 100.0
    
    def setup_renderer(self):
        """Configure rendering settings."""
        max_samples = self.rendering_config.get('max_samples', 32)
        denoiser = self.rendering_config.get('denoiser', 'OPTIX')
        
        # Set number of samples
        bproc.renderer.set_max_amount_of_samples(max_samples)
        
        # Set denoiser
        # FIXME: This should detect available denoisers at runtime and select the best one
        # Currently we hardcode a fallback from OPTIX to OPENIMAGEDENOISE since OPTIX requires NVIDIA GPU
        if denoiser:
            try:
                bproc.renderer.set_denoiser(denoiser)
            except RuntimeError:
                # Fallback to OPENIMAGEDENOISE if OPTIX is not available (e.g., CPU rendering, non-NVIDIA GPU)
                try:
                    bproc.renderer.set_denoiser('OPENIMAGEDENOISE')
                except RuntimeError:
                    # If neither denoiser is available, continue without denoising
                    print(f"⚠ Warning: Denoiser '{denoiser}' and fallback 'OPENIMAGEDENOISE' unavailable. Rendering without denoiser.")
        
        # Enable normal and depth rendering
        bproc.renderer.enable_normals_output()
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        
        # Enable segmentation (required for bbox extraction)
        bproc.renderer.enable_segmentation_output(map_by=["class", "instance"])
    
    def compute_room_size(self, object_sizes: List[float]) -> float:
        """
        Compute room size based on largest object.
        
        Args:
            object_sizes: List of object sizes (max dimension)
            
        Returns:
            Room size
        """
        max_obj_size = max(object_sizes) if object_sizes else 1.0
        
        multiplier_min = self.scene_config.get('room_size_multiplier_min', 5.0)
        multiplier_max = self.scene_config.get('room_size_multiplier_max', 10.0)
        
        multiplier = np.random.uniform(multiplier_min, multiplier_max)
        room_size = max_obj_size * multiplier
        
        return room_size
    
    def apply_base_color(self, obj: bproc.types.MeshObject):
        """Ensure objects are visible by creating/applying colors to materials."""
        try:
            materials = obj.get_materials()
            obj_name = obj.get_name()
            
            # If object has no materials, create one
            if not materials:
                # Create material using BlenderProc natively
                mat = bproc.material.create("Material_" + obj_name)
                obj.add_material(mat)
                # Reload materials from BlenderProc
                materials = obj.get_materials()
            
            # Apply a visible color to each material
            if materials:
                # Use bright, saturated colors for good visibility against light backgrounds
                colors = [
                    (1.0, 0.0, 0.0, 1.0),   # Bright red
                    (0.0, 1.0, 0.0, 1.0),   # Bright green
                    (0.0, 0.0, 1.0, 1.0),   # Bright blue
                    (1.0, 1.0, 0.0, 1.0),   # Bright yellow
                    (1.0, 0.0, 1.0, 1.0),   # Bright magenta
                    (0.0, 1.0, 1.0, 1.0),   # Bright cyan
                    (1.0, 0.5, 0.0, 1.0),   # Bright orange
                    (0.5, 0.0, 1.0, 1.0),   # Bright purple
                ]
                color_rgba = colors[hash(obj_name) % len(colors)]
                
                for mat in materials:
                    try:
                        # BlenderProc materials use set_principled_shader_value
                        mat.set_principled_shader_value("Base Color", list(color_rgba))
                        mat.set_principled_shader_value("Roughness", 0.8)
                        mat.set_principled_shader_value("Metallic", 0.0)
                    except Exception:
                        pass  # Silently fail for individual materials
        except Exception:
            pass  # Silently fail if we can't apply colors
    
    def apply_neutral_color(self, obj: bproc.types.MeshObject):
        """Apply neutral light colors to room surfaces for better contrast with objects."""
        try:
            materials = obj.get_materials()
            obj_name = obj.get_name()
            
            # If object has no materials, create one
            if not materials:
                mat = bproc.material.create("Material_" + obj_name)
                obj.add_material(mat)
                materials = obj.get_materials()
            
            # Apply neutral colors based on surface type
            if materials:
                # Assign lighter, neutral colors to different room surfaces for contrast
                neutral_colors = {
                    "Floor": (0.75, 0.75, 0.75, 1.0),    # Light gray
                    "Ceiling": (0.85, 0.85, 0.85, 1.0),  # Lighter gray
                    "Wall": (0.80, 0.80, 0.80, 1.0),     # Light gray with slight warmth
                }
                
                # Find matching color by object name
                color_rgba = (0.8, 0.8, 0.8, 1.0)  # Default light gray
                for surface_name, color in neutral_colors.items():
                    if surface_name in obj_name:
                        color_rgba = color
                        break
                
                for mat in materials:
                    try:
                        mat.set_principled_shader_value("Base Color", list(color_rgba))
                        mat.set_principled_shader_value("Roughness", 0.8)
                        mat.set_principled_shader_value("Metallic", 0.0)
                    except Exception:
                        pass
        except Exception:
            pass
    
    def clean_scene(self):
        """Remove all objects from scene."""
        bproc.clean_up()
        self.cc_materials = []



if __name__ == '__main__':
    # Test scene generator
    print("Testing SceneGenerator...")
    
    config = {
        'seed': {'numpy': 42, 'blenderproc': '123'},
        'camera': {
            'px': 600, 'py': 600, 'u0': 320, 'v0': 240,
            'width': 640, 'height': 480
        },
        'rendering': {
            'max_samples': 32,
            'denoiser': None
        },
        'scene': {
            'room_size_multiplier_min': 5.0,
            'room_size_multiplier_max': 10.0,
            'distractors': {
                'min_count': 5,
                'max_count': 10
            },
            'lights': {
                'min_count': 2,
                'max_count': 4
            }
        }
    }
    
    try:
        generator = SceneGenerator(config)
        print("✓ Scene generator initialized")
        
        room_size = generator.compute_room_size([1.0])
        print(f"✓ Room size computed: {room_size:.2f}")
        
        room_objects = generator.create_room(room_size)
        print(f"✓ Room created with {len(room_objects)} objects")
        
        distractors = generator.add_distractors(room_size, 5)
        print(f"✓ Added {len(distractors)} distractors")
        
        generator.add_lights(room_size, 3)
        print("✓ Lights added")
        
        generator.setup_camera()
        print("✓ Camera configured")
        
        generator.setup_renderer()
        print("✓ Renderer configured")
        
        print("\nScene generator test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
