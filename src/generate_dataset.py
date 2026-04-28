import blenderproc as bproc
"""
Main Dataset Generation Script.

This script coordinates the entire pipeline:
1. Load configuration
2. Load 3D models
3. Generate scenes with BlenderProc
4. Extract bounding boxes
5. Convert to YOLO format
6. Organize into train/val/test splits
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add src directory to path so we can import our modules
# This is needed when running via 'blenderproc run'
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
from tqdm import tqdm
import imageio

# Import our modules
from config_parser import load_config, ConfigError
from object_loader import ObjectLoader
from scene_generator import SceneGenerator
from randomization import SceneRandomizer
from bbox_extractor import BBoxExtractor
from yolo_converter import YOLOConverter, YOLODatasetFormatter
from dataset_manager import DatasetSplitter, DatasetOrganizer, DatasetValidator


class DatasetGenerator:
    """Main dataset generation coordinator."""
    
    def __init__(self, config_path: str, num_scenes: Optional[int] = None):
        """
        Initialize dataset generator.
        
        Args:
            config_path: Path to configuration file
            num_scenes: Optional override for number of scenes
        """
        print("=" * 70)
        print("Blender YOLO Dataset Generator")
        print("=" * 70)
        
        # Load configuration
        print(f"\nLoading configuration from: {config_path}")
        try:
            self.config = load_config(config_path)
            
            # Override num_scenes if provided
            if num_scenes is not None:
                self.config['dataset']['num_scenes'] = num_scenes
                print(f"✓ Configuration loaded (num_scenes overridden to {num_scenes})")
            else:
                print("✓ Configuration loaded successfully")
        except ConfigError as e:
            print(f"✗ Configuration error: {e}")
            sys.exit(1)
        
        # Initialize components
        print("\nInitializing components...")
        
        # Load object models
        self.object_loader = ObjectLoader(self.config['models_path'], self.config.to_dict())
        print(f"✓ Loaded {len(self.object_loader.object_classes)} object classes")

        # Initialize scene generator
        self.scene_generator = SceneGenerator(self.config.to_dict())
        print("✓ Scene generator initialized")

        
        # Setup renderer (only once at initialization, not per scene)
        self.scene_generator.setup_renderer()
        
        # Initialize randomizer
        self.randomizer = SceneRandomizer(self.config.to_dict())
        print("✓ Randomizer initialized")
        
        # Initialize bbox extractor
        self.bbox_extractor = BBoxExtractor(self.config.to_dict())
        print("✓ Bounding box extractor initialized")
        
        # Initialize YOLO converter
        camera_config = self.config.get('camera', {})
        self.yolo_converter = YOLOConverter(
            self.config['output']['yolo_format'],
            camera_config.get('width', 640),
            camera_config.get('height', 480)
        )
        print(f"✓ YOLO converter initialized (format: {self.config['output']['yolo_format']})")
        
        # Initialize dataset formatter
        self.dataset_formatter = YOLODatasetFormatter(
            self.config['output']['save_path'],
            self.object_loader.get_class_names(),
            self.config['output']['yolo_format']
        )
        print("✓ Dataset formatter initialized")
        
        # Create data.yaml and classes.txt
        self.dataset_formatter.create_data_yaml()
        self.dataset_formatter.create_classes_file()
    
    def generate_scene(self, scene_idx: int) -> Tuple[List[Path], List[Path]]:
        """
        Generate a single scene with multiple camera views.
        
        Args:
            scene_idx: Scene index
            
        Returns:
            Tuple of (image_paths, label_paths)
        """
        # Clean previous scene (skip on first iteration since we just initialized)
        if scene_idx > 0:
            self.scene_generator.clean_scene()
            
        # Re-load CC0 materials because clean_up() removes them, 
        # and they are needed for room creation and objects.
        self.scene_generator.load_cc_materials()
        
        # Note: bproc.init() was already called in SceneGenerator.__init__
        # Calling it again causes an error. Use bproc.clean_up() between scenes instead.
        
        # Sample objects for this scene
        obj_config = self.config.get('scene', {}).get('objects', {})
        min_objs = obj_config.get('min_count', 1)
        max_objs = obj_config.get('max_count', 5)
        num_objects = np.random.randint(min_objs, max_objs + 1)
        allow_duplicates = obj_config.get('multiple_occurrences', True)
        
        object_classes = self.object_loader.get_random_classes(
            num_objects,
            allow_duplicates=allow_duplicates
        )
        
        # Load and place target objects
        target_objects = []
        obj_sizes = []
        for obj_class in object_classes:
            # Load object
            objs = bproc.loader.load_obj(obj_class.model_path)
            
            # If multi-part object, join into one
            if len(objs) > 1:
                objs[0].join_with_other_objects(objs[1:])
            
            obj = objs[0]
            
            # Apply category_id *after* joining in case metadata is lost
            obj.blender_obj["category_id"] = obj_class.class_id
            obj.blender_obj["class_name"] = obj_class.name
            
            # Set pass_index for segmentation (BlenderProc uses this)
            obj.blender_obj.pass_index = obj_class.class_id + 1
            
            # Get object size
            bbox = obj.get_bound_box()
            obj_size = np.max(np.ptp(bbox, axis=0))
            obj_sizes.append(obj_size)
            
            # Apply specific texture if requested, otherwise randomize material
            texture_name = None
            if hasattr(obj_class, 'texture') and obj_class.texture:
                texture_name = obj_class.texture
            elif hasattr(obj_class, 'textures') and obj_class.textures:
                texture_name = np.random.choice(obj_class.textures)

            cc_mat = self.scene_generator.cc_materials_dict.get(texture_name) if texture_name else None
            if texture_name:
                print(f"    DEBUG: Object {obj.get_name()}, requested texture={texture_name}, mat_found={cc_mat is not None}")

            # Apply base color to object ONLY if no CC materials are available AND no texture specified
            if not self.scene_generator.cc_materials and not cc_mat:
                self.scene_generator.apply_base_color(obj)

            # Randomize object
            self.randomizer.randomize_object(
                obj,
                cc_materials=self.scene_generator.cc_materials,
                target_mat=cc_mat
            )

            target_objects.append(obj)

        
        # Compute room size based on actual objects
        room_size = self.scene_generator.compute_room_size(obj_sizes)
        print(f"    DEBUG: Room size: {room_size:.2f}")
        
        # Create room
        room_objects = self.scene_generator.create_room(room_size)
        
        # Place objects in room
        for i, obj in enumerate(target_objects):
            # Random position in room
            x = np.random.uniform(-room_size/4, room_size/4)
            y = np.random.uniform(-room_size/4, room_size/4)
            # Ensure it's above the floor
            z = np.random.uniform(obj_sizes[i]/2 + 0.1, room_size/2)
            obj.set_location([x, y, z])
            print(f"    DEBUG: Object {i} location: {obj.get_location()}")
        
        # Add distractors
        distractor_config = self.config.get('scene', {}).get('distractors', {})
        num_distractors = np.random.randint(
            distractor_config.get('min_count', 20),
            distractor_config.get('max_count', 50) + 1
        )
        self.scene_generator.add_distractors(room_size, num_distractors)
        
        # Add lights
        light_config = self.config.get('scene', {}).get('lights', {})
        num_lights = np.random.randint(
            light_config.get('min_count', 3),
            light_config.get('max_count', 6) + 1
        )
        self.scene_generator.add_lights(room_size, num_lights)
        
        # Setup camera
        self.scene_generator.setup_camera()
        
        # Apply physics if enabled (move physics after cam setup for clarity, but doesn't change much)
        self.randomizer.apply_physics()
        
        # Ensure objects haven't dropped through the floor and filter deleted ones
        valid_target_objects = []
        for obj in target_objects:
            try:
                loc = obj.get_location()
                if loc[2] < 0:
                    print(f"Warning: Object {obj.get_name()} fell below floor, resetting Z")
                    loc[2] = 0.05
                    obj.set_location(loc)
                valid_target_objects.append(obj)
            except ReferenceError:
                print("Warning: Object fell out of bounds and was deleted during physics simulation")
        
        target_objects = valid_target_objects
        
        if not target_objects:
            print(f"  Warning: All target objects fell out of bounds for scene {scene_idx}")
            return [], []
        
        # Sample camera poses
        dataset_config = self.config.get('dataset', {})
        num_images = dataset_config.get('images_per_scene', 10)
        
        num_poses = self.randomizer.sample_cameras(
            target_objects,
            room_size,
            num_images
        )
        
        if num_poses == 0:
            print(f"  Warning: No valid camera poses found for scene {scene_idx}")
            return [], []
        
        # Render
        print(f"  Rendering {num_poses} images...")
        
        data = bproc.renderer.render()
        
        # Save renders and extract annotations
        temp_dir = Path(self.config['output']['save_path']) / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        label_paths = []
        
        class_mapping = self.object_loader.create_class_mapping_dict()
        
        for img_idx in range(len(data['colors'])):
            # Save image
            image_name = f"scene_{scene_idx:05d}_img_{img_idx:03d}.png"
            image_path = temp_dir / image_name
            
            # Extract bounding boxes from rendered data
            bboxes = self.bbox_extractor.extract_from_dict(
                data,
                class_mapping,
                img_idx
            )
            print(f"    Extracted {len(bboxes)} bboxes for image {img_idx}")
            
            # Save labels
            label_name = f"scene_{scene_idx:05d}_img_{img_idx:03d}.txt"
            label_path = temp_dir / label_name
            
            # If no bboxes extracted, delete files
            if not bboxes:
                continue
            
            # Save image
            image_data = data['colors'][img_idx]
            if image_data.dtype != np.uint8:
                image_data = np.clip(image_data, 0, 255).astype(np.uint8)
            imageio.imwrite(str(image_path), image_data)
            
            self.yolo_converter.save_annotations(bboxes, str(label_path))
            
            image_paths.append(image_path)
            label_paths.append(label_path)
        
        return image_paths, label_paths
    
    def generate_dataset(self):
        """Generate complete dataset."""
        print("\n" + "=" * 70)
        print("Starting Dataset Generation")
        print("=" * 70)
        
        dataset_config = self.config.get('dataset', {})
        num_scenes = dataset_config.get('num_scenes', 100)
        
        all_image_paths = []
        all_label_paths = []
        
        # Generate scenes
        print(f"\nGenerating {num_scenes} scenes...")
        for scene_idx in tqdm(range(num_scenes), desc="Scenes"):
            try:
                image_paths, label_paths = self.generate_scene(scene_idx)
                all_image_paths.extend(image_paths)
                all_label_paths.extend(label_paths)
            except Exception as e:
                print(f"\nError generating scene {scene_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n✓ Generated {len(all_image_paths)} images total")
        
        # Split dataset
        print("\nSplitting dataset...")
        splitter = DatasetSplitter(
            train_ratio=dataset_config.get('train_split', 0.7),
            val_ratio=dataset_config.get('val_split', 0.2),
            test_ratio=dataset_config.get('test_split', 0.1),
            random_seed=self.config.get('seed', {}).get('numpy')
        )
        
        splits = splitter.split_indices(len(all_image_paths))
        print(f"  Train: {len(splits['train'])} images")
        print(f"  Val:   {len(splits['val'])} images")
        print(f"  Test:  {len(splits['test'])} images")
        
        # Organize dataset
        organizer = DatasetOrganizer(self.config['output']['save_path'])
        organizer.organize_dataset(all_image_paths, all_label_paths, splits)
        organizer.print_stats()
        organizer.save_stats()
        
        # Validate dataset
        print("\nValidating dataset...")
        validator = DatasetValidator(self.config['output']['save_path'])
        is_valid, issues = validator.validate()
        
        if is_valid:
            print("✓ Dataset validation passed")
        else:
            print(f"⚠ Dataset validation found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        
        # Print class distribution
        distribution = validator.get_class_distribution()
        print("\nClass distribution:")
        for class_id, count in sorted(distribution.items()):
            class_name = self.object_loader.create_class_mapping_dict().get(class_id, f"class_{class_id}")
            print(f"  {class_name} (ID {class_id}): {count} objects")
        
        print("\n" + "=" * 70)
        print("Dataset Generation Complete!")
        print("=" * 70)
        print(f"\nDataset location: {self.config['output']['save_path']}")
        print(f"Format: {self.config['output']['yolo_format']}")
        print("\nTo train YOLO:")
        print(f"  yolo train data={self.config['output']['save_path']}/data.yaml model=yolo11n.pt")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate YOLO training datasets using Blender and BlenderProc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dataset with standard config
  python generate_dataset.py --config configs/example_config.json
  
  # Generate OBB dataset
  python generate_dataset.py --config configs/example_obb_config.json
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to configuration file (JSON or YAML)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing dataset without generating'
    )
    
    parser.add_argument(
        '--num-scenes',
        type=int,
        help='Override number of scenes to generate'
    )
    
    args = parser.parse_args()
    
    if args.validate_only:
        # Validate existing dataset
        config = load_config(args.config)
        validator = DatasetValidator(config['output']['save_path'])
        is_valid, issues = validator.validate()
        
        if is_valid:
            print("✓ Dataset is valid")
            sys.exit(0)
        else:
            print(f"✗ Dataset validation failed with {len(issues)} issues")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
    else:
        # Generate dataset
        try:
            generator = DatasetGenerator(args.config, num_scenes=args.num_scenes)
            generator.generate_dataset()
        except KeyboardInterrupt:
            print("\n\nGeneration interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\nFatal error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()
