"""
3D Object Loader and Manager for Blender YOLO Dataset Generator.

This module handles loading .obj models, managing object classes,
and providing objects for scene generation.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class ObjectClass:
    """Represents a single object class with its model and metadata."""
    
    def __init__(self, name: str, model_path: str, class_id: int, 
                 class_name: Optional[str] = None,
                 texture: Optional[str] = None, 
                 textures: Optional[List[str]] = None,
                 randomize_materials: bool = True,
                 initial_rotation: Optional[List[float]] = None,
                 min_rotation: Optional[List[float]] = None,
                 max_rotation: Optional[List[float]] = None,
                 initial_height: Optional[float] = None):
        """
        Initialize object class.
        
        Args:
            name: Object model name (directory name)
            model_path: Path to .obj file
            class_id: Numeric class ID (0-indexed for YOLO training)
            class_name: Optional training class name for YOLO (defaults to name)
            texture: Optional single texture
            textures: Optional list of textures
            randomize_materials: Whether to apply random materials
            initial_rotation: Optional initial rotation in degrees [x, y, z]
            initial_height: Optional initial Z height
        """
        self.name = name
        self.model_path = model_path
        self.class_id = class_id
        self.class_name = class_name if class_name is not None else name
        self.texture = texture
        self.textures = textures
        self.randomize_materials = randomize_materials
        self.initial_rotation = initial_rotation
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.initial_height = initial_height
        self.material_paths = self._find_materials()
    
    def _find_materials(self) -> Dict[str, str]:
        """Find associated material files (.mtl, textures)."""
        materials = {}
        model_path = Path(self.model_path)
        model_dir = model_path.parent
        
        # Find matching .mtl file with same stem first
        matching_mtl = model_path.with_suffix('.mtl')
        if matching_mtl.exists():
            materials['mtl'] = str(matching_mtl)
        else:
            mtl_files = list(model_dir.glob('*.mtl'))
            if mtl_files:
                materials['mtl'] = str(mtl_files[0])
        
        # Find texture files
        texture_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tga']
        for ext in texture_extensions:
            matching_tex = model_path.with_suffix(ext)
            if matching_tex.exists():
                materials['textures'] = [str(matching_tex)]
                break
            texture_files = list(model_dir.glob(f'*{ext}'))
            if texture_files:
                materials['textures'] = [str(f) for f in texture_files]
                break
        
        return materials
    
    def __repr__(self) -> str:
        return f"ObjectClass(name='{self.name}', class_name='{self.class_name}', id={self.class_id}, model='{self.model_path}')"


class ObjectLoader:
    """Manages loading and accessing 3D object models."""
    
    def __init__(self, models_path: str, config: Dict[str, Any] = None):
        """
        Initialize object loader.
        
        Args:
            models_path: Path to directory containing object class subdirectories
            config: Optional configuration dictionary
        """
        self.models_path = Path(models_path)
        self.object_classes: List[ObjectClass] = []
        self.class_name_to_id: Dict[str, int] = {}
        self.object_classes_by_name: Dict[str, ObjectClass] = {}
        self.config = config or {}
        
        if not self.models_path.exists():
            raise ValueError(f"Models path does not exist: {models_path}")
        
        self._load_objects()
    
    def _load_objects(self):
        """Scan models directory and load all object classes specified in config."""
        if not self.models_path.is_dir():
            raise ValueError(f"Models path is not a directory: {self.models_path}")
        
        # Load object configurations from config
        obj_configs = self.config.get('scene', {}).get('objects', {})
        
        # Define reserved keys that are NOT class names
        reserved_keys = [
            'min_count', 'max_count', 'multiple_occurrences', 
            'scale_noise', 'displacement_max', 'pbr_noise',
            'cam_min_dist_rel', 'cam_max_dist_rel',
            'cam_min_elev_deg', 'cam_max_elev_deg',
            'cam_min_roll_deg', 'cam_max_roll_deg',
            'cam_min_pitch_deg', 'cam_max_pitch_deg',
            'cam_min_yaw_deg', 'cam_max_yaw_deg'
        ]
        
        # Get all subdirectories (each represents one class)
        all_dirs = [
            d for d in self.models_path.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ]
        
        # Determine which classes to load
        # We only load directories that have a corresponding entry in obj_configs
        # and are not reserved keys.
        class_dirs = []
        for d in all_dirs:
            if d.name in obj_configs and d.name not in reserved_keys:
                class_dirs.append(d)
        
        # If no specific class folders mentioned in config, fall back to loading all
        if not class_dirs:
            # But only if no class names were provided (to avoid loading everything when one typo exists)
            has_class_names = any(k not in reserved_keys for k in obj_configs.keys())
            if not has_class_names:
                class_dirs = all_dirs
            
        class_dirs.sort()  # Ensure deterministic order
        
        if not class_dirs:
            raise ValueError(f"No valid object class directories found in {self.models_path} matching config")
        
        # Collect candidate models and per-object configs
        raw_items = []
        for class_dir in class_dirs:
            obj_files = sorted(class_dir.glob('*.obj'))
            
            if not obj_files:
                print(f"Warning: No .obj file found in {class_dir.name}, skipping...")
                continue
            
            # Get class-specific config
            obj_cfg = obj_configs.get(class_dir.name, {})
            c_id = obj_cfg.get('class_id')
            c_name = obj_cfg.get('class_name')
            
            for obj_file in obj_files:
                if len(obj_files) == 1:
                    model_name = class_dir.name
                else:
                    model_name = f"{class_dir.name}/{obj_file.stem}"
                
                raw_items.append({
                    'dir': class_dir,
                    'model_name': model_name,
                    'obj_file': obj_file,
                    'obj_cfg': obj_cfg,
                    'raw_id': c_id,
                    'raw_name': c_name
                })
        
        # Step 1: Validate and link explicit class_id and class_name
        explicit_ids = set()
        for item in raw_items:
            if item['raw_id'] is not None:
                if not isinstance(item['raw_id'], int) or item['raw_id'] < 0:
                    raise ValueError(f"Invalid class_id for {item['dir'].name}: {item['raw_id']}. Must be a non-negative integer.")
                explicit_ids.add(item['raw_id'])

        # 1a. Propagate class_id for matching class_name
        name_to_id = {}
        for item in raw_items:
            name = item['raw_name']
            cid = item['raw_id']
            if name is not None and cid is not None:
                if name in name_to_id and name_to_id[name] != cid:
                    raise ValueError(f"Conflicting class_id for class_name '{name}': {name_to_id[name]} vs {cid}")
                name_to_id[name] = cid

        for item in raw_items:
            name = item['raw_name']
            if name is not None and item['raw_id'] is None and name in name_to_id:
                item['raw_id'] = name_to_id[name]

        # 1b. Propagate class_name for matching class_id
        id_to_name = {}
        for item in raw_items:
            cid = item['raw_id']
            name = item['raw_name']
            if cid is not None and name is not None:
                if cid in id_to_name and id_to_name[cid] != name:
                    raise ValueError(f"Conflicting class_name for class_id {cid}: '{id_to_name[cid]}' vs '{name}'")
                id_to_name[cid] = name

        for item in raw_items:
            cid = item['raw_id']
            if cid is not None and item['raw_name'] is None:
                if cid in id_to_name:
                    item['raw_name'] = id_to_name[cid]
                else:
                    id_to_name[cid] = item['dir'].name
                    item['raw_name'] = item['dir'].name

        # Step 2: Assign sequential IDs to items without explicit IDs
        next_id = 0
        def get_next_available_id():
            nonlocal next_id
            while next_id in explicit_ids:
                next_id += 1
            assigned = next_id
            explicit_ids.add(assigned)
            next_id += 1
            return assigned

        for item in raw_items:
            if item['raw_name'] is None:
                item['raw_name'] = item['dir'].name

            if item['raw_id'] is None:
                if item['raw_name'] in name_to_id:
                    item['raw_id'] = name_to_id[item['raw_name']]
                else:
                    assigned_id = get_next_available_id()
                    name_to_id[item['raw_name']] = assigned_id
                    id_to_name[assigned_id] = item['raw_name']
                    item['raw_id'] = assigned_id

        # Instantiate ObjectClasses
        for item in raw_items:
            obj_cfg = item['obj_cfg']
            obj_class = ObjectClass(
                name=item['model_name'],
                model_path=str(item['obj_file']),
                class_id=item['raw_id'],
                class_name=item['raw_name'],
                texture=obj_cfg.get('texture'),
                textures=obj_cfg.get('textures'),
                randomize_materials=obj_cfg.get('randomize_materials', True),
                initial_rotation=obj_cfg.get('initial_rotation'),
                min_rotation=obj_cfg.get('min_rotation'),
                max_rotation=obj_cfg.get('max_rotation'),
                initial_height=obj_cfg.get('initial_height')
            )
            
            self.object_classes.append(obj_class)
            self.class_name_to_id[item['model_name']] = item['raw_id']
            self.class_name_to_id[item['dir'].name] = item['raw_id']
            self.object_classes_by_name[item['model_name']] = obj_class
            if item['dir'].name not in self.object_classes_by_name:
                self.object_classes_by_name[item['dir'].name] = obj_class
        
        if not self.object_classes:
            raise ValueError(f"No valid object classes loaded from {self.models_path}")
        
        unique_classes = len(self.create_class_mapping_dict())
        print(f"Loaded {len(self.object_classes)} object model(s) across {unique_classes} YOLO class(es):")
        for obj_class in self.object_classes:
            print(f"  - Model '{obj_class.name}' -> YOLO Class '{obj_class.class_name}' (ID: {obj_class.class_id})")
    
    def get_class_by_name(self, name: str) -> Optional[ObjectClass]:
        """Get object class by model name."""
        return self.object_classes_by_name.get(name)
    
    def get_class_by_id(self, class_id: int) -> Optional[ObjectClass]:
        """Get object class by ID (returns first model matching this class ID)."""
        for obj_class in self.object_classes:
            if obj_class.class_id == class_id:
                return obj_class
        return None

    def get_classes_by_id(self, class_id: int) -> List[ObjectClass]:
        """Get all object classes sharing this class ID."""
        return [obj_class for obj_class in self.object_classes if obj_class.class_id == class_id]
    
    def get_random_classes(self, n: int, allow_duplicates: bool = True, 
                          rng: Optional[np.random.Generator] = None) -> List[ObjectClass]:
        """
        Get random object classes.
        
        Args:
            n: Number of classes to sample
            allow_duplicates: If True, same class can appear multiple times
            rng: NumPy random generator (optional)
            
        Returns:
            List of object classes
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if allow_duplicates:
            indices = rng.integers(0, len(self.object_classes), size=n)
            return [self.object_classes[i] for i in indices]
        else:
            if n > len(self.object_classes):
                raise ValueError(
                    f"Cannot sample {n} unique classes from {len(self.object_classes)} available"
                )
            indices = rng.choice(len(self.object_classes), size=n, replace=False)
            return [self.object_classes[i] for i in indices]
    
    def get_all_classes(self) -> List[ObjectClass]:
        """Get all loaded object classes."""
        return self.object_classes.copy()
    
    def get_class_names(self) -> List[str]:
        """Get list of unique YOLO class names sorted by class ID."""
        mapping = self.create_class_mapping_dict()
        if not mapping:
            return []
        max_id = max(mapping.keys())
        return [mapping.get(i, f"class_{i}") for i in range(max_id + 1)]
    
    def get_model_names(self) -> List[str]:
        """Get list of all model directory names."""
        return [obj_class.name for obj_class in self.object_classes]
    
    def get_num_classes(self) -> int:
        """Get number of loaded classes."""
        return len(self.object_classes)
    
    def validate_models(self) -> Tuple[bool, List[str]]:
        """
        Validate that all models can be accessed.
        
        Returns:
            Tuple of (all_valid, error_messages)
        """
        errors = []
        
        for obj_class in self.object_classes:
            # Check model file exists
            if not os.path.exists(obj_class.model_path):
                errors.append(f"Model file not found: {obj_class.model_path}")
            
            # Check file is readable
            try:
                with open(obj_class.model_path, 'r') as f:
                    f.read(100)  # Read first 100 bytes
            except Exception as e:
                errors.append(f"Cannot read {obj_class.model_path}: {e}")
        
        return len(errors) == 0, errors
    
    def create_yolo_classes_file(self, output_path: str):
        """
        Create classes.txt file for YOLO (unique class names in class_id order).
        
        Args:
            output_path: Path to save classes.txt
        """
        with open(output_path, 'w') as f:
            for name in self.get_class_names():
                f.write(f"{name}\n")
    
    def create_class_mapping_dict(self) -> Dict[int, str]:
        """Create mapping from class ID to YOLO class name."""
        mapping = {}
        for obj_class in self.object_classes:
            mapping[obj_class.class_id] = obj_class.class_name
        return dict(sorted(mapping.items()))


def test_object_loader(models_path: str):
    """Test the object loader with a models directory."""
    print(f"Testing ObjectLoader with path: {models_path}")
    print("-" * 60)
    
    try:
        loader = ObjectLoader(models_path)
        
        print(f"\nTotal classes loaded: {loader.get_num_classes()}")
        print(f"Class names: {loader.get_class_names()}")
        
        print("\nValidating models...")
        valid, errors = loader.validate_models()
        if valid:
            print("✓ All models are valid")
        else:
            print("✗ Validation errors:")
            for error in errors:
                print(f"  - {error}")
        
        print("\nSampling random classes:")
        rng = np.random.default_rng(42)
        random_classes = loader.get_random_classes(3, allow_duplicates=True, rng=rng)
        for obj_class in random_classes:
            print(f"  - {obj_class}")
        
        print("\nClass mapping:")
        mapping = loader.create_class_mapping_dict()
        for class_id, name in mapping.items():
            print(f"  {class_id}: {name}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python object_loader.py <models_path>")
        sys.exit(1)
    
    test_object_loader(sys.argv[1])
