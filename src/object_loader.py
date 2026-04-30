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
    
    def __init__(self, name: str, model_path: str, class_id: int, texture: Optional[str] = None, textures: Optional[List[str]] = None):
        """
        Initialize object class.
        
        Args:
            name: Class name (directory name)
            model_path: Path to .obj file
            class_id: Numeric class ID (0-indexed for YOLO)
            texture: Optional single texture
            textures: Optional list of textures
        """
        self.name = name
        self.model_path = model_path
        self.class_id = class_id
        self.texture = texture
        self.textures = textures
        self.material_paths = self._find_materials()
    
    def _find_materials(self) -> Dict[str, str]:
        """Find associated material files (.mtl, textures)."""
        materials = {}
        model_dir = Path(self.model_path).parent
        
        # Find .mtl file
        mtl_files = list(model_dir.glob('*.mtl'))
        if mtl_files:
            materials['mtl'] = str(mtl_files[0])
        
        # Find texture files
        texture_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tga']
        for ext in texture_extensions:
            texture_files = list(model_dir.glob(f'*{ext}'))
            if texture_files:
                materials['textures'] = [str(f) for f in texture_files]
                break
        
        return materials
    
    def __repr__(self) -> str:
        return f"ObjectClass(name='{self.name}', id={self.class_id}, model='{self.model_path}')"


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
        self.config = config or {}
        
        if not self.models_path.exists():
            raise ValueError(f"Models path does not exist: {models_path}")
        
        self._load_objects()
    
    def _load_objects(self):
        """Scan models directory and load all object classes."""
        if not self.models_path.is_dir():
            raise ValueError(f"Models path is not a directory: {self.models_path}")
        
        # Get all subdirectories (each represents one class)
        class_dirs = sorted([
            d for d in self.models_path.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        if not class_dirs:
            raise ValueError(f"No object class directories found in {self.models_path}")
        
        # Load object configurations from config
        obj_configs = self.config.get('scene', {}).get('objects', {})

        # Load each class
        for class_id, class_dir in enumerate(class_dirs):
            obj_files = list(class_dir.glob('*.obj'))
            
            if not obj_files:
                print(f"Warning: No .obj file found in {class_dir.name}, skipping...")
                continue
            
            # Use first .obj file found
            obj_file = obj_files[0]
            
            # Get class-specific config
            obj_cfg = obj_configs.get(class_dir.name, {})

            obj_class = ObjectClass(
                name=class_dir.name,
                model_path=str(obj_file),
                class_id=class_id,
                texture=obj_cfg.get('texture'),
                textures=obj_cfg.get('textures')
            )
            
            self.object_classes.append(obj_class)
            self.class_name_to_id[class_dir.name] = class_id
        
        if not self.object_classes:
            raise ValueError(f"No valid object classes loaded from {self.models_path}")
        
        print(f"Loaded {len(self.object_classes)} object classes:")
        for obj_class in self.object_classes:
            print(f"  - {obj_class.name} (ID: {obj_class.class_id})")
    
    def get_class_by_name(self, name: str) -> Optional[ObjectClass]:
        """Get object class by name."""
        class_id = self.class_name_to_id.get(name)
        if class_id is not None:
            return self.object_classes[class_id]
        return None
    
    def get_class_by_id(self, class_id: int) -> Optional[ObjectClass]:
        """Get object class by ID."""
        if 0 <= class_id < len(self.object_classes):
            return self.object_classes[class_id]
        return None
    
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
        """Get list of all class names (sorted by ID)."""
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
        Create classes.txt file for YOLO (class names in order).
        
        Args:
            output_path: Path to save classes.txt
        """
        with open(output_path, 'w') as f:
            for obj_class in self.object_classes:
                f.write(f"{obj_class.name}\n")
    
    def create_class_mapping_dict(self) -> Dict[int, str]:
        """Create mapping from class ID to class name."""
        return {obj_class.class_id: obj_class.name for obj_class in self.object_classes}


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
