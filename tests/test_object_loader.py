"""
Unit tests for object loader.

Tests 3D model loading, class mapping, and validation.
"""

import pytest
from pathlib import Path
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from object_loader import ObjectLoader, ObjectClass


class TestObjectLoader:
    """Test suite for ObjectLoader class."""
    
    @pytest.fixture
    def mock_models_dir(self, tmp_path):
        """Create a mock models directory with test OBJ files."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        # Create class directories with minimal OBJ files
        for class_name in ['cube', 'sphere', 'cylinder']:
            class_dir = models_dir / class_name
            class_dir.mkdir()
            
            # Create minimal OBJ file
            obj_file = class_dir / f"{class_name}.obj"
            with open(obj_file, 'w') as f:
                f.write("# Minimal OBJ file\n")
                f.write("v 0.0 0.0 0.0\n")
                f.write("v 1.0 0.0 0.0\n")
                f.write("v 0.0 1.0 0.0\n")
                f.write("f 1 2 3\n")
            
            # Create MTL file
            mtl_file = class_dir / f"{class_name}.mtl"
            with open(mtl_file, 'w') as f:
                f.write("# Minimal MTL file\n")
                f.write("newmtl material\n")
        
        return models_dir
    
    def test_load_models(self, mock_models_dir):
        """Test loading models from directory."""
        loader = ObjectLoader(str(mock_models_dir))
        
        assert loader.get_num_classes() == 3
        assert set(loader.get_class_names()) == {'cube', 'sphere', 'cylinder'}
    
    def test_class_sorting(self, mock_models_dir):
        """Test that classes are sorted alphabetically by directory name."""
        loader = ObjectLoader(str(mock_models_dir))
        class_names = loader.get_class_names()
        
        # Should be sorted: cube (0), cylinder (1), sphere (2)
        assert class_names[0] == 'cube'
        assert class_names[1] == 'cylinder'
        assert class_names[2] == 'sphere'
        
        # Verify class IDs
        assert loader.get_class_by_name('cube').class_id == 0
        assert loader.get_class_by_name('cylinder').class_id == 1
        assert loader.get_class_by_name('sphere').class_id == 2
    
    def test_get_class_by_name(self, mock_models_dir):
        """Test retrieving class by name."""
        loader = ObjectLoader(str(mock_models_dir))
        
        obj_class = loader.get_class_by_name('cube')
        assert obj_class is not None
        assert obj_class.name == 'cube'
        assert obj_class.class_id == 0
    
    def test_get_class_by_id(self, mock_models_dir):
        """Test retrieving class by ID."""
        loader = ObjectLoader(str(mock_models_dir))
        
        obj_class = loader.get_class_by_id(1)  # cylinder
        assert obj_class is not None
        assert obj_class.name == 'cylinder'
        assert obj_class.class_id == 1
    
    def test_get_random_classes(self, mock_models_dir):
        """Test random class sampling."""
        loader = ObjectLoader(str(mock_models_dir))
        
        # With duplicates
        classes = loader.get_random_classes(5, allow_duplicates=True)
        assert len(classes) == 5
        assert all(isinstance(c, ObjectClass) for c in classes)
        
        # Without duplicates
        classes = loader.get_random_classes(3, allow_duplicates=False)
        assert len(classes) == 3
        assert len(set(c.class_id for c in classes)) == 3
    
    def test_random_classes_too_many(self, mock_models_dir):
        """Test error when sampling too many unique classes."""
        loader = ObjectLoader(str(mock_models_dir))
        
        with pytest.raises(ValueError, match="Cannot sample"):
            loader.get_random_classes(5, allow_duplicates=False)
    
    def test_validate_models(self, mock_models_dir):
        """Test model validation."""
        loader = ObjectLoader(str(mock_models_dir))
        
        is_valid, errors = loader.validate_models()
        assert is_valid
        assert len(errors) == 0
    
    def test_create_class_mapping(self, mock_models_dir):
        """Test class mapping creation."""
        loader = ObjectLoader(str(mock_models_dir))
        
        mapping = loader.create_class_mapping_dict()
        assert mapping[0] == 'cube'
        assert mapping[1] == 'cylinder'
        assert mapping[2] == 'sphere'
    
    def test_create_yolo_classes_file(self, mock_models_dir, tmp_path):
        """Test YOLO classes file creation."""
        loader = ObjectLoader(str(mock_models_dir))
        
        output_file = tmp_path / "classes.txt"
        loader.create_yolo_classes_file(str(output_file))
        
        with open(output_file, 'r') as f:
            lines = f.read().strip().split('\n')
        
        assert lines == ['cube', 'cylinder', 'sphere']
    
    def test_empty_models_dir(self, tmp_path):
        """Test error when models directory is empty."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="No valid object class directories found"):
            ObjectLoader(str(empty_dir))
    
    def test_missing_obj_files(self, tmp_path):
        """Test handling of directories without OBJ files."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        # Create directory without OBJ file
        (models_dir / "empty_class").mkdir()
        
        # Create valid class
        valid_class = models_dir / "valid"
        valid_class.mkdir()
        with open(valid_class / "model.obj", 'w') as f:
            f.write("v 0 0 0\n")
        
        loader = ObjectLoader(str(models_dir))
        
        # Should only load the valid class
        assert loader.get_num_classes() == 1
        assert loader.get_class_names()[0] == 'valid'
    
    def test_material_detection(self, mock_models_dir):
        """Test detection of material files."""
        loader = ObjectLoader(str(mock_models_dir))
        
        obj_class = loader.get_class_by_name('cube')
        assert 'mtl' in obj_class.material_paths
        assert obj_class.material_paths['mtl'].endswith('.mtl')

    def test_shared_class_name(self, mock_models_dir):
        """Test grouping multiple models under the same class_name."""
        config = {
            'scene': {
                'objects': {
                    'cube': {'class_name': 'polyhedron'},
                    'cylinder': {'class_name': 'curved'},
                    'sphere': {'class_name': 'curved'}
                }
            }
        }
        loader = ObjectLoader(str(mock_models_dir), config)
        
        # Should have 3 models loaded
        assert loader.get_num_classes() == 3
        
        # But only 2 unique YOLO classes: 'polyhedron' and 'curved'
        mapping = loader.create_class_mapping_dict()
        assert len(mapping) == 2
        assert loader.get_class_names() == ['polyhedron', 'curved']
        
        cube = loader.get_class_by_name('cube')
        cylinder = loader.get_class_by_name('cylinder')
        sphere = loader.get_class_by_name('sphere')
        
        assert cube.class_id == 0
        assert cube.class_name == 'polyhedron'
        assert cylinder.class_id == 1
        assert cylinder.class_name == 'curved'
        assert sphere.class_id == 1
        assert sphere.class_name == 'curved'

    def test_shared_explicit_class_id(self, mock_models_dir, tmp_path):
        """Test grouping multiple models under explicit class_id."""
        config = {
            'scene': {
                'objects': {
                    'cube': {'class_id': 0, 'class_name': 'box'},
                    'cylinder': {'class_id': 0},  # inherits 'box'
                    'sphere': {'class_id': 1, 'class_name': 'ball'}
                }
            }
        }
        loader = ObjectLoader(str(mock_models_dir), config)
        
        mapping = loader.create_class_mapping_dict()
        assert mapping == {0: 'box', 1: 'ball'}
        assert loader.get_class_names() == ['box', 'ball']
        
        # Verify classes.txt
        classes_file = tmp_path / "classes.txt"
        loader.create_yolo_classes_file(str(classes_file))
        with open(classes_file, 'r') as f:
            lines = f.read().strip().split('\n')
        assert lines == ['box', 'ball']

    def test_conflicting_class_mapping(self, mock_models_dir):
        """Test error when conflicting class IDs are configured for the same class name."""
        config = {
            'scene': {
                'objects': {
                    'cube': {'class_name': 'same_name', 'class_id': 0},
                    'cylinder': {'class_name': 'same_name', 'class_id': 1}
                }
            }
        }
        with pytest.raises(ValueError, match="Conflicting class_id"):
            ObjectLoader(str(mock_models_dir), config)

    def test_multiple_obj_files_in_folder(self, tmp_path):
        """Test loading multiple OBJ variations from a single directory."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        # Directory with 3 variations
        gates_dir = models_dir / "blue_gates_with_variations"
        gates_dir.mkdir()
        for size in ['400x400x30', '500x600x30', '600x500x45']:
            with open(gates_dir / f"gate_{size}.obj", 'w') as f:
                f.write("v 0 0 0\n")
            with open(gates_dir / f"gate_{size}.mtl", 'w') as f:
                f.write(f"# MTL for {size}\n")
        
        config = {
            'scene': {
                'objects': {
                    'blue_gates_with_variations': {
                        'class_id': 0,
                        'class_name': 'blue_gate'
                    }
                }
            }
        }
        loader = ObjectLoader(str(models_dir), config)
        
        # 3 variations loaded
        assert loader.get_num_classes() == 3
        
        # All 3 variations belong to YOLO class 0 ('blue_gate')
        mapping = loader.create_class_mapping_dict()
        assert mapping == {0: 'blue_gate'}
        assert loader.get_class_names() == ['blue_gate']
        
        # Names are namespaced by folder
        for obj in loader.object_classes:
            assert obj.class_id == 0
            assert obj.class_name == 'blue_gate'
            assert obj.name.startswith('blue_gates_with_variations/')
            assert obj.material_paths['mtl'].endswith('.mtl')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
