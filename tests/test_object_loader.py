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
        
        with pytest.raises(ValueError, match="No object class directories found"):
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
