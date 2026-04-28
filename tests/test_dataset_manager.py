"""
Unit tests for dataset manager.

Tests splitting, organization, and validation.
"""

import pytest
import numpy as np
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataset_manager import DatasetSplitter, DatasetOrganizer, DatasetValidator


class TestDatasetSplitter:
    """Test suite for DatasetSplitter class."""
    
    def test_random_split(self):
        """Test random splitting without stratification."""
        splitter = DatasetSplitter(
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            random_seed=42
        )
        
        splits = splitter.split_indices(100)
        
        # Check all splits exist
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        # Check approximate sizes
        assert len(splits['train']) == 70
        assert len(splits['val']) == 20
        assert len(splits['test']) == 10
        
        # Check no overlap
        all_indices = set(splits['train'] + splits['val'] + splits['test'])
        assert len(all_indices) == 100
    
    def test_stratified_split(self):
        """Test stratified splitting by class."""
        # Create class labels: 50 of class 0, 30 of class 1, 20 of class 2
        class_labels = [0] * 50 + [1] * 30 + [2] * 20
        
        splitter = DatasetSplitter(
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            stratify=True,
            random_seed=42
        )
        
        splits = splitter.split_indices(100, class_labels)
        
        # Check that each class is represented in all splits
        train_classes = set(class_labels[i] for i in splits['train'])
        val_classes = set(class_labels[i] for i in splits['val'])
        test_classes = set(class_labels[i] for i in splits['test'])
        
        assert train_classes == {0, 1, 2}
        assert val_classes == {0, 1, 2}
        assert test_classes == {0, 1, 2}
    
    def test_invalid_ratios(self):
        """Test error on invalid split ratios."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            DatasetSplitter(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
    
    def test_reproducibility(self):
        """Test that splits are reproducible with same seed."""
        splitter1 = DatasetSplitter(random_seed=42)
        splits1 = splitter1.split_indices(100)
        
        splitter2 = DatasetSplitter(random_seed=42)
        splits2 = splitter2.split_indices(100)
        
        assert splits1['train'] == splits2['train']
        assert splits1['val'] == splits2['val']
        assert splits1['test'] == splits2['test']


class TestDatasetOrganizer:
    """Test suite for DatasetOrganizer class."""
    
    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create mock image and label files."""
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        
        image_files = []
        label_files = []
        
        for i in range(10):
            # Create image file
            img_file = temp_dir / f"image_{i:03d}.png"
            img_file.write_text("fake image data")
            image_files.append(img_file)
            
            # Create label file
            lbl_file = temp_dir / f"image_{i:03d}.txt"
            lbl_file.write_text(f"0 0.5 0.5 0.3 0.3\n1 0.7 0.7 0.2 0.2\n")
            label_files.append(lbl_file)
        
        return image_files, label_files
    
    def test_organize_dataset(self, mock_dataset, tmp_path):
        """Test dataset organization into splits."""
        image_files, label_files = mock_dataset
        
        organizer = DatasetOrganizer(str(tmp_path / "output"))
        
        splits = {
            'train': [0, 1, 2, 3, 4, 5, 6],
            'val': [7, 8],
            'test': [9]
        }
        
        organizer.organize_dataset(image_files, label_files, splits)
        
        # Check files were copied to correct locations
        output_path = tmp_path / "output"
        
        assert (output_path / 'images' / 'train' / 'image_000.png').exists()
        assert (output_path / 'labels' / 'train' / 'image_000.txt').exists()
        assert (output_path / 'images' / 'val' / 'image_007.png').exists()
        assert (output_path / 'images' / 'test' / 'image_009.png').exists()
        
        # Check stats
        assert organizer.stats['train']['images'] == 7
        assert organizer.stats['val']['images'] == 2
        assert organizer.stats['test']['images'] == 1


class TestDatasetValidator:
    """Test suite for DatasetValidator class."""
    
    @pytest.fixture
    def valid_dataset(self, tmp_path):
        """Create a valid dataset structure."""
        dataset_path = tmp_path / "dataset"
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (dataset_path / 'images' / split).mkdir(parents=True)
            (dataset_path / 'labels' / split).mkdir(parents=True)
            
            # Create sample files
            img_file = dataset_path / 'images' / split / 'image_001.png'
            img_file.write_text("fake image")
            
            lbl_file = dataset_path / 'labels' / split / 'image_001.txt'
            lbl_file.write_text("0 0.5 0.5 0.3 0.3\n")
        
        # Create data.yaml
        data_yaml = dataset_path / 'data.yaml'
        data_yaml.write_text("path: .\ntrain: images/train\nnc: 1\nnames:\n  0: object\n")
        
        return dataset_path
    
    def test_validate_valid_dataset(self, valid_dataset):
        """Test validation of valid dataset."""
        validator = DatasetValidator(str(valid_dataset))
        is_valid, issues = validator.validate()
        
        assert is_valid
        assert len(issues) == 0
    
    def test_missing_directory(self, tmp_path):
        """Test detection of missing directories."""
        dataset_path = tmp_path / "incomplete"
        dataset_path.mkdir()
        
        # Only create train, not val/test
        (dataset_path / 'images' / 'train').mkdir(parents=True)
        
        validator = DatasetValidator(str(dataset_path))
        is_valid, issues = validator.validate()
        
        assert not is_valid
        assert any('Missing directory' in issue for issue in issues)
    
    def test_missing_label(self, valid_dataset):
        """Test detection of missing label file."""
        # Add image without corresponding label
        img_file = valid_dataset / 'images' / 'train' / 'orphan.png'
        img_file.write_text("fake image")
        
        validator = DatasetValidator(str(valid_dataset))
        is_valid, issues = validator.validate()
        
        assert not is_valid
        assert any('Missing label' in issue for issue in issues)
    
    def test_invalid_label_format(self, valid_dataset):
        """Test detection of invalid label format."""
        # Create label with invalid format
        lbl_file = valid_dataset / 'labels' / 'train' / 'invalid.txt'
        lbl_file.write_text("not valid yolo format\n")
        
        # Add corresponding image
        img_file = valid_dataset / 'images' / 'train' / 'invalid.png'
        img_file.write_text("fake image")
        
        validator = DatasetValidator(str(valid_dataset))
        is_valid, issues = validator.validate()
        
        assert not is_valid
        assert any('Invalid format' in issue for issue in issues)
    
    def test_out_of_range_coordinates(self, valid_dataset):
        """Test detection of coordinates out of [0,1] range."""
        # Create label with out-of-range coordinates
        lbl_file = valid_dataset / 'labels' / 'train' / 'outofrange.txt'
        lbl_file.write_text("0 1.5 0.5 0.3 0.3\n")  # x > 1
        
        img_file = valid_dataset / 'images' / 'train' / 'outofrange.png'
        img_file.write_text("fake image")
        
        validator = DatasetValidator(str(valid_dataset))
        is_valid, issues = validator.validate()
        
        assert not is_valid
        assert any('out of range' in issue for issue in issues)
    
    def test_class_distribution(self, valid_dataset):
        """Test class distribution counting."""
        # Add more labels with different classes
        for i in range(3):
            lbl_file = valid_dataset / 'labels' / 'train' / f'image_{i:03d}.txt'
            lbl_file.write_text(f"0 0.5 0.5 0.3 0.3\n{i} 0.7 0.7 0.2 0.2\n")
            
            img_file = valid_dataset / 'images' / 'train' / f'image_{i:03d}.png'
            img_file.write_text("fake image")
        
        validator = DatasetValidator(str(valid_dataset))
        distribution = validator.get_class_distribution()
        
        # Class 0 should appear most frequently
        assert 0 in distribution
        assert distribution[0] >= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
