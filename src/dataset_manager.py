"""
Dataset Manager for organizing and splitting YOLO datasets.

This module handles dataset organization, train/val/test splitting,
and dataset structure validation.
"""

import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict


class DatasetSplitter:
    """Handles splitting dataset into train/val/test sets."""
    
    def __init__(self,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 stratify: bool = True,
                 random_seed: Optional[int] = None):
        """
        Initialize dataset splitter.
        
        Args:
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            stratify: Whether to stratify split by class
            random_seed: Random seed for reproducibility
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.stratify = stratify
        self.rng = np.random.default_rng(random_seed)
    
    def split_indices(self, 
                     num_samples: int,
                     class_labels: Optional[List[int]] = None) -> Dict[str, List[int]]:
        """
        Split sample indices into train/val/test sets.
        
        Args:
            num_samples: Total number of samples
            class_labels: Class labels for each sample (for stratification)
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing indices
        """
        indices = np.arange(num_samples)
        
        if self.stratify and class_labels is not None:
            return self._stratified_split(indices, class_labels)
        else:
            return self._random_split(indices)
    
    def _random_split(self, indices: np.ndarray) -> Dict[str, List[int]]:
        """Perform random split."""
        # Shuffle indices
        self.rng.shuffle(indices)
        
        # Calculate split points
        n_total = len(indices)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        # Split
        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train:n_train + n_val].tolist()
        test_indices = indices[n_train + n_val:].tolist()
        
        return {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
    
    def _stratified_split(self,
                         indices: np.ndarray,
                         class_labels: List[int]) -> Dict[str, List[int]]:
        """Perform stratified split by class."""
        # Group indices by class
        class_to_indices = defaultdict(list)
        for idx, label in zip(indices, class_labels):
            class_to_indices[label].append(idx)
        
        # Split each class
        train_indices = []
        val_indices = []
        test_indices = []
        
        for class_label, class_indices in class_to_indices.items():
            class_indices = np.array(class_indices)
            self.rng.shuffle(class_indices)
            
            n_total = len(class_indices)
            n_train = max(1, int(n_total * self.train_ratio))
            n_val = max(1, int(n_total * self.val_ratio))
            
            train_indices.extend(class_indices[:n_train].tolist())
            val_indices.extend(class_indices[n_train:n_train + n_val].tolist())
            test_indices.extend(class_indices[n_train + n_val:].tolist())
        
        return {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }


class DatasetOrganizer:
    """Organizes generated images and labels into YOLO dataset structure."""
    
    def __init__(self, output_path: str):
        """
        Initialize dataset organizer.
        
        Args:
            output_path: Root path for organized dataset
        """
        self.output_path = Path(output_path)
        self.stats = {
            'train': {'images': 0, 'annotations': 0, 'objects': 0},
            'val': {'images': 0, 'annotations': 0, 'objects': 0},
            'test': {'images': 0, 'annotations': 0, 'objects': 0}
        }
    
    def organize_dataset(self,
                        image_files: List[Path],
                        label_files: List[Path],
                        split_assignments: Dict[str, List[int]]):
        """
        Organize images and labels into train/val/test splits.
        
        Args:
            image_files: List of image file paths
            label_files: List of label file paths
            split_assignments: Dict mapping split names to indices
        """
        print("Organizing dataset...")
        
        for split_name, indices in split_assignments.items():
            print(f"  Processing {split_name} split ({len(indices)} samples)...")
            
            for idx in indices:
                # Copy image
                src_image = image_files[idx]
                dst_image = self.output_path / 'images' / split_name / src_image.name
                dst_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_image, dst_image)
                
                # Copy label
                src_label = label_files[idx]
                dst_label = self.output_path / 'labels' / split_name / src_label.name
                dst_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_label, dst_label)
                
                # Update stats
                self.stats[split_name]['images'] += 1
                self.stats[split_name]['annotations'] += 1
                
                # Count objects in label file
                if dst_label.exists():
                    with open(dst_label, 'r') as f:
                        num_objects = len(f.readlines())
                    self.stats[split_name]['objects'] += num_objects
        
        print("✓ Dataset organization complete")
    
    def print_stats(self):
        """Print dataset statistics."""
        print("\nDataset Statistics:")
        print("=" * 60)
        
        for split_name, stats in self.stats.items():
            print(f"{split_name.upper()} set:")
            print(f"  Images:      {stats['images']}")
            print(f"  Annotations: {stats['annotations']}")
            print(f"  Objects:     {stats['objects']}")
            if stats['images'] > 0:
                print(f"  Avg objects/image: {stats['objects'] / stats['images']:.2f}")
            print()
    
    def save_stats(self):
        """Save statistics to JSON file."""
        stats_path = self.output_path / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"Saved statistics to {stats_path}")


class DatasetValidator:
    """Validates YOLO dataset structure and annotations."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize dataset validator.
        
        Args:
            dataset_path: Path to dataset root
        """
        self.dataset_path = Path(dataset_path)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate dataset structure and annotations.
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check directory structure
        required_dirs = [
            'images/train', 'images/val', 'images/test',
            'labels/train', 'labels/val', 'labels/test'
        ]
        
        for dir_path in required_dirs:
            full_path = self.dataset_path / dir_path
            if not full_path.exists():
                issues.append(f"Missing directory: {dir_path}")
        
        # Check data.yaml exists
        if not (self.dataset_path / 'data.yaml').exists():
            issues.append("Missing data.yaml file")
        
        # Check image-label pairs
        for split in ['train', 'val', 'test']:
            img_dir = self.dataset_path / 'images' / split
            lbl_dir = self.dataset_path / 'labels' / split
            
            if not img_dir.exists() or not lbl_dir.exists():
                continue
            
            # Get image files
            image_files = list(img_dir.glob('*.jpg')) + \
                         list(img_dir.glob('*.png')) + \
                         list(img_dir.glob('*.jpeg'))
            
            for img_file in image_files:
                # Check corresponding label exists
                label_file = lbl_dir / (img_file.stem + '.txt')
                if not label_file.exists():
                    issues.append(f"Missing label for {split}/{img_file.name}")
                else:
                    # Validate label format
                    label_issues = self._validate_label_file(label_file)
                    issues.extend(label_issues)
        
        return len(issues) == 0, issues
    
    def _validate_label_file(self, label_path: Path) -> List[str]:
        """Validate individual label file format."""
        issues = []
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                # Check number of parts (5 for standard, 9 for OBB)
                if len(parts) not in [5, 9]:
                    issues.append(
                        f"{label_path.name}:{line_num} - Invalid format "
                        f"(expected 5 or 9 values, got {len(parts)})"
                    )
                    continue
                
                # Check class ID is integer
                try:
                    class_id = int(parts[0])
                    if class_id < 0:
                        issues.append(f"{label_path.name}:{line_num} - Negative class ID")
                except ValueError:
                    issues.append(f"{label_path.name}:{line_num} - Invalid class ID")
                
                # Check coordinates are floats in [0, 1]
                try:
                    coords = [float(x) for x in parts[1:]]
                    for coord in coords:
                        if coord < 0 or coord > 1.001: # Allow slight floating point overshoot
                            issues.append(
                                f"{label_path.name}:{line_num} - "
                                f"Coordinate out of range [0, 1]: {coord}"
                            )
                except ValueError:
                    issues.append(f"{label_path.name}:{line_num} - Invalid coordinate format")
        
        except Exception as e:
            issues.append(f"Error reading {label_path.name}: {e}")
        
        return issues
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of classes across dataset."""
        distribution = defaultdict(int)
        
        for split in ['train', 'val', 'test']:
            lbl_dir = self.dataset_path / 'labels' / split
            
            if not lbl_dir.exists():
                continue
            
            for label_file in lbl_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            distribution[class_id] += 1
        
        return dict(distribution)


if __name__ == '__main__':
    print("Dataset management module")
    print("This module handles dataset organization and validation")
