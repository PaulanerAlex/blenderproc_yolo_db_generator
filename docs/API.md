# API Documentation

## Architecture Overview

The Blender YOLO Dataset Generator is organized into modular components:

```
src/
├── config_parser.py      # Configuration loading and validation
├── object_loader.py      # 3D model management
├── scene_generator.py    # BlenderProc scene composition
├── randomization.py      # Scene randomization utilities
├── bbox_extractor.py     # Bounding box extraction
├── yolo_converter.py     # YOLO format conversion
├── dataset_manager.py    # Dataset organization and validation
└── generate_dataset.py   # Main pipeline orchestrator
```

---

## Core Modules

### config_parser.py

Handles configuration file loading, validation, and defaults.

#### Classes

**`Config`**
```python
class Config:
    """Configuration container with dot-notation access."""
    
    def __init__(self, config_dict: Dict[str, Any])
    def get(self, key: str, default: Any = None) -> Any
    def __getitem__(self, key: str) -> Any
```

**Usage:**
```python
from config_parser import ConfigParser

parser = ConfigParser()
config = parser.load('configs/example_config.json')

# Access with dot notation
width = config.get('camera.width')  # Returns 640
scenes = config['dataset']['num_scenes']  # Returns 100
```

**`ConfigParser`**
```python
class ConfigParser:
    """Parser for configuration files with validation."""
    
    def __init__(self, schema_path: Optional[str] = None)
    def load(self, config_path: str) -> Config
    def save(self, config: Union[Config, Dict], path: str)
```

**Methods:**
- `load(config_path)`: Load and validate configuration file
- `save(config, path)`: Save configuration to file
- `load_schema(schema_path)`: Load JSON schema for validation

---

### object_loader.py

Manages 3D object loading and class mapping.

#### Classes

**`ObjectClass`**
```python
@dataclass
class ObjectClass:
    """Represents a single object class."""
    name: str          # Class name (directory name)
    model_path: str    # Path to .obj file
    class_id: int      # 0-indexed class ID
```

**`ObjectLoader`**
```python
class ObjectLoader:
    """Loads and manages 3D object models."""
    
    def __init__(self, models_path: str)
    def get_class_by_name(self, name: str) -> Optional[ObjectClass]
    def get_class_by_id(self, class_id: int) -> Optional[ObjectClass]
    def get_random_classes(self, n: int, allow_duplicates: bool = True) -> List[ObjectClass]
    def get_all_classes() -> List[ObjectClass]
    def get_class_names() -> List[str]
    def get_num_classes() -> int
```

**Usage:**
```python
from object_loader import ObjectLoader

loader = ObjectLoader('./models')

# Get class information
cube_class = loader.get_class_by_name('cube')
print(f"Cube ID: {cube_class.class_id}")  # 0 (alphabetical)

# Sample random classes
random_objs = loader.get_random_classes(5, allow_duplicates=True)

# Get all class names (for data.yaml)
class_names = loader.get_class_names()  # ['cube', 'cylinder', 'sphere']
```

---

### bbox_extractor.py

Extracts bounding boxes from BlenderProc HDF5 output.

#### Classes

**`BoundingBox`**
```python
class BoundingBox:
    """Represents a 2D bounding box annotation."""
    
    def __init__(self,
                 class_id: int,
                 class_name: str,
                 bbox_2d: np.ndarray,      # [x_min, y_min, x_max, y_max] or corners
                 angle: Optional[float] = None,  # Rotation angle (radians)
                 visibility: float = 1.0,
                 area_px: float = 0.0)
```

**Properties:**
- `is_obb()`: Returns True if this is an oriented bounding box
- `get_center()`: Returns (x_center, y_center)
- `get_dimensions()`: Returns (width, height)

**`BBoxExtractor`**
```python
class BBoxExtractor:
    """Extracts bounding boxes from HDF5 files."""
    
    def __init__(self, min_visibility: float = 0.3,
                 min_bbox_side_px: int = 10)
    
    def extract_from_hdf5(self,
                         hdf5_path: str,
                         class_mapping: Dict[int, str]) -> List[BoundingBox]
```

**Usage:**
```python
from bbox_extractor import BBoxExtractor

extractor = BBoxExtractor(
    min_visibility=0.3,    # Filter out heavily occluded objects
    min_bbox_side_px=10    # Minimum box size
)

bboxes = extractor.extract_from_hdf5(
    'output/scene_0000.hdf5',
    class_mapping={0: 'cube', 1: 'sphere'}
)

for bbox in bboxes:
    print(f"{bbox.class_name}: {bbox.get_center()}, visible={bbox.visibility}")
```

---

### yolo_converter.py

Converts bounding boxes to YOLO format.

#### Classes

**`YOLOConverter`**
```python
class YOLOConverter:
    """Converts bounding boxes to YOLO format."""
    
    def __init__(self,
                 yolo_format: str,    # 'yolov11', 'yolov26', 'yolov11-obb', 'yolov26-obb'
                 image_width: int,
                 image_height: int)
    
    def convert_bbox(self, bbox: BoundingBox) -> str
    def convert_bboxes(self, bboxes: List[BoundingBox]) -> str
    def save_annotations(self, bboxes: List[BoundingBox], output_path: str)
```

**Usage:**
```python
from yolo_converter import YOLOConverter
from bbox_extractor import BoundingBox
import numpy as np

# Standard YOLO
converter = YOLOConverter('yolov11', image_width=640, image_height=480)

bbox = BoundingBox(
    class_id=0,
    class_name='cube',
    bbox_2d=np.array([100, 100, 200, 200])
)

yolo_line = converter.convert_bbox(bbox)
# Output: "0 0.234375 0.3125 0.15625 0.208333"

# OBB YOLO
obb_converter = YOLOConverter('yolov11-obb', image_width=640, image_height=480)
obb_bbox = BoundingBox(
    class_id=0,
    class_name='cube',
    bbox_2d=corners,  # 4x2 array of corners
    angle=0.785       # 45 degrees in radians
)

obb_line = obb_converter.convert_bbox(obb_bbox)
# Output: "0 0.5 0.5 0.15625 0.208333 0.785"
```

**`YOLODatasetFormatter`**
```python
class YOLODatasetFormatter:
    """Formats complete YOLO dataset structure."""
    
    def __init__(self,
                 output_path: str,
                 class_names: List[str],
                 yolo_format: str)
    
    def create_data_yaml()
    def create_classes_file()
    def get_image_path(split: str, image_name: str) -> Path
    def get_label_path(split: str, label_name: str) -> Path
```

**Usage:**
```python
from yolo_converter import YOLODatasetFormatter

formatter = YOLODatasetFormatter(
    output_path='./output/dataset',
    class_names=['cube', 'sphere', 'cylinder'],
    yolo_format='yolov11'
)

# Creates directory structure automatically in __init__
formatter.create_data_yaml()      # Creates data.yaml
formatter.create_classes_file()   # Creates classes.txt

# Get paths for saving
img_path = formatter.get_image_path('train', 'scene_0000_0.png')
# Returns: ./output/dataset/images/train/scene_0000_0.png

label_path = formatter.get_label_path('train', 'scene_0000_0.txt')
# Returns: ./output/dataset/labels/train/scene_0000_0.txt
```

---

### dataset_manager.py

Handles dataset splitting, organization, and validation.

#### Classes

**`DatasetSplitter`**
```python
class DatasetSplitter:
    """Splits dataset into train/val/test sets."""
    
    def __init__(self,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 stratify: bool = True,
                 random_seed: Optional[int] = None)
    
    def split_indices(self,
                     num_samples: int,
                     class_labels: Optional[List[int]] = None) -> Dict[str, List[int]]
```

**Usage:**
```python
from dataset_manager import DatasetSplitter

splitter = DatasetSplitter(
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    random_seed=42  # For reproducibility
)

# Split 1000 samples
splits = splitter.split_indices(num_samples=1000)

print(f"Train: {len(splits['train'])}")  # 700
print(f"Val: {len(splits['val'])}")      # 200
print(f"Test: {len(splits['test'])}")    # 100

# Stratified split (maintains class distribution)
class_labels = [0, 0, 1, 1, 2, 2, ...]  # One per sample
splits = splitter.split_indices(
    num_samples=1000,
    class_labels=class_labels
)
```

**`DatasetValidator`**
```python
class DatasetValidator:
    """Validates YOLO dataset structure and content."""
    
    def __init__(self, dataset_path: str)
    
    def validate() -> Tuple[bool, List[str]]
    def get_class_distribution() -> Dict[int, int]
```

**Usage:**
```python
from dataset_manager import DatasetValidator

validator = DatasetValidator('./output/dataset')

is_valid, errors = validator.validate()

if not is_valid:
    print("Dataset validation failed:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Dataset valid!")
    
    # Get statistics
    dist = validator.get_class_distribution()
    print(f"Class distribution: {dist}")
```

---

## Main Pipeline

### generate_dataset.py

Main entry point for dataset generation.

#### Classes

**`DatasetGenerator`**
```python
class DatasetGenerator:
    """Main pipeline for dataset generation."""
    
    def __init__(self, config: Config)
    
    def generate() -> None
    def generate_scene(scene_id: int) -> List[str]
```

**Command Line Usage:**
```bash
python src/generate_dataset.py --config configs/example_config.json [OPTIONS]

Options:
  --config PATH       Configuration file (required)
  --output PATH       Override output path
  --num-scenes N      Override number of scenes
  --verbose           Enable verbose logging
  --help              Show help message
```

**Programmatic Usage:**
```python
from config_parser import ConfigParser
from generate_dataset import DatasetGenerator

# Load configuration
parser = ConfigParser()
config = parser.load('configs/my_config.json')

# Create generator
generator = DatasetGenerator(config)

# Generate dataset
generator.generate()

print(f"Dataset generated at: {config['output.save_path']}")
```

---

## Extending the Framework

### Adding Custom Scene Elements

Modify `src/scene_generator.py`:

```python
class SceneGenerator:
    def add_custom_lighting(self):
        """Add your custom lighting setup."""
        import bproc
        
        # Add spot light
        light = bproc.types.Light()
        light.set_type("SPOT")
        light.set_location([0, 0, 5])
        light.set_energy(1000)
```

### Custom Randomization

Modify `src/randomization.py`:

```python
def randomize_weather_conditions(scene):
    """Add weather effects."""
    fog_density = np.random.uniform(0.01, 0.1)
    # Apply fog shader
    ...
```

### Custom Annotation Formats

Extend `yolo_converter.py`:

```python
class CustomConverter(YOLOConverter):
    def convert_to_custom_format(self, bbox: BoundingBox) -> str:
        """Convert to your custom format."""
        return f"{bbox.class_id},{bbox.get_center()[0]},{bbox.get_center()[1]}"
```

---

## Best Practices

### 1. Configuration Management

```python
# Good: Load once, reuse
parser = ConfigParser()
config = parser.load('config.json')

# Bad: Load multiple times
config1 = ConfigParser().load('config.json')
config2 = ConfigParser().load('config.json')
```

### 2. Error Handling

```python
from config_parser import ConfigError

try:
    config = parser.load('config.json')
except ConfigError as e:
    print(f"Configuration error: {e}")
    sys.exit(1)
```

### 3. Resource Management

```python
# Good: Process in batches
for batch in range(0, total_scenes, batch_size):
    process_batch(batch, batch_size)
    
# Bad: Load everything in memory
all_scenes = [generate_scene(i) for i in range(10000)]
```

### 4. Validation

```python
# Always validate before training
validator = DatasetValidator(dataset_path)
is_valid, errors = validator.validate()

if not is_valid:
    print("Fix these issues before training:")
    for error in errors:
        print(f"  - {error}")
```

---

## Performance Tips

### 1. Use GPU Rendering

Ensure CUDA is available:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### 2. Batch Processing

```json
{
  "dataset": {
    "scenes_per_run": 10  // Process 10 scenes at a time
  }
}
```

### 3. Optimize Models

- Reduce polygon count
- Remove unnecessary details
- Use simplified collision meshes

### 4. Parallel Generation

Run multiple instances:
```bash
python src/generate_dataset.py --config config1.json &
python src/generate_dataset.py --config config2.json &
wait
```

---

## Testing

### Unit Tests

```python
import pytest
from config_parser import ConfigParser

def test_config_loading():
    parser = ConfigParser()
    config = parser.load('test_config.json')
    assert config['dataset.num_scenes'] == 10
```

### Integration Tests

```python
def test_full_pipeline():
    from generate_dataset import DatasetGenerator
    from config_parser import ConfigParser
    
    config = ConfigParser().load('test_config.json')
    generator = DatasetGenerator(config)
    generator.generate()
    
    # Validate output
    assert Path(config['output.save_path']).exists()
```

---

## Reference

### YOLO Format Specifications

**Standard (YOLOv11/v26):**
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1]

**OBB (YOLOv11-OBB/v26-OBB):**
```
<class_id> <x_center> <y_center> <width> <height> <angle>
```
Coordinates normalized to [0, 1], angle in radians [-π/2, π/2]

### File Formats

**data.yaml:**
```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 3  # Number of classes
names:
  0: cube
  1: sphere
  2: cylinder
```

**classes.txt:**
```
cube
sphere
cylinder
```

---

For more examples, see the `examples/` directory and the User Guide.
