# Blender YOLO Dataset Generator - User Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Configuration](#configuration)
5. [Advanced Topics](#advanced-topics)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

---

## Quick Start

### 1. Install (One Command)
```bash
cd /home/paul/akamav/dev/projects/blenderproc
./install.sh
```

### 2. Prepare Your 3D Models
```bash
# Create model directories (one per class)
mkdir -p models/cube models/sphere models/cylinder

# Add .obj files (one per directory)
cp /path/to/cube.obj models/cube/model.obj
cp /path/to/sphere.obj models/sphere/model.obj
```

### 3. Generate Dataset
```bash
source .venv/bin/activate
blenderproc run src/generate_dataset.py --config configs/example_config.json
```

That's it! Your YOLO dataset will be in `output/yolo_dataset/`

---

## Installation

### Prerequisites
- Linux system (Ubuntu 20.04+ recommended)
- Python 3.10 or higher
- ~5GB disk space minimum (61GB with full CC0 textures)
- GPU recommended for faster rendering

**⚠ Important Note on Textures:**
- Full CC0 texture pack is **56GB+** (much larger than expected!)
- **Recommendation:** Start without textures - they're optional
- Solid colors work well for most YOLO training tasks
- If needed, download minimal (~5GB) or medium (~15GB) subset

### Automated Installation

The `install.sh` script handles everything:

```bash
# Recommended: No textures (fastest, good for testing)
./install.sh --skip-textures

# Alternative: Minimal textures (~5GB)
./install.sh --textures minimal

# Alternative: Medium textures (~15GB)
./install.sh --textures medium
```

**What it does:**
1. Installs `uv` package manager (if needed)
2. Creates Python virtual environment (`.venv`)
3. Installs all dependencies (BlenderProc, NumPy, etc.)
4. Optionally downloads texture subset
5. Runs verification tests

**Options:**
```bash
./install.sh --skip-textures     # No textures (recommended!)
./install.sh --textures minimal  # ~5GB, 50 textures
./install.sh --textures medium   # ~15GB, 150 textures
./install.sh --python 3.11       # Use specific Python version
./install.sh --help              # Show all options
```

### Manual Installation

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download textures (optional)
blenderproc download cc_textures resources/cc_textures
```

### Verification

Test your installation:

```bash
source .venv/bin/activate
pytest tests/ -v
```

All 70 tests should pass ✅

---

## Basic Usage

### Project Structure

```
blenderproc/
├── models/               # Your 3D object models
│   ├── class1/
│   │   └── model.obj     # One .obj file per class
│   ├── class2/
│   │   └── model.obj
│   └── ...
├── configs/              # Configuration files
│   ├── example_config.json
│   └── example_obb_config.json
├── output/               # Generated datasets (created automatically)
└── src/                  # Source code
```

### Preparing 3D Models

**Requirements:**
- One directory per class in `models/`
- One `.obj` file per directory (any name, e.g., `model.obj`)
- Optional: `.mtl` file for materials
- Supported formats: Wavefront OBJ

**Example:**
```bash
models/
├── apple/
│   ├── apple.obj
│   └── apple.mtl
├── banana/
│   └── banana.obj
└── orange/
    └── orange.obj
```

**Class IDs:** Automatically assigned alphabetically:
- `apple` → ID 0
- `banana` → ID 1
- `orange` → ID 2

### Generating Your First Dataset

1. **Create/edit configuration** (start with `configs/example_config.json`):

```json
{
  "models_path": "./models",
  "camera": {
    "px": 600, "py": 600,
    "u0": 320, "v0": 240,
    "width": 640, "height": 480
  },
  "dataset": {
    "num_scenes": 100,
    "images_per_scene": 10
  },
  "output": {
    "save_path": "./output/my_dataset",
    "yolo_format": "yolov11"
  }
}
```

2. **Run generation:**

```bash
source .venv/bin/activate
blenderproc run src/generate_dataset.py --config configs/my_config.json
```

3. **Output structure:**

```
output/my_dataset/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
├── labels/
│   ├── train/          # Training annotations
│   ├── val/            # Validation annotations
│   └── test/           # Test annotations
├── data.yaml           # YOLO configuration
└── classes.txt         # Class names
```

---

## Configuration

### Configuration File Format

Supports both JSON and YAML:

**JSON:**
```json
{
  "models_path": "./models",
  "camera": { ... },
  "dataset": { ... },
  "output": { ... }
}
```

**YAML:**
```yaml
models_path: "./models"
camera:
  px: 600
  py: 600
dataset:
  num_scenes: 100
```

### Core Parameters

#### Camera Settings

```json
"camera": {
  "px": 600,              // Focal length X (pixels)
  "py": 600,              // Focal length Y (pixels)
  "u0": 320,              // Principal point X
  "v0": 240,              // Principal point Y
  "width": 640,           // Image width
  "height": 480           // Image height
}
```

#### Dataset Generation

```json
"dataset": {
  "num_scenes": 100,              // Number of scenes to generate
  "images_per_scene": 10,         // Images per scene
  "scenes_per_run": 1,            // Scenes per batch (for memory)
  "train_split": 0.7,             // Training set ratio
  "val_split": 0.2,               // Validation set ratio
  "test_split": 0.1,              // Test set ratio
  "empty_images_per_scene": 0     // Empty images (no objects)
}
```

**Note:** Splits must sum to 1.0

#### Output Settings

```json
"output": {
  "save_path": "./output/dataset",
  "yolo_format": "yolov11",        // or "yolov26", "yolov11-obb", "yolov26-obb"
  "save_depth": false,             // Save depth maps
  "save_normals": false,           // Save normal maps
  "save_segmentation": false,      // Save segmentation masks
  "save_pose": false,              // Save object poses
  "detection_params": {
    "min_bbox_side_px": 10,        // Minimum bbox size
    "min_visibility": 0.3,         // Minimum visibility ratio
    "occlusion_samples": 100       // Occlusion sampling density
  }
}
```

**YOLO Formats:**
- `yolov11`: Standard bounding boxes (class x y w h)
- `yolov26`: Same as v11
- `yolov11-obb`: Oriented bounding boxes (class x y w h angle)
- `yolov26-obb`: Same as v11-obb

### Scene Randomization

```json
"scene": {
  "room_size_multiplier_min": 2.0,
  "room_size_multiplier_max": 4.0,
  "floor_material_prob": 0.8,
  "use_physics": false,           // Enable physics simulation
  "distractors": {
    "min_count": 5,
    "max_count": 15,
    "min_size_rel_scene": 0.02,
    "max_size_rel_scene": 0.1,
    "emissive_prob": 0.1,
    "emissive_strength_min": 2.0,
    "emissive_strength_max": 5.0
  },
  "lights": {
    "min_count": 3,
    "max_count": 6,
    "min_intensity": 50,
    "max_intensity": 200
  },
  "objects": {
    "min_count": 1,               // Min objects per scene
    "max_count": 5,               // Max objects per scene
    "multiple_occurrences": true, // Allow same class multiple times
    "scale_noise": 0.2,           // Scale variation (±20%)
    "displacement_max": 0.0,      // Geometry displacement
    "pbr_noise": 0.3,             // PBR property variation
    "cam_min_dist_rel": 1.0,      // Min camera distance (relative)
    "cam_max_dist_rel": 3.0       // Max camera distance (relative)
  }
}
```

### Example Configurations

**Quick Test (Fast generation):**
```json
{
  "models_path": "./models",
  "camera": {
    "px": 600, "py": 600, "u0": 320, "v0": 240,
    "width": 640, "height": 480
  },
  "dataset": {
    "num_scenes": 10,
    "images_per_scene": 5
  },
  "output": {
    "save_path": "./output/test_dataset",
    "yolo_format": "yolov11"
  }
}
```

**Production (High quality):**
```json
{
  "models_path": "./models",
  "camera": {
    "px": 800, "py": 800, "u0": 512, "v0": 512,
    "width": 1024, "height": 1024
  },
  "dataset": {
    "num_scenes": 1000,
    "images_per_scene": 20
  },
  "scene": {
    "objects": {
      "min_count": 2,
      "max_count": 8,
      "scale_noise": 0.3
    },
    "distractors": {
      "min_count": 10,
      "max_count": 20
    }
  },
  "output": {
    "save_path": "./output/production_dataset",
    "yolo_format": "yolov11"
  }
}
```

**OBB (Oriented Bounding Boxes):**
```json
{
  "models_path": "./models",
  "camera": {
    "px": 600, "py": 600, "u0": 320, "v0": 240,
    "width": 640, "height": 480
  },
  "dataset": {
    "num_scenes": 200,
    "images_per_scene": 15
  },
  "scene": {
    "use_physics": true,          // Objects can rotate
    "objects": {
      "scale_noise": 0.4
    }
  },
  "output": {
    "save_path": "./output/obb_dataset",
    "yolo_format": "yolov11-obb"   // Enable oriented bounding boxes
  }
}
```

---

## Advanced Topics

### Oriented Bounding Boxes (OBB)

OBB format adds rotation angle to standard YOLO format:

**Standard YOLO:** `class x_center y_center width height`  
**OBB YOLO:** `class x_center y_center width height angle`

**When to use OBB:**
- Objects with significant rotation
- Elongated objects (bottles, pens, etc.)
- Dense scenes with overlapping objects
- Better localization precision needed

**Usage:**
```json
"output": {
  "yolo_format": "yolov11-obb"
}
```

**Training with OBB:**
```python
from ultralytics import YOLO

# Load OBB model
model = YOLO('yolov8n-obb.pt')

# Train
model.train(
    data='output/obb_dataset/data.yaml',
    epochs=100,
    imgsz=640
)
```

### CC0 Textures

CC0 textures improve realism:

**Benefits:**
- Photorealistic materials
- Better sim-to-real transfer
- Variety in scenes

**Usage:**
```json
"cc_textures_path": "./resources/cc_textures"
```

**Download (if not installed):**
```bash
blenderproc download cc_textures resources/cc_textures
```

### Physics Simulation

Enable realistic object placement:

```json
"scene": {
  "use_physics": true
}
```

**Note:** Slower rendering, but more realistic object interactions

### Batch Generation

For large datasets, use multiple runs:

```json
"dataset": {
  "num_scenes": 1000,
  "scenes_per_run": 10    // Generate in batches of 10
}
```

### Custom Materials

Add `.mtl` files alongside `.obj` files:

```
models/
├── object1/
│   ├── model.obj
│   └── model.mtl    # Material definition
```

---

## Troubleshooting

### Common Issues

**1. "No object classes found"**
```
Error: No object class directories found in ./models
```
**Solution:** Ensure each class has its own directory with at least one `.obj` file

**2. "Dataset splits must sum to 1.0"**
```
Error: Dataset splits must sum to 1.0, got 0.9
```
**Solution:** Adjust train/val/test splits to sum exactly to 1.0:
```json
"train_split": 0.7,
"val_split": 0.2,
"test_split": 0.1
```

**3. "BlenderProc not found"**
```
ModuleNotFoundError: No module named 'blenderproc'
```
**Solution:** Activate virtual environment:
```bash
source .venv/bin/activate
```

**4. Out of memory during rendering**
```
Error: CUDA out of memory
```
**Solution:** Reduce batch size or image resolution:
```json
"dataset": {
  "scenes_per_run": 1    // Reduce batch size
},
"camera": {
  "width": 512,          // Smaller resolution
  "height": 512
}
```

**5. Slow rendering**

**Solutions:**
- Use GPU acceleration (automatic with CUDA)
- Reduce `images_per_scene`
- Disable physics: `"use_physics": false`
- Reduce distractors: `"max_count": 5`

### Performance Tips

1. **Start small:** Test with 10 scenes before generating 1000
2. **Use GPU:** NVIDIA GPU with CUDA significantly faster
3. **Optimize models:** Simplify 3D meshes for faster rendering
4. **Disable extras:** Turn off depth/normals/segmentation if not needed
5. **Parallel processing:** Run multiple instances on different configs

### Debugging

**Enable verbose logging:**
```bash
blenderproc run src/generate_dataset.py --config config.json --verbose
```

**Check configuration validity:**
```bash
python src/config_parser.py configs/my_config.json
```

**Validate dataset:**
```bash
python src/dataset_manager.py --validate output/my_dataset
```

---

## FAQ

**Q: How long does generation take?**  
A: Depends on hardware and settings. Typical: ~10-30 seconds per image with GPU, 1-3 minutes per image with CPU.

**Q: What file formats are supported?**  
A: Currently .obj files. Support for .fbx, .dae planned.

**Q: Can I use my own backgrounds?**  
A: Yes, modify scene generation in `src/scene_generator.py` or use HDRI environments.

**Q: How accurate are the annotations?**  
A: Very accurate for visible objects. Occlusion handling may need tuning via `min_visibility` parameter.

**Q: Can I train YOLOv8/v9/v10 with this?**  
A: Yes! All modern YOLO versions use the same format. Just use the generated `data.yaml`.

**Q: How do I add more variety?**  
A: Increase `scale_noise`, add more distractors, enable CC0 textures, increase light variation.

**Q: Can I generate only specific classes?**  
A: Not directly, but you can create a config with only those class directories in `models_path`.

**Q: What about data augmentation?**  
A: BlenderProc generates diverse scenes. Additional augmentation (flips, crops, color jitter) can be done during training.

**Q: How many images do I need?**  
A: Typically 500-5000 per class, depending on complexity and target accuracy.

**Q: Can I use this commercially?**  
A: Yes, the framework is open source. Check individual 3D model licenses.

---

## Next Steps

- Read [Configuration Reference](CONFIGURATION.md) for all parameters
- See [API Documentation](API.md) for extending the framework
- Check [Examples](../examples/) for more use cases
- Visit [AKAMAV Wiki](WIKI.md) for best practices

**Need help?** Contact your AKAMAV supervisor or check the troubleshooting section above.
