# AKAMAV Wiki: Blender YOLO Dataset Generator

## Complete Setup and Usage Guide

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start (5 Minutes)](#quick-start)
3. [Installation](#installation)
4. [Preparing 3D Models](#preparing-3d-models)
5. [Basic Usage](#basic-usage)
6. [Configuration Guide](#configuration-guide)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Performance Tuning](#performance-tuning)

---

## Overview

The Blender YOLO Dataset Generator automates the creation of synthetic training datasets for YOLO object detection models. It uses Blender's powerful 3D rendering engine via BlenderProc to generate photorealistic images with accurate annotations.

### Key Features

- ✅ **Automated Generation:** One command creates complete YOLO datasets
- ✅ **4 YOLO Formats:** YOLOv11, YOLOv26, YOLOv11-OBB, YOLOv26-OBB
- ✅ **Oriented Bounding Boxes (OBB):** For rotated objects
- ✅ **Fully Parameterizable:** JSON/YAML configuration
- ✅ **One-Command Installation:** `./install.sh`
- ✅ **Production Ready:** Tested with 70 automated tests

### When to Use

**✅ Good for:**
- Creating large training datasets quickly
- Objects with clear 3D models available
- Scenarios where real data is expensive/dangerous to collect
- Testing YOLO models before real-world deployment
- Augmenting limited real-world datasets

**❌ Not ideal for:**
- Objects with complex textures hard to model
- Scenarios requiring perfect photo-realism
- When real data is abundant and easy to annotate

---

## Quick Start

### 5-Minute Setup

```bash
# 1. Navigate to project
cd /home/paul/akamav/dev/projects/blenderproc

# 2. Run installation
./install.sh

# 3. Activate environment
source .venv/bin/activate

# 4. Add your 3D models
mkdir -p models/object1 models/object2
cp /path/to/model1.obj models/object1/
cp /path/to/model2.obj models/object2/

# 5. Generate dataset
python src/generate_dataset.py --config configs/example_config.json

# Done! Dataset is in output/yolo_dataset/
```

---

## Installation

### Automated Installation (Recommended)

```bash
./install.sh
```

This installs:
- `uv` package manager
- Python virtual environment
- BlenderProc and Blender
- All dependencies
- CC0 textures (optional, ~30GB)

### Installation Options

```bash
./install.sh --skip-textures    # Skip texture download
./install.sh --python 3.11      # Use specific Python version
./install.sh --help            # Show all options
```

### Manual Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Verify Installation

```bash
source .venv/bin/activate
pytest tests/ -v
```

Expected: **70 tests passing** ✅

---

## Preparing 3D Models

### Directory Structure

```
models/
├── class1/           # One directory per class
│   └── model.obj     # One .obj file per directory
├── class2/
│   ├── model.obj
│   └── model.mtl     # Optional material file
└── class3/
    └── model.obj
```

### Requirements

- **Format:** Wavefront OBJ (.obj)
- **One model per class:** One .obj file in each subdirectory
- **Naming:** Directory name becomes class name
- **Materials:** Optional .mtl file for textures

### Class ID Assignment

Classes are assigned IDs **alphabetically by directory name**:

```
models/
├── apple/    → ID 0
├── banana/   → ID 1
└── orange/   → ID 2
```

### Finding 3D Models

**Free Sources:**
- [Poly Haven](https://polyhaven.com/) - High quality, CC0
- [Sketchfab](https://sketchfab.com/) - Check licenses
- [TurboSquid](https://www.turbosquid.com/) - Freebies section
- [CGTrader](https://www.cgtrader.com/) - Free models

**Creating Your Own:**
- Blender (modeling)
- Photogrammetry (RealityCapture, Meshroom)
- CAD software (Fusion 360, FreeCAD)

---

## Basic Usage

### Generate Your First Dataset

1. **Create configuration:**

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
    "images_per_scene": 10,
    "train_split": 0.7,
    "val_split": 0.2,
    "test_split": 0.1
  },
  "output": {
    "save_path": "./output/my_dataset",
    "yolo_format": "yolov11"
  }
}
```

2. **Run generation:**

```bash
python src/generate_dataset.py --config my_config.json
```

3. **Output structure:**

```
output/my_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml
└── classes.txt
```

### Training YOLO with Generated Data

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train
model.train(
    data='output/my_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

---

## Configuration Guide

### Minimal Configuration

```json
{
  "models_path": "./models",
  "camera": {
    "px": 600, "py": 600, "u0": 320, "v0": 240,
    "width": 640, "height": 480
  },
  "dataset": {"num_scenes": 10, "images_per_scene": 5},
  "output": {"save_path": "./output", "yolo_format": "yolov11"}
}
```

### Full Configuration with Defaults

See `configs/example_config.json` for a complete example with all available parameters.

### Key Parameters

**Camera (Required):**
- `px, py`: Focal length in pixels
- `u0, v0`: Principal point (usually image center)
- `width, height`: Image resolution

**Dataset (Required):**
- `num_scenes`: Total scenes to generate
- `images_per_scene`: Images per scene (different camera angles)
- `train_split, val_split, test_split`: Must sum to 1.0

**Output (Required):**
- `save_path`: Where to save dataset
- `yolo_format`: `yolov11`, `yolov26`, `yolov11-obb`, `yolov26-obb`

---

## Advanced Features

### Oriented Bounding Boxes (OBB)

For objects that rotate significantly:

```json
{
  "output": {
    "yolo_format": "yolov11-obb"
  },
  "scene": {
    "use_physics": true  // Enables rotation
  }
}
```

OBB format: `class x_center y_center width height angle`

### Scene Randomization

```json
{
  "scene": {
    "room_size_multiplier_min": 2.0,
    "room_size_multiplier_max": 4.0,
    "objects": {
      "min_count": 1,
      "max_count": 5,
      "scale_noise": 0.2,     // ±20% size variation
      "multiple_occurrences": true
    },
    "distractors": {
      "min_count": 5,
      "max_count": 15
    },
    "lights": {
      "min_count": 3,
      "max_count": 6,
      "min_intensity": 50,
      "max_intensity": 200
    }
  }
}
```

### CC0 Textures

For photorealistic materials:

```json
{
  "cc_textures_path": "./resources/cc_textures"
}
```

Download: `blenderproc download cc_textures resources/cc_textures`

---

## Best Practices

### 1. Start Small, Scale Up

```
Test: 10 scenes × 5 images = 50 images
Validate: 100 scenes × 10 images = 1000 images
Production: 1000 scenes × 20 images = 20000 images
```

### 2. Balance Dataset

- Similar number of images per class
- Varied scenes (lighting, angles, distances)
- Mix of simple and complex scenes

### 3. Quality Over Quantity

Better to have 1000 diverse, high-quality images than 10000 similar ones.

### 4. Validate Before Training

```bash
python src/dataset_manager.py --validate output/dataset
```

### 5. Iterate Based on Results

1. Generate small dataset
2. Train YOLO model
3. Test on real images
4. Adjust parameters based on performance
5. Regenerate with improvements

### 6. Monitor Generation

- Check first few images visually
- Verify annotations are correct
- Monitor render times
- Check for errors in logs

---

## Troubleshooting

### Common Issues

**1. "No object classes found"**

```bash
# Check directory structure
ls -la models/
# Each class should be a directory with .obj file
```

**2. "CUDA out of memory"**

```json
{
  "camera": {"width": 512, "height": 512},  // Reduce resolution
  "dataset": {"scenes_per_run": 1}          // Reduce batch size
}
```

**3. "Splits must sum to 1.0"**

```json
{
  "dataset": {
    "train_split": 0.7,
    "val_split": 0.2,
    "test_split": 0.1  // Total = 1.0
  }
}
```

**4. Slow rendering**

- Enable GPU acceleration (automatic with CUDA)
- Reduce image resolution
- Disable physics simulation
- Reduce object/distractor count

### Getting Help

1. Check logs: `output/generation.log`
2. Run tests: `pytest tests/ -v`
3. Validate config: `python src/config_parser.py config.json`
4. Contact AKAMAV supervisor

---

## Performance Tuning

### Quick Performance Tips

| Optimization | Speedup | Quality Impact |
|--------------|---------|----------------|
| Enable GPU | 5-10x | None |
| 512x512 vs 1024x1024 | 4x | Moderate |
| Disable physics | 1.5-2x | Low |
| Max 3 objects vs 8 | 2-3x | Moderate |

### Recommended Configurations

**Fast Prototyping:**
```json
{
  "camera": {"width": 320, "height": 320},
  "scene": {
    "use_physics": false,
    "objects": {"max_count": 2}
  }
}
```

**Production:**
```json
{
  "camera": {"width": 640, "height": 480},
  "scene": {
    "use_physics": false,
    "objects": {"max_count": 5}
  }
}
```

**High Quality:**
```json
{
  "camera": {"width": 1024, "height": 1024},
  "scene": {
    "use_physics": true,
    "objects": {"max_count": 8}
  }
}
```

---

## Additional Resources

- **User Guide:** Complete usage documentation
- **API Documentation:** Programmatic access
- **Testing Guide:** Test suite information
- **Performance Guide:** Detailed optimization strategies

---

**Last Updated:** 2026-04-06  
**Version:** 1.0.0  
**Maintainer:** AKAMAV Team
