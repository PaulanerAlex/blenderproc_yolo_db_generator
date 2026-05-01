# Blender Framework for YOLO Training Dataset Generation

A powerful framework for generating high-quality synthetic training datasets for YOLO object detection using Blender and BlenderProc.

## Features

- 🎨 **Photorealistic Rendering**: Leverages Blender's ray-tracing for realistic synthetic images
- 🔄 **Full Parameterization**: Configure scene, lighting, camera, and object properties via JSON/YAML
- 📦 **Multiple YOLO Formats**: Supports YOLOv11, YOLOv26, YOLOv11-OBB, and YOLOv26-OBB
- 🎯 **Oriented Bounding Boxes**: Generate OBB annotations with rotation angles
- 🚀 **Automated Pipeline**: Batch generation with train/val/test splitting
- ⚡ **One-Command Setup**: Automated installation script for quick start

## Supported YOLO Formats

1. **YOLOv11** - Standard bounding boxes
2. **YOLOv26** - Standard bounding boxes
3. **YOLOv11-OBB** - Oriented bounding boxes with rotation
4. **YOLOv26-OBB** - Oriented bounding boxes with rotation

## Quick Start

### Installation

Simply run the installation script:

```bash
# Recommended: No textures (fastest, works great!)
./install.sh --skip-textures

# Alternative: Minimal textures (~5GB)
./install.sh --textures minimal
```

The script will:
- ✅ Install uv package manager if needed
- ✅ Create Python 3.10 virtual environment (.venv)
- ✅ Install BlenderProc 2.5.0+ and all dependencies
- ✅ Optionally download texture subset
- ✅ Verify the installation

**⚠ Important:** Full CC0 textures are **56GB+**! 
- Recommended: Use `--skip-textures` (solid colors work fine)
- See [TEXTURE_OPTIONS.md](docs/TEXTURE_OPTIONS.md) for full details

#### Installation Options

```bash
# No textures (recommended!)
./install.sh --skip-textures

# Minimal textures (~5GB)
./install.sh --textures minimal

# Medium textures (~15GB) 
./install.sh --textures medium

# Show help
./install.sh --help
```

### Basic Usage

1. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Prepare your 3D models**:
   Place your .obj models in the `models/` directory:
   ```
   models/
   ├── object_class_1/
   │   ├── model.obj
   │   ├── model.mtl
   │   └── texture.png
   └── object_class_2/
       └── model.obj
   ```

3. **Configure generation**:
   Edit or create a config file in `configs/`:
   ```bash
   cp configs/example_config.json configs/my_config.json
   # Edit my_config.json with your parameters
   ```

4. **Generate dataset**:
   ```bash
   blenderproc run src/generate_dataset.py --config configs/my_config.json
   ```

5. **Output**:
   Your YOLO-ready dataset will be in `output/`:
   ```
   output/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   ├── labels/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── data.yaml
   ```

## Project Structure

```
blenderproc/
├── install.sh              # Automated installation script
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── src/                   # Source code
├── scripts/               # Utility tools and helper scripts
│   └── visualize_model.py     # 3D model orientation visualizer
├── configs/               # Configuration files
├── models/                # 3D models (user-provided)
├── output/                # Generated datasets
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Utility Tools

### 3D Model Visualizer

Before generating a dataset, it is often useful to check the default orientation of your `.obj` models to determine if they need initial rotation correction.

```bash
blenderproc run scripts/visualize_model.py models/your_model/model.obj
```

This tool:
- Renders the model from 4 standard views: **ISO**, **Front**, **Side**, and **Top**.
- Draws colored axis indicators: **Red (+X)**, **Green (+Y)**, **Blue (+Z)**.
- Saves the results to the `model_visualizations/` directory.

Use these images to determine the correct values for the `initial_rotation` property in your configuration.

## Configuration

All generation parameters are controlled via JSON/YAML configuration files. Key sections:

- **scene**: Room size, materials, physics simulation
- **objects**: Count, scale, color/texture variations
- **lighting**: Number, type, intensity, color
- **camera**: Intrinsics, position constraints
- **rendering**: Quality, resolution, denoiser
- **dataset**: Number of images, splits, output format

See `configs/example_config.json` for a complete example with documentation.

## Requirements

- Linux (tested on Ubuntu 20.04+)
- conda or miniconda
- Python 3.10
- ~5GB disk space (or ~35GB with CC0 textures)
- GPU with CUDA support recommended (for faster rendering)

## Documentation

- [Installation Guide](docs/installation.md) - Detailed installation instructions
- [Configuration Reference](docs/configuration.md) - Complete configuration options
- [Tutorial](docs/tutorial.md) - Step-by-step guide
- [API Documentation](docs/api.md) - Code documentation
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## Resources

- [BlenderProc GitHub](https://github.com/DLR-RM/BlenderProc)
- [BlenderProc Documentation](https://dlr-rm.github.io/BlenderProc/)
- [ViSP Tutorial](https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-synthetic-blenderproc.html)
- [YOLO Documentation](https://docs.ultralytics.com/)

## License

[Add your license here]

## Citation

If you use this framework in your research, please cite:

```bibtex
[Add citation information]
```

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## Contact

For questions or support, contact: [Add contact information]
