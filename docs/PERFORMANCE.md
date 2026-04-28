# Performance Optimization Guide

## Overview

This guide provides strategies for optimizing the Blender YOLO Dataset Generator for maximum throughput and efficiency.

---

## Performance Metrics

### Typical Performance (640x480 images)

| Configuration | GPU (NVIDIA RTX) | CPU (8-core) |
|--------------|------------------|--------------|
| Basic scene (1-3 objects) | 10-15 sec/image | 60-120 sec/image |
| Medium scene (3-5 objects) | 15-25 sec/image | 120-180 sec/image |
| Complex scene (5-8 objects) | 25-40 sec/image | 180-300 sec/image |
| With physics simulation | +50% time | +100% time |
| With CC0 textures | +10% time | +15% time |

### Scaling

- **Resolution:** ~O(n²) - doubling resolution = 4x rendering time
- **Objects per scene:** ~O(n) - linear scaling
- **Distractors:** ~O(n) - but minimal impact with instancing

---

## Quick Wins

### 1. Enable GPU Acceleration ⚡

BlenderProc automatically uses GPU if available. Speedup: **5-10x**

Check GPU:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Optimize Image Resolution

```json
{
  "camera": {
    "width": 512,
    "height": 512
  }
}
```

Speedup for 512x512 vs 1024x1024: **4x**

### 3. Reduce Scene Complexity

```json
{
  "scene": {
    "objects": {"max_count": 3},
    "distractors": {"max_count": 8}
  }
}
```

Speedup: **2-3x**

### 4. Disable Physics

```json
{
  "scene": {
    "use_physics": false
  }
}
```

Speedup: **1.5-2x**

---

## Advanced Optimizations

### Parallel Generation

Run multiple instances:

```bash
python src/generate_dataset.py --config config1.json &
python src/generate_dataset.py --config config2.json &
wait
```

Use different output paths!

### Model Optimization

- Reduce polygon count (<10k polygons)
- Optimize textures (max 2048x2048)
- Remove unnecessary details

---

## Hardware Recommendations

### Minimum
- CPU: 4-core, 3.0+ GHz
- RAM: 16GB
- GPU: NVIDIA GTX 1060 (6GB)
- Storage: 100GB SSD

### Recommended
- CPU: 8-core, 3.5+ GHz
- RAM: 32GB
- GPU: NVIDIA RTX 3060 (12GB)
- Storage: 500GB NVMe SSD

---

## Optimization Checklist

Before Generation:
- [ ] GPU drivers updated
- [ ] CUDA available
- [ ] Models optimized
- [ ] Config validated

During Generation:
- [ ] Monitor GPU usage
- [ ] Check memory
- [ ] Verify output quality

---

For more details, see the User Guide and API Documentation.
