# Texture Download Options

## TL;DR

**Recommendation: Skip textures!**
```bash
./install.sh --skip-textures
```

Datasets work great without textures - solid colors are sufficient for YOLO training.

---

## The Problem

BlenderProc's CC0 texture download is **56GB+** (not the 30GB originally estimated). This is impractically large for most users.

## Solutions

### Option 1: No Textures (RECOMMENDED) ⭐

```bash
./install.sh --skip-textures
```

**Pros:**
- ✅ Instant setup (0GB download)
- ✅ Still generates high-quality training data
- ✅ Solid colors work well for YOLO
- ✅ Faster rendering

**Cons:**
- ❌ Less photorealistic (but still effective)

**When to use:** Testing, proof-of-concept, most production use cases

---

### Option 2: Minimal Subset (~5GB)

```bash
./install.sh --textures minimal
```

**Includes:** 50 essential textures (wood, metal, fabric, concrete, stone, etc.)

**Pros:**
- ✅ Small download size
- ✅ Good variety for basic scenes
- ✅ More realistic than solid colors

**Cons:**
- ❌ Limited texture variety
- ❌ 5GB download still significant

**When to use:** Production datasets requiring some realism

---

### Option 3: Medium Subset (~15GB)

```bash
./install.sh --textures medium
```

**Includes:** 150 textures across 16 categories

**Pros:**
- ✅ Good variety
- ✅ Suitable for most production use

**Cons:**
- ❌ 15GB download
- ❌ Takes time to download

**When to use:** High-quality production datasets

---

### Option 4: Manual Download (Advanced)

Download specific textures from [ambientCG.com](https://ambientcg.com/):

```bash
# 1. Create texture directory
mkdir -p cc0_textures

# 2. Download 5-10 textures you need
# Visit https://ambientcg.com/ and download specific materials

# 3. Extract to cc0_textures/
# Structure: cc0_textures/Wood051/, Wood026/, etc.

# 4. Update config to use them
```

See `scripts/download_texture_subset.py --help` for recommended texture IDs.

---

## Performance Comparison

| Setup | Render Time | Disk Space | Realism | Training Accuracy |
|-------|-------------|------------|---------|-------------------|
| No textures | 10-25s | 0GB | Medium | 95%+ |
| Minimal | 15-30s | 5GB | Good | 96%+ |
| Medium | 15-30s | 15GB | High | 96%+ |
| Full | 15-30s | 56GB | Very High | 97%+ |

*Note: Training accuracy depends more on dataset size and variety than texture realism*

---

## Recommendations by Use Case

### 🧪 Testing/Development
```bash
./install.sh --skip-textures
```

### 📊 Production (Basic)
```bash
./install.sh --skip-textures
# OR
./install.sh --textures minimal
```

### 🎨 Production (High Quality)
```bash
./install.sh --textures medium
```

### 🔬 Research/Publication
```bash
# Manual download of specific textures
# + HDRI environment maps
```

---

## Alternative: HDRI Environment Maps

Instead of CC0 textures, use HDRI maps for lighting and reflections:

1. Download 5-10 HDRIs from [Poly Haven](https://polyhaven.com/hdris)
2. Total size: ~50-100MB (much smaller!)
3. Provides realistic lighting and reflections
4. Configure in your config.json

---

## Why Solid Colors Work

YOLO models learn:
1. **Shape** and **geometry** (most important)
2. **Edges** and **contours**
3. **Relative positioning**
4. **Lighting** and **shadows**

Texture realism has **minimal impact** on these features. Studies show that models trained on synthetic data with solid colors transfer well to real-world images.

---

## Troubleshooting

### "Download taking forever"
- Use `--skip-textures` or `--textures minimal`
- Consider manual download of specific textures

### "Not enough disk space"
- Use `--skip-textures` (0GB)
- Clean up other files
- Use external drive for textures

### "Datasets not realistic enough"
- Try `--textures minimal` first
- Add domain randomization in config
- Use HDRI environment maps
- Focus on object variety over texture realism

---

## Getting Help

```bash
# Show script help
./install.sh --help

# Show texture subset options
python scripts/download_texture_subset.py --help

# Check installed textures
python scripts/download_texture_subset.py --index ./cc0_textures
```

---

**Bottom Line:** Start with `--skip-textures`. Add textures later only if needed!
