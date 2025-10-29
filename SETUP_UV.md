# UV Environment Setup Guide

This guide shows how to set up the Ditto TalkingHead project using UV package manager with CUDA 12.8.

## Prerequisites

- **Python 3.10** (NOT 3.11 or higher)
- **CUDA 12.8** (or CUDA 12.x - PyTorch CUDA 12.1 builds are forward compatible)
- **UV package manager** installed

## Why UV?

UV is a fast, reliable Python package manager that:
- ✅ Much faster than pip/conda
- ✅ Better dependency resolution
- ✅ Simpler environment management
- ✅ No conda environment issues

## Installation Steps

### Step 1: Install UV

If you don't have UV installed:

```bash
# On Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv

# Or using pipx
pipx install uv
```

### Step 2: Verify Python Version

Ensure you have Python 3.10:

```bash
python3.10 --version
```

If you don't have Python 3.10, install it:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# Or use pyenv
pyenv install 3.10.13
pyenv local 3.10.13
```

### Step 3: Create UV Environment

Navigate to the project directory and create the environment:

```bash
cd /path/to/ditto-talkinghead

# Create virtual environment with Python 3.10
uv venv --python 3.10

# Activate the environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

### Step 4: Install Dependencies

Install all project dependencies using UV:

```bash
# Install from pyproject.toml
uv pip install -e .
```

This will:
- Install PyTorch 2.5.1 with CUDA 12.1 support (compatible with CUDA 12.8)
- Install all audio/video processing libraries
- Install TensorRT 8.6.1
- Install all scientific computing dependencies

**Note**: The installation may take 5-10 minutes depending on your internet speed.

### Step 5: Verify Installation

Run the verification script:

```bash
python verify_installation.py
```

You should see:
```
✓ Python version: 3.10.x
✓ PyTorch installed: 2.5.1
✓ CUDA available: True
✓ CUDA version: 12.8
✓ TensorRT available: True
✓ All core dependencies installed
```

## CUDA Compatibility Notes

### CUDA 12.8 with PyTorch CUDA 12.1

The project uses PyTorch built with CUDA 12.1, which is **fully compatible** with CUDA 12.8:

- ✅ PyTorch CUDA 12.1 builds work with CUDA 12.0-12.8
- ✅ Forward compatibility within CUDA 12.x series
- ✅ No need to match exact CUDA version

### Checking Your CUDA Version

```bash
# Check NVIDIA driver and CUDA version
nvidia-smi

# Should show CUDA Version: 12.8 in top-right
```

### If You Have Different CUDA Version

If you have CUDA 11.x:

1. Edit `pyproject.toml`:
   ```toml
   [tool.uv]
   index-url = "https://download.pytorch.org/whl/cu118"  # Change to cu118
   ```

2. Update dependencies:
   ```toml
   dependencies = [
       "torch==2.5.1",
       "torchvision==0.20.1",
       "torchaudio==2.5.1",
       # Update CUDA packages to cu11 versions
   ]
   ```

3. Reinstall:
   ```bash
   uv pip install -e . --reinstall
   ```

## Quick Start After Installation

### 1. Download Models

```bash
# Download PyTorch models (recommended for first-time users)
wget https://example.com/ditto_pytorch.zip
unzip ditto_pytorch.zip -d checkpoints/

# Download config files
wget https://example.com/ditto_cfg.zip
unzip ditto_cfg.zip -d checkpoints/
```

### 2. Run Basic Inference

```bash
# Test with image source
python inference.py \
    --source_path examples/image.jpg \
    --audio_path examples/audio.wav \
    --output_path result.mp4

# Test with video source (recommended)
python inference_video.py \
    --source_video examples/avatar.mp4 \
    --audio examples/audio.wav \
    --output result.mp4
```

### 3. Verify Output

Check that `result.mp4` was created and plays correctly:

```bash
# Play with ffplay
ffplay result.mp4

# Check video info
ffprobe result.mp4
```

## Troubleshooting

### Issue 1: UV Command Not Found

```bash
# Add UV to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Or reinstall UV
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Issue 2: Python 3.10 Not Found

```bash
# Install Python 3.10 using pyenv
curl https://pyenv.run | bash
pyenv install 3.10.13
pyenv local 3.10.13

# Or use system package manager
sudo apt install python3.10 python3.10-venv
```

### Issue 3: CUDA Not Available

Check CUDA installation:

```bash
# Check CUDA version
nvidia-smi

# Check LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# Add CUDA to path if needed
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.8/bin:$PATH
```

### Issue 4: PyTorch Not Using GPU

```bash
# Test PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

If False:
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch: `uv pip install torch torchvision torchaudio --reinstall`
3. Verify CUDA toolkit installed: `nvcc --version`

### Issue 5: TensorRT Import Error

```bash
# TensorRT requires specific version
uv pip install tensorrt==8.6.1 tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1

# Verify installation
python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
```

### Issue 6: librosa/soundfile Errors

```bash
# Install system audio libraries (Ubuntu/Debian)
sudo apt-get install libsndfile1 ffmpeg

# Reinstall audio packages
uv pip install librosa soundfile audioread --reinstall
```

### Issue 7: opencv-python Issues

```bash
# Install system dependencies
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# Use headless version (included in pyproject.toml)
uv pip install opencv-python-headless
```

## Performance Tips

### 1. Use TensorRT for Faster Inference

TensorRT provides 2-3x speedup:

```bash
# Download TensorRT models
wget https://example.com/ditto_trt_Ampere_Plus.zip
unzip ditto_trt_Ampere_Plus.zip -d checkpoints/

# Use TensorRT config
python inference_video.py \
    --data_root ./checkpoints/ditto_trt_Ampere_Plus \
    --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl \
    --source_video avatar.mp4 \
    --audio audio.wav \
    --output result.mp4
```

### 2. Optimize Parameters

For faster processing:

```bash
python inference_video.py \
    --source_video avatar.mp4 \
    --audio audio.wav \
    --output result.mp4 \
    --sampling_timesteps 25 \
    --overlap_v2 5 \
    --max_size 1024
```

### 3. Use Online Mode for Real-time

```bash
python inference_online_example.py \
    --source image.png \
    --audio audio.wav \
    --output result.mp4 \
    --chunksize 3 5 2
```

## Environment Management

### Activating Environment

```bash
# Always activate before running scripts
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Deactivating Environment

```bash
deactivate
```

### Updating Dependencies

```bash
# Update all packages
uv pip install -e . --upgrade

# Update specific package
uv pip install torch --upgrade
```

### Removing Environment

```bash
# Simply delete the .venv directory
rm -rf .venv
```

### Creating New Environment

```bash
# If you need to start fresh
rm -rf .venv
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

## Comparing with Conda

| Feature | Conda | UV |
|---------|-------|-----|
| Installation Speed | 5-15 min | 2-5 min |
| Dependency Resolution | Slow | Fast |
| Disk Space | ~5 GB | ~2 GB |
| Environment Activation | conda activate | source .venv/bin/activate |
| Updates | conda update | uv pip install --upgrade |
| Compatibility | Sometimes conflicts | Clean resolution |

## Next Steps

After successful installation:

1. **Read Documentation**
   - `personal_docs/README.md` - Start here
   - `personal_docs/VIDEO_INFERENCE_GUIDE.md` - Video sources
   - `personal_docs/BLENDING_CONTROL_GUIDE.md` - Blending modes

2. **Try Examples**
   - Basic: `inference.py`
   - Video: `inference_video.py`
   - Advanced: `inference_video_advanced.py`

3. **Optimize**
   - Switch to TensorRT for speed
   - Adjust quality parameters
   - Try different blend modes

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify installation: `python verify_installation.py`
3. Read relevant documentation in `personal_docs/`
4. Check CUDA compatibility
5. Ensure Python 3.10 is being used

## Summary

```bash
# Complete setup in 5 commands:
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install UV
cd /path/to/ditto-talkinghead                    # Navigate
uv venv --python 3.10                            # Create env
source .venv/bin/activate                        # Activate
uv pip install -e .                              # Install deps
```

That's it! Your environment is ready to use. 🎉

For detailed usage, see `personal_docs/README.md`.
