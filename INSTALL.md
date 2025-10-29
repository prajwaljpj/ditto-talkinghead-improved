# One-Click Installation Guide

## Quick Start (3 Commands)

```bash
# 1. Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync all dependencies (creates venv automatically)
uv sync

# 3. Verify installation
uv run python verify_installation.py
```

That's it! Everything is installed and ready to use.

**Note**: The default installation includes TensorRT 10.13.3.9 for CUDA 12.x. If you have pre-built TensorRT models from the original repository (built with TensorRT 8.6.1), you'll need to rebuild them. See [TensorRT Setup](#tensorrt-setup) below.

## What `uv sync` Does

When you run `uv sync`:
1. ✅ Creates a virtual environment with Python 3.10 automatically
2. ✅ Installs PyTorch 2.5.1 with CUDA 12.1
3. ✅ Installs TensorRT 10.13.3.9 for CUDA 12.x
4. ✅ Installs all audio/video processing libraries
5. ✅ Installs all dependencies from `pyproject.toml`

**No manual steps needed!**

## Usage

### Run commands with UV

```bash
# Run inference directly
uv run python inference_video.py --source_video avatar.mp4 --audio audio.wav --output result.mp4

# Run verification
uv run python verify_installation.py

# Run any script
uv run python inference_video_advanced.py --help
```

### Or activate the environment

```bash
# Activate
source .venv/bin/activate

# Now run commands normally
python inference_video.py --source_video avatar.mp4 --audio audio.wav --output result.mp4

# Deactivate when done
deactivate
```

## What Gets Installed

All packages from `pyproject.toml`:
- Python 3.10.x (managed by UV)
- PyTorch 2.5.1+cu121
- TensorRT 10.13.3.9 (cu12)
- librosa, soundfile, audioread
- opencv-python-headless
- scipy, scikit-learn, numba
- imageio, pillow
- And all other dependencies

## Verification

After installation, verify everything works:

```bash
uv run python verify_installation.py
```

You should see:
```
✓ ALL REQUIRED CHECKS PASSED!
✓ Your environment is ready to use.
```

## System Requirements

- **OS**: Linux, macOS, or Windows
- **CUDA**: 12.x or 11.x (for GPU acceleration)
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **Disk**: 10GB+ free space
- **RAM**: 16GB+ recommended

**Note**: You don't need Python 3.10 pre-installed. UV will download and manage it for you!

## Troubleshooting

### Issue: UV not found

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

### Issue: CUDA not available

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Issue: TensorRT not working

```bash
# Check TensorRT
uv run python -c "import tensorrt; print(f'TensorRT: {tensorrt.__version__}')"

# If fails, reinstall
uv pip install tensorrt-cu12==10.13.3.9 --index-url https://pypi.nvidia.com
```

### Issue: Fresh start needed

```bash
# Remove environment and reinstall
rm -rf .venv
uv sync
```

## Next Steps

After installation:

1. **Download Models**
   ```bash
   git lfs install
   git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
   ```

2. **Run Inference**
   ```bash
   uv run python inference_video.py \
       --source_video avatar.mp4 \
       --audio audio.wav \
       --output result.mp4
   ```

3. **Read Documentation**
   - `personal_docs/README.md` - Complete guide
   - `personal_docs/VIDEO_INFERENCE_GUIDE.md` - Video inference
   - `personal_docs/BLENDING_CONTROL_GUIDE.md` - Blending control

## Advanced Usage

### Update dependencies

```bash
uv sync --upgrade
```

### Install development tools

```bash
uv sync --extra dev
```

### Add new package

```bash
uv add package-name
```

## Comparison: UV vs Other Methods

| Method | Setup Time | Commands | Automatic |
|--------|-----------|----------|-----------|
| **UV** | **2-5 min** | **1 command** | **✓ Yes** |
| Conda | 10-20 min | 2+ commands | Partial |
| Pip | 5-10 min | 5+ commands | ✗ No |

## TensorRT Setup

### Understanding TensorRT Versions

This project now uses **TensorRT 10.13.3.9** for CUDA 12.x compatibility. Pre-built models from HuggingFace were built with TensorRT 8.6.1 and are **incompatible**.

### Option 1: Use PyTorch Models (Easiest)

```bash
uv run python inference.py \
    --data_root ./checkpoints/ditto_pytorch \
    --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl \
    --source_video avatar.mp4 \
    --audio audio.wav \
    --output result.mp4
```

PyTorch models work perfectly and require no TensorRT setup!

### Option 2: Build TensorRT 10 Models

If you need TensorRT for faster inference:

#### Step 1: Build GridSample3D Plugin

```bash
# Download TensorRT 10.x from NVIDIA
# https://developer.nvidia.com/tensorrt-download
# Extract to: TensorRT-10.13.3.9/

cd grid-sample3d-trt-plugin
rm -rf build && mkdir build && cd build

# Configure
cmake .. -DTensorRT_ROOT=../TensorRT-10.13.3.9 -DCMAKE_CUDA_ARCHITECTURES="89"

# Build
make -j$(nproc)

# Copy to checkpoints
cp libgrid_sample_3d_plugin.so ../../checkpoints/ditto_onnx/
```

#### Step 2: Convert ONNX to TensorRT

```bash
cd ../..
source .venv/bin/activate

python scripts/cvt_onnx_to_trt.py \
    --onnx_dir "./checkpoints/ditto_onnx" \
    --trt_dir "./checkpoints/ditto_trt_custom"
```

This will take ~20 minutes and create TensorRT 10 engines.

#### Step 3: Run Inference with TensorRT

```bash
python inference.py \
    --data_root ./checkpoints/ditto_trt_custom \
    --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl \
    --source_video avatar.mp4 \
    --audio audio.wav \
    --output result.mp4
```

### Performance Comparison

| Backend | Speed | Setup | Recommended |
|---------|-------|-------|-------------|
| PyTorch | 1x | ✅ Easy | ✓ Yes (default) |
| TensorRT | ~2.5x | ❌ Complex | Only if needed |

## Summary

**One command to rule them all:**
```bash
uv sync
```

Everything else is handled automatically by UV! 🚀

**For TensorRT**: See [TensorRT Setup](#tensorrt-setup) section above.
