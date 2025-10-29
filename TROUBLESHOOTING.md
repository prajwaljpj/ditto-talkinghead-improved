# Troubleshooting Guide

## Common Issues and Solutions

### 1. ImportError: cannot import name 'cuda' from 'cuda'

**Error:**
```
ImportError: cannot import name 'cuda' from 'cuda' (unknown location)
```

**Cause:** cuda-python version 13.x has breaking API changes

**Solution:**
```bash
source .venv/bin/activate
uv pip uninstall cuda-python
uv pip install "cuda-python>=12.0,<13.0"
```

**Verification:**
```bash
python -c "from cuda import cuda, cudart, nvrtc; print('✓ CUDA imports working')"
```

---

### 2. TensorRT Model Version Mismatch

**Error:**
```
Error Code 1: Internal Error (Failed due to an old deserialization call on a newer plan file)
```

**Cause:** TensorRT models built with TensorRT 8.6.1, but you have TensorRT 10.13.3.9 installed

**Solution Options:**

**A. Use PyTorch Models (Recommended):**
```bash
python inference.py \
    --data_root ./checkpoints/ditto_pytorch \
    --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl \
    --source_video avatar.mp4 \
    --audio audio.wav \
    --output result.mp4
```

**B. Rebuild TensorRT Models:**
See [INSTALL.md - TensorRT Setup](#tensorrt-setup) for detailed instructions.

---

### 3. GridSample3D Plugin Missing

**Error:**
```
[TRT] [E] Plugin not found, are the plugin name, version, and namespace correct?
```

**Cause:** GridSample3D plugin not built or not in correct location

**Solution:**

1. Build the plugin:
```bash
cd grid-sample3d-trt-plugin
rm -rf build && mkdir build && cd build
cmake .. -DTensorRT_ROOT=../TensorRT-10.13.3.9 -DCMAKE_CUDA_ARCHITECTURES="89"
make -j$(nproc)
```

2. Copy to checkpoints:
```bash
cp libgrid_sample_3d_plugin.so ../../checkpoints/ditto_onnx/
```

3. Verify:
```bash
ls -lh ../../checkpoints/ditto_onnx/libgrid_sample_3d_plugin.so
```

---

### 4. CUDA Not Available

**Error:**
```
CUDA available: False
```

**Checks:**

1. **Verify NVIDIA driver:**
```bash
nvidia-smi
```

2. **Check PyTorch CUDA:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

3. **Check CUDA paths:**
```bash
echo $LD_LIBRARY_PATH
echo $PATH
```

**Solution:**
```bash
# Add CUDA to environment
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.8/bin:$PATH

# Verify
nvcc --version
```

---

### 5. Missing System Dependencies

**Error:**
```
libsndfile.so.1: cannot open shared object file
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0

# Verify
ffmpeg -version
```

---

### 6. UV Command Not Found

**Error:**
```
bash: uv: command not found
```

**Solution:**
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Add to ~/.bashrc for persistence
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify
uv --version
```

---

### 7. Python 3.10 Issues

**Error:**
```
Python 3.10 not found
```

**Solution:**

UV handles Python installation automatically! Just run:
```bash
uv sync
```

UV will download and install Python 3.10.14 for you.

**Manual installation (if needed):**
```bash
# Ubuntu/Debian
sudo apt install python3.10 python3.10-venv python3.10-dev

# Or use pyenv
curl https://pyenv.run | bash
pyenv install 3.10.14
pyenv local 3.10.14
```

---

### 8. Out of Memory (OOM) Errors

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size / resolution:**
```bash
python inference_video.py \
    --source_video avatar.mp4 \
    --audio audio.wav \
    --output result.mp4 \
    --max_size 1024  # Lower resolution
```

2. **Free GPU memory:**
```bash
# Kill other processes using GPU
nvidia-smi
kill <PID>

# Or restart
sudo systemctl restart gdm  # For desktop
```

3. **Use smaller models:**
```bash
# Use PyTorch instead of TensorRT
python inference.py \
    --data_root ./checkpoints/ditto_pytorch \
    --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl \
    # ...
```

---

### 9. Config/Model Mismatch

**Error:**
```
FileNotFoundError: Model file not found
```

**Cause:** Using wrong config for available models

**Solution:**

Match config to model directory:

| Model Directory | Config File |
|----------------|-------------|
| `ditto_pytorch` | `v0.4_hubert_cfg_pytorch.pkl` |
| `ditto_trt_custom` | `v0.4_hubert_cfg_trt_online.pkl` |
| `ditto_trt_Ampere_Plus` | Not compatible (TensorRT 8) |

**Correct usage:**
```bash
# PyTorch
python inference.py \
    --data_root ./checkpoints/ditto_pytorch \
    --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl

# TensorRT (custom built)
python inference.py \
    --data_root ./checkpoints/ditto_trt_custom \
    --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl
```

---

### 10. CMake Build Errors

**Error:**
```
NvInfer.h: No such file or directory
```

**Cause:** TensorRT headers not found

**Solution:**

1. **Download TensorRT from NVIDIA:**
   - Visit: https://developer.nvidia.com/tensorrt-download
   - Download: TensorRT 10.x for Ubuntu and CUDA 12.x (tar.gz)
   - Extract to project directory

2. **Configure with correct path:**
```bash
cd grid-sample3d-trt-plugin/build
cmake .. -DTensorRT_ROOT=/path/to/TensorRT-10.13.3.9
```

3. **Verify TensorRT structure:**
```bash
ls TensorRT-10.13.3.9/
# Should contain: include/ lib/ bin/ etc.
```

---

### 11. Verification Script Failures

**Error:**
```
✗ SOME CHECKS FAILED
```

**Solution:**

1. **Check which component failed:**
```bash
python verify_installation.py
```

2. **Reinstall failed component:**
```bash
source .venv/bin/activate
uv pip install <package-name> --reinstall
```

3. **Complete reinstall:**
```bash
rm -rf .venv
uv sync
python verify_installation.py
```

---

### 12. Performance Issues

**Problem:** Inference is slow

**Solutions:**

1. **Use TensorRT models** (2-3x faster):
   - Build TensorRT models (see INSTALL.md)
   - Use `--data_root ./checkpoints/ditto_trt_custom`

2. **Reduce quality settings:**
```bash
python inference_video.py \
    --source_video avatar.mp4 \
    --audio audio.wav \
    --output result.mp4 \
    --sampling_timesteps 25 \  # Lower = faster
    --overlap_v2 5 \            # Lower = faster
    --max_size 1024             # Lower resolution
```

3. **Optimize video source:**
```bash
# Use only neutral frames
python inference_video.py \
    --source_video avatar.mp4 \
    --audio audio.wav \
    --output result.mp4 \
    --template_n_frames 100  # Use only first 100 frames
```

---

### 13. Video Output Issues

**Problem:** No output file or corrupted video

**Checks:**

1. **Verify ffmpeg:**
```bash
ffmpeg -version
```

2. **Check output directory exists:**
```bash
mkdir -p output
```

3. **Check disk space:**
```bash
df -h
```

4. **Test with simple output path:**
```bash
python inference.py \
    --source_path image.jpg \
    --audio_path audio.wav \
    --output_path result.mp4  # No subdirectory
```

---

### 14. Module Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'xxx'
```

**Solution:**

1. **Ensure virtual environment is activated:**
```bash
source .venv/bin/activate
```

2. **Reinstall dependencies:**
```bash
uv sync
```

3. **Check you're using the right Python:**
```bash
which python
# Should show: /path/to/project/.venv/bin/python
```

---

## Getting Help

If your issue isn't listed here:

1. **Check verification:**
```bash
source .venv/bin/activate
python verify_installation.py
```

2. **Check environment:**
```bash
nvidia-smi
python --version
which python
uv --version
```

3. **Review documentation:**
   - `INSTALL.md` - Installation guide
   - `personal_docs/README.md` - Complete documentation
   - `UV_SETUP_SUMMARY.md` - Setup details

4. **Fresh install:**
```bash
rm -rf .venv
uv sync
python verify_installation.py
```

---

## Quick Diagnostic Commands

```bash
# Check everything
source .venv/bin/activate
python verify_installation.py

# Check CUDA
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check TensorRT
python -c "import tensorrt; print(f'TensorRT: {tensorrt.__version__}')"

# Check cuda-python
python -c "from cuda import cuda; print('cuda-python: OK')"

# List installed packages
uv pip list

# Check models
ls -lh checkpoints/ditto_pytorch/
ls -lh checkpoints/ditto_trt_custom/

# Check plugin
ls -lh checkpoints/ditto_onnx/libgrid_sample_3d_plugin.so
```

---

## Summary

Most issues are solved by:

1. ✅ Using UV environment: `source .venv/bin/activate`
2. ✅ Using PyTorch models (avoid TensorRT complexity)
3. ✅ Correct config for model type
4. ✅ Fresh install: `rm -rf .venv && uv sync`

**Still stuck?** Check the error message carefully and match it to the sections above!
