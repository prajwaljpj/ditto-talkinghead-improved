# UV Quick Start Guide

Quick reference for setting up and using Ditto TalkingHead with UV.

## Initial Setup (One Time)

```bash
# 1. Clone repository
git clone https://github.com/antgroup/ditto-talkinghead
cd ditto-talkinghead

# 2. Run automated setup
bash setup_uv.sh
```

That's it! The script will:
- ✅ Install UV if needed
- ✅ Check Python 3.10
- ✅ Verify CUDA installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Verify installation

## Daily Usage

```bash
# Activate environment
cd /path/to/ditto-talkinghead
source .venv/bin/activate

# Run inference
python inference_video.py \
    --source_video avatar.mp4 \
    --audio speech.wav \
    --output result.mp4

# Deactivate when done
deactivate
```

## Common Commands

### Environment Management

```bash
# Activate
source .venv/bin/activate

# Deactivate
deactivate

# Recreate environment
rm -rf .venv
bash setup_uv.sh

# Update dependencies
source .venv/bin/activate
uv pip install -e . --upgrade
```

### Verification

```bash
# Check installation
python verify_installation.py

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU
nvidia-smi
```

### Inference Scripts

```bash
# Basic image inference
python inference.py \
    --source_path image.jpg \
    --audio_path audio.wav \
    --output_path result.mp4

# Video inference (optimized)
python inference_video.py \
    --source_video avatar.mp4 \
    --audio audio.wav \
    --output result.mp4

# Preserve natural movements
python inference_video_preserve_motion.py \
    --source_video avatar.mp4 \
    --audio audio.wav \
    --output result.mp4 \
    --motion_frames 100

# Control blending
python inference_video_blend_control.py \
    --source_video avatar.mp4 \
    --audio audio.wav \
    --output result.mp4 \
    --blend_mode strong

# Full control (motion + blending)
python inference_video_advanced.py \
    --source_video avatar.mp4 \
    --audio audio.wav \
    --output result.mp4 \
    --motion_frames 100 \
    --blend_mode strong

# Online/streaming mode
python inference_online_example.py \
    --source image.png \
    --audio audio.wav \
    --output result.mp4
```

## Troubleshooting

### UV not found
```bash
export PATH="$HOME/.cargo/bin:$PATH"
# Or reinstall: curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Python 3.10 not found
```bash
sudo apt install python3.10 python3.10-venv python3.10-dev
```

### CUDA not available
```bash
# Check driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch
source .venv/bin/activate
uv pip install torch torchvision torchaudio --reinstall
```

### Missing ffmpeg
```bash
sudo apt-get install ffmpeg
```

### ImportError
```bash
# Reinstall all dependencies
source .venv/bin/activate
uv pip install -e . --reinstall
```

## Performance Tips

### Use TensorRT (2-3x faster)
```bash
python inference_video.py \
    --data_root ./checkpoints/ditto_trt_Ampere_Plus \
    --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl \
    --source_video avatar.mp4 \
    --audio audio.wav \
    --output result.mp4
```

### Reduce quality for speed
```bash
python inference_video.py \
    --source_video avatar.mp4 \
    --audio audio.wav \
    --output result.mp4 \
    --sampling_timesteps 25 \
    --overlap_v2 5 \
    --max_size 1024
```

### Batch processing
```bash
for video in *.mp4; do
    python inference_video.py \
        --source_video "$video" \
        --audio audio.wav \
        --output "output_${video}"
done
```

## File Locations

```
ditto-talkinghead/
├── setup_uv.sh              # Automated setup script
├── pyproject.toml           # UV dependencies
├── verify_installation.py   # Installation verification
├── SETUP_UV.md             # Detailed setup guide
├── UV_QUICK_START.md       # This file
│
├── inference*.py           # Inference scripts
├── stream_pipeline_offline.py
│
├── .venv/                  # Virtual environment (created)
├── checkpoints/            # Model files (download)
│
└── personal_docs/          # Documentation
    ├── README.md
    ├── VIDEO_INFERENCE_GUIDE.md
    ├── BLENDING_CONTROL_GUIDE.md
    └── ...
```

## System Requirements

| Component | Requirement |
|-----------|------------|
| OS | Linux (tested), macOS, Windows |
| Python | 3.10.x (NOT 3.11+) |
| CUDA | 12.x or 11.x |
| GPU | NVIDIA GPU with 8GB+ VRAM |
| RAM | 16GB+ recommended |
| Disk | 10GB+ for models |

## Next Steps

After setup:

1. **Download Models**
   ```bash
   git lfs install
   git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
   ```

2. **Read Documentation**
   - Start: `personal_docs/README.md`
   - Video: `personal_docs/VIDEO_INFERENCE_GUIDE.md`
   - Blending: `personal_docs/BLENDING_CONTROL_GUIDE.md`

3. **Try Examples**
   - Basic: `inference.py`
   - Video: `inference_video.py`
   - Advanced: `inference_video_advanced.py`

## Quick Decision Tree

**What's your source?**

- **Image** → Use `inference.py`
- **Neutral video** → Use `inference_video.py`
- **Smiling video** → Use `inference_video_blend_control.py` (strong mode)
- **Video with movements to preserve** → Use `inference_video_preserve_motion.py`
- **Need full control** → Use `inference_video_advanced.py`

## Environment Variables (Optional)

```bash
# Add to ~/.bashrc for convenience
export DITTO_ROOT="/path/to/ditto-talkinghead"
export DITTO_CHECKPOINTS="$DITTO_ROOT/checkpoints"

# Quick activation alias
alias ditto='cd $DITTO_ROOT && source .venv/bin/activate'
```

Then just run:
```bash
ditto  # Activates environment
```

## Comparing UV vs Conda

| Feature | UV | Conda |
|---------|-----|-------|
| Setup time | 2-5 min ⚡ | 10-20 min |
| Disk space | ~2 GB | ~5 GB |
| Activation | `source .venv/bin/activate` | `conda activate ditto` |
| Speed | Very fast | Slower |
| Issues | Rare | Sometimes conflicts |

## Support

For detailed help:
- Full guide: `SETUP_UV.md`
- Documentation: `personal_docs/README.md`
- Troubleshooting: `SETUP_UV.md` (Troubleshooting section)

## Summary

```bash
# One-line setup
bash setup_uv.sh

# Daily usage
source .venv/bin/activate
python inference_video.py --source_video avatar.mp4 --audio audio.wav --output result.mp4
deactivate
```

That's all you need! 🚀
