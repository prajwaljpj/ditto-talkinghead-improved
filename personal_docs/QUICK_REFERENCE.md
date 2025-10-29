# Ditto Inference - Quick Reference Guide

## Quick Start Commands

### Basic Inference (PyTorch)
```bash
python inference.py \
    --data_root "./checkpoints/ditto_pytorch" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./output/result.mp4"
```

### Fast Inference (TensorRT)
```bash
python inference.py \
    --data_root "./checkpoints/ditto_trt_Ampere_Plus" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./output/result.mp4"
```

### Convert Models for Your GPU
```bash
python scripts/cvt_onnx_to_trt.py \
    --onnx_dir "./checkpoints/ditto_onnx" \
    --trt_dir "./checkpoints/ditto_trt_custom"
```

---

## Key Parameters Reference

### Avatar/Source Control

| Parameter | Default | Description |
|-----------|---------|-------------|
| `crop_scale` | 2.3 | Face crop scale (higher = tighter) |
| `crop_vx_ratio` | 0 | Horizontal shift (-1 to 1) |
| `crop_vy_ratio` | -0.125 | Vertical shift (-1 to 1) |
| `crop_flag_do_rot` | True | Align face rotation |
| `max_size` | 1920 | Maximum image dimension |
| `template_n_frames` | -1 | Number of source frames (-1 = all) |
| `smo_k_s` | 13 | Source smoothing (for videos) |

### Motion Generation (Audio2Motion)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling_timesteps` | 50 | Diffusion steps (25-100) |
| `overlap_v2` | 10 | Overlap frames for fusion (5-20) |
| `smo_k_d` | 3 | Motion smoothing kernel (1-5) |
| `fix_kp_cond` | 0 | Reset keypoint frequency (0 = never) |
| `online_mode` | False | Enable streaming mode |

### Expression Control

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emo` | 4 | Emotion value (0-7) |
| `drive_eye` | None | Override eye driving (None/True/False) |
| `delta_eye_open_n` | 0 | Eye openness adjustment |
| `eye_f0_mode` | False | Eye mode for videos |

### Motion Stitching

| Parameter | Default | Description |
|-----------|---------|-------------|
| `relative_d` | True | Use relative motion |
| `flag_stitching` | True | Enable stitching network |
| `fade_type` | "" | Fade type ("", "d0", "s") |
| `fade_out_keys` | ("exp",) | Keys to fade out |

---

## Per-Frame Control (`ctrl_info`)

Control individual frames with a dictionary:

```python
ctrl_info = {
    frame_id: {
        "fade_alpha": 0.0-1.0,      # Fade effect
        "delta_pitch": float,        # Pitch adjustment
        "delta_yaw": float,          # Yaw adjustment
        "delta_roll": float,         # Roll adjustment
        "fade_out_keys": tuple,      # Keys to fade
    }
}
```

### Example: Fade In/Out
```python
more_kwargs = {
    "run_kwargs": {
        "fade_in": 25,    # Fade in over 25 frames (1 second)
        "fade_out": 25,   # Fade out over 25 frames
    }
}
```

---

## Pipeline Stages

1. **AvatarRegistrar**: Face detection & feature extraction
2. **Wav2Feat**: Audio → features (HuBERT/WavLM)
3. **ConditionHandler**: Combine audio + control signals
4. **Audio2Motion**: Features → facial motion (diffusion)
5. **MotionStitch**: Combine source + driven motion
6. **WarpF3D**: Warp 3D features
7. **DecodeF3D**: Render 2D face image
8. **PutBack**: Composite into original frame
9. **VideoWriter**: Save to video file

---

## Model Formats Comparison

| Format | Speed | Compatibility | Best For |
|--------|-------|---------------|----------|
| PyTorch | Medium | High | Development, debugging |
| ONNX | Medium | High | Cross-platform |
| TensorRT | Fast | GPU-specific | Production, real-time |

---

## Mode Comparison

| Feature | Offline | Online |
|---------|---------|--------|
| Quality | Higher | Good |
| Latency | High | Low |
| Audio Processing | Full context | Streaming chunks |
| Use Case | Video generation | Real-time streaming |
| Audio Model | HuBERT/WavLM | HuBERT only |

---

## Performance Presets

### Maximum Quality
```python
setup_kwargs = {
    "sampling_timesteps": 100,
    "overlap_v2": 20,
    "smo_k_d": 5,
    "smo_k_s": 15,
    "max_size": 1920,
}
```

### Balanced (Default)
```python
setup_kwargs = {
    "sampling_timesteps": 50,
    "overlap_v2": 10,
    "smo_k_d": 3,
    "smo_k_s": 13,
    "max_size": 1920,
}
```

### Maximum Speed
```python
setup_kwargs = {
    "sampling_timesteps": 25,
    "overlap_v2": 5,
    "smo_k_d": 1,
    "smo_k_s": 1,
    "max_size": 512,
}
```

### Real-Time (Online)
```python
setup_kwargs = {
    "online_mode": True,
    "sampling_timesteps": 30,
    "overlap_v2": 5,
    "smo_k_d": 1,
}
```

---

## Python API Usage

### Basic Usage
```python
from stream_pipeline_offline import StreamSDK

# Initialize
SDK = StreamSDK(cfg_pkl, data_root)

# Setup
SDK.setup(source_path, output_path)

# Process audio
audio, sr = librosa.load(audio_path, sr=16000)
num_f = math.ceil(len(audio) / 16000 * 25)
SDK.setup_Nd(N_d=num_f)

# Run
aud_feat = SDK.wav2feat.wav2feat(audio)
SDK.audio2motion_queue.put(aud_feat)
SDK.close()

# Add audio track
os.system(f'ffmpeg -i {SDK.tmp_output_path} -i {audio_path} -c:v copy -c:a aac {output_path}')
```

### Advanced Usage with Controls
```python
# Setup with custom parameters
setup_kwargs = {
    "emo": 5,
    "sampling_timesteps": 75,
    "crop_scale": 2.5,
}
SDK.setup(source_path, output_path, **setup_kwargs)

# Frame-level control
ctrl_info = {
    0: {"fade_alpha": 0.0},
    25: {"fade_alpha": 1.0},
    200: {"delta_pitch": 0.1, "delta_yaw": -0.05},
}

num_f = math.ceil(len(audio) / 16000 * 25)
SDK.setup_Nd(N_d=num_f, fade_in=25, fade_out=25, ctrl_info=ctrl_info)

# Process
aud_feat = SDK.wav2feat.wav2feat(audio)
SDK.audio2motion_queue.put(aud_feat)
SDK.close()
```

### Online Streaming Mode
```python
from stream_pipeline_online import StreamSDK

SDK = StreamSDK(cfg_pkl_online, data_root)
SDK.setup(source_path, output_path, online_mode=True)

# Calculate frames
num_f = math.ceil(len(audio) / 16000 * 25)
SDK.setup_Nd(N_d=num_f)

# Add padding for online mode
chunksize = (3, 5, 2)  # (past_context, current, future_context)
audio = np.concatenate([np.zeros((chunksize[0] * 640,)), audio], 0)

# Process in chunks
split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480 samples
for i in range(0, len(audio), chunksize[1] * 640):
    audio_chunk = audio[i:i + split_len]
    if len(audio_chunk) < split_len:
        audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)))
    SDK.run_chunk(audio_chunk, chunksize)

SDK.close()
```

---

## Emotion Values

The `emo` parameter controls the base emotional expression:

- **0-7**: Different emotion presets (experiment to find desired expression)
- **4**: Default neutral expression
- Higher values typically increase expressiveness

Note: Exact mapping depends on training data and is best determined empirically.

---

## Chunksize for Online Mode

Format: `(past, current, future)`

- **past**: Context frames from history (e.g., 3)
- **current**: Frames to process in this chunk (e.g., 5)
- **future**: Look-ahead frames (e.g., 2)

Default: `(3, 5, 2)` - good balance of quality and latency

For lower latency: `(2, 5, 1)`
For higher quality: `(5, 5, 3)`

---

## Motion Parameters (265D)

The motion vector contains:

1. **scale** (1D): Face scale adjustment
2. **pitch** (66D): Head pitch rotation (nodding)
3. **yaw** (66D): Head yaw rotation (shaking)
4. **roll** (66D): Head roll rotation (tilting)
5. **t** (3D): Translation (x, y, z)
6. **exp** (63D): Facial expression blend shapes
7. **kp** (63D): Internal keypoints (usually not modified)

Total: 1 + 66 + 66 + 66 + 3 + 63 + 63 = 265 dimensions

---

## Troubleshooting Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| TensorRT error | Run conversion script for your GPU |
| Face not detected | Use frontal face, increase resolution |
| Choppy motion | Increase `overlap_v2` and `smo_k_d` |
| Slow inference | Use TensorRT, reduce `sampling_timesteps` |
| Out of memory | Reduce `max_size` or use smaller source |
| Unnatural expressions | Adjust `emo` value, try different values |
| Eye issues | Set `drive_eye=False` or adjust `delta_eye_open_n` |

---

## File Locations

### Models
- PyTorch: `checkpoints/ditto_pytorch/models/*.pth`
- TensorRT: `checkpoints/ditto_trt_Ampere_Plus/*.engine`
- ONNX: `checkpoints/ditto_onnx/*.onnx`

### Configs
- `checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl` - PyTorch
- `checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl` - TensorRT offline
- `checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl` - TensorRT online

### Code
- Main: `inference.py`
- Offline pipeline: `stream_pipeline_offline.py`
- Online pipeline: `stream_pipeline_online.py`
- Components: `core/atomic_components/*.py`
- Models: `core/models/*.py`

---

## Useful Code Snippets

### Load Configuration
```python
import pickle

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

cfg = load_pkl("./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl")
```

### Calculate Number of Frames
```python
import librosa
import math

audio, sr = librosa.load(audio_path, sr=16000)
num_frames = math.ceil(len(audio) / 16000 * 25)  # 25 FPS
```

### Set Random Seed
```python
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(1024)
```

---

## Common Workflows

### 1. Single Image → Talking Head
```bash
python inference.py \
    --data_root "./checkpoints/ditto_pytorch" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl" \
    --audio_path "speech.wav" \
    --source_path "portrait.jpg" \
    --output_path "result.mp4"
```

### 2. Video Source (Lip-Sync)
```bash
python inference.py \
    --data_root "./checkpoints/ditto_pytorch" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl" \
    --audio_path "new_audio.wav" \
    --source_path "original_video.mp4" \
    --output_path "resynced.mp4"
```

### 3. Batch Processing
```python
import glob

audio_files = glob.glob("audios/*.wav")
for audio_path in audio_files:
    output_path = f"outputs/{os.path.basename(audio_path)[:-4]}.mp4"
    run(SDK, audio_path, source_path, output_path)
```

### 4. Different Emotions
```python
for emo_id in range(8):
    more_kwargs = {
        "setup_kwargs": {"emo": emo_id}
    }
    output_path = f"output_emo_{emo_id}.mp4"
    run(SDK, audio_path, source_path, output_path, more_kwargs)
```

---

## System Requirements

### Minimum
- GPU: NVIDIA GPU with 4GB VRAM
- RAM: 8GB
- CUDA: 11.0+
- Python: 3.10

### Recommended
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- RAM: 16GB+
- CUDA: 11.8+
- TensorRT: 8.6.1

### For TensorRT
- GPU: Ampere architecture or newer (RTX 30-series, A100, etc.)
- Or: Compile your own TensorRT models for your GPU

---

## Tips and Best Practices

1. **Start with PyTorch models** - easier to debug and understand
2. **Use TensorRT for production** - significant speed improvements
3. **Test different emotion values** - find what works for your use case
4. **Adjust crop_scale** - tighter crops focus on face, wider includes more context
5. **Use offline mode** - unless you specifically need low latency
6. **Enable smoothing** - default values work well for most cases
7. **Higher sampling_timesteps** - diminishing returns beyond 50-75
8. **Process videos as source** - for maintaining original video style
9. **Frame-level control** - powerful for artistic control and transitions
10. **Monitor GPU memory** - reduce max_size if running into OOM errors

