# Ditto Talking Head - Inference Documentation

## Overview

Ditto is a Motion-Space Diffusion model for Controllable Realtime Talking Head Synthesis. This documentation covers the complete inference pipeline, capabilities, and usage instructions.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Inference Modes](#inference-modes)
3. [Pipeline Components](#pipeline-components)
4. [Configuration Files](#configuration-files)
5. [Usage Guide](#usage-guide)
6. [Advanced Controls](#advanced-controls)
7. [Model Formats](#model-formats)

---

## Architecture Overview

The Ditto inference system uses a **multi-threaded pipeline architecture** with the following stages:

```
Audio Input → Audio2Motion → MotionStitch → WarpF3D → DecodeF3D → PutBack → VideoWriter
```

### Key Design Principles

- **Multi-threaded Processing**: Each pipeline stage runs in a separate thread for maximum throughput
- **Queue-based Communication**: Thread-safe queues connect pipeline stages
- **Streaming Support**: Both offline (batch) and online (streaming) modes available
- **Flexible Control**: Frame-level control over face movements, expressions, and effects

---

## Inference Modes

### 1. Offline Mode (Default)

**File**: `stream_pipeline_offline.py`

**Characteristics**:
- Processes entire audio at once
- Better quality due to full audio context
- Suitable for video generation from pre-recorded audio
- Uses full overlap fusion for smoother results

**When to Use**:
- Generating high-quality talking head videos
- Non-real-time applications
- When you have the complete audio file

### 2. Online Mode (Streaming)

**File**: `stream_pipeline_online.py`

**Characteristics**:
- Processes audio in chunks
- Lower latency for real-time applications
- Requires HuBERT audio feature extractor
- Uses reduced overlap for faster processing

**When to Use**:
- Real-time video conferencing
- Live streaming applications
- Interactive avatars
- Low-latency requirements

---

## Pipeline Components

### 1. AvatarRegistrar

**Location**: `core/atomic_components/avatar_registrar.py`

**Purpose**: Processes the source image/video to extract facial features

**Key Functions**:
- Face detection and alignment
- Landmark extraction (106, 203, 478 points)
- Appearance feature extraction
- Motion feature extraction
- Cropping and scaling

**Parameters**:
- `crop_scale`: Face crop scale (default: 2.3)
- `crop_vx_ratio`: Horizontal shift ratio (default: 0)
- `crop_vy_ratio`: Vertical shift ratio (default: -0.125)
- `crop_flag_do_rot`: Enable rotation alignment (default: True)
- `smo_k_s`: Smoothing kernel size for video sources (default: 13)

### 2. Wav2Feat (Audio Feature Extraction)

**Location**: `core/atomic_components/wav2feat.py`

**Purpose**: Converts audio waveform to features

**Supported Models**:
- **HuBERT**: Supports both offline and online modes
- **WavLM**: Offline mode only

**Output**: Audio features (dimension: 1024 for HuBERT)

### 3. ConditionHandler

**Location**: `core/atomic_components/condition_handler.py`

**Purpose**: Combines audio features with control signals

**Control Types**:
- **Emotion (emo)**: Emotion control (default: 4)
- **Source Condition (sc)**: Source face characteristics
- **Eye Open**: Eye opening control
- **Eye Ball**: Eye gaze direction

### 4. Audio2Motion (LMDM)

**Location**: `core/atomic_components/audio2motion.py`

**Purpose**: Converts audio features to facial motion using diffusion model

**Key Features**:
- Latent Motion Diffusion Model (LMDM)
- Configurable sampling steps (default: 50)
- Overlap-based fusion for smooth transitions
- Adaptive keypoint conditioning

**Parameters**:
- `seq_frames`: Sequence length (default: 80)
- `overlap_v2`: Overlap frames for fusion (default: 10)
- `sampling_timesteps`: Diffusion sampling steps (default: 50)
- `fix_kp_cond`: Reset keypoint condition frequency (0 = never)
- `smo_k_d`: Smoothing kernel for motion (default: 3)

**Motion Parameters** (265 dimensions):
- `scale` (1): Face scale
- `pitch` (66): Head pitch rotation
- `yaw` (66): Head yaw rotation
- `roll` (66): Head roll rotation
- `t` (3): Translation
- `exp` (63): Facial expression
- `kp` (63): Keypoints (internal use)

### 5. MotionStitch

**Location**: `core/atomic_components/motion_stitch.py`

**Purpose**: Stitches source and driven motions together

**Features**:
- Relative motion transformation
- Eye control (open/close, gaze)
- Fade in/out effects
- Per-frame control via `ctrl_info`

**Parameters**:
- `relative_d`: Use relative motion (default: True)
- `drive_eye`: Override eye driving (None/True/False)
- `delta_eye_arr`: Custom eye adjustments
- `fade_type`: Fade effect type ("", "d0", "s")
- `flag_stitching`: Enable stitching network (default: True)

### 6. WarpF3D

**Location**: `core/atomic_components/warp_f3d.py`

**Purpose**: Warps 3D feature maps based on motion

**Function**: Applies geometric transformations to appearance features

### 7. DecodeF3D

**Location**: `core/atomic_components/decode_f3d.py`

**Purpose**: Decodes 3D features to 2D image

**Function**: Generates the final rendered face image

### 8. PutBack

**Location**: `core/atomic_components/putback.py`

**Purpose**: Composites rendered face back to original frame

**Function**: Handles alignment, blending, and background preservation

### 9. VideoWriter

**Location**: `core/atomic_components/writer.py`

**Purpose**: Writes frames to video file

**Features**:
- ImageIO-based video writing
- Progress tracking
- RGB format support

---

## Configuration Files

Configuration files (`.pkl`) contain all model paths and default parameters.

### Available Configurations

1. **v0.4_hubert_cfg_trt.pkl**: TensorRT offline mode
2. **v0.4_hubert_cfg_trt_online.pkl**: TensorRT online mode
3. **v0.4_hubert_cfg_pytorch.pkl**: PyTorch mode

### Configuration Structure

```python
{
    "base_cfg": {
        "insightface_det_cfg": {...},      # Face detection
        "landmark106_cfg": {...},           # Landmark detection (106 points)
        "landmark203_cfg": {...},           # Landmark detection (203 points)
        "landmark478_cfg": {...},           # Landmark detection (478 points)
        "appearance_extractor_cfg": {...},  # Appearance features
        "motion_extractor_cfg": {...},      # Motion features
        "stitch_network_cfg": {...},        # Motion stitching
        "warp_network_cfg": {...},          # 3D warping
        "decoder_cfg": {...},               # Image decoder
        "hubert_cfg": {...},                # Audio feature extractor
    },
    "audio2motion_cfg": {
        "model_path": "...",
        "seq_frames": 80,
        "motion_feat_dim": 265,
        "audio_feat_dim": 1059,
        "use_emo": True,
        "use_sc": True,
        "use_eye_open": True,
        "use_eye_ball": True,
        ...
    },
    "default_kwargs": {
        # Default runtime parameters
        ...
    }
}
```

---

## Usage Guide

### Basic Usage

#### 1. PyTorch Model (Recommended for Development)

```bash
python inference.py \
    --data_root "./checkpoints/ditto_pytorch" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./output/result.mp4"
```

#### 2. TensorRT Model (Recommended for Production)

```bash
python inference.py \
    --data_root "./checkpoints/ditto_trt_Ampere_Plus" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./output/result.mp4"
```

### Input Requirements

**Audio**:
- Format: WAV file
- Sample rate: Automatically resampled to 16kHz
- Duration: Any length

**Source**:
- Format: Image (PNG, JPG) or Video
- Requirements: Clear frontal face
- Resolution: Any (automatically resized, max 1920px by default)

### Output

- Format: MP4 video
- Frame rate: 25 FPS (derived from audio: 16000 / 640)
- Resolution: Matches source image/video
- Audio: Original audio synchronized with video

---

## Advanced Controls

### Frame-Level Control

You can control facial movements on a per-frame basis using `ctrl_info`:

```python
ctrl_info = {
    0: {
        "fade_alpha": 0.0,      # Fade effect (0.0 to 1.0)
        "delta_pitch": 0.1,     # Pitch adjustment
        "delta_yaw": 0.0,       # Yaw adjustment
        "delta_roll": 0.0,      # Roll adjustment
    },
    50: {
        "fade_alpha": 1.0,
    }
}

more_kwargs = {
    "setup_kwargs": {},
    "run_kwargs": {
        "ctrl_info": ctrl_info,
        "fade_in": 10,      # Fade in duration (frames)
        "fade_out": 10,     # Fade out duration (frames)
    }
}

run(SDK, audio_path, source_path, output_path, more_kwargs)
```

### Emotion Control

Control the emotional expression:

```python
setup_kwargs = {
    "emo": 4,  # Emotion value (typically 0-7)
}
```

### Eye Control

```python
setup_kwargs = {
    "drive_eye": True,              # Enable eye driving from audio
    "delta_eye_open_n": 0,          # Eye openness adjustment
    "delta_eye_arr": None,          # Custom eye position array
}
```

### Motion Quality Controls

```python
setup_kwargs = {
    "sampling_timesteps": 50,       # Diffusion steps (higher = better quality, slower)
    "overlap_v2": 10,               # Overlap for fusion (higher = smoother)
    "smo_k_d": 3,                   # Motion smoothing kernel
    "smo_k_s": 13,                  # Source smoothing (for videos)
}
```

### Cropping and Framing

```python
setup_kwargs = {
    "crop_scale": 2.3,              # Crop scale (higher = tighter crop)
    "crop_vx_ratio": 0.0,           # Horizontal shift
    "crop_vy_ratio": -0.125,        # Vertical shift
    "crop_flag_do_rot": True,       # Align face rotation
    "max_size": 1920,               # Maximum dimension
}
```

---

## Model Formats

### 1. PyTorch Models

**Location**: `checkpoints/ditto_pytorch/`

**Advantages**:
- Easy to modify and debug
- Cross-platform compatibility
- No compilation required

**Models**:
- `appearance_extractor.pth`
- `motion_extractor.pth`
- `lmdm_v0.4_hubert.pth`
- `stitch_network.pth`
- `warp_network.pth`
- `decoder.pth`

**Auxiliary Models** (ONNX):
- `hubert_streaming_fix_kv.onnx`: Audio feature extractor
- `det_10g.onnx`: Face detection
- `2d106det.onnx`: 106-point landmarks
- `landmark203.onnx`: 203-point landmarks
- `face_landmarker.task`: MediaPipe face mesh (478 points)

### 2. TensorRT Models

**Location**: `checkpoints/ditto_trt_Ampere_Plus/`

**Advantages**:
- Fastest inference speed
- Optimized for production
- GPU-specific optimization

**Models**:
- All `.engine` files (compiled for specific GPU architecture)

**Requirements**:
- TensorRT 8.6.1
- CUDA-compatible GPU
- Matching GPU architecture (Ampere or newer for provided models)

### 3. ONNX Models

**Location**: `checkpoints/ditto_onnx/`

**Purpose**:
- Intermediate format for conversion to TensorRT
- Cross-platform inference (with ONNX Runtime)

**Conversion**:
```bash
python scripts/cvt_onnx_to_trt.py \
    --onnx_dir "./checkpoints/ditto_onnx" \
    --trt_dir "./checkpoints/ditto_trt_custom"
```

---

## Workflow Summary

### Complete Inference Flow

1. **Initialize SDK**
   ```python
   SDK = StreamSDK(cfg_pkl, data_root)
   ```

2. **Setup**
   ```python
   SDK.setup(source_path, output_path, **setup_kwargs)
   ```
   - Loads source image/video
   - Extracts facial features
   - Initializes all pipeline components
   - Starts worker threads

3. **Configure Duration**
   ```python
   SDK.setup_Nd(N_d=num_frames, fade_in=10, fade_out=10, ctrl_info={})
   ```

4. **Process Audio**

   **Offline Mode**:
   ```python
   audio, sr = librosa.load(audio_path, sr=16000)
   aud_feat = SDK.wav2feat.wav2feat(audio)
   SDK.audio2motion_queue.put(aud_feat)
   ```

   **Online Mode**:
   ```python
   for audio_chunk in audio_chunks:
       SDK.run_chunk(audio_chunk, chunksize=(3, 5, 2))
   ```

5. **Close and Finalize**
   ```python
   SDK.close()  # Waits for all threads to complete
   ```

6. **Add Audio**
   ```bash
   ffmpeg -i temp_video.mp4 -i audio.wav -c:v copy -c:a aac output.mp4
   ```

---

## Threading Architecture

The pipeline uses 6 worker threads processing in parallel:

1. **audio2motion_worker**: Converts audio features to motion
2. **motion_stitch_worker**: Stitches source and driven motion
3. **warp_f3d_worker**: Warps 3D features
4. **decode_f3d_worker**: Decodes to 2D image
5. **putback_worker**: Composites into original frame
6. **writer_worker**: Writes frames to video

**Queue Flow**:
```
audio2motion_queue → motion_stitch_queue → warp_f3d_queue →
decode_f3d_queue → putback_queue → writer_queue
```

Each thread:
- Waits for input from previous stage
- Processes the data
- Sends output to next stage
- Handles exceptions gracefully

---

## Performance Considerations

### Speed Optimization

1. **Use TensorRT models**: 2-5x faster than PyTorch
2. **Reduce sampling_timesteps**: 25-30 for faster inference (trade-off: quality)
3. **Smaller crop_scale**: Less pixels to process
4. **Disable smoothing**: Set `smo_k_d=1`, `smo_k_s=1`
5. **Online mode**: Lower latency for streaming

### Quality Optimization

1. **Increase sampling_timesteps**: 50-100 for best quality
2. **Higher overlap_v2**: 15-20 for smoother transitions
3. **Enable smoothing**: Use default or higher values
4. **Offline mode**: Better results than online mode
5. **Higher resolution source**: Better detail preservation

---

## Common Use Cases

### 1. High-Quality Video Generation

```python
setup_kwargs = {
    "sampling_timesteps": 75,
    "overlap_v2": 15,
    "smo_k_d": 5,
    "max_size": 1920,
}
```

### 2. Fast Prototyping

```python
setup_kwargs = {
    "sampling_timesteps": 25,
    "overlap_v2": 5,
    "smo_k_d": 1,
    "max_size": 512,
}
```

### 3. Real-Time Streaming

Use online mode with TensorRT:
```python
SDK = StreamSDK(cfg_pkl_online, data_root_trt)
setup_kwargs = {
    "online_mode": True,
    "sampling_timesteps": 30,
    "overlap_v2": 5,
}
```

### 4. Multiple Emotions

Create multiple outputs with different emotions:
```python
for emo in range(8):
    SDK.setup(source_path, f"output_emo_{emo}.mp4", emo=emo)
    run(SDK, audio_path, source_path, output_path)
```

---

## Troubleshooting

### Common Issues

1. **GPU Compatibility Error (TensorRT)**
   - Solution: Convert ONNX models to TensorRT for your GPU
   ```bash
   python scripts/cvt_onnx_to_trt.py --onnx_dir "./checkpoints/ditto_onnx" --trt_dir "./checkpoints/ditto_trt_custom"
   ```

2. **Face Detection Fails**
   - Ensure clear, frontal face in source image
   - Try adjusting crop parameters
   - Use higher resolution source

3. **Choppy Output**
   - Increase `overlap_v2`
   - Enable smoothing (`smo_k_d` > 1)
   - Use offline mode

4. **Out of Memory**
   - Reduce `max_size`
   - Use smaller `seq_frames`
   - Process shorter audio segments

5. **Slow Inference**
   - Use TensorRT models
   - Reduce `sampling_timesteps`
   - Disable unnecessary smoothing

---

## File Structure Reference

```
ditto-talkinghead/
├── inference.py                    # Main inference script
├── stream_pipeline_offline.py      # Offline pipeline
├── stream_pipeline_online.py       # Online pipeline
├── core/
│   ├── atomic_components/
│   │   ├── avatar_registrar.py     # Source processing
│   │   ├── wav2feat.py             # Audio feature extraction
│   │   ├── condition_handler.py    # Condition processing
│   │   ├── audio2motion.py         # Motion generation
│   │   ├── motion_stitch.py        # Motion stitching
│   │   ├── warp_f3d.py             # 3D warping
│   │   ├── decode_f3d.py           # Image decoding
│   │   ├── putback.py              # Frame composition
│   │   ├── writer.py               # Video writing
│   │   ├── cfg.py                  # Config parsing
│   │   └── ...
│   ├── models/
│   │   ├── lmdm.py                 # Diffusion model
│   │   ├── decoder.py              # Decoder network
│   │   ├── warp_network.py         # Warp network
│   │   └── ...
│   └── utils/
│       └── ...
├── checkpoints/
│   ├── ditto_cfg/                  # Config files
│   ├── ditto_pytorch/              # PyTorch models
│   ├── ditto_trt_Ampere_Plus/      # TensorRT models
│   └── ditto_onnx/                 # ONNX models
├── example/
│   ├── audio.wav
│   └── image.png
└── scripts/
    └── cvt_onnx_to_trt.py          # Model conversion
```

---

## Summary

Ditto provides a powerful and flexible inference system for talking head synthesis:

- **Two modes**: Offline (high quality) and Online (low latency)
- **Three model formats**: PyTorch, ONNX, TensorRT
- **Multi-threaded pipeline**: Efficient parallel processing
- **Rich controls**: Frame-level control, emotions, eye movements, fades
- **Flexible inputs**: Images or videos as source
- **High quality**: Diffusion-based motion generation

For most users:
- Start with **PyTorch models** for ease of use
- Use **offline mode** for best quality
- Switch to **TensorRT + online mode** for production/real-time applications

