# Pipeline Deep Dive

This document provides an in-depth look at the StreamSDK pipeline internals - how audio becomes video frames.

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Component Analysis](#component-analysis)
3. [Data Transformations](#data-transformations)
4. [Threading Architecture](#threading-architecture)
5. [Online vs Offline Mode](#online-vs-offline-mode)
6. [Performance Optimization](#performance-optimization)

---

## Pipeline Overview

The StreamSDK pipeline transforms audio into talking head video through a series of stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           StreamSDK Pipeline                                 │
│                                                                             │
│  Audio Input                                                                │
│      │                                                                      │
│      ▼                                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Wav2Feat │──▶│Condition │──▶│  Audio   │──▶│  Motion  │──▶│  Warp    │ │
│  │ (HuBERT) │   │ Handler  │   │ 2Motion  │   │  Stitch  │   │  F3D     │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       │              │              │              │              │         │
│       ▼              ▼              ▼              ▼              ▼         │
│    Features      Conditioned     Motion        Stitched      Warped       │
│    (N, 768)      Features       Keypoints     Motion        Features      │
│                  (N, 4096)      (N, 21×3)     (x_s, x_d)    (f_3d)        │
│                                                    │              │         │
│                                                    ▼              ▼         │
│                                              ┌──────────┐   ┌──────────┐   │
│                                              │ Decode   │──▶│ Putback  │   │
│                                              │  F3D     │   │          │   │
│                                              └──────────┘   └──────────┘   │
│                                                    │              │         │
│                                                    ▼              ▼         │
│                                              Rendered         Final        │
│                                              Face             Frame        │
│                                              (256×256)        (Original)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Analysis

### 1. Wav2Feat (Audio Feature Extraction)

**File:** `core/atomic_components/wav2feat.py`

Converts raw audio waveform to semantic features using HuBERT.

```python
# Input: Audio waveform
audio_chunk: np.ndarray  # Shape: (N_samples,), float32, 16kHz

# Output: Audio features  
audio_features: np.ndarray  # Shape: (N_frames, 768)
```

**Key Parameters:**
- Sample rate: 16,000 Hz
- Feature dim: 768 (HuBERT hidden size)
- Frame rate: 25 fps (640 samples per frame)

**Processing:**
```
Raw Audio (16kHz) ──▶ HuBERT Encoder ──▶ Hidden States ──▶ Audio Features
     │                    │                   │
     │                    │                   └─▶ (N, 768) per frame
     │                    │
     │                    └─▶ Transformer-based acoustic model
     │
     └─▶ Resampled to 16kHz if needed
```

### 2. Condition Handler (Conditioning)

**File:** `core/atomic_components/condition_handler.py`

Adds emotion, eye control, and other conditioning information to audio features.

```python
# Input
audio_feat: np.ndarray     # Shape: (seq_frames, 768)
frame_idx: int             # Current frame index

# Output
conditioned: np.ndarray    # Shape: (seq_frames, 4096)
```

**Conditioning Dimensions:**
```
Total: 4096 dimensions
├── Audio Features: 768
├── Emotion Embedding: ~256
├── Eye Control: ~128
├── Position Encoding: ~256
└── Other Conditioning: remaining
```

**Emotion System:**
```python
# Emotion indices
EMO_MAPPING = {
    0: "angry",
    1: "contempt", 
    2: "disgusted",
    3: "fear",
    4: "happy",     # Default/neutral
    5: "sad",
    6: "surprised"
}
```

### 3. Audio2Motion (LMDM Diffusion Model)

**File:** `core/atomic_components/audio2motion.py`

The core diffusion model that generates motion keypoints from conditioned audio features.

```python
# Input
conditioned_feat: np.ndarray  # Shape: (1, seq_frames, 4096)

# Output  
motion_keypoints: np.ndarray  # Shape: (1, seq_frames, 63)
                              # 21 keypoints × 3 (x, y, z)
```

**LMDM Architecture:**
```
Conditioned Features
        │
        ▼
┌─────────────────────┐
│  Diffusion Model    │
│  (Latent Motion DM) │
│                     │
│  • T=50 steps       │
│  • Noise prediction │
│  • Sequence model   │
└─────────────────────┘
        │
        ▼
Motion Keypoints (21 landmarks × 3D)
```

**Key Parameters:**
```python
seq_frames = 80           # Sequence length for diffusion
overlap_v2 = 10           # Overlap between chunks
valid_clip_len = 70       # Output frames per chunk (seq - overlap)
sampling_timesteps = 50   # Diffusion steps
```

**Online Mode Processing:**
```
Chunk 1: [────────────────|overlap]
                          ↘
Chunk 2:          [overlap|────────────────|overlap]
                                           ↘
Chunk 3:                          [overlap|────────────────]
```

### 4. Motion Stitch (Motion Blending)

**File:** `core/atomic_components/motion_stitch.py`

Combines source identity motion with driving motion to create final motion parameters.

```python
# Input
x_s_info: dict   # Source identity info (from avatar image)
x_d_info: dict   # Driving motion info (from audio2motion)
ctrl_kwargs: dict  # Per-frame control (fade, rotation, etc.)

# Output
x_s: np.ndarray  # Source keypoints
x_d: np.ndarray  # Driving keypoints
```

**Stitching Process:**
```
Source Motion (x_s_info)          Driving Motion (x_d_info)
        │                                  │
        │                                  │
        ▼                                  ▼
┌───────────────────────────────────────────────────┐
│              Motion Stitching                      │
│                                                   │
│  1. Extract relative motion from driving         │
│  2. Apply to source identity                     │
│  3. Handle expression/pose separately            │
│  4. Apply any per-frame adjustments              │
└───────────────────────────────────────────────────┘
        │                    │
        ▼                    ▼
   Stitched x_s         Stitched x_d
```

**Control Options:**
```python
ctrl_kwargs = {
    "fade_alpha": 0.5,       # Blend with neutral
    "delta_pitch": 5.0,      # Head rotation adjustments
    "delta_yaw": 0.0,
    "delta_roll": 0.0,
    "fade_out_keys": ("exp",),  # Keys to fade
}
```

### 5. Warp F3D (Feature Warping)

**File:** `core/atomic_components/warp_f3d.py`

Warps the source appearance features according to the motion.

```python
# Input
f_s: np.ndarray   # Source features (from avatar registration)
x_s: np.ndarray   # Source keypoints
x_d: np.ndarray   # Driving keypoints

# Output
f_3d: np.ndarray  # Warped 3D features
```

**Warping Architecture:**
```
┌─────────────────────────────────────────────┐
│           Warping Network                    │
│                                             │
│  Source Features (f_s) ──────┐              │
│                              ▼              │
│  Source KP (x_s) ─────▶  Warp Grid ◀──┐     │
│                              │        │     │
│  Driving KP (x_d) ──────────────────┘      │
│                              │              │
│                              ▼              │
│                        Grid Sample          │
│                              │              │
│                              ▼              │
│                         f_3d (warped)       │
└─────────────────────────────────────────────┘
```

### 6. Decode F3D (Image Synthesis)

**File:** `core/atomic_components/decode_f3d.py`

Decodes the warped 3D features into an RGB image.

```python
# Input
f_3d: np.ndarray   # Warped features

# Output
render_img: np.ndarray  # RGB image (256×256×3)
```

**Decoder Architecture:**
```
Warped Features (f_3d)
        │
        ▼
┌─────────────────────┐
│    SPADE Decoder    │
│                     │
│  • Upsampling       │
│  • Skip connections │
│  • SPADE norm       │
└─────────────────────┘
        │
        ▼
Face Image (256×256×3 RGB)
```

### 7. Putback (Compositing)

**File:** `core/atomic_components/putback.py`

Composites the rendered face back onto the original image.

```python
# Input
frame_rgb: np.ndarray    # Original frame
render_img: np.ndarray   # Rendered face (256×256)
M_c2o: np.ndarray        # Crop-to-original transform matrix

# Output
result: np.ndarray       # Composited frame (original resolution)
```

**Compositing Process:**
```
Original Frame                     Rendered Face
      │                                  │
      │                                  │
      ▼                                  ▼
┌─────────────────────────────────────────────────┐
│              Putback Compositing                 │
│                                                 │
│  1. Apply inverse transform (M_c2o)            │
│  2. Create blending mask                        │
│  3. Blend face region seamlessly               │
│  4. Return full resolution frame               │
└─────────────────────────────────────────────────┘
                    │
                    ▼
            Final Output Frame
```

---

## Data Transformations

### Dimension Summary

| Stage | Input Shape | Output Shape | Notes |
|-------|------------|--------------|-------|
| Wav2Feat | (N_samples,) | (N_frames, 768) | 640 samples/frame |
| CondHandler | (N, 768) | (N, 4096) | Adds conditioning |
| Audio2Motion | (1, N, 4096) | (1, N, 63) | 21 keypoints × 3 |
| MotionStitch | (63,), (63,) | (21, 3), (21, 3) | Stitched motion |
| WarpF3D | f_s, x_s, x_d | f_3d | 3D features |
| DecodeF3D | f_3d | (256, 256, 3) | RGB image |
| Putback | orig, face, M | (H, W, 3) | Final frame |

### Audio to Frame Timing

```
Audio @ 16kHz:  [─────────────────────────────────]
                640 samples = 1 frame @ 25fps = 40ms

Video @ 25fps:  [F0][F1][F2][F3][F4][F5][F6][F7]...
                 40  40  40  40  40  40  40  40ms

Chunk Processing:
  split_len = 6480 samples
  = 6480 / 640 ≈ 10 frames of context
  
  chunksize = (3, 5, 2)
  = 3 warmup + 5 output + 2 overlap
  = 5 new frames per chunk

  Processing cadence:
  Every 5 × 640 = 3200 samples (200ms)
  Output: 5 frames
```

---

## Threading Architecture

### Queue-Based Pipeline

```
Main Thread              Worker Threads
───────────              ──────────────

run_chunk()
    │
    │ audio_feat
    ▼
┌───────────────────┐
│audio2motion_queue │───▶ audio2motion_worker ───┐
└───────────────────┘                             │
                                                  │ [frame_idx, x_d_info, ctrl]
                                                  ▼
                      ┌───────────────────┐
                      │motion_stitch_queue│───▶ motion_stitch_worker ───┐
                      └───────────────────┘                              │
                                                                         │ [frame_idx, x_s, x_d]
                                                                         ▼
                                          ┌───────────────────┐
                                          │  warp_f3d_queue   │───▶ warp_f3d_worker ───┐
                                          └───────────────────┘                         │
                                                                                        │ [frame_idx, f_3d]
                                                                                        ▼
                                                              ┌───────────────────┐
                                                              │ decode_f3d_queue  │───▶ decode_f3d_worker ───┐
                                                              └───────────────────┘                           │
                                                                                                              │ [frame_idx, render_img]
                                                                                                              ▼
                                                                              ┌───────────────────┐
                                                                              │   putback_queue   │───▶ putback_worker ───┐
                                                                              └───────────────────┘                        │
                                                                                                                           │ res_frame_rgb
                                                                                                                           ▼
                                                                                              ┌───────────────────┐
                                                                                              │   writer_queue    │───▶ writer_worker
                                                                                              └───────────────────┘         │
                                                                                                                            │
                                                                                                                            ▼
                                                                                                                      frame_callback()
```

### Worker Implementation Pattern

```python
def worker(self):
    """Exception-safe worker wrapper."""
    try:
        self._worker_impl()
    except Exception as e:
        self.worker_exception = e
        self.stop_event.set()

def _worker_impl(self):
    """Actual worker logic."""
    while not self.stop_event.is_set():
        try:
            item = self.input_queue.get(timeout=1)
        except queue.Empty:
            continue
            
        if item is None:  # Sentinel for shutdown
            self.output_queue.put(None)
            break
            
        # Process item
        result = self.process(item)
        self.output_queue.put(result)
```

### Synchronization

```python
# Queue settings
QUEUE_MAX_SIZE = 100
QUEUE_TIMEOUT = 1.0  # seconds

# Shutdown mechanism
stop_event = threading.Event()

# Error propagation
worker_exception = None  # Set by any failing worker
```

---

## Online vs Offline Mode

### Offline Mode (Default)

Used for pre-recorded audio processing:

```python
sdk.setup(..., online_mode=False)

# All audio available upfront
# Process in optimal batch sizes
# Higher quality, not real-time
```

**Characteristics:**
- Full audio available
- Optimal chunk boundaries
- Best quality output
- No latency constraints

### Online Mode (Streaming)

Used for real-time processing:

```python
sdk.setup(..., online_mode=True)

# Audio arrives in chunks
# Process incrementally
# Lower latency, slight quality trade-off
```

**Characteristics:**
- Audio arrives incrementally
- Chunk overlap for continuity
- Warmup frames discarded
- d0 (initial motion) set from first chunk

**Online Mode Buffer Management:**

```python
# Buffer starts with overlap padding
self.audio_feat = self.wav2feat.wav2feat(
    np.zeros((self.overlap_v2 * 640,), dtype=np.float32), sr=16000
)

# First chunk establishes d0 (reference motion)
if res_kp_seq_valid_start is None:
    d0 = self.audio2motion.cvt_fmt(res_kp_seq[0:1])[0]
    self.motion_stitch.d0 = d0
```

---

## Performance Optimization

### Bottleneck Analysis

Typical processing times (RTX 3080):

| Stage | Time (ms) | % of Total |
|-------|-----------|------------|
| Wav2Feat | 5-10 | 5% |
| CondHandler | 2-3 | 2% |
| Audio2Motion | 60-80 | 70% |
| MotionStitch | 1-2 | 1% |
| WarpF3D | 10-15 | 12% |
| DecodeF3D | 8-12 | 10% |
| Putback | 1-2 | 1% |
| **Total** | **~100** | **100%** |

### Optimization Strategies

**1. Reduce Diffusion Steps:**
```python
# Default: 50 steps, ~80ms
sampling_timesteps=50

# Fast: 25 steps, ~40ms (slight quality loss)
sampling_timesteps=25
```

**2. TensorRT Optimization:**
```bash
# Build with FP16 for 2x speedup
python scripts/cvt_onnx_to_trt.py --fp16
```

**3. Batch Processing:**
```python
# Process multiple frames at once
# (requires model modification)
```

**4. GPU Memory Management:**
```python
# Ensure models stay on GPU
# Avoid CPU-GPU transfers
```

### Latency Budget

For real-time 25fps operation:

```
Available time per frame: 40ms

Pipeline latency breakdown:
├── Audio accumulation: 200ms (5 frames for quality)
├── Ditto processing: 100ms
├── Queue overhead: 10ms
├── Frame pairing: 5ms
└── WebRTC encoding: 20ms

Minimum end-to-end: ~335ms
Typical end-to-end: ~400-500ms
```

### Memory Footprint

```
GPU Memory Usage (RTX 3080):

Models:
├── HuBERT: ~500 MB
├── LMDM: ~800 MB  
├── Warping: ~400 MB
├── Decoder: ~600 MB
└── Other: ~200 MB

Total Static: ~2.5 GB

Runtime Buffers:
├── Feature tensors: ~500 MB
├── Queue buffers: ~200 MB
└── Working memory: ~300 MB

Total Runtime: ~3.5 GB
Total Peak: ~4-5 GB
```

---

## Next Steps

- [API Reference](./API_REFERENCE.md) - Detailed API docs
- [Troubleshooting](./TROUBLESHOOTING.md) - Common issues
- [Architecture](./ARCHITECTURE.md) - System overview

