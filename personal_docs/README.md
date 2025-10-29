# Ditto Talking Head - Personal Documentation

This folder contains comprehensive documentation about the Ditto inference system for generating talking head videos.

## Documentation Files

### 1. [INFERENCE_DOCUMENTATION.md](./INFERENCE_DOCUMENTATION.md)
**Complete technical documentation covering:**
- Architecture overview and design principles
- Detailed explanation of inference modes (offline vs online)
- All pipeline components with parameters
- Configuration file structure
- Threading architecture
- Performance considerations
- Troubleshooting guide
- Complete file structure reference

**Best for:** Understanding how the system works, architecture details, and technical reference.

### 2. [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
**Quick lookup guide with:**
- Quick start commands
- Parameter reference tables
- Performance presets
- Mode comparisons
- Python API usage patterns
- Common workflows
- Troubleshooting quick fixes
- Code snippets

**Best for:** Quick lookups, finding specific parameters, and copy-paste commands.

### 3. [CODE_EXAMPLES.md](./CODE_EXAMPLES.md)
**Practical examples including:**
- Basic usage examples
- Advanced control examples (fades, emotions, head movements)
- Production use cases (batch processing, streaming, quality pipelines)
- Custom pipeline integrations
- Optimization examples (memory, parallel processing)
- Complete application example (web API)

**Best for:** Learning by example, implementing specific features, and production scenarios.

### 4. [VIDEO_INFERENCE_GUIDE.md](./VIDEO_INFERENCE_GUIDE.md)
**Complete guide for using video as source:**
- Video-to-video inference workflow
- Video source requirements and best practices
- Optimized parameters for video sources
- Practical examples with different scenarios
- Comparison with image sources
- Troubleshooting video-specific issues
- Batch processing scripts

**Best for:** Using a neutral avatar video with new audio, video dubbing/lip-sync, maintaining video style.

### 5. [BLENDING_CONTROL_GUIDE.md](./BLENDING_CONTROL_GUIDE.md)
**Control blending between rendered and original face:**
- Understanding face compositing and blending
- Five blend modes (blend, strong, replace, weak, custom)
- Visual comparisons and mask visualizations
- When to use each mode
- Custom alpha parameter guide
- Common scenarios and solutions
- Technical details of compositing process

**Best for:** Fine-tuning output quality, handling problematic sources (smiles, teeth), maximizing lip-sync visibility.

---

## Quick Navigation

### I want to...

#### Use Video as Source (Video-to-Video)
→ See [VIDEO_INFERENCE_GUIDE.md](./VIDEO_INFERENCE_GUIDE.md)

#### Control Face Blending (Rendered vs Original)
→ See [BLENDING_CONTROL_GUIDE.md](./BLENDING_CONTROL_GUIDE.md)

#### Fix Smiling/Teeth-Showing Avatar
→ See [BLENDING_CONTROL_GUIDE.md - Common Scenarios](./BLENDING_CONTROL_GUIDE.md#scenario-1-avatar-with-smile)

#### Preserve Natural Movements from Original Video
→ See [inference_video_preserve_motion.py](#preserve-original-motion-script)

#### Get Started Quickly
→ See [QUICK_REFERENCE.md - Quick Start Commands](./QUICK_REFERENCE.md#quick-start-commands)

#### Understand the Architecture
→ See [INFERENCE_DOCUMENTATION.md - Architecture Overview](./INFERENCE_DOCUMENTATION.md#architecture-overview)

#### Run Basic Inference
→ See [CODE_EXAMPLES.md - Example 1](./CODE_EXAMPLES.md#example-1-simple-image-to-video)

#### Control Emotions/Expressions
→ See [CODE_EXAMPLES.md - Example 4](./CODE_EXAMPLES.md#example-4-emotion-control)

#### Add Fade Effects
→ See [CODE_EXAMPLES.md - Example 5](./CODE_EXAMPLES.md#example-5-fade-inout-effects)

#### Process Multiple Videos
→ See [CODE_EXAMPLES.md - Example 2](./CODE_EXAMPLES.md#example-2-batch-processing-multiple-audios)

#### Optimize for Speed
→ See [INFERENCE_DOCUMENTATION.md - Performance Considerations](./INFERENCE_DOCUMENTATION.md#performance-considerations)

#### Build a Production System
→ See [CODE_EXAMPLES.md - Example 8](./CODE_EXAMPLES.md#example-8-high-quality-production-pipeline)

#### Implement Real-Time Streaming
→ See [CODE_EXAMPLES.md - Example 9](./CODE_EXAMPLES.md#example-9-real-time-streaming)

#### Find Specific Parameters
→ See [QUICK_REFERENCE.md - Key Parameters](./QUICK_REFERENCE.md#key-parameters-reference)

#### Understand Threading
→ See [INFERENCE_DOCUMENTATION.md - Threading Architecture](./INFERENCE_DOCUMENTATION.md#threading-architecture)

#### Debug Issues
→ See [INFERENCE_DOCUMENTATION.md - Troubleshooting](./INFERENCE_DOCUMENTATION.md#troubleshooting) or [QUICK_REFERENCE.md - Quick Fixes](./QUICK_REFERENCE.md#troubleshooting-quick-fixes)

---

## System Overview

**Ditto** is a Motion-Space Diffusion model for generating controllable, real-time talking head videos from a single image and audio input.

### Key Capabilities

✅ Generate talking head videos from single image + audio
✅ Support for both image and video sources
✅ Two modes: Offline (high quality) and Online (low latency)
✅ Three model formats: PyTorch, ONNX, TensorRT
✅ Frame-level control over expressions and head movements
✅ Emotion control
✅ Fade in/out effects
✅ Custom head movements (pitch, yaw, roll)
✅ Eye control (open/close, gaze direction)
✅ Multi-threaded pipeline for efficient processing
✅ Batch processing support
✅ Real-time streaming capability

### Basic Workflow

```
Input Image + Audio → Face Detection → Feature Extraction →
Audio Processing → Motion Generation → Motion Stitching →
3D Warping → Decoding → Compositing → Output Video
```

### Typical Use Cases

1. **Video Dubbing/Lip-Sync**: Replace audio in existing videos
2. **Avatar Creation**: Animate static portraits with speech
3. **Content Creation**: Generate video content from text-to-speech
4. **Multi-lingual Videos**: Create videos in multiple languages
5. **Interactive Avatars**: Real-time talking avatars for applications
6. **Presentation Videos**: Automated presentation with virtual speakers
7. **Educational Content**: Animated instructors for e-learning

---

## Getting Started

### Prerequisites

```bash
# Python 3.10
# CUDA 11.0+
# 4GB+ GPU VRAM (8GB+ recommended)
```

### Installation

```bash
# Clone repository
git clone https://github.com/antgroup/ditto-talkinghead
cd ditto-talkinghead

# Create environment
conda env create -f environment.yaml
conda activate ditto

# Download checkpoints
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
```

### Basic Usage

```bash
# PyTorch (easiest to start)
python inference.py \
    --data_root "./checkpoints/ditto_pytorch" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./output/result.mp4"

# TensorRT (fastest)
python inference.py \
    --data_root "./checkpoints/ditto_trt_Ampere_Plus" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./output/result.mp4"
```

---

## Available Inference Scripts

Ditto provides several specialized inference scripts for different use cases:

### 1. `inference.py` - Standard Inference
**Basic inference for images or videos**

```bash
python inference.py \
    --audio_path audio.wav \
    --source_path image.png \
    --output_path result.mp4
```

**Use for**: Simple image/video to talking head conversion

### 2. `inference_video.py` - Video-Optimized Inference
**Optimized for video sources with video-specific parameters**

```bash
python inference_video.py \
    --source_video avatar.mp4 \
    --audio speech.wav \
    --output result.mp4 \
    --template_n_frames 100
```

**Use for**: Video sources, better parameter defaults, limiting to neutral frames

**Key features**:
- Video-specific default parameters
- `--template_n_frames` to use only neutral portion
- Better for looping videos

### 3. `inference_video_preserve_motion.py` - Preserve Original Motion
**Generates lip-sync from neutral frames but composites to ALL original frames**

```bash
python inference_video_preserve_motion.py \
    --source_video looping_avatar.mp4 \
    --audio speech.wav \
    --output result.mp4 \
    --motion_frames 100
```

**Use for**: Preserving natural movements (blinks, head tilts, smiles) from original video

**How it works**:
- Uses first N frames for motion generation (neutral part)
- Uses ALL frames for compositing (preserves natural movements)
- Best of both: clean lip-sync + natural original movements

### 4. `inference_video_blend_control.py` - Blending Control
**Full control over rendered vs original face blending**

```bash
# Default blending (50/50)
python inference_video_blend_control.py \
    --source_video avatar.mp4 \
    --audio speech.wav \
    --output result.mp4

# Strong blending (90% rendered, hide smile)
python inference_video_blend_control.py \
    --source_video smiling_avatar.mp4 \
    --audio speech.wav \
    --output result.mp4 \
    --blend_mode strong

# Custom blending (70% rendered)
python inference_video_blend_control.py \
    --source_video avatar.mp4 \
    --audio speech.wav \
    --output result.mp4 \
    --blend_mode custom \
    --blend_alpha 0.7
```

**Use for**:
- Controlling how much rendered vs original face appears
- Fixing smile/teeth issues
- Fine-tuning output quality

**Blend modes**:
- `blend`: Default (50% center, smooth edges)
- `strong`: More rendered (90%, hides original issues)
- `replace`: Full replacement (100% rendered)
- `weak`: More original (subtle lip-sync)
- `custom`: Your percentage (use `--blend_alpha`)

---

## Script Comparison

| Script | Best For | Key Feature |
|--------|----------|-------------|
| `inference.py` | Simple use | Standard inference |
| `inference_video.py` | Video sources | Template frame control |
| `inference_video_preserve_motion.py` | Natural movements | Preserves original motion |
| `inference_video_blend_control.py` | Fine-tuning | Blending control |

---

## Common Workflows

### Workflow 1: Simple Image to Video
```bash
python inference.py \
    --source_path portrait.jpg \
    --audio_path speech.wav \
    --output_path result.mp4
```

### Workflow 2: Video with Smile (Hide Smile)
```bash
# Option A: Use only neutral frames
python inference_video.py \
    --source_video smiling_avatar.mp4 \
    --audio speech.wav \
    --output result.mp4 \
    --template_n_frames 100

# Option B: Use strong blending
python inference_video_blend_control.py \
    --source_video smiling_avatar.mp4 \
    --audio speech.wav \
    --output result.mp4 \
    --blend_mode strong
```

### Workflow 3: Preserve Natural Movements
```bash
# Avatar has natural blinks, head tilts you want to keep
python inference_video_preserve_motion.py \
    --source_video natural_avatar.mp4 \
    --audio speech.wav \
    --output result.mp4 \
    --motion_frames 100
```

### Workflow 4: Maximum Quality
```bash
python inference_video_blend_control.py \
    --data_root ./checkpoints/ditto_trt_Ampere_Plus \
    --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl \
    --source_video avatar.mp4 \
    --audio speech.wav \
    --output result.mp4 \
    --sampling_timesteps 75 \
    --overlap_v2 15 \
    --blend_mode blend
```

---

## Architecture Summary

### Pipeline Stages

1. **AvatarRegistrar**: Face detection, alignment, and feature extraction
2. **Wav2Feat**: Audio waveform to feature conversion (HuBERT/WavLM)
3. **ConditionHandler**: Combines audio features with control signals
4. **Audio2Motion (LMDM)**: Generates facial motion using diffusion model
5. **MotionStitch**: Stitches source and driven motions
6. **WarpF3D**: Applies 3D geometric transformations
7. **DecodeF3D**: Decodes 3D features to 2D image
8. **PutBack**: Composites rendered face into original frame
9. **VideoWriter**: Writes frames to video file

### Threading Model

All stages run in parallel using dedicated threads communicating via queues:

```
Thread 1: Audio → Motion
Thread 2: Motion Stitching
Thread 3: 3D Warping
Thread 4: Decoding
Thread 5: Compositing
Thread 6: Video Writing
```

This architecture enables efficient pipeline processing with maximum GPU utilization.

---

## Key Parameters

### Quality vs Speed Trade-offs

| Parameter | Quality Impact | Speed Impact | Default |
|-----------|---------------|--------------|---------|
| `sampling_timesteps` | High | High | 50 |
| `overlap_v2` | Medium | Low | 10 |
| `smo_k_d` | Medium | Low | 3 |
| `max_size` | Medium | High | 1920 |
| Model Format | Low | Very High | - |

### Common Configurations

**Maximum Quality:**
```python
sampling_timesteps=100, overlap_v2=20, smo_k_d=5, max_size=1920
```

**Balanced (Default):**
```python
sampling_timesteps=50, overlap_v2=10, smo_k_d=3, max_size=1920
```

**Maximum Speed:**
```python
sampling_timesteps=25, overlap_v2=5, smo_k_d=1, max_size=512
```

---

## Model Formats

### PyTorch
- **Pros**: Easy to modify, debug, cross-platform
- **Cons**: Slower than TensorRT
- **Best for**: Development, prototyping, debugging

### ONNX
- **Pros**: Cross-platform, good compatibility
- **Cons**: Medium speed
- **Best for**: Deployment on diverse hardware

### TensorRT
- **Pros**: Fastest inference (2-5x faster)
- **Cons**: GPU-specific, requires compilation
- **Best for**: Production, real-time applications

### Converting Models

```bash
python scripts/cvt_onnx_to_trt.py \
    --onnx_dir "./checkpoints/ditto_onnx" \
    --trt_dir "./checkpoints/ditto_trt_custom"
```

---

## Modes

### Offline Mode (Default)

**Characteristics:**
- Processes entire audio at once
- Higher quality (full context)
- Higher latency
- Supports both HuBERT and WavLM

**Use Cases:**
- Video generation
- Content creation
- High-quality outputs

### Online Mode (Streaming)

**Characteristics:**
- Processes audio in chunks
- Lower latency
- Good quality
- Requires HuBERT only

**Use Cases:**
- Real-time streaming
- Interactive avatars
- Video conferencing

---

## Advanced Features

### Frame-Level Control

Control individual frames with `ctrl_info`:

```python
ctrl_info = {
    0: {"fade_alpha": 0.0},
    25: {"fade_alpha": 1.0},
    100: {"delta_pitch": 0.1, "delta_yaw": -0.05},
}
```

### Emotion Control

```python
setup_kwargs = {"emo": 5}  # 0-7
```

### Fade Effects

```python
SDK.setup_Nd(N_d=num_frames, fade_in=25, fade_out=25)
```

### Eye Control

```python
setup_kwargs = {
    "drive_eye": True,
    "delta_eye_open_n": 0.1,
}
```

### Head Movements

```python
ctrl_info = {
    frame_id: {
        "delta_pitch": 0.1,  # Nod
        "delta_yaw": -0.05,  # Shake
        "delta_roll": 0.02,  # Tilt
    }
}
```

---

## Performance Tips

1. **Use TensorRT models** for production
2. **Reduce `sampling_timesteps`** for faster generation (trade-off: quality)
3. **Lower `max_size`** to reduce memory usage
4. **Disable smoothing** (`smo_k_d=1`) for speed
5. **Use online mode** for low-latency applications
6. **Batch process** multiple videos efficiently
7. **Multi-GPU processing** for large-scale operations

---

## Common Issues & Solutions

### TensorRT Compatibility Error
**Solution:** Convert ONNX models for your specific GPU architecture

### Face Not Detected
**Solution:** Ensure clear frontal face, good lighting, higher resolution

### Choppy Motion
**Solution:** Increase `overlap_v2` and `smo_k_d`

### Out of Memory
**Solution:** Reduce `max_size`, use smaller resolution source

### Slow Inference
**Solution:** Use TensorRT models, reduce `sampling_timesteps`

---

## Code Examples Summary

### Basic Examples
- Simple image-to-video
- Batch processing
- Video re-dubbing

### Advanced Control
- Emotion control
- Fade effects
- Custom head movements
- Expression transitions

### Production Use Cases
- High-quality pipeline
- Real-time streaming
- Multi-language support
- Web API service

### Optimization
- Memory management
- Parallel GPU processing
- Custom post-processing

---

## Resources

### Official Links
- [GitHub Repository](https://github.com/antgroup/ditto-talkinghead)
- [HuggingFace Models](https://huggingface.co/digital-avatar/ditto-talkinghead)
- [Project Page](https://digital-avatar.github.io/ai/Ditto/)
- [Paper](https://arxiv.org/abs/2411.19509)
- [Colab Demo](https://colab.research.google.com/drive/19SUi1TiO32IS-Crmsu9wrkNspWE8tFbs?usp=sharing)

### Documentation Files
- [INFERENCE_DOCUMENTATION.md](./INFERENCE_DOCUMENTATION.md) - Complete technical reference
- [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - Quick lookup guide
- [CODE_EXAMPLES.md](./CODE_EXAMPLES.md) - Practical code examples

---

## Summary

Ditto provides a powerful, flexible system for generating high-quality talking head videos. With support for multiple model formats, inference modes, and extensive control options, it can be adapted for a wide range of applications from simple video generation to complex production pipelines.

**Choose PyTorch** for development and experimentation.
**Choose TensorRT** for production and real-time applications.
**Use offline mode** for maximum quality.
**Use online mode** for low-latency streaming.

Start with the basic examples and gradually explore advanced features as needed!

---

## License

Apache-2.0 License

## Citation

```bibtex
@article{li2024ditto,
    title={Ditto: Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis},
    author={Li, Tianqi and Zheng, Ruobing and Yang, Minghui and Chen, Jingdong and Yang, Ming},
    journal={arXiv preprint arXiv:2411.19509},
    year={2024}
}
```

---

**Last Updated:** 2025-10-27
