# Video-to-Video Inference Guide

This guide explains how to use Ditto to generate talking head videos using a **video as source** (instead of a static image).

## Use Case

You have:
- ✅ A **neutral avatar video** (e.g., person with neutral expression)
- ✅ An **audio file** with speech you want the avatar to say

You want:
- 🎯 Generate a video where the avatar speaks the audio with proper lip-sync
- 🎯 Maintain the original video style and characteristics
- 🎯 Natural-looking facial movements

---

## Quick Start

### Basic Command

```bash
python inference_video.py \
    --source_video path/to/neutral_avatar.mp4 \
    --audio path/to/speech.wav \
    --output path/to/result.mp4
```

### With PyTorch Models (Recommended for First Time)

```bash
python inference_video.py \
    --data_root ./checkpoints/ditto_pytorch \
    --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl \
    --source_video ./input/avatar_neutral.mp4 \
    --audio ./input/speech.wav \
    --output ./output/talking_avatar.mp4
```

### With TensorRT (Faster, for Production)

```bash
python inference_video.py \
    --data_root ./checkpoints/ditto_trt_Ampere_Plus \
    --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl \
    --source_video ./input/avatar_neutral.mp4 \
    --audio ./input/speech.wav \
    --output ./output/talking_avatar.mp4
```

---

## Video Source Requirements

### Recommended Source Video Characteristics

1. **Content**:
   - Clear, frontal face (or mostly frontal)
   - Neutral expression (calm, no strong emotions)
   - Good lighting
   - Minimal motion (head can move slightly, but not too much)

2. **Technical**:
   - Resolution: 720p or higher (1080p recommended)
   - Format: MP4, AVI, MOV (any format supported by OpenCV)
   - Duration: Can be any length (system will loop/repeat frames as needed)
   - Frame rate: Any (will be processed at 25 FPS)

3. **Quality**:
   - Clear, in-focus face
   - No occlusions (glasses are OK, but avoid masks or hands covering face)
   - Consistent lighting throughout

### What Happens with the Source Video

The system will:
1. Extract facial features from all frames of your source video
2. Use these features as a template for the generated video
3. Apply lip movements from the audio
4. Maintain the overall style and characteristics of your source video
5. If your audio is longer than the source video, it will **loop/mirror** through the source frames

---

## Key Parameters for Video Sources

### Video-Specific Parameters (Optimized Defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--smo_k_s` | 13 | Source video smoothing (higher = smoother) |
| `--template_n_frames` | -1 | Use all frames from source (-1 = all) |
| `--drive_eye` | auto | Eye control (auto = False for videos) |

### Quality Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sampling_timesteps` | 50 | Quality (25=fast, 50=balanced, 75=high) |
| `--overlap_v2` | 10 | Smoothness (5=fast, 10=balanced, 15=smooth) |
| `--smo_k_d` | 3 | Motion smoothing (1=none, 3=balanced, 5=high) |
| `--max_size` | 1920 | Max resolution |

### Expression Control

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--emo` | 4 | Emotion (0-7, 4=neutral) |
| `--fade_in` | -1 | Fade in frames (-1=disabled, 25=1 sec) |
| `--fade_out` | -1 | Fade out frames (-1=disabled, 25=1 sec) |

---

## Usage Examples

### Example 1: Basic Usage

```bash
python inference_video.py \
    --source_video avatar_neutral.mp4 \
    --audio speech.wav \
    --output result.mp4
```

This uses default settings optimized for video sources.

### Example 2: High Quality Output

```bash
python inference_video.py \
    --source_video avatar_neutral.mp4 \
    --audio speech.wav \
    --output high_quality_result.mp4 \
    --sampling_timesteps 75 \
    --overlap_v2 15 \
    --smo_k_d 5
```

Better quality, but slower processing.

### Example 3: Fast Preview

```bash
python inference_video.py \
    --source_video avatar_neutral.mp4 \
    --audio speech.wav \
    --output preview.mp4 \
    --sampling_timesteps 25 \
    --overlap_v2 5 \
    --max_size 512
```

Faster processing for quick previews.

### Example 4: With Fade Effects

```bash
python inference_video.py \
    --source_video avatar_neutral.mp4 \
    --audio speech.wav \
    --output result_with_fades.mp4 \
    --fade_in 25 \
    --fade_out 25
```

Adds 1-second fade in and fade out (25 frames @ 25 FPS).

### Example 5: Different Emotion

```bash
python inference_video.py \
    --source_video avatar_neutral.mp4 \
    --audio speech.wav \
    --output happy_result.mp4 \
    --emo 6
```

Try different emotion values (0-7) to find the best expression.

### Example 6: Multiple Source Frames

```bash
python inference_video.py \
    --source_video avatar_neutral.mp4 \
    --audio speech.wav \
    --output result.mp4 \
    --template_n_frames 100
```

Use only first 100 frames from source video (useful if you have a long source video but want consistency).

### Example 7: Custom Eye Control

```bash
python inference_video.py \
    --source_video avatar_neutral.mp4 \
    --audio speech.wav \
    --output result.mp4 \
    --drive_eye false
```

Disable eye driving to preserve original video's eye movements.

---

## Understanding the Workflow

### Step-by-Step Process

1. **Load Source Video**
   ```
   Your neutral avatar video → Frame extraction → Facial feature detection
   ```

2. **Extract Features**
   ```
   Each frame → Face landmarks → Appearance features → Motion features
   ```

3. **Process Audio**
   ```
   Speech audio → Audio features (HuBERT) → Motion information
   ```

4. **Generate Motion**
   ```
   Audio motion + Source features → Diffusion model → Lip movements
   ```

5. **Combine**
   ```
   Source video style + Generated lip movements → Final video
   ```

6. **Output**
   ```
   Rendered frames + Original audio → MP4 video
   ```

### Frame Looping

If your audio is **longer** than your source video:
- System will **loop** through source frames
- Uses mirroring technique (forward → backward → forward)
- Example: If source has 50 frames and you need 200 frames:
  - Frames 1-50: Use source frames 0-49
  - Frames 51-100: Use source frames 49-0 (reversed)
  - Frames 101-150: Use source frames 0-49
  - Frames 151-200: Use source frames 49-0

This creates smooth transitions when cycling through source frames.

---

## Differences: Video vs Image Source

| Aspect | Image Source | Video Source |
|--------|--------------|--------------|
| **Input** | Single image | Multiple frames |
| **Features** | One set of features | Features per frame |
| **Eye Control** | Usually driven by audio | Usually from source video |
| **Smoothing** | Less critical | Important (`smo_k_s`) |
| **Style** | Static | Dynamic (maintains video style) |
| **Best For** | Portraits, photos | Natural movements, existing videos |

### When to Use Video Source

✅ You want to maintain a specific video style
✅ Your source has natural head movements
✅ You want more dynamic results
✅ You're doing video dubbing/re-dubbing
✅ Source video has good quality footage

### When to Use Image Source

✅ You only have a photo
✅ You want maximum control over movements
✅ Simpler, faster processing
✅ Creating avatar from scratch

---

## Tips for Best Results

### 1. Source Video Preparation

✅ **DO:**
- Use high-quality, well-lit footage
- Keep face centered and clear
- Use neutral expression
- Ensure stable footage (not too shaky)
- Trim to relevant portion (if very long)

❌ **DON'T:**
- Use heavily compressed videos
- Use videos with strong expressions
- Use videos with poor lighting
- Use videos where face is occluded

### 2. Parameter Tuning

**For Realistic Results:**
```bash
--sampling_timesteps 50 \
--overlap_v2 10 \
--smo_k_d 3 \
--smo_k_s 13
```

**For Expressive Results:**
```bash
--sampling_timesteps 50 \
--overlap_v2 10 \
--smo_k_d 2 \
--emo 6
```

**For Stable/Smooth Results:**
```bash
--sampling_timesteps 50 \
--overlap_v2 15 \
--smo_k_d 5 \
--smo_k_s 15
```

### 3. Audio Preparation

✅ **Recommended:**
- Clear speech audio
- WAV format (16kHz or will be resampled)
- Minimal background noise
- Proper volume levels

### 4. Testing Workflow

1. **Start with a short clip** (5-10 seconds) for testing
2. **Try default parameters** first
3. **Adjust one parameter at a time** to see effects
4. **Once satisfied, process full video**

---

## Common Issues and Solutions

### Issue 1: Face Not Detected

**Symptoms:** Error about face detection failing

**Solutions:**
- Ensure face is clearly visible and frontal
- Check lighting in source video
- Try a different portion of the video
- Increase resolution if too low

### Issue 2: Unnatural Movements

**Symptoms:** Jerky or unnatural facial movements

**Solutions:**
```bash
# Increase smoothing
--overlap_v2 15 --smo_k_d 5 --smo_k_s 15
```

### Issue 3: Lip Sync Issues

**Symptoms:** Lips don't match audio well

**Solutions:**
- Use offline mode (default)
- Increase `sampling_timesteps` (50 → 75)
- Check audio quality and clarity

### Issue 4: Style Mismatch

**Symptoms:** Generated video looks different from source

**Solutions:**
- Use more frames: `--template_n_frames -1` (use all)
- Increase source smoothing: `--smo_k_s 15`
- Check if emotion setting matches: `--emo 4` (neutral)

### Issue 5: Slow Processing

**Symptoms:** Taking too long to generate

**Solutions:**
```bash
# Use TensorRT models
--data_root ./checkpoints/ditto_trt_Ampere_Plus

# Or reduce quality for preview
--sampling_timesteps 25 --max_size 512
```

### Issue 6: Out of Memory

**Symptoms:** CUDA out of memory error

**Solutions:**
```bash
# Reduce resolution
--max_size 1080

# Or reduce batch processing
--template_n_frames 50  # Use fewer source frames
```

---

## Python API Usage

If you want to use the video inference in your own Python scripts:

```python
from stream_pipeline_offline import StreamSDK
from inference_video import run_video_inference

# Initialize SDK
cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
data_root = "./checkpoints/ditto_pytorch"
SDK = StreamSDK(cfg_pkl, data_root)

# Configure parameters
more_kwargs = {
    "setup_kwargs": {
        "smo_k_s": 13,
        "sampling_timesteps": 50,
        "overlap_v2": 10,
        "smo_k_d": 3,
        "emo": 4,
    },
    "run_kwargs": {
        "fade_in": 25,
        "fade_out": 25,
    }
}

# Run inference
run_video_inference(
    SDK=SDK,
    audio_path="./audio/speech.wav",
    source_video_path="./videos/avatar_neutral.mp4",
    output_path="./output/result.mp4",
    more_kwargs=more_kwargs
)
```

---

## Batch Processing Multiple Videos

Process multiple videos with the same audio:

```bash
#!/bin/bash

# Array of source videos
VIDEOS=("avatar1.mp4" "avatar2.mp4" "avatar3.mp4")
AUDIO="speech.wav"

# Process each video
for video in "${VIDEOS[@]}"; do
    output="output_$(basename $video)"
    echo "Processing $video..."

    python inference_video.py \
        --source_video "$video" \
        --audio "$AUDIO" \
        --output "$output"

    echo "Completed: $output"
done
```

Or different audios for the same avatar:

```bash
#!/bin/bash

VIDEO="neutral_avatar.mp4"
AUDIOS=("speech1.wav" "speech2.wav" "speech3.wav")

for i in "${!AUDIOS[@]}"; do
    audio="${AUDIOS[$i]}"
    output="result_$((i+1)).mp4"
    echo "Processing audio $((i+1))..."

    python inference_video.py \
        --source_video "$VIDEO" \
        --audio "$audio" \
        --output "$output"

    echo "Completed: $output"
done
```

---

## Performance Benchmarks

Approximate processing times (on NVIDIA A100):

| Configuration | Quality | Speed (25 sec audio) |
|---------------|---------|---------------------|
| PyTorch + Fast | Low | ~2 minutes |
| PyTorch + Balanced | Medium | ~4 minutes |
| PyTorch + High | High | ~8 minutes |
| TensorRT + Fast | Low | ~45 seconds |
| TensorRT + Balanced | Medium | ~1.5 minutes |
| TensorRT + High | High | ~3 minutes |

*Times vary based on GPU, resolution, and source video length*

---

## Comparison with Original inference.py

Both scripts work with videos, but `inference_video.py` provides:

✅ **Better defaults for video sources**
✅ **Clearer parameter explanations**
✅ **Video-specific optimizations**
✅ **More helpful command-line interface**
✅ **Better documentation and examples**

You can still use `inference.py` for videos:

```bash
python inference.py \
    --data_root ./checkpoints/ditto_pytorch \
    --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl \
    --audio_path speech.wav \
    --source_path avatar_video.mp4 \
    --output_path result.mp4
```

But `inference_video.py` makes it easier and more intuitive!

---

## Summary

**Key Points:**

1. ✅ Ditto **supports both image and video** as source
2. ✅ Video sources maintain the original video style
3. ✅ Use `inference_video.py` for video-optimized experience
4. ✅ Key parameters: `smo_k_s` (source smoothing), `template_n_frames` (frame usage)
5. ✅ System automatically loops source frames if audio is longer
6. ✅ Best results with clear, neutral, well-lit source videos

**Quick Command:**
```bash
python inference_video.py \
    --source_video your_avatar.mp4 \
    --audio your_speech.wav \
    --output result.mp4
```

That's it! 🎬
