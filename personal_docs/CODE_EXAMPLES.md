# Ditto Inference - Code Examples and Use Cases

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Advanced Control Examples](#advanced-control-examples)
3. [Production Use Cases](#production-use-cases)
4. [Custom Pipeline Integration](#custom-pipeline-integration)
5. [Optimization Examples](#optimization-examples)

---

## Basic Examples

### Example 1: Simple Image-to-Video

```python
import librosa
import math
from stream_pipeline_offline import StreamSDK

# Configuration
cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
data_root = "./checkpoints/ditto_pytorch"
audio_path = "./input/audio.wav"
source_path = "./input/portrait.png"
output_path = "./output/talking_head.mp4"

# Initialize SDK
SDK = StreamSDK(cfg_pkl, data_root)

# Setup
SDK.setup(source_path, output_path)

# Load audio and calculate frames
audio, sr = librosa.load(audio_path, sr=16000)
num_frames = math.ceil(len(audio) / 16000 * 25)

# Configure duration
SDK.setup_Nd(N_d=num_frames)

# Process
aud_feat = SDK.wav2feat.wav2feat(audio)
SDK.audio2motion_queue.put(aud_feat)
SDK.close()

# Add audio track
import os
cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
os.system(cmd)

print(f"Video saved to: {output_path}")
```

### Example 2: Batch Processing Multiple Audios

```python
import glob
import os
from pathlib import Path

def batch_process(source_image, audio_folder, output_folder):
    # Initialize SDK once
    SDK = StreamSDK(cfg_pkl, data_root)

    # Get all audio files
    audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))

    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for audio_path in audio_files:
        # Generate output filename
        basename = os.path.basename(audio_path).replace('.wav', '.mp4')
        output_path = os.path.join(output_folder, basename)

        print(f"Processing: {basename}")

        # Setup for this audio
        SDK.setup(source_image, output_path)

        # Load and process audio
        audio, sr = librosa.load(audio_path, sr=16000)
        num_frames = math.ceil(len(audio) / 16000 * 25)
        SDK.setup_Nd(N_d=num_frames)

        # Generate video
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
        SDK.close()

        # Add audio
        cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
        os.system(cmd)

        print(f"Saved: {output_path}")

# Usage
batch_process(
    source_image="./input/avatar.jpg",
    audio_folder="./audios/",
    output_folder="./outputs/"
)
```

### Example 3: Video Re-dubbing (Lip-Sync)

```python
def redub_video(original_video_path, new_audio_path, output_path):
    """
    Replace audio in a video with new audio and generate matching lip movements
    """
    SDK = StreamSDK(cfg_pkl, data_root)

    # Use the original video as source
    SDK.setup(original_video_path, output_path)

    # Load new audio
    audio, sr = librosa.load(new_audio_path, sr=16000)
    num_frames = math.ceil(len(audio) / 16000 * 25)
    SDK.setup_Nd(N_d=num_frames)

    # Generate
    aud_feat = SDK.wav2feat.wav2feat(audio)
    SDK.audio2motion_queue.put(aud_feat)
    SDK.close()

    # Add new audio
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{new_audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    os.system(cmd)

    return output_path

# Usage
redub_video(
    original_video_path="./input/original.mp4",
    new_audio_path="./input/new_speech.wav",
    output_path="./output/redubbed.mp4"
)
```

---

## Advanced Control Examples

### Example 4: Emotion Control

```python
def generate_with_emotion(source_path, audio_path, output_folder):
    """
    Generate videos with different emotions
    """
    SDK = StreamSDK(cfg_pkl, data_root)

    # Load audio once
    audio, sr = librosa.load(audio_path, sr=16000)
    num_frames = math.ceil(len(audio) / 16000 * 25)

    # Try different emotions
    for emo in range(8):
        output_path = os.path.join(output_folder, f"emotion_{emo}.mp4")

        # Setup with specific emotion
        SDK.setup(source_path, output_path, emo=emo)
        SDK.setup_Nd(N_d=num_frames)

        # Generate
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
        SDK.close()

        # Add audio
        cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
        os.system(cmd)

        print(f"Generated emotion {emo}: {output_path}")

# Usage
generate_with_emotion(
    source_path="./input/face.jpg",
    audio_path="./input/speech.wav",
    output_folder="./output/emotions/"
)
```

### Example 5: Fade In/Out Effects

```python
def generate_with_fades(source_path, audio_path, output_path,
                        fade_in_duration=1.0, fade_out_duration=1.0):
    """
    Generate video with smooth fade in and fade out

    Args:
        fade_in_duration: Fade in duration in seconds
        fade_out_duration: Fade out duration in seconds
    """
    SDK = StreamSDK(cfg_pkl, data_root)
    SDK.setup(source_path, output_path)

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    num_frames = math.ceil(len(audio) / 16000 * 25)

    # Convert durations to frames (25 FPS)
    fade_in_frames = int(fade_in_duration * 25)
    fade_out_frames = int(fade_out_duration * 25)

    # Setup with fades
    SDK.setup_Nd(
        N_d=num_frames,
        fade_in=fade_in_frames,
        fade_out=fade_out_frames
    )

    # Generate
    aud_feat = SDK.wav2feat.wav2feat(audio)
    SDK.audio2motion_queue.put(aud_feat)
    SDK.close()

    # Add audio
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    os.system(cmd)

# Usage
generate_with_fades(
    source_path="./input/avatar.png",
    audio_path="./input/speech.wav",
    output_path="./output/with_fades.mp4",
    fade_in_duration=2.0,  # 2 seconds fade in
    fade_out_duration=1.5  # 1.5 seconds fade out
)
```

### Example 6: Custom Head Movements

```python
import numpy as np

def generate_with_head_movement(source_path, audio_path, output_path):
    """
    Add custom head movements (nodding, shaking) at specific times
    """
    SDK = StreamSDK(cfg_pkl, data_root)
    SDK.setup(source_path, output_path)

    audio, sr = librosa.load(audio_path, sr=16000)
    num_frames = math.ceil(len(audio) / 16000 * 25)

    # Define custom head movements
    # Frames 50-75: Nod (pitch down then up)
    # Frames 100-125: Shake head (yaw left then right)
    ctrl_info = {}

    # Nodding motion
    for i in range(50, 75):
        progress = (i - 50) / 25
        # Smooth sine wave for natural motion
        pitch_delta = 0.15 * np.sin(progress * 2 * np.pi)
        ctrl_info[i] = {"delta_pitch": pitch_delta}

    # Head shake motion
    for i in range(100, 125):
        progress = (i - 100) / 25
        yaw_delta = 0.1 * np.sin(progress * 2 * np.pi)
        ctrl_info[i] = {"delta_yaw": yaw_delta}

    # Setup with control info
    SDK.setup_Nd(N_d=num_frames, ctrl_info=ctrl_info)

    # Generate
    aud_feat = SDK.wav2feat.wav2feat(audio)
    SDK.audio2motion_queue.put(aud_feat)
    SDK.close()

    # Add audio
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    os.system(cmd)

# Usage
generate_with_head_movement(
    source_path="./input/face.jpg",
    audio_path="./input/speech.wav",
    output_path="./output/with_movements.mp4"
)
```

### Example 7: Dynamic Expression Transitions

```python
def generate_with_expression_transitions(source_path, audio_path, output_path):
    """
    Create dynamic expression transitions during speaking
    """
    SDK = StreamSDK(cfg_pkl, data_root)

    audio, sr = librosa.load(audio_path, sr=16000)
    num_frames = math.ceil(len(audio) / 16000 * 25)

    # Define expression keyframes
    # Start neutral (emo=4), transition to happy (emo=6) at 3s, back to neutral at 6s
    expression_keyframes = {
        0: 4,      # Neutral
        75: 6,     # Happy (3 seconds in)
        150: 4,    # Back to neutral (6 seconds in)
    }

    # Setup with initial emotion
    SDK.setup(source_path, output_path, emo=4)

    # Create smooth transitions
    ctrl_info = {}
    fade_duration = 25  # 1 second transition

    for frame_id, target_emo in expression_keyframes.items():
        if frame_id > 0:  # Skip first frame
            # Fade out old expression
            for i in range(max(0, frame_id - fade_duration), frame_id):
                alpha = 1.0 - (i - (frame_id - fade_duration)) / fade_duration
                ctrl_info[i] = {
                    "fade_alpha": alpha,
                    "fade_out_keys": ("exp",)
                }

    SDK.setup_Nd(N_d=num_frames, ctrl_info=ctrl_info)

    # Generate
    aud_feat = SDK.wav2feat.wav2feat(audio)
    SDK.audio2motion_queue.put(aud_feat)
    SDK.close()

    # Add audio
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    os.system(cmd)

# Usage
generate_with_expression_transitions(
    source_path="./input/face.jpg",
    audio_path="./input/speech.wav",
    output_path="./output/with_transitions.mp4"
)
```

---

## Production Use Cases

### Example 8: High-Quality Production Pipeline

```python
class ProductionPipeline:
    def __init__(self, cfg_pkl, data_root, quality="high"):
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.quality_presets = {
            "low": {
                "sampling_timesteps": 25,
                "overlap_v2": 5,
                "smo_k_d": 1,
                "max_size": 512,
            },
            "medium": {
                "sampling_timesteps": 50,
                "overlap_v2": 10,
                "smo_k_d": 3,
                "max_size": 1080,
            },
            "high": {
                "sampling_timesteps": 75,
                "overlap_v2": 15,
                "smo_k_d": 5,
                "max_size": 1920,
            }
        }
        self.quality = quality

    def process(self, source_path, audio_path, output_path, **kwargs):
        """Process with quality preset"""
        SDK = StreamSDK(self.cfg_pkl, self.data_root)

        # Merge quality preset with custom kwargs
        setup_kwargs = self.quality_presets[self.quality].copy()
        setup_kwargs.update(kwargs)

        # Setup
        SDK.setup(source_path, output_path, **setup_kwargs)

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        num_frames = math.ceil(len(audio) / 16000 * 25)
        SDK.setup_Nd(N_d=num_frames, fade_in=25, fade_out=25)

        # Generate
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
        SDK.close()

        # Add audio with high quality encoding
        cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v libx264 -preset slow -crf 18 -c:a aac -b:a 256k "{output_path}"'
        os.system(cmd)

        return output_path

    def process_batch(self, jobs):
        """
        Process multiple jobs

        Args:
            jobs: List of (source_path, audio_path, output_path, kwargs) tuples
        """
        results = []
        for source_path, audio_path, output_path, kwargs in jobs:
            print(f"Processing: {output_path}")
            result = self.process(source_path, audio_path, output_path, **kwargs)
            results.append(result)
        return results

# Usage
pipeline = ProductionPipeline(
    cfg_pkl="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
    data_root="./checkpoints/ditto_trt_Ampere_Plus",
    quality="high"
)

# Single job
pipeline.process(
    source_path="./input/avatar.jpg",
    audio_path="./input/speech.wav",
    output_path="./output/result.mp4",
    emo=5  # Custom parameter
)

# Batch jobs
jobs = [
    ("avatar1.jpg", "audio1.wav", "output1.mp4", {"emo": 4}),
    ("avatar2.jpg", "audio2.wav", "output2.mp4", {"emo": 6}),
    ("avatar3.jpg", "audio3.wav", "output3.mp4", {"emo": 5}),
]
pipeline.process_batch(jobs)
```

### Example 9: Real-Time Streaming

```python
from stream_pipeline_online import StreamSDK as StreamSDK_Online
import sounddevice as sd

class RealtimeTalkingHead:
    def __init__(self, cfg_pkl, data_root, source_path):
        self.SDK = StreamSDK_Online(cfg_pkl, data_root)
        self.source_path = source_path
        self.chunksize = (3, 5, 2)
        self.sample_rate = 16000
        self.initialized = False

    def initialize(self, output_path):
        """Initialize the streaming pipeline"""
        self.SDK.setup(
            self.source_path,
            output_path,
            online_mode=True,
            sampling_timesteps=30,
            overlap_v2=5
        )
        # Estimate large enough frame count for streaming
        self.SDK.setup_Nd(N_d=10000)
        self.initialized = True

    def process_audio_chunk(self, audio_chunk):
        """Process a single audio chunk"""
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized")

        # Ensure correct length
        required_len = int(sum(self.chunksize) * 0.04 * self.sample_rate) + 80
        if len(audio_chunk) < required_len:
            audio_chunk = np.pad(
                audio_chunk,
                (0, required_len - len(audio_chunk))
            )

        self.SDK.run_chunk(audio_chunk, self.chunksize)

    def finalize(self, audio_path):
        """Finalize and save video"""
        self.SDK.close()

        # Add audio
        cmd = f'ffmpeg -loglevel error -y -i "{self.SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{self.SDK.output_path}"'
        os.system(cmd)

    def stream_from_file(self, audio_path, output_path):
        """Stream processing from audio file"""
        self.initialize(output_path)

        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Add initial padding
        audio = np.concatenate([
            np.zeros((self.chunksize[0] * 640,), dtype=np.float32),
            audio
        ], 0)

        # Process in chunks
        chunk_step = self.chunksize[1] * 640
        chunk_len = int(sum(self.chunksize) * 0.04 * self.sample_rate) + 80

        for i in range(0, len(audio), chunk_step):
            audio_chunk = audio[i:i + chunk_len]
            self.process_audio_chunk(audio_chunk)

        self.finalize(audio_path)
        print(f"Saved: {output_path}")

# Usage
realtime = RealtimeTalkingHead(
    cfg_pkl="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
    data_root="./checkpoints/ditto_trt_Ampere_Plus",
    source_path="./input/avatar.jpg"
)

realtime.stream_from_file(
    audio_path="./input/speech.wav",
    output_path="./output/streaming_result.mp4"
)
```

### Example 10: Multi-Language Support with Translation

```python
from googletrans import Translator
import pyttsx3

def generate_multilingual_video(source_path, text, languages, output_folder):
    """
    Generate talking head videos in multiple languages

    Args:
        source_path: Avatar image
        text: Original text (in English)
        languages: List of language codes (e.g., ['es', 'fr', 'de'])
        output_folder: Output directory
    """
    translator = Translator()
    tts_engine = pyttsx3.init()

    SDK = StreamSDK(cfg_pkl, data_root)

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for lang in languages:
        print(f"Generating video for language: {lang}")

        # Translate text
        translated = translator.translate(text, dest=lang)
        print(f"Translation: {translated.text}")

        # Generate audio using TTS
        audio_path = os.path.join(output_folder, f"audio_{lang}.wav")
        tts_engine.save_to_file(translated.text, audio_path)
        tts_engine.runAndWait()

        # Generate video
        output_path = os.path.join(output_folder, f"video_{lang}.mp4")

        SDK.setup(source_path, output_path)
        audio, sr = librosa.load(audio_path, sr=16000)
        num_frames = math.ceil(len(audio) / 16000 * 25)
        SDK.setup_Nd(N_d=num_frames)

        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
        SDK.close()

        # Add audio
        cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
        os.system(cmd)

        print(f"Saved: {output_path}")

# Usage
generate_multilingual_video(
    source_path="./input/avatar.jpg",
    text="Hello, welcome to our presentation.",
    languages=['es', 'fr', 'de', 'ja'],
    output_folder="./output/multilingual/"
)
```

---

## Custom Pipeline Integration

### Example 11: Custom Post-Processing Pipeline

```python
import cv2
from PIL import Image, ImageFilter

def custom_postprocess_pipeline(source_path, audio_path, output_path):
    """
    Generate video with custom post-processing effects
    """
    # Generate base video
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    SDK = StreamSDK(cfg_pkl, data_root)
    SDK.setup(source_path, temp_output)

    audio, sr = librosa.load(audio_path, sr=16000)
    num_frames = math.ceil(len(audio) / 16000 * 25)
    SDK.setup_Nd(N_d=num_frames)

    aud_feat = SDK.wav2feat.wav2feat(audio)
    SDK.audio2motion_queue.put(aud_feat)
    SDK.close()

    # Post-process frames
    cap = cv2.VideoCapture(SDK.tmp_output_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply custom effects
        # Example: Add vignette effect
        frame = apply_vignette(frame)

        # Example: Color grading
        frame = adjust_color_temperature(frame, temperature=6500)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # Add audio
    cmd = f'ffmpeg -loglevel error -y -i "{temp_output}" -i "{audio_path}" -map 0:v -map 1:a -c:v libx264 -c:a aac "{output_path}"'
    os.system(cmd)

    # Cleanup
    os.remove(temp_output)
    os.remove(SDK.tmp_output_path)

def apply_vignette(frame, intensity=0.5):
    """Apply vignette effect to frame"""
    rows, cols = frame.shape[:2]

    # Create vignette mask
    X_resultant_kernel = cv2.getGaussianKernel(cols, cols / 2)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, rows / 2)
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = resultant_kernel / resultant_kernel.max()
    mask = (mask * (1 - intensity) + intensity)

    # Apply mask
    frame = frame.astype(float)
    for i in range(3):
        frame[:, :, i] = frame[:, :, i] * mask

    return frame.astype(np.uint8)

def adjust_color_temperature(frame, temperature=6500):
    """Adjust color temperature of frame"""
    # Simple color temperature adjustment
    # Warm (high temp) = more red/yellow
    # Cool (low temp) = more blue

    if temperature > 6500:
        # Warm
        factor = (temperature - 6500) / 2000
        frame[:, :, 2] = np.clip(frame[:, :, 2] * (1 + 0.1 * factor), 0, 255)
        frame[:, :, 1] = np.clip(frame[:, :, 1] * (1 + 0.05 * factor), 0, 255)
    else:
        # Cool
        factor = (6500 - temperature) / 2000
        frame[:, :, 0] = np.clip(frame[:, :, 0] * (1 + 0.1 * factor), 0, 255)

    return frame

# Usage
custom_postprocess_pipeline(
    source_path="./input/avatar.jpg",
    audio_path="./input/speech.wav",
    output_path="./output/enhanced.mp4"
)
```

### Example 12: Integration with Face Swap

```python
import insightface
from insightface.app import FaceAnalysis

def generate_with_face_swap(target_face_path, body_video_path, audio_path, output_path):
    """
    Generate talking head and swap face onto another body
    """
    # Step 1: Generate talking head video
    temp_talking_head = "./temp/talking_head.mp4"
    SDK = StreamSDK(cfg_pkl, data_root)
    SDK.setup(target_face_path, temp_talking_head)

    audio, sr = librosa.load(audio_path, sr=16000)
    num_frames = math.ceil(len(audio) / 16000 * 25)
    SDK.setup_Nd(N_d=num_frames)

    aud_feat = SDK.wav2feat.wav2feat(audio)
    SDK.audio2motion_queue.put(aud_feat)
    SDK.close()

    # Step 2: Initialize face swapper
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx')

    # Step 3: Swap faces
    talking_cap = cv2.VideoCapture(SDK.tmp_output_path)
    body_cap = cv2.VideoCapture(body_video_path)

    fps = body_cap.get(cv2.CAP_PROP_FPS)
    width = int(body_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(body_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_talking_head + ".swap.mp4", fourcc, fps, (width, height))

    while True:
        ret1, talking_frame = talking_cap.read()
        ret2, body_frame = body_cap.read()

        if not ret1 or not ret2:
            break

        # Detect faces
        talking_faces = app.get(talking_frame)
        body_faces = app.get(body_frame)

        if len(talking_faces) > 0 and len(body_faces) > 0:
            # Swap face
            result = swapper.get(body_frame, body_faces[0], talking_faces[0], paste_back=True)
            out.write(result)
        else:
            out.write(body_frame)

    talking_cap.release()
    body_cap.release()
    out.release()

    # Add audio
    cmd = f'ffmpeg -loglevel error -y -i "{temp_talking_head}.swap.mp4" -i "{audio_path}" -map 0:v -map 1:a -c:v libx264 -c:a aac "{output_path}"'
    os.system(cmd)

    # Cleanup
    os.remove(temp_talking_head)
    os.remove(SDK.tmp_output_path)
    os.remove(temp_talking_head + ".swap.mp4")

# Usage
generate_with_face_swap(
    target_face_path="./input/face.jpg",
    body_video_path="./input/body_video.mp4",
    audio_path="./input/speech.wav",
    output_path="./output/face_swapped.mp4"
)
```

---

## Optimization Examples

### Example 13: GPU Memory Optimization

```python
import torch

class MemoryEfficientPipeline:
    def __init__(self, cfg_pkl, data_root):
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root

    def process_large_batch(self, jobs, batch_size=1):
        """
        Process large batches with memory management
        """
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]

            # Clear CUDA cache before each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Process batch
            for source_path, audio_path, output_path, kwargs in batch:
                # Initialize SDK for each job to free memory
                SDK = StreamSDK(self.cfg_pkl, self.data_root)

                # Setup with memory-efficient params
                setup_kwargs = {
                    "max_size": 512,  # Reduce resolution
                    **kwargs
                }
                SDK.setup(source_path, output_path, **setup_kwargs)

                # Process
                audio, sr = librosa.load(audio_path, sr=16000)
                num_frames = math.ceil(len(audio) / 16000 * 25)
                SDK.setup_Nd(N_d=num_frames)

                aud_feat = SDK.wav2feat.wav2feat(audio)
                SDK.audio2motion_queue.put(aud_feat)
                SDK.close()

                # Add audio
                cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -c:v copy -c:a aac "{output_path}"'
                os.system(cmd)

                # Force cleanup
                del SDK
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# Usage
pipeline = MemoryEfficientPipeline(cfg_pkl, data_root)
large_job_list = [...]  # Many jobs
pipeline.process_large_batch(large_job_list, batch_size=1)
```

### Example 14: Parallel Processing Multiple Avatars

```python
from multiprocessing import Process, Queue
import multiprocessing as mp

def worker_process(gpu_id, job_queue, cfg_pkl, data_root):
    """Worker process for parallel generation"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    while True:
        try:
            job = job_queue.get(timeout=1)
            if job is None:  # Poison pill
                break

            source_path, audio_path, output_path, kwargs = job

            # Process job
            SDK = StreamSDK(cfg_pkl, data_root)
            SDK.setup(source_path, output_path, **kwargs)

            audio, sr = librosa.load(audio_path, sr=16000)
            num_frames = math.ceil(len(audio) / 16000 * 25)
            SDK.setup_Nd(N_d=num_frames)

            aud_feat = SDK.wav2feat.wav2feat(audio)
            SDK.audio2motion_queue.put(aud_feat)
            SDK.close()

            cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -c:v copy -c:a aac "{output_path}"'
            os.system(cmd)

            print(f"GPU {gpu_id} completed: {output_path}")

        except Exception as e:
            print(f"GPU {gpu_id} error: {e}")

def parallel_generate(jobs, cfg_pkl, data_root, num_gpus=2):
    """
    Process jobs in parallel across multiple GPUs
    """
    # Create job queue
    job_queue = Queue()

    # Add jobs to queue
    for job in jobs:
        job_queue.put(job)

    # Add poison pills
    for _ in range(num_gpus):
        job_queue.put(None)

    # Start worker processes
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(target=worker_process, args=(gpu_id, job_queue, cfg_pkl, data_root))
        p.start()
        processes.append(p)

    # Wait for completion
    for p in processes:
        p.join()

# Usage
jobs = [
    ("avatar1.jpg", "audio1.wav", "output1.mp4", {}),
    ("avatar2.jpg", "audio2.wav", "output2.mp4", {}),
    # ... more jobs
]

parallel_generate(
    jobs,
    cfg_pkl="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
    data_root="./checkpoints/ditto_trt_Ampere_Plus",
    num_gpus=2
)
```

---

## Complete Application Example

### Example 15: Web API Service

```python
from flask import Flask, request, send_file, jsonify
import uuid
import tempfile
from pathlib import Path

app = Flask(__name__)

# Initialize SDK globally
CFG_PKL = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
DATA_ROOT = "./checkpoints/ditto_trt_Ampere_Plus"
TEMP_DIR = "./temp"
Path(TEMP_DIR).mkdir(exist_ok=True)

@app.route('/generate', methods=['POST'])
def generate_talking_head():
    """
    API endpoint to generate talking head video

    Request:
        - source_image: Image file
        - audio: Audio file (WAV)
        - parameters: JSON with optional parameters

    Response:
        - video file (MP4)
    """
    try:
        # Get files
        if 'source_image' not in request.files or 'audio' not in request.files:
            return jsonify({"error": "Missing files"}), 400

        source_file = request.files['source_image']
        audio_file = request.files['audio']

        # Get parameters
        params = request.form.get('parameters', '{}')
        params = json.loads(params) if params else {}

        # Save uploaded files
        job_id = str(uuid.uuid4())
        source_path = os.path.join(TEMP_DIR, f"{job_id}_source.jpg")
        audio_path = os.path.join(TEMP_DIR, f"{job_id}_audio.wav")
        output_path = os.path.join(TEMP_DIR, f"{job_id}_output.mp4")

        source_file.save(source_path)
        audio_file.save(audio_path)

        # Generate video
        SDK = StreamSDK(CFG_PKL, DATA_ROOT)
        SDK.setup(source_path, output_path, **params)

        audio, sr = librosa.load(audio_path, sr=16000)
        num_frames = math.ceil(len(audio) / 16000 * 25)
        SDK.setup_Nd(N_d=num_frames)

        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
        SDK.close()

        # Add audio
        cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -c:v copy -c:a aac "{output_path}"'
        os.system(cmd)

        # Send result
        response = send_file(output_path, mimetype='video/mp4')

        # Cleanup
        os.remove(source_path)
        os.remove(audio_path)
        os.remove(SDK.tmp_output_path)

        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

These examples demonstrate the full range of capabilities of the Ditto inference system, from basic usage to advanced production scenarios. You can mix and match these techniques based on your specific requirements.
