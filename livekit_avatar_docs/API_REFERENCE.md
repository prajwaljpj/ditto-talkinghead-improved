# API Reference

Complete API documentation for all components in the LiveKit Avatar system.

## Table of Contents

1. [DittoVideoGeneratorDecoupled](#dittovideogeneratordecoupled)
2. [StreamSDK](#streamsdk)
3. [Agent Worker](#agent-worker)
4. [Avatar Worker](#avatar-worker)
5. [Token Server](#token-server)
6. [Configuration Options](#configuration-options)

---

## DittoVideoGeneratorDecoupled

`livekit_avatar/ditto_video_generator_decoupled.py`

The main video generator class that implements LiveKit's `VideoGenerator` interface.

### Class Definition

```python
class DittoVideoGeneratorDecoupled(VideoGenerator):
    """
    Decoupled architecture: Processing and yielding are INDEPENDENT.
    
    Three concurrent tasks:
    - Task 1 (background): Accumulate audio continuously
    - Task 2 (background): Process chunks continuously  
    - Task 3 (main): Yield frames as they become available
    """
```

### Constructor

```python
def __init__(
    self, 
    options: AvatarOptions, 
    data_root: str, 
    cfg_pkl: str, 
    source_path: str
):
    """
    Initialize the Ditto video generator.
    
    Args:
        options: AvatarOptions with video/audio settings
        data_root: Path to TensorRT model directory
        cfg_pkl: Path to configuration pickle file
        source_path: Path to avatar source image
    """
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `options` | `AvatarOptions` | Video dimensions, FPS, audio settings |
| `data_root` | `str` | Directory containing TRT engines |
| `cfg_pkl` | `str` | Config pickle file path |
| `source_path` | `str` | Avatar image path (JPG/PNG) |

### Internal Configuration

```python
# Audio processing parameters
self.chunksize = (3, 5, 2)      # (padding, valid, overlap) frames
self.split_len = 6480           # Audio samples per chunk
self.ditto_sample_rate = 16000  # Ditto expects 16kHz audio

# Queue sizes
self.MAX_QUEUE_SIZE = 30        # Max buffered frames

# Timing
self._target_frame_time_ms = 1000.0 / options.video_fps  # 40ms for 25fps
```

### Methods

#### `push_audio(frame)`

```python
async def push_audio(
    self, 
    frame: rtc.AudioFrame | AudioSegmentEnd
) -> None:
    """
    Push an audio frame for processing.
    
    Args:
        frame: LiveKit AudioFrame or AudioSegmentEnd marker
        
    Note:
        Audio is automatically resampled to 16kHz mono if needed.
    """
```

#### `clear_buffer()`

```python
def clear_buffer(self) -> None:
    """
    Clear all pending audio from the buffer.
    
    Use when:
        - User interrupts the agent
        - Conversation context changes
        - Starting a new response
    """
```

#### `__aiter__()`

```python
def __aiter__(
    self
) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
    """
    Async iterator that yields paired (audio, video) frames.
    
    Yields:
        Alternating rtc.AudioFrame and rtc.VideoFrame pairs
        
    Note:
        Audio is always yielded before its corresponding video frame
        to maintain lip sync.
    """
```

#### `aclose()`

```python
async def aclose(self) -> None:
    """
    Clean up resources and stop all background tasks.
    
    Should be called when:
        - Avatar worker is shutting down
        - Room disconnects
        - Error recovery
    """
```

### Diagnostics

The generator tracks timing statistics for debugging:

```python
# Access diagnostic data
generator._log_wait_statistics()

# Statistics tracked:
# - _audio_queue_wait_times: Time waiting for audio
# - _audio_lock_wait_times: Time waiting for buffer lock
# - _buffer_event_wait_times: Time waiting for enough audio
# - _ditto_processing_times: Actual Ditto inference time
# - _pair_event_wait_times: Time waiting for complete pairs
```

---

## StreamSDK

`stream_pipeline_online.py`

The core streaming pipeline that processes audio and generates video frames.

### Class Definition

```python
class StreamSDK:
    """
    Streaming SDK for real-time audio-to-video generation.
    
    Features:
        - Online streaming mode for real-time processing
        - Multi-threaded pipeline for parallelism
        - Frame callback support for integration
    """
```

### Constructor

```python
def __init__(self, cfg_pkl: str, data_root: str, **kwargs):
    """
    Initialize the StreamSDK.
    
    Args:
        cfg_pkl: Path to configuration pickle file
        data_root: Path to model checkpoint directory
        **kwargs: Override default configuration options
    """
```

### Methods

#### `setup()`

```python
def setup(
    self,
    source_path: str,
    output_path: str,
    frame_callback: Callable = None,
    **kwargs
) -> None:
    """
    Setup the pipeline for a specific avatar and output.
    
    Args:
        source_path: Path to avatar image or video
        output_path: Output video path (or "/dev/null" for streaming)
        frame_callback: Optional callback for each frame
            Signature: callback(frame_rgb, frame_idx, timestamp)
        **kwargs: Pipeline configuration options
        
    Configuration Options:
        max_size (int): Maximum image dimension (default: 1920)
        crop_scale (float): Face crop scale (default: 2.3)
        fps (int): Output frame rate (default: 25)
        online_mode (bool): Enable streaming mode (default: False)
        emo (int): Emotion index (default: 4 = neutral)
        sampling_timesteps (int): LMDM diffusion steps (default: 50)
    """
```

#### `setup_Nd()`

```python
def setup_Nd(
    self,
    N_d: int,
    fade_in: int = -1,
    fade_out: int = -1,
    ctrl_info: dict = None
) -> None:
    """
    Configure the expected number of output frames.
    
    Args:
        N_d: Expected number of frames (-1 for unlimited)
        fade_in: Frames for fade-in effect (-1 to disable)
        fade_out: Frames for fade-out effect (-1 to disable)
        ctrl_info: Per-frame control information
    """
```

#### `run_chunk()`

```python
def run_chunk(
    self,
    audio_chunk: np.ndarray,
    chunksize: tuple = (3, 5, 2)
) -> None:
    """
    Process a chunk of audio.
    
    Args:
        audio_chunk: Float32 audio samples at 16kHz
        chunksize: (padding, valid, overlap) frame counts
            - padding: Initial warmup frames
            - valid: Actual output frames
            - overlap: Context for next chunk
            
    Note:
        Audio length should be: (sum(chunksize) - overlap) * 640 samples
        Default: (3+5+2-2) * 640 = 5120 samples
        
        Actual split_len used: 6480 samples for best quality
    """
```

#### `close()`

```python
def close(self) -> None:
    """
    Shutdown the pipeline and wait for all workers.
    
    Raises:
        Exception: Re-raises any exception from worker threads
    """
```

### Pipeline Configuration Options

#### Avatar Registration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_size` | int | 1920 | Max image dimension |
| `template_n_frames` | int | -1 | Number of template frames |
| `crop_scale` | float | 2.3 | Face crop scale factor |
| `crop_vx_ratio` | float | 0 | Horizontal crop offset |
| `crop_vy_ratio` | float | -0.125 | Vertical crop offset |
| `crop_flag_do_rot` | bool | True | Enable rotation correction |
| `smo_k_s` | int | 13 | Source smoothing kernel |

#### Audio-to-Motion Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `overlap_v2` | int | 10 | Frame overlap for continuity |
| `sampling_timesteps` | int | 50 | Diffusion sampling steps |
| `online_mode` | bool | False | Enable streaming mode |
| `smo_k_d` | int | 3 | Driving smoothing kernel |
| `fix_kp_cond` | int | 0 | Fixed keypoint conditioning |

#### Motion Stitch Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `N_d` | int | -1 | Expected frame count |
| `relative_d` | bool | True | Use relative motion |
| `drive_eye` | bool | None | Drive eye motion |
| `flag_stitching` | bool | True | Enable motion stitching |
| `fade_type` | str | "" | Fade type ("", "d0", "s") |

#### Emotion and Conditioning

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `emo` | int/list | 4 | Emotion index or array |
| `eye_f0_mode` | bool | False | Eye motion from F0 |
| `ch_info` | dict | None | Custom conditioning info |

#### Emotion Index Values

| Index | Emotion |
|-------|---------|
| 0 | Angry |
| 1 | Contempt |
| 2 | Disgusted |
| 3 | Fear |
| 4 | Happy (Neutral) |
| 5 | Sad |
| 6 | Surprised |

---

## Agent Worker

`livekit_avatar/agent_worker.py`

The conversation agent that handles STT, LLM, and TTS.

### Entry Point

```python
async def entrypoint(ctx: agents.JobContext):
    """
    Main agent entrypoint.
    
    Called by LiveKit when a participant joins a room.
    
    Args:
        ctx: JobContext with room and connection info
    """
```

### Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT_ID` | - | Google Cloud project ID |
| `GCP_REGION` | us-central1 | Vertex AI region |
| `GEMINI_MODEL` | gemini-live-2.5-flash-preview-native-audio-09-2025 | Gemini model name |
| `DATA_ROOT` | ./checkpoints/ditto_trt_Ampere_Plus | Ditto model path |
| `CFG_PKL` | ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl | Config path |
| `SOURCE_PATH` | ./assets/source_image.png | Avatar image |
| `AVATAR_WIDTH` | 1280 | Video width |
| `AVATAR_HEIGHT` | 720 | Video height |
| `AVATAR_FPS` | 25 | Video frame rate |

### Functions

#### `launch_avatar_worker()`

```python
async def launch_avatar_worker(
    ctx: agents.JobContext,
    avatar_identity: str
) -> subprocess.Popen:
    """
    Launch the avatar worker as a subprocess.
    
    Args:
        ctx: JobContext with room info
        avatar_identity: Identity for avatar in room
        
    Returns:
        Popen object for the subprocess
        
    Note:
        Creates a JWT token for the avatar with:
        - Room join permission
        - Publish-on-behalf attribute
        - Agent participant kind
    """
```

### Agent State Events

The agent broadcasts state changes to the avatar:

```python
@session.on("agent_state_changed")
def on_agent_state_changed(event):
    # States: "idle", "listening", "thinking", "speaking"
    state_msg = {
        "type": "agent_state",
        "state": event.new_state,
        "old_state": event.old_state,
    }
    # Sent via data channel to avatar worker
```

---

## Avatar Worker

`livekit_avatar/avatar_worker.py`

The video generation worker that receives audio and publishes synchronized media.

### Entry Point

```python
async def main(api_url: str, api_token: str):
    """
    Main avatar worker function.
    
    Args:
        api_url: LiveKit server URL
        api_token: JWT token from agent worker
    """
```

### Configuration (Environment Variables)

Same as Agent Worker, plus:

| Variable | Default | Description |
|----------|---------|-------------|
| `LIVEKIT_URL` | - | LiveKit server URL |
| `LIVEKIT_TOKEN` | - | JWT token (set by agent) |

### Event Handlers

```python
@room.on("data_received")
def _on_data_received(data: rtc.DataPacket):
    """Handle agent state updates via data channel."""
    
@room.on("participant_disconnected")
def _on_participant_disconnected(participant):
    """Stop when agent disconnects."""
    
@room.on("disconnected")
def _on_disconnected():
    """Handle room disconnection."""
```

---

## Token Server

`livekit_client/token_server.py`

Development server for generating LiveKit tokens and serving the web client.

### Usage

```bash
python livekit_client/token_server.py
```

### Endpoints

#### `POST /token`

Generate a LiveKit access token.

**Request:**
```json
{
    "room": "room-name",
    "identity": "user-identity"
}
```

**Response:**
```json
{
    "token": "eyJ..."
}
```

#### `GET /simple_client.html`

Serves the web client HTML page.

### Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `LIVEKIT_API_KEY` | devkey | LiveKit API key |
| `LIVEKIT_API_SECRET` | devsecret | LiveKit API secret |

---

## Configuration Options

### AvatarOptions

```python
from livekit.agents.voice.avatar import AvatarOptions

options = AvatarOptions(
    video_width=1280,        # Output video width
    video_height=720,        # Output video height
    video_fps=25,            # Frames per second
    audio_sample_rate=16000, # Audio sample rate
    audio_channels=1,        # Audio channels (mono)
)
```

### RoomOutputOptions

```python
from livekit.agents.voice.room_io import RoomOutputOptions

output_options = RoomOutputOptions(
    audio_enabled=False,        # Don't publish audio (avatar does)
    transcription_enabled=True, # Enable transcription
)
```

### AvatarRunner Configuration

```python
runner = AvatarRunner(
    room,
    audio_recv=DataStreamAudioReceiver(room),
    video_gen=video_gen,
    options=avatar_options,
    _queue_size_ms=500,  # Audio queue size (default: 100ms)
)
```

---

## Helper Functions

### `rgb_to_i420()`

```python
def rgb_to_i420(
    frame_rgb: np.ndarray,
    width: int,
    height: int
) -> bytes:
    """
    Convert RGB frame to I420 (YUV420p) format for LiveKit.
    
    Args:
        frame_rgb: RGB numpy array (H, W, 3)
        width: Target width
        height: Target height
        
    Returns:
        I420 encoded bytes
    """
```

---

## Type Definitions

### AudioSegmentEnd

Marker indicating the end of an audio segment (TTS complete):

```python
from livekit.agents.voice.avatar import AudioSegmentEnd

# Usage in video generator
if isinstance(frame, AudioSegmentEnd):
    # Switch to idle animation
    self._is_generating_idle = True
```

### VideoFrame

LiveKit video frame:

```python
video = rtc.VideoFrame(
    data=i420_bytes,
    width=width,
    height=height,
    type=rtc.VideoBufferType.I420,
)
```

### AudioFrame

LiveKit audio frame:

```python
audio = rtc.AudioFrame(
    data=pcm_bytes,
    sample_rate=16000,
    num_channels=1,
    samples_per_channel=640,
)
```

