# API Reference

Documentation for the custom classes and methods in the LiveKit Avatar Agent.

## Module: `livekit_avatar.main_agent`

### `entrypoint(ctx: agents.JobContext)`

Main agent entrypoint function called by LiveKit Agents framework.

**Parameters:**
- `ctx` (agents.JobContext): Context object containing room and participant info

**Flow:**
1. Creates video source and track
2. Initializes CustomAvatarWorker
3. Sets up Gemini LLM model
4. Starts AgentSession for conversation
5. Captures TTS audio and feeds to avatar
6. Handles graceful shutdown

**Example:**
```python
# Called automatically by agents.cli.run_app()
# No need to call directly
```

**Environment Variables Used:**
- `GCP_PROJECT_ID`: Google Cloud project ID
- `GCP_REGION`: Vertex AI region
- `GEMINI_MODEL`: Model name (default: gemini-live-2.5-flash-preview-native-audio-09-2025)
- `DATA_ROOT`: Path to Ditto model files
- `CFG_PKL`: Path to Ditto configuration pickle
- `SOURCE_PATH`: Path to avatar source image
- `AVATAR_WIDTH`: Video width in pixels
- `AVATAR_HEIGHT`: Video height in pixels

---

## Module: `livekit_avatar.custom_avatar_worker`

### Class: `CustomAvatarWorker`

Worker class that bridges LiveKit audio input to Ditto avatar generation.

#### Constructor

```python
def __init__(
    self,
    data_root: str,
    cfg_pkl: str,
    source_path: str,
    frame_width: int,
    frame_height: int,
    av_sync: rtc.AVSynchronizer,
)
```

**Parameters:**
- `data_root` (str): Path to Ditto model checkpoint directory
- `cfg_pkl` (str): Path to Ditto configuration pickle file
- `source_path` (str): Path to avatar source image/video
- `frame_width` (int): Output video width (e.g., 1280)
- `frame_height` (int): Output video height (e.g., 720)
- `av_sync` (rtc.AVSynchronizer): LiveKit AVSynchronizer for coordinated audio/video output

**Raises:**
- `ImportError`: If StreamSDK cannot be imported
- `FileNotFoundError`: If model files or source image not found

**Initialization Process:**
1. Initializes Ditto StreamSDK with model files
2. Runs warmup phase (generates 3 dummy frames to initialize GPU)
3. Sets up audio processing queues and threading
4. Ready to receive audio via `feed_audio()`

**Example:**
```python
# Create synchronized sources
video_source = rtc.VideoSource(1280, 720)
audio_source = rtc.AudioSource(sample_rate=16000, num_channels=1)
av_sync = rtc.AVSynchronizer(
    audio_source=audio_source,
    video_source=video_source,
    video_fps=50,
)

worker = CustomAvatarWorker(
    data_root="checkpoints/ditto_trt_custom2/",
    cfg_pkl="checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
    source_path="avatars/my_avatar.jpg",
    frame_width=1280,
    frame_height=720,
    av_sync=av_sync,
)
```

#### Method: `start()`

Starts the avatar worker's processing loops.

```python
def start(self) -> None
```

**Description:**
Begins audio processing and frame generation. Must be called before `feed_audio()`.

**Example:**
```python
worker.start()
# Worker now ready to receive audio
```

#### Method: `feed_audio(audio_frame)`

Feeds a TTS audio frame to the avatar for lip-sync animation.

```python
async def feed_audio(self, audio_frame: rtc.AudioFrame) -> None
```

**Parameters:**
- `audio_frame` (rtc.AudioFrame): Audio frame from TTS output
  - Expected format: PCM int16
  - Sample rate: Any (will resample to 16kHz)
  - Channels: Mono or stereo (converted to mono)

**Description:**
Buffers and processes audio for avatar animation. Audio is automatically:
1. Converted from int16 to float32
2. Resampled to 16kHz if needed
3. Chunked to appropriate size for Ditto
4. Fed to the generation pipeline

**Example:**
```python
async for frame_event in audio_stream:
    await worker.feed_audio(frame_event.frame)
```

**Thread Safety:** This method is async-safe and can be called from the main event loop.

#### Method: `set_state(state)`

Sets the avatar's animation state.

```python
def set_state(self, state: str) -> None
```

**Parameters:**
- `state` (str): One of:
  - `"idle"`: Default state with subtle breathing/blinking
  - `"listening"`: Active listening pose when user speaks
  - `"thinking"`: Contemplative expression during processing
  - `"speaking"`: Lip-synced speech animation

**Example:**
```python
# When user starts speaking
worker.set_state("listening")

# When agent starts speaking
worker.set_state("speaking")

# When conversation is idle
worker.set_state("idle")
```

**Note:** State changes affect which audio is generated:
- `idle/listening/thinking`: Silent audio (subtle animations)
- `speaking`: TTS-driven audio (lip-sync)

#### Method: `close()`

Gracefully shuts down the worker and releases resources.

```python
async def close(self) -> None
```

**Description:**
- Stops audio processing loop
- Flushes pending frames
- Closes Ditto SDK
- Releases GPU resources

**Example:**
```python
try:
    # ... use worker ...
finally:
    await worker.close()
```

**Important:** Always call `close()` to prevent resource leaks.

#### Private Methods

##### `_handle_generated_frame(frame_rgb, frame_idx, timestamp)`

Internal callback from Ditto SDK (runs in worker thread).

**Note:** Do not call directly. This is invoked by StreamSDK.

##### `_run_audio_processing()`

Internal async loop for audio processing.

##### `_idle_audio_generator()`

Internal generator for silent audio chunks.

##### `_warmup_model()`

Internal method called during initialization to pre-warm the Ditto model.

**Purpose:**
- Generates 3 dummy frames with silent audio
- Initializes CUDA memory allocations
- Loads TensorRT engines into GPU
- Eliminates cold start lag during first real frame

**When Called:**
- Automatically during `__init__()` after SDK setup
- Runs synchronously before worker is ready

**Impact:**
- Adds 1-2 seconds to initialization time
- Eliminates "frame capture behind schedule" warnings
- Ensures smooth real-time performance from first frame

**Note:** This is called automatically; do not invoke manually.

---

## Module: `stream_pipeline_online` (Ditto SDK Wrapper)

### Class: `StreamSDK`

Wrapper for Ditto's audio-driven avatar generation pipeline.

**Note:** This is part of the Ditto project. See Ditto documentation for full API.

#### Key Methods Used

##### `setup(source_path, output_path, frame_callback, online_mode, fps)`

Configures the avatar generation pipeline.

**Parameters:**
- `source_path` (str): Path to avatar source
- `output_path` (str): Path for video output (use "/dev/null" for streaming)
- `frame_callback` (callable): Function called for each generated frame
- `online_mode` (bool): Enable streaming mode (True for real-time)
- `fps` (int): Target frame rate

##### `setup_Nd(N_d, fade_in, fade_out, ctrl_info)`

Sets up frame count and fade effects.

**Parameters:**
- `N_d` (int): Total number of frames (use large number for continuous)
- `fade_in` (int, optional): Frames for fade-in effect
- `fade_out` (int, optional): Frames for fade-out effect
- `ctrl_info` (dict, optional): Per-frame control parameters

##### `run_chunk(audio_chunk, chunksize)`

Processes an audio chunk and generates frames.

**Parameters:**
- `audio_chunk` (np.ndarray): Float32 audio samples at 16kHz
- `chunksize` (tuple): Chunking parameters (3, 5, 2)

**Thread Safety:** Blocking call, run in executor.

##### `close()`

Closes the SDK and releases resources.

---

## Module: `livekit_client.token_server`

### Class: `TokenHTTPRequestHandler`

HTTP handler for token generation and file serving.

#### Endpoint: `POST /token`

Generates a LiveKit access token.

**Request Body:**
```json
{
  "room": "my-room-name",
  "identity": "user-id"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Example:**
```javascript
const response = await fetch('http://localhost:8000/token', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    room: 'my-avatar-room',
    identity: 'user123'
  })
});
const { token } = await response.json();
```

---

## Environment Variables Reference

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON | `/path/to/key.json` |
| `VERTEX_PROJECT_ID` | GCP project ID | `my-project-123` |

### Optional (Agent)

| Variable | Default | Description |
|----------|---------|-------------|
| `VERTEX_LOCATION` | `us-central1` | Vertex AI region |
| `GEMINI_MODEL` | `gemini-live-2.5-flash-preview-native-audio-09-2025` | Gemini model name |
| `DATA_ROOT` | `./checkpoints/ditto_trt_Ampere_Plus` | Ditto models path |
| `CFG_PKL` | `./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl` | Ditto config path |
| `SOURCE_PATH` | `./assets/source_image.png` | Avatar source image |
| `AVATAR_WIDTH` | `1280` | Video width |
| `AVATAR_HEIGHT` | `720` | Video height |

### Optional (LiveKit)

| Variable | Default | Description |
|----------|---------|-------------|
| `LIVEKIT_URL` | `ws://localhost:7880` | LiveKit server URL |
| `LIVEKIT_API_KEY` | `devkey` | API key |
| `LIVEKIT_API_SECRET` | `devsecret` | API secret |

---

## Event Handlers

### AgentSession Events

Events emitted by `agents.AgentSession`:

#### `user_turn_started`

Fired when user begins speaking (VAD detected).

```python
@agent_session.on("user_turn_started")
def on_user_started(data):
    print("User is speaking")
```

#### `user_turn_completed`

Fired when user stops speaking.

```python
@agent_session.on("user_turn_completed")
def on_user_stopped(data):
    print("User finished speaking")
```

#### `agent_started_speaking`

Fired when agent begins TTS output.

```python
@agent_session.on("agent_started_speaking")
def on_agent_speaking(data):
    print("Agent is speaking")
```

#### `agent_stopped_speaking`

Fired when agent finishes TTS output.

```python
@agent_session.on("agent_stopped_speaking")
def on_agent_stopped(data):
    print("Agent finished speaking")
```

---

## LiveKit Room Events (Client-Side)

### `RoomEvent.TrackSubscribed`

Fired when a new track is subscribed.

```javascript
room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
  if (track.kind === Track.Kind.Video) {
    // Attach video track to element
    track.attach(videoElement);
  }
});
```

### `RoomEvent.Disconnected`

Fired when disconnected from room.

```javascript
room.on(RoomEvent.Disconnected, () => {
  console.log('Disconnected from room');
  // Clean up UI
});
```

---

## Data Types

### `rtc.AudioFrame`

Audio frame from LiveKit.

**Properties:**
- `data` (bytes): Raw PCM audio data (int16)
- `sample_rate` (int): Samples per second (e.g., 24000)
- `num_channels` (int): Number of audio channels (1 or 2)
- `samples_per_channel` (int): Number of samples per channel

### `rtc.VideoFrame`

Video frame for LiveKit.

**Constructor:**
```python
rtc.VideoFrame(
    data: bytes,
    width: int,
    height: int,
    type: rtc.VideoBufferType,
)
```

**VideoBufferType Options:**
- `VideoBufferType.RGBA`: 4-channel RGBA
- `VideoBufferType.I420`: YUV420 planar (most efficient)

### `agents.JobContext`

Context passed to agent entrypoint.

**Properties:**
- `room` (rtc.Room): The LiveKit room
- `job` (agents.Job): Job information

---

## Utility Functions

### `rgb_to_i420(frame_rgb, width, height)`

Converts RGB numpy array to I420 format.

```python
def rgb_to_i420(frame_rgb: np.ndarray, width: int, height: int) -> bytes
```

**Parameters:**
- `frame_rgb` (np.ndarray): RGB image array [H, W, 3]
- `width` (int): Target width
- `height` (int): Target height

**Returns:**
- `bytes`: I420-encoded frame data

**Example:**
```python
frame_rgb = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
i420_data = rgb_to_i420(frame_rgb, 1280, 720)
```

---

## Constants

### Audio Configuration

```python
FPS = 50                    # Target frame rate
SAMPLE_RATE = 16000        # Ditto audio sample rate (Hz)
SPLIT_LEN = 6480           # Audio chunk size (samples)
CHUNKSIZE = (3, 5, 2)      # Ditto chunking parameters
```

### Queue Sizes

```python
QUEUE_MAX_SIZE = 100       # Max audio frames in buffer
```

---

## Error Handling

### Common Exceptions

#### `ImportError: StreamSDK not found`

**Cause:** `stream_pipeline_online.py` not in Python path

**Solution:**
```python
import sys
sys.path.append("/path/to/project/root")
```

#### `FileNotFoundError: Avatar source not found`

**Cause:** `SOURCE_PATH` points to non-existent file

**Solution:** Verify path and file existence

#### `RuntimeError: CUDA out of memory`

**Cause:** Insufficient GPU VRAM

**Solution:** Reduce resolution or close other GPU applications

---

## Best Practices

### Resource Management

```python
# Always use context managers or try/finally
worker = CustomAvatarWorker(...)
worker.start()
try:
    # ... use worker ...
finally:
    await worker.close()
```

### Error Logging

```python
import logging
logger = logging.getLogger(__name__)

try:
    await worker.feed_audio(frame)
except Exception as e:
    logger.error(f"Audio feed error: {e}", exc_info=True)
```

### Performance Monitoring

```python
import time

start = time.time()
await worker.feed_audio(frame)
latency = time.time() - start
logger.debug(f"Audio feed latency: {latency*1000:.1f}ms")
```
