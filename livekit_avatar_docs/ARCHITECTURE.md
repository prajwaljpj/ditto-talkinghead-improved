# System Architecture

This document provides a comprehensive overview of the LiveKit Avatar system architecture, explaining how each component works and interacts with others.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Two-Worker Design](#two-worker-design)
3. [Data Flow](#data-flow)
4. [Component Deep Dive](#component-deep-dive)
5. [Threading Model](#threading-model)
6. [Synchronization Strategy](#synchronization-strategy)

---

## High-Level Architecture

The system follows a **two-worker architecture** that separates concerns for optimal performance:

```
                                   LIVEKIT SERVER
                                   ┌──────────────┐
                                   │   Signaling  │
                                   │   + TURN     │
                                   └──────┬───────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           │                           ▼
    ┌─────────────────┐                   │               ┌─────────────────┐
    │     CLIENT      │                   │               │  AGENT WORKER   │
    │   (Browser)     │                   │               │   (Process 1)   │
    │                 │                   │               │                 │
    │ ┌─────────────┐ │                   │               │ ┌─────────────┐ │
    │ │ Microphone  │─┼───user audio──────┼───────────────┼▶│ AgentSession│ │
    │ └─────────────┘ │                   │               │ │  (Gemini)   │ │
    │                 │                   │               │ └──────┬──────┘ │
    │ ┌─────────────┐ │                   │               │        │        │
    │ │   Video     │◀┼───────────────────┼───────────────┼────────┼────────┤
    │ │   Display   │ │                   │               │        │ TTS    │
    │ └─────────────┘ │                   │               │        │ Audio  │
    │                 │                   │               │        ▼        │
    │ ┌─────────────┐ │                   │               │ ┌─────────────┐ │
    │ │   Audio     │◀┼───────────────────┼───────────────┼─│ DataStream  │ │
    │ │   Speaker   │ │                   │               │ │   Output    │ │
    │ └─────────────┘ │                   │               │ └──────┬──────┘ │
    └─────────────────┘                   │               └────────┼────────┘
              ▲                           │                        │
              │                           │                        │ DataChannel
              │        avatar             │                        │ (audio bytes)
              │        audio+video        │                        ▼
              │                           │               ┌─────────────────┐
              │                           │               │  AVATAR WORKER  │
              │                           │               │   (Process 2)   │
              │                           │               │                 │
              │                           │               │ ┌─────────────┐ │
              │                           │               │ │ DataStream  │ │
              │                           │               │ │  Receiver   │ │
              │                           │               │ └──────┬──────┘ │
              │                           │               │        │        │
              │                           │               │        ▼        │
              │                           │               │ ┌─────────────┐ │
              │                           │               │ │   Ditto     │ │
              │                           │               │ │ VideoGen    │ │
              │                           │               │ └──────┬──────┘ │
              │                           │               │        │        │
              │                           │               │        ▼        │
              └───────────────────────────┼───────────────┼──audio+video────┘
                                          │               │ Published to   │
                                          │               │ LiveKit Room   │
                                          │               └─────────────────┘
```

### Why Two Workers?

The two-worker design provides several advantages:

1. **Resource Isolation**: GPU-intensive video generation doesn't block conversation processing
2. **Fault Tolerance**: If video generation fails, conversation can continue (gracefully)
3. **Scalability**: Workers can potentially run on different machines
4. **Simplified Development**: Each worker has a single responsibility

---

## Two-Worker Design

### Agent Worker (`agent_worker.py`)

The Agent Worker handles all conversation logic:

```python
# Core responsibilities:
# 1. Connect to LiveKit room
# 2. Initialize Gemini model for conversation
# 3. Launch Avatar Worker subprocess
# 4. Route TTS audio to Avatar via DataStream
# 5. Handle conversation state changes

async def entrypoint(ctx: agents.JobContext):
    # Connect to room
    await ctx.connect()
    
    # Create Gemini model
    llm_model = google.beta.realtime.RealtimeModel(
        vertexai=True,
        project=GCP_PROJECT_ID,
        model=GEMINI_MODEL,
    )
    
    # Create voice agent
    session = AgentSession(llm=llm_model)
    
    # Launch avatar subprocess
    avatar_process = await launch_avatar_worker(ctx, AVATAR_IDENTITY)
    
    # Configure DataStream output (sends TTS to avatar)
    session.output.audio = DataStreamAudioOutput(
        ctx.room,
        destination_identity=AVATAR_IDENTITY,
    )
    
    # Start conversation
    await session.start(agent=voice_agent, room=ctx.room)
```

### Avatar Worker (`avatar_worker.py`)

The Avatar Worker handles video generation:

```python
# Core responsibilities:
# 1. Connect to LiveKit room (with avatar token)
# 2. Receive TTS audio via DataStream
# 3. Generate synchronized video frames
# 4. Publish both audio and video to room

async def main(api_url: str, api_token: str):
    # Create avatar options
    avatar_options = AvatarOptions(
        video_width=1280,
        video_height=720,
        video_fps=25,
    )
    
    # Initialize Ditto video generator
    video_gen = DittoVideoGenerator(
        options=avatar_options,
        source_path=SOURCE_PATH,
    )
    
    # Connect to room
    room = rtc.Room()
    await room.connect(api_url, api_token)
    
    # Create runner with DataStream receiver
    runner = AvatarRunner(
        room,
        audio_recv=DataStreamAudioReceiver(room),
        video_gen=video_gen,
    )
    
    await runner.start()
```

---

## Data Flow

### Complete Request-Response Flow

```
1. USER SPEAKS
   └─▶ Browser captures microphone audio
       └─▶ WebRTC audio track published to LiveKit
           └─▶ Agent Worker receives audio via AgentSession
               └─▶ Gemini processes speech (STT → LLM → TTS)

2. AGENT RESPONDS
   └─▶ Gemini generates TTS audio
       └─▶ DataStreamAudioOutput sends audio to Avatar
           └─▶ Avatar Worker receives via DataStreamAudioReceiver

3. AVATAR GENERATES VIDEO
   └─▶ DittoVideoGenerator receives audio chunks
       └─▶ Audio buffering (split_len = 6480 samples)
           └─▶ Chunk processing via StreamSDK
               └─▶ Audio → Motion (LMDM)
                   └─▶ Motion → Video frames (Warp + Decode)

4. AVATAR PUBLISHES
   └─▶ Video frames paired with audio frames
       └─▶ AvatarRunner publishes both tracks to room
           └─▶ WebRTC streams to Client
               └─▶ Browser displays video + plays audio
```

### Audio Processing Timeline

```
Time (ms)    0        40       80      120      160      200      240
             │        │        │        │        │        │        │
TTS Audio    ├────────┼────────┼────────┼────────┼────────┼────────┤
             │ Frame 0│ Frame 1│ Frame 2│ Frame 3│ Frame 4│ Frame 5│
             │        │        │        │        │        │        │
Buffering    ├────────────────────────────────────────┐
             │     Accumulate 6480 samples            │
             │     (~405ms at 16kHz)                  │
             └────────────────────────────────────────┤
                                                      │
Ditto Chunk  ─────────────────────────────────────────┼────────────┐
Processing                                            │  ~100ms    │
                                                      │            │
Video Out    ─────────────────────────────────────────────────────▶├──┼──┼──
                                                                   │F0│F1│F2│...
```

---

## Component Deep Dive

### DittoVideoGeneratorDecoupled

The heart of video generation, implementing LiveKit's `VideoGenerator` interface:

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

#### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `split_len` | 6480 | Audio samples per chunk (~405ms at 16kHz) |
| `chunksize` | (3, 5, 2) | Padding, process, overlap frames |
| `MAX_QUEUE_SIZE` | 30 | Maximum buffered frames |

#### Internal Queues

```
                        Audio Accumulation Task
                        ┌─────────────────────┐
    push_audio() ──────▶│ _audio_queue        │
                        │ (asyncio.Queue)     │
                        └─────────┬───────────┘
                                  │
                                  ▼
                        ┌─────────────────────┐
                        │ _sdk_audio_buffer   │◀──── Resampled + chunked
                        │ (numpy array)       │
                        └─────────┬───────────┘
                                  │
                                  │ buffer >= split_len
                                  ▼
                        Chunk Processing Task
                        ┌─────────────────────┐
                        │ sdk.run_chunk()     │
                        │ (StreamSDK)         │
                        └─────────┬───────────┘
                                  │
                                  │ frame callback
                                  ▼
                        ┌─────────────────────┐
                        │ _video_queue_internal│
                        └─────────┬───────────┘
                                  │
                                  │ pair with audio
                                  ▼
                        ┌─────────────────────┐
                        │ _paired_frames      │
                        │ [(audio, video)]    │
                        └─────────┬───────────┘
                                  │
                                  │ main loop
                                  ▼
                              yield frames
```

### StreamSDK Pipeline

The `StreamSDK` class orchestrates the video generation pipeline with multiple worker threads:

```
Audio Input
    │
    ▼
┌─────────────────┐
│ wav2feat        │  Convert audio to features (Hubert)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ condition_handler│  Add emotion/conditioning
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ audio2motion    │  LMDM diffusion model
│ (LMDM)          │  Audio → Motion keypoints
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ motion_stitch   │  Combine source + driving motion
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ warp_f3d        │  Warp source features
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ decode_f3d      │  Decode to RGB image
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ putback         │  Composite onto original frame
└────────┬────────┘
         │
         ▼
    Video Frame
```

---

## Threading Model

### StreamSDK Thread Architecture

```
Main Thread                     Worker Threads
────────────                    ──────────────

setup()
    │
    ├──▶ audio2motion_worker ──────────────────▶ Thread 1
    ├──▶ motion_stitch_worker ─────────────────▶ Thread 2
    ├──▶ warp_f3d_worker ──────────────────────▶ Thread 3
    ├──▶ decode_f3d_worker ────────────────────▶ Thread 4
    ├──▶ putback_worker ───────────────────────▶ Thread 5
    └──▶ writer_worker ────────────────────────▶ Thread 6
         │
run_chunk()                     
    │                           Thread 1: audio2motion_queue
    └──▶ audio_feat ──────────▶ │
                                ▼
                                Thread 2: motion_stitch_queue
                                │
                                ▼
                                Thread 3: warp_f3d_queue
                                │
                                ▼
                                Thread 4: decode_f3d_queue
                                │
                                ▼
                                Thread 5: putback_queue
                                │
                                ▼
                                Thread 6: writer_queue
                                │
                                ▼
                            frame_callback(rgb, idx, ts)
```

### Async Architecture (DittoVideoGenerator)

```
Event Loop
──────────

_main_loop() ◀────────────────────────────────────────┐
    │                                                  │
    │ asyncio.create_task()                           │
    ├──────────────────────▶ _audio_accumulation_task │
    │                                │                 │
    │ asyncio.create_task()          │                 │
    └──────────────────────▶ _chunk_processing_task   │
                                     │                 │
                                     │ _pair_ready_event
                                     └─────────────────┘
```

---

## Synchronization Strategy

### Audio-Video Pairing

The system uses **order-based pairing** (FIFO) rather than timestamp-based:

```python
def _try_pair_frames(self):
    """Pair audio and video frames by order (FIFO)."""
    while self._audio_queue_internal and self._video_queue_internal:
        audio = self._audio_queue_internal.pop(0)  # First audio
        video = self._video_queue_internal.pop(0)  # First video
        self._paired_frames.append((audio, video))
        self._loop.call_soon_threadsafe(self._pair_ready_event.set)
```

### Why Order-Based Pairing Works

1. **Audio chunks produce predictable video frames**: Each audio chunk → N video frames
2. **Queues maintain order**: Both audio and video queues are FIFO
3. **Timing is preserved**: Audio sample rate matches video frame rate (640 samples per frame at 25fps)

### Key Synchronization Events

| Event | Purpose |
|-------|---------|
| `_pair_ready_event` | Signals complete (audio, video) pair available |
| `_buffer_ready_event` | Signals audio buffer has enough samples |
| `_stop_event` | Signals shutdown to all tasks |
| `_sdk_lock` | Protects audio buffer access |

### Frame Rate Math

```
Audio: 16,000 Hz (16,000 samples/second)
Video: 25 fps (25 frames/second)

Samples per video frame = 16,000 / 25 = 640 samples

Chunk size (split_len) = 6,480 samples
Frames per chunk = 6,480 / 640 ≈ 10 frames (with overlap)

Actual: chunksize = (3, 5, 2)
  - 3 frames padding (beginning warmup)
  - 5 frames valid output
  - 2 frames overlap (context for next chunk)
  
Output = 5 frames × 640 = 3,200 samples advanced
```

---

## Error Handling

### Worker Exception Propagation

```python
# StreamSDK pattern
def worker_function(self):
    try:
        self._worker_function_impl()
    except Exception as e:
        self.worker_exception = e
        self.stop_event.set()  # Signal all workers to stop

# In close():
if self.worker_exception is not None:
    raise self.worker_exception
```

### Graceful Shutdown

```
close() called
    │
    ├──▶ Put None in audio2motion_queue (sentinel)
    │
    ├──▶ Wait for all threads to join
    │       │
    │       ├─▶ audio2motion_worker sees None → puts None in next queue
    │       ├─▶ motion_stitch_worker sees None → puts None in next queue
    │       ├─▶ ... cascades through all workers
    │       └─▶ writer_worker sees None → exits
    │
    └──▶ Check worker_exception, re-raise if any
```

---

## Next Steps

- [Setup Guide](./SETUP_GUIDE.md) - Get the system running
- [Pipeline Deep Dive](./PIPELINE_DEEP_DIVE.md) - Understand StreamSDK internals
- [API Reference](./API_REFERENCE.md) - Detailed API documentation

