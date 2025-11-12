# WebRTC Audio-Video Synchronization Architecture

## Overview

This document explains how `signaling_server_v2.py` achieves perfect audio-video synchronization for the Ditto avatar system using **FIFO pairing with shared timestamps**.

## The Fundamental Principle

**For audio passthrough systems (echoing back input audio with generated video):**

> Audio chunk N should play at the same time as the video frame N that it helped generate.

This is achieved through **FIFO (First-In-First-Out) pairing** + **shared timestamps**.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Browser (Client)                                            │
│  Audio Input: 48kHz stereo →                               │
│  Audio Output: ← 48kHz mono (synchronized)                 │
│  Video Output: ← 1280x720 @ 25fps                          │
└─────────────────────────────────────────────────────────────┘
                    ↓ WebRTC                    ↑
┌─────────────────────────────────────────────────────────────┐
│ AudioPassthrough                                            │
│  ┌──────────────────────────────────────────────┐          │
│  │ 1. Receive 48kHz stereo frame                │          │
│  │ 2. Convert stereo → mono (averaging)         │          │
│  │ 3. Downsample to 16kHz for Ditto            │          │
│  │ 4. Store 48kHz mono for playback (NO timestamp) │      │
│  └──────────────────────────────────────────────┘          │
│                                                             │
│  Buffer: deque[np.ndarray]  ← Audio chunks without PTS     │
│  Event: audio_ready         ← Signals when chunks available│
└─────────────────────────────────────────────────────────────┘
        ↓ 16kHz chunks                     ↓ 48kHz chunks
┌─────────────────────────┐    ┌───────────────────────────────┐
│ Ditto Model             │    │ AVSynchronizer                │
│  ┌──────────────────┐   │    │  ┌────────────────────────┐  │
│  │ Audio → Motion   │   │    │  │ Wait for video frame   │  │
│  │ Motion → Video   │   │    │  │ Wait for audio chunk   │  │
│  │ Timestamp = f/25 │   │    │  │ Pair them (FIFO)       │  │
│  └──────────────────┘   │    │  │ Both get SAME PTS      │  │
│                         │    │  └────────────────────────┘  │
└─────────────────────────┘    └───────────────────────────────┘
        ↓ Video frames                  ↓ Paired (V, A, PTS)
┌─────────────────────────────────────────────────────────────┐
│ BufferedVideoTrack / BufferedAudioTrack                     │
│  ┌──────────────────────────────────────────────┐          │
│  │ Video: frame.pts = int(timestamp * 25)       │          │
│  │ Audio: frame.pts = int(timestamp * 48000)    │          │
│  │                                               │          │
│  │ Both resolve to same playback time:          │          │
│  │   Video: 100/25 = 4.0s                      │          │
│  │   Audio: 192000/48000 = 4.0s                │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                    ↓ RTP packets
┌─────────────────────────────────────────────────────────────┐
│ WebRTC Jitter Buffer (Client)                              │
│  Plays frames based on PTS → Perfect synchronization       │
└─────────────────────────────────────────────────────────────┘
```

## Why FIFO Pairing Works

### The Causality Relationship

The Ditto model creates a direct cause-and-effect relationship between audio and video:

1. **Input**: Audio chunk 0 (samples 0-6399 @ 16kHz = 400ms)
2. **Processing**: Model generates motion from audio
3. **Output**: Video frames 0-9 (timestamps 0.00s-0.36s @ 25fps)

For perfect lip-sync, we must play back:
- Audio chunk 0 with video frames 0-9
- Audio chunk 1 with video frames 10-19
- Audio chunk N with video frames (N*10) through (N*10+9)

### FIFO Preserves This Relationship

```python
# Audio arrives and is stored in FIFO order
audio_chunks = [chunk_0, chunk_1, chunk_2, ...]

# Video is generated in FIFO order
video_frames = [
    frame_0 (from chunk_0), frame_1 (from chunk_0), ..., frame_9 (from chunk_0),
    frame_10 (from chunk_1), frame_11 (from chunk_1), ..., frame_19 (from chunk_1),
    ...
]

# FIFO pairing maintains the relationship
pair(frame_0, chunk_0.sub_chunk_0)  # ✓ Correct
pair(frame_1, chunk_0.sub_chunk_1)  # ✓ Correct
pair(frame_10, chunk_1.sub_chunk_0) # ✓ Correct
```

**Key insight**: We split each 400ms audio chunk into 10× 40ms sub-chunks (for 25fps). FIFO pairing automatically matches video frame N with audio sub-chunk N.

## The Shared Timestamp Mechanism

### Problem with Independent Timestamps

**Wrong approach (old code):**
```python
# Audio gets timestamp from when it was received
audio_ts = samples_received / 48000  # e.g., 2.123s

# Video gets timestamp from model
video_ts = frame_idx / 25            # e.g., 2.000s

# Try to match them → diff = 123ms → Drift!
```

**Issues**:
- Timestamps have different meanings (capture time vs playback position)
- Rounding errors accumulate
- Network delays cause drift
- Complex matching logic required

### Correct Approach (Current Implementation)

**FIFO pairing with shared timestamp:**
```python
# 1. Audio stored WITHOUT timestamp
audio_chunks = deque([chunk_0, chunk_1, chunk_2, ...])

# 2. Video generated WITH timestamp from model
video, frame_idx, timestamp = video_generator.get_next_frame()
# timestamp = frame_idx / 25 (e.g., 2.000s)

# 3. Pop next audio chunk (FIFO order)
audio = audio_chunks.popleft()  # Gets chunk_N

# 4. Both use SAME timestamp
video.pts = int(timestamp * 25)      # = 50
audio.pts = int(timestamp * 48000)   # = 96000

# 5. Both resolve to same playback time
# Video: 50 / 25 = 2.000s
# Audio: 96000 / 48000 = 2.000s
```

**Benefits**:
- ✅ **Zero drift**: Mathematically impossible (same timestamp value)
- ✅ **Simple code**: Just FIFO pop, no matching logic
- ✅ **Self-documenting**: Order reflects causality
- ✅ **Robust**: Works regardless of network delays or model speed

## Implementation Details

### Component 1: AudioPassthrough

**Purpose**: Convert and buffer audio without assigning timestamps

```python
class AudioPassthrough:
    def __init__(self):
        self.playback_chunks = deque(maxlen=500)  # Audio without PTS
        self.audio_ready = asyncio.Event()        # Signaling

    def get_16khz_chunk(self):
        # ... Convert 48kHz → 16kHz for model ...
        # ... Split 48kHz into 40ms chunks ...

        for sub_chunk in split_to_40ms(chunk_48k):
            self.playback_chunks.append(sub_chunk)  # NO timestamp!

        self.audio_ready.set()  # Signal availability
```

**Key**: Audio is just raw data, waiting to be paired with video.

### Component 2: VideoGenerator

**Purpose**: Generate video with timestamps from Ditto model

```python
class VideoGenerator:
    def on_frame(self, frame_rgb, frame_idx, timestamp):
        # Timestamp comes from model: frame_idx / 25
        self.frame_queue.put_nowait((frame_rgb, frame_idx, timestamp))
```

**Key**: Timestamp is authoritative (from model's audio processing).

### Component 3: AVSynchronizer

**Purpose**: Pair audio and video, both inherit video's timestamp

```python
class AVSynchronizer:
    async def get_next(self):
        # Get video with timestamp
        video, frame_idx, timestamp = await video_gen.get_next_frame()

        # Wait for audio using Event (efficient)
        while True:
            audio = audio_pass.get_next_playback_chunk()
            if audio:
                break
            await audio_pass.audio_ready.wait()  # Efficient waiting
            audio_pass.audio_ready.clear()

        # Return pair - audio gets video's timestamp
        return (video, frame_idx, timestamp, audio)
```

**Key**: Event-based waiting (no busy polling), FIFO pairing.

### Component 4: WebRTC Tracks

**Purpose**: Assign PTS based on shared timestamp

```python
class BufferedVideoTrack:
    async def recv(self):
        video, frame_idx, timestamp, audio = await sync.get_next()

        # Add audio with video's timestamp
        audio_track.add_chunk(audio, timestamp)

        # Create video frame with same timestamp
        frame.pts = int(timestamp * 25)
        frame.time_base = Fraction(1, 25)
        return frame

class BufferedAudioTrack:
    def add_chunk(self, audio, timestamp):
        frame.pts = int(timestamp * 48000)
        frame.time_base = Fraction(1, 48000)
        self.queue.put_nowait(frame)
```

**Key**: Both tracks use the same timestamp, converted to appropriate PTS units.

## Performance Characteristics

### For 2x Real-Time Model (Typical)

**Model Speed**: Generates ~50 FPS (2x faster than 25 FPS target)

**Expected Behavior**:
- Video queue grows to ~25-50 frames (buffering ahead)
- Audio queue grows to ~25-50 chunks (buffering ahead)
- Synchronizer wait time: ~0-100ms (audio usually ready)
- Output: Smooth 25 FPS with perfect sync

**Logs**:
```
✅ Sync #100: frame=99, ts=3.960s, video_queue=30/200, audio_queue=28/500, wait=5ms
✅ Sync #200: frame=199, ts=7.960s, video_queue=35/200, audio_queue=32/500, wait=8ms
```

### For Slower Model (<1x Real-Time)

**Model Speed**: Generates <25 FPS (slower than real-time)

**Expected Behavior**:
- Video queue stays small (0-5 frames)
- Audio queue grows (audio arrives faster than consumption)
- Synchronizer wait time: Variable (occasionally waits for video)
- Output: May stutter (model can't keep up)

**Logs**:
```
⚠️ Sync #100: frame=99, ts=3.960s, video_queue=2/200, audio_queue=50/500, wait=150ms
❌ Sync #200: frame=199, ts=7.960s, video_queue=0/200, audio_queue=75/500, wait=800ms
```

**Solution**: Optimize model or reduce video quality.

## Critical Design Decisions

### Decision 1: Why NOT Timestamp-Based Matching?

**Rejected approach:**
```python
# Find audio with timestamp closest to video
for audio_chunk, audio_ts in audio_queue:
    if abs(audio_ts - video_ts) < 0.050:  # 50ms tolerance
        use(audio_chunk)
```

**Problems**:
- Requires precise timestamp calculation (error-prone)
- O(N) search for each frame
- Doesn't guarantee causality (could match wrong audio)
- Complex edge case handling

**FIFO is better**: O(1), guarantees causality, simple code.

### Decision 2: Why NOT Drop Frames?

**Rejected approach:**
```python
if frame_queue.full():
    old_frame = frame_queue.get()  # Drop oldest
    frame_queue.put(new_frame)
```

**Problem**: Breaks FIFO ordering → Breaks audio pairing → Desync

**Better**: Increase queue size or apply backpressure (block on put).

### Decision 3: Why Event-Based Waiting?

**Old approach (polling):**
```python
while not audio:
    audio = get_audio()
    await asyncio.sleep(0.01)  # Poll every 10ms
```

**Problems**:
- CPU waste (wakes up 100 times per second)
- 10ms latency even when audio ready
- Not scalable

**Event-based (current):**
```python
while not audio:
    audio = get_audio()
    await audio_ready.wait()  # Instant wake-up
    audio_ready.clear()
```

**Benefits**:
- Zero CPU when waiting
- Instant wake-up (no polling latency)
- Scalable (OS-level primitives)

## Validation and Monitoring

### Key Metrics

**1. Wait Time**
```
✅ <100ms:  Excellent (audio ready before video)
⚠️ 100-500ms: Acceptable (occasional wait)
❌ >500ms: Problem (model too slow or network issues)
```

**2. Queue Sizes**
```
Video queue 25-50/200: Healthy (model 1.5-2x real-time)
Audio queue 25-50/500: Healthy (buffering ahead)
Video queue 180+/200: Warning (near full, may drop)
Audio queue 0-5: Warning (starving, may wait)
```

**3. PTS Offset**
```
Should always be 0.0ms (same timestamp by design)
If not zero: Bug in timestamp assignment!
```

### chrome://webrtc-internals Metrics

Monitor these in browser developer tools:

- **googTimingFrameInfo**: Should show <50ms A/V offset
- **jitterBufferDelay**: Normal is 30-200ms
- **packetsLost**: Should be <1%
- **googFrameRateOutput**: Should be stable at 25 FPS

### Test Protocol

```python
# Generate test pattern
def test_av_sync():
    # 1. Create audio with 1Hz beep
    audio = generate_beeps(duration=10, frequency=1)

    # 2. Feed to avatar system
    run_avatar(audio)

    # 3. Record output
    record_output()

    # 4. Analyze: beep should align with mouth opening
    offset = analyze_lip_sync()

    # 5. Verify offset <50ms
    assert offset < 0.050, f"Sync offset {offset*1000:.1f}ms too large!"
```

## Troubleshooting

### Symptom: Audio ahead of video

**Cause**: Video generation is too slow
**Solution**: Optimize model, reduce quality, or increase buffering

### Symptom: Video ahead of audio

**Cause**: Audio not arriving fast enough from client
**Solution**: Check network, verify client is sending continuously

### Symptom: Intermittent desync

**Cause**: Frame dropping breaking FIFO order
**Solution**: Increase queue sizes, don't drop frames

### Symptom: High CPU usage

**Cause**: Polling instead of event-based waiting
**Solution**: Use asyncio.Event (current implementation)

## Conclusion

The **FIFO pairing + shared timestamp** approach is:

✅ **Simple**: ~50 lines of core logic
✅ **Correct**: Mathematically impossible to drift
✅ **Efficient**: Event-based waiting, minimal CPU
✅ **Robust**: Works for any model speed or network condition
✅ **Maintainable**: Self-documenting, easy to debug

This is the **standard pattern for audio passthrough in WebRTC systems** and the correct approach for the Ditto avatar use case.
