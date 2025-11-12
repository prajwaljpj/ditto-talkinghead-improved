# Ditto WebRTC Server v3 - Clean Queue-Based Architecture

## Overview

This is a completely redesigned WebRTC signaling server for Ditto talking head avatars, built with clean separation of concerns and simple FIFO queue-based processing.

### Key Improvements over v1/v2

- ✅ **No Frame Dropping**: Pure FIFO queues, all data is processed
- ✅ **No Artificial Waits**: WebRTC handles timing via PTS timestamps
- ✅ **World Clock Timing**: Single monotonic clock for all timestamps
- ✅ **Perfect Sync**: Audio and video matched via timestamp pairing
- ✅ **Modular Design**: Easy integration into conversational systems (Gemini, etc.)

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                       WebRTC Client (Browser)                    │
│                     Audio Input → Video Output                   │
└────────────┬────────────────────────────────────┬────────────────┘
             │ Audio (48kHz)                       │ Video + Audio
             ▼                                     ▲
┌────────────────────────────────────────────────────────────────┐
│                    DittoWebRTCSession                           │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐  │
│  │ WorldClock   │───▶│ AudioInput   │───▶│ VideoGenerator  │  │
│  │ (Timestamps) │    │ Processor    │    │ (Ditto SDK)     │  │
│  └──────────────┘    └──────┬───────┘    └────────┬────────┘  │
│                             │                      │            │
│                             │ Playback Chunks     │ Frames     │
│                             │ (timestamped)       │ (generated)│
│                             ▼                      ▼            │
│                      ┌─────────────────────────────────┐       │
│                      │   AVOutputSynchronizer          │       │
│                      │   (Match via timestamps)        │       │
│                      └────────────┬────────────────────┘       │
│                                   │ Synchronized pairs         │
│                                   ▼                             │
│                      ┌─────────────────────────┐               │
│                      │  WebRTC Tracks          │               │
│                      │  (BufferedVideoTrack +  │               │
│                      │   BufferedAudioTrack)   │               │
│                      └─────────────────────────┘               │
└────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input Audio Processing**
   ```
   Client (48kHz stereo) → AudioInputProcessor
     ├─ Convert to mono
     ├─ Downsample to 16kHz for Ditto model
     ├─ Keep 48kHz for playback
     ├─ Buffer 400ms chunks (6400 samples @ 16kHz)
     └─ Timestamp with world clock
   ```

2. **Video Generation**
   ```
   AudioInputProcessor (16kHz chunks) → Ditto SDK
     └─ Generates video frames (25 FPS)
     └─ Timestamps with world clock
   ```

3. **Output Synchronization**
   ```
   Playback Chunks (48kHz, timestamped) + Video Frames (timestamped)
     → AVOutputSynchronizer
     → Pairs via FIFO + timestamp matching
     → WebRTC Tracks (same PTS for audio and video)
   ```

## Usage

### Standalone Server

```bash
# Start the signaling server
python webrtc/signaling_server_v3.py \
  --cfg_pkl outputs/cfg_f_model.pkl \
  --data_root ./ \
  --port 8080
```

### Web Client

```bash
# Serve the web client
cd webrtc/client/web
python -m http.server 8000

# Open browser to http://localhost:8000
# Connect to ws://localhost:8080
```

### Integration with Gemini (Conversational System)

```python
from webrtc.ditto_avatar_module import DittoAvatarModule

# Initialize avatar
avatar = DittoAvatarModule(
    cfg_pkl="outputs/cfg_f_model.pkl",
    data_root="./",
    source_path="avatars/person.jpg",
    emo=4  # neutral emotion
)
await avatar.initialize()

# In your Gemini TTS callback:
def on_audio_from_gemini(audio_16khz: np.ndarray):
    # Feed to Ditto (automatically queues output)
    avatar.process_audio(audio_16khz, upsample_for_output=True)

# In your WebRTC video track:
async def get_next_frame():
    video_rgb, audio_48khz, timestamp = await avatar.get_next_frame()
    return video_rgb  # Send to WebRTC

# In your WebRTC audio track:
async def get_next_audio():
    video_rgb, audio_48khz, timestamp = await avatar.get_next_frame()
    return audio_48khz  # Send to WebRTC
```

## Key Design Decisions

### 1. World Clock (Single Source of Truth)

All timestamps come from a single monotonic clock:

```python
class WorldClock:
    def now(self) -> float:
        return time.monotonic() - self._start_time
```

Benefits:
- No clock drift between components
- Deterministic timing
- Easy debugging (all times relative to session start)

### 2. FIFO Queues (No Dropping)

Audio and video are queued and consumed in order:

```python
# Audio chunks queued when captured
audio_queue.append((audio_data, timestamp))

# Video frames queued when generated
video_queue.append((frame_rgb, frame_idx, timestamp))

# Consumed in FIFO order
audio_chunk, audio_ts = audio_queue.popleft()
video_frame, frame_idx, video_ts = video_queue.popleft()
```

Benefits:
- No frame drops = smooth playback
- Preserves all audio = perfect lip sync
- Backpressure naturally controls timing

### 3. PTS-Based Timing (WebRTC Handles Playback)

We set PTS (Presentation Timestamp) on all frames:

```python
# Audio frame
audio_frame.pts = int(timestamp * 48000)  # PTS in samples
audio_frame.time_base = Fraction(1, 48000)

# Video frame
video_frame.pts = int(timestamp * 25)  # PTS in frames
video_frame.time_base = Fraction(1, 25)
```

Benefits:
- WebRTC handles playback timing automatically
- No artificial sleeps or pacing needed
- Browser controls buffering and jitter

### 4. Timestamp Matching (Audio Duration, Not Wall Clock!)

**CRITICAL INSIGHT**: Timestamps must be based on **accumulated audio duration**, NOT wall clock time!

```python
# WRONG: Using wall clock (causes drift with model latency)
timestamp = clock.now()  # ❌ Time when audio captured

# CORRECT: Using accumulated audio duration
timestamp = accumulated_duration  # ✅ Playback time for this audio
accumulated_duration += chunk_duration  # Advance by audio length
```

Why this matters:
- Model has latency (4-5 seconds in your case)
- If we timestamp audio at capture time (t=3s) but video generates at t=8s, drift grows
- Using accumulated duration: audio at playback_time=0s matches video that will play at playback_time=0s
- Result: **Perfect sync regardless of model latency** ✅

Example timeline:
```
Wall Clock:  t=3s        t=4s        t=5s        t=8s
             Audio       Audio       Audio       Video
             arrives     arrives     arrives     generated

With wall clock timestamps:
             ts=3s       ts=4s       ts=5s       ts=8s  ❌ Drift!

With accumulated duration:
             ts=0s       ts=0.4s     ts=0.8s     ts=0s  ✅ Sync!
             (0-400ms)   (400-800ms) (800-1200ms) (for 0-400ms audio)
```

Benefits:
- Audio is the input that drove generation
- Perfect sync by construction
- **Model latency doesn't affect sync**
- Deterministic pairing

## Configuration

### Ditto Parameters

```python
server = DittoSignalingServer(
    cfg_pkl="outputs/cfg_f_model.pkl",  # Model config
    data_root="./",                      # Model data root
    host="0.0.0.0",                      # Server host
    port=8080,                           # Server port
    max_size=1920,                       # Max image dimension
    emo=4,                               # Emotion (0-7)
    crop_scale=2.3,                      # Face crop scale
)
```

### Emotion Codes

- 0: Happy
- 1: Angry
- 2: Sad
- 3: Fear
- 4: Neutral (default)
- 5: Surprised
- 6: Disgusted
- 7: Contemptuous

## Troubleshooting

### Audio-Video Desync

Check the sync logs:

```
✅ Sync #100: frame=100, audio_ts=4.000s, video_gen_at=4.050s, latency=50ms, avg_abs_drift=45ms
```

**Understanding the metrics:**
- `audio_ts`: Playback timestamp for this audio chunk (from accumulated duration)
- `video_gen_at`: Wall clock time when video was generated
- `latency`: Time between audio timestamp and video generation (expected to be positive)
- `avg_abs_drift`: Average latency over all frames

**Expected behavior:**
- **latency < 200ms**: Perfect sync ✅ (model is fast)
- **latency 200-1000ms**: Acceptable ⚠️ (model has moderate latency)
- **latency > 1000ms**: High latency ⚠️ (but sync still works if stable!)
- **latency GROWING**: Problem ❌ (timestamps not based on audio duration)

**If you see GROWING drift** (like your logs showed: 1.3s → 69s):
- This means timestamps are using wall clock instead of audio duration
- Make sure you're using the FIXED version of signaling_server_v3.py
- The fix uses `accumulated_duration` instead of `clock.now()` for timestamps

Common causes of other sync issues:
1. Model too slow (frames queue up) → Reduce max_size or use faster GPU
2. Network too slow (audio queues up) → Check bandwidth
3. Client buffering issues → Check browser logs

### Choppy Playback

Check queue sizes:

```python
video_queue_size, audio_queue_size = avatar.get_queue_sizes()
```

- **Both high (>100)**: Model generating faster than network → Normal
- **Video high, audio low**: Client not sending audio fast enough
- **Audio high, video low**: Model too slow → Reduce max_size

### No Audio Output

1. Check client logs for audio track:
   ```javascript
   console.log('Audio tracks:', stream.getAudioTracks())
   ```

2. Verify browser audio unmuted:
   ```javascript
   videoElement.muted = false
   videoElement.volume = 1.0
   ```

3. Check server logs:
   ```
   🔊 First audio frame queued: PTS=0, ts=0.000s
   ```

## Performance

### Expected Metrics

- **Latency**: 200-500ms (audio input → video output)
  - Audio buffering: 400ms (required by Ditto)
  - Model inference: 50-100ms (depends on GPU)
  - Network: 50-100ms (depends on connection)

- **Throughput**: 25 FPS (video), 48kHz (audio)

- **Memory**: ~2GB (Ditto model + queues)

### Optimization Tips

1. **Reduce Latency**
   - Use faster GPU
   - Reduce max_size (512 or 768 instead of 1920)
   - Use local network (avoid internet)

2. **Improve Quality**
   - Increase max_size (1920 or 2048)
   - Use better source image (high-res, frontal, good lighting)

3. **Reduce Memory**
   - Reduce queue sizes (maxsize=100 instead of 500)
   - Use smaller max_size

## Comparison with v1/v2

| Feature | v1 | v2 | v3 |
|---------|----|----|-----|
| Frame dropping | Yes | No | No |
| Artificial waits | Yes | Yes | No |
| Timing source | Multiple | Multiple | Single (WorldClock) |
| Audio sync | Timestamp-based | Event-based waiting | FIFO + timestamps |
| Complexity | High | Medium | Low |
| Modularity | Low | Medium | High |
| Gemini integration | Hard | Hard | Easy |

## Future Work

- [ ] Dynamic emotion updates (from Gemini emotion detection)
- [ ] Multi-client support (separate sessions)
- [ ] Recording/playback of sessions
- [ ] Metrics and monitoring dashboard
- [ ] TURN server support for NAT traversal

## License

Same as parent project (Ditto).
