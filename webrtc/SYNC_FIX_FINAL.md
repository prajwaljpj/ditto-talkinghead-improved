# Final Audio-Video Sync Fix

## Your Question: "Do we need a target FPS?"

**YES! The target FPS (25) is essential** and cannot be changed. Here's why:

### Why 25 FPS is Fixed

1. **Model's timestamp system**: Ditto generates frames with `timestamp = frame_idx / 25`
   - Frame 0 → ts=0.000s
   - Frame 25 → ts=1.000s
   - Frame 100 → ts=4.000s

2. **Audio synchronization**: The model processes 400ms of audio at a time (10 frames @ 25 FPS)
   - Each audio chunk → 10 video frames
   - Audio at ts=1.00s corresponds to frame 25

3. **1:1 mapping required**: For perfect sync, we need:
   - 25 video frames per second of playback
   - 25 audio chunks per second of playback
   - Timestamps must match exactly

**You cannot change the target FPS** without retraining the model. The 25 FPS is baked into how Ditto processes audio.

## The REAL Problem (Finally!)

### Two Bugs Fixed

#### Bug #1: Incorrect Audio Timestamping ❌

**Old code (WRONG)**:
```python
chunk_idx = len(self.playback_chunks)  # Just counting chunks
timestamp = chunk_idx * 0.04           # 0.00, 0.04, 0.08, ...
```

**Problem**: Timestamps based on chunk count, not actual audio duration received!

If we receive audio in bursts:
- T=0s: Receive 2s of audio → Create 50 chunks with ts=[0.00-1.96s]
- T=5s: Receive 2s more audio → Create 50 chunks with ts=[2.00-3.96s]

But chunk_idx keeps incrementing: 0, 1, 2, ..., 99
So timestamps become: 0.00, 0.04, ..., 3.96s ✓ (accidentally correct)

BUT if audio arrives continuously:
- We process in chunks, timestamps drift based on processing order
- Not based on actual sample timing!

**New code (CORRECT)**:
```python
base_offset = total_samples_received - len(chunk_48k)
sample_offset = base_offset + i
timestamp = sample_offset / 48000  # Based on actual samples!
```

**Fix**: Timestamps based on actual sample count from start of stream.

#### Bug #2: Not Actually Waiting for Matching Audio ❌

**Old code (WRONG)**:
```python
while audio_chunk is None:
    audio_chunk = get_next_audio()  # Just pop whatever's next!
    if not audio_chunk:
        await asyncio.sleep(0.01)
```

**Problem**: Popped audio chunks in FIFO order without checking timestamps!

Result:
- Video frame 100 (ts=3.96s) gets audio chunk 69 (ts=2.76s)
- Desync!

**New code (CORRECT)**:
```python
while wait_count < max_iterations:
    if len(playback_chunks) > 0:
        chunk, ts = playback_chunks[0]  # PEEK, don't pop

        if abs(ts - video_ts) < 0.050:
            # Timestamps match! Use this chunk
            audio_chunk = playback_chunks.popleft()
            break
        elif ts < video_ts - 0.050:
            # Audio too old, skip it
            playback_chunks.popleft()
            continue
        # else: Audio in future, wait for it

    await asyncio.sleep(0.01)  # Wait for matching audio to arrive
```

**Fix**: Actually checks timestamps and waits for matching audio!

## Expected Behavior Now

### Healthy Operation ✅

```
✅ Sync #1: video_ts=0.000s, audio_ts=0.000s, diff=0.0ms, waited=0ms
✅ Sync #100: video_ts=3.960s, audio_ts=3.960s, diff=10.0ms, waited=20ms
📊 Generated 100 frames, queue size: 25/100
⏳ Waiting for audio for video ts=4.000s, have 8 chunks, next_ts=3.960s, waited=500ms
```

**What this means**:
- **diff < 50ms**: Excellent sync ✅
- **waited > 0ms**: Video is waiting for audio (correct behavior!) ✅
- **queue size 25/100**: Model generating faster than consumption (good!) ✅
- **Waiting logs**: Normal - model generated ahead, waiting for client to send more audio ✅

### What Each Wait Time Means

```
waited=0-100ms    ✅ Perfect - audio arrives quickly
waited=100-500ms  ✅ Normal - model slightly ahead of audio
waited=500-2000ms ⚠️  Acceptable - model much faster than audio
waited>2000ms     ❌ Problem - check network or model speed
```

### Why Waiting is GOOD

**Waiting means the model is fast!**

Example timeline:
```
T=0.0s: Client starts, sends audio
T=0.5s: Received 0.5s of audio
T=0.5s: Model generates frames 0-20 (ts=0.0-0.8s)
        Frame 0-12 (ts=0.0-0.48s) have matching audio → sent
        Frame 13-20 (ts=0.52-0.8s) NO matching audio → WAIT
T=1.0s: Received 1.0s of audio total
        Frame 13-20 now have matching audio → sent
```

The wait is the model **getting ahead** and then **pausing** for audio to catch up. This is perfect throttling!

## Performance Characteristics

### Video Queue Size

```
size ~0-5:    ⚠️  Model barely keeping up (might stutter)
size ~10-30:  ✅ Healthy (model 1.5-2x faster than needed)
size ~50-80:  ⚠️  Model very fast OR audio very slow
size ~100:    ❌ Queue full (problem with audio consumption)
```

### Model Speed vs Real-Time

If model generates 45 FPS:
- Generates 45 frames per second (wall-clock)
- Each frame has ts = frame_idx / 25
- So 45 frames cover 1.8s of playback time
- **Model is 1.8x faster than real-time** ✅

This is GOOD! The waiting throttles output to match real-time audio.

### Network Impact

- **Bandwidth**: Always 25 FPS output (consistent)
- **Latency**: ~500ms startup (model warmup) + ~50ms sync tolerance
- **Jitter**: Minimal (WebRTC handles jitter buffering)

## What "Target FPS" Means

### What It Controls

- **Video frame timestamps**: frame_pts = int(timestamp * 25)
- **Audio frame pairing**: 1 audio chunk per video frame
- **Playback rate**: 25 frames displayed per second on client

### What It Does NOT Control

- **Model generation speed**: Model generates as fast as it can (45-50 FPS)
- **Network transmission**: WebRTC encodes/sends as needed
- **Client playback**: Browser plays based on PTS timestamps

### Why 25 FPS Specifically

1. **Standard video rate**: 24-30 FPS is cinematic/broadcast standard
2. **Audio chunk size**: 40ms chunks @ 48kHz = 1920 samples (WebRTC standard)
3. **Model training**: Ditto was trained with 25 FPS assumption
4. **Bandwidth efficiency**: Good balance of quality vs data rate

## Summary

### The Fix (2 Parts)

1. **Correct audio timestamping**: Based on actual sample count, not chunk count
2. **Timestamp-aware synchronization**: Wait for audio with matching timestamp

### Why It Works

- **Model timestamps** = ideal playback time (frame_idx / 25)
- **Audio timestamps** = actual capture time (samples / 48000)
- **Synchronizer** = matches them within 50ms tolerance
- **Waiting** = natural backpressure when model generates ahead

### Expected Results

- ✅ **diff < 50ms** - Perfect sync
- ✅ **waited > 0ms** - Model faster than real-time (good!)
- ✅ **queue growing** - Buffer building up (good!)
- ✅ **No silence frames** - Audio never runs dry

The 25 FPS is **not negotiable** - it's fundamental to how Ditto processes audio and generates video. The fix ensures video output is paced to match real-time audio arrival, regardless of how fast the model actually generates.
