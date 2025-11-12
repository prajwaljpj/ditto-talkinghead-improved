# The Correct Timestamp Approach for Audio Passthrough

## The Fundamental Insight

You were absolutely right - we were not using the right timestamps!

### Wrong Approach (What I Was Doing)

```
Audio received → Timestamp = samples_received / 48000  (capture time)
Video generated → Timestamp = frame_idx / 25           (playback position)
Try to match them → Only works by accident!
```

**Problem**: These timestamps represent DIFFERENT things:
- Audio timestamp = "when audio was captured/received"
- Video timestamp = "where in the video this frame belongs"

### Correct Approach (What We Do Now)

```
Audio received → Buffer (NO timestamp yet)
Video generated → Timestamp T from model
Pop next audio chunk (FIFO) → Give it SAME timestamp T
Send (video, audio) pair → Both have identical timestamp T
```

**Why this is correct**: We're doing audio **passthrough** (echoing back the input audio). The audio chunk used to generate video frame N should be played back at the same time as video frame N.

## The Key Realization

### What Timestamps Actually Mean

In WebRTC, timestamps define **playback position**, not capture time:

- **PTS (Presentation Time Stamp)**: "Play this frame at this time"
- Video frame with PTS=100 → Play at t=4.0s (100/25 fps)
- Audio frame with PTS=192000 → Play at t=4.0s (192000/48000 Hz)

### Why FIFO Pairing is Correct

The Ditto model:
1. Takes audio chunk 0 (samples 0-6399 @ 16kHz)
2. Generates video frames 0-9 (timestamps 0.00s-0.36s)
3. Takes audio chunk 1 (samples 6400-12799 @ 16kHz)
4. Generates video frames 10-19 (timestamps 0.40s-0.76s)

For passthrough, we want:
- Video frame 0 (ts=0.00s) paired with audio chunk 0 (samples 0-1919 @ 48kHz)
- Video frame 1 (ts=0.04s) paired with audio chunk 1 (samples 1920-3839 @ 48kHz)
- Both should have **matching timestamps** for synchronized playback

The audio chunks are extracted in the same order they were fed to the model, so **FIFO pairing is correct**!

## Implementation

### AudioPassthrough

```python
# Store audio chunks WITHOUT timestamps
def get_16khz_chunk(self):
    # Extract 48kHz audio for playback
    for sub_chunk in split_into_40ms_chunks(chunk_48k):
        self.playback_chunks.append(sub_chunk)  # NO timestamp!
```

**Key**: Audio chunks are just buffers, waiting to be paired with video.

### AVSynchronizer

```python
async def get_next(self):
    video, frame_idx, timestamp = await get_video_frame()  # Timestamp from model
    audio_chunk = audio_pass.get_next_playback_chunk()     # Pop next (FIFO)

    # Audio gets video's timestamp!
    return (video, frame_idx, timestamp, audio_chunk)
```

**Key**: Simple FIFO pairing, audio inherits video's timestamp.

### BufferedVideoTrack

```python
async def recv(self):
    video, frame_idx, timestamp, audio_chunk = await sync.get_next()

    # Add audio with SAME timestamp as video
    audio_track.add_chunk(audio_chunk, timestamp)

    # Add video with its original timestamp
    video_frame.pts = int(timestamp * 25)
```

**Key**: Both audio and video get the same timestamp.

## Why This Works

### Perfect Sync Guarantee

```
Frame 0:  video_pts=0,    audio_pts=0      → Both play at t=0.00s
Frame 1:  video_pts=1,    audio_pts=1920   → Both play at t=0.04s
Frame 25: video_pts=25,   audio_pts=48000  → Both play at t=1.00s
```

Since both have matching playback times, WebRTC's jitter buffer keeps them perfectly synchronized!

### No Drift Possible

Old approach:
- Audio timestamp calculated independently
- Could drift due to timing differences, rounding errors, etc.

New approach:
- Audio gets video's timestamp directly
- **Mathematically impossible to drift** (they're the same value!)

### Handles Variable Model Speed

If model generates:
- 50 FPS: Video queue grows, but each frame still paired with correct audio chunk
- 20 FPS: Video queue shrinks, but pairing still correct
- Variable: Doesn't matter, FIFO order is always correct

## Expected Logs

### Healthy Operation

```
✅ Sync #100: frame=99, ts=3.960s, audio_queue=25, waited=0ms
📊 Generated 100 frames, queue size: 15/100
```

**Interpretation**:
- **ts=3.960s**: Video frame 99 has playback time 3.96s
- **audio_queue=25**: Have 25 audio chunks buffered (1 second worth)
- **waited=0ms**: Audio already available (plenty buffered)

### When Model is Faster

```
✅ Sync #100: frame=99, ts=3.960s, audio_queue=50, waited=0ms
```

Audio queue grows → Model generating faster than audio arrives (good!)

### When Model is Slower

```
⏳ Waiting for audio chunk for frame 99 (ts=3.960s), have 0 chunks buffered, waited=500ms
✅ Sync #100: frame=99, ts=3.960s, audio_queue=0, waited=500ms
```

Audio queue empty, had to wait → Model slower than audio arrival (might cause stuttering)

## Comparison to Wrong Approach

### Before (Timestamp Matching)

```python
# Audio: ts = sample_offset / 48000 = 0.040s (when received)
# Video: ts = frame_idx / 25 = 0.040s (from model)
# Match by timestamp → Works if timing is perfect, breaks otherwise
```

**Problems**:
- Depends on timing coincidence
- Can drift over time
- Doesn't handle bursts or delays
- Complex matching logic

### After (FIFO Pairing)

```python
# Audio: NO timestamp (just buffer)
# Video: ts = frame_idx / 25 = 0.040s
# Audio gets video's timestamp → Always perfect
```

**Benefits**:
- Simple FIFO queue
- Zero drift (timestamps identical)
- Handles any timing pattern
- Self-documenting code

## Why "25 FPS" is Still Required

The 25 FPS target defines:
- **Video PTS calculation**: `pts = int(timestamp * 25)`
- **Audio frame pairing**: 1 audio chunk per video frame (25 chunks/sec)
- **Playback rate**: Browser plays 25 frames per second

But it does NOT define:
- **Generation rate**: Model can generate at any speed
- **Transmission rate**: WebRTC sends as needed
- **Encoding rate**: Encoder runs at its own pace

The 25 FPS is the **playback rate**, not the generation rate.

## Summary

**The fix**: Audio chunks get their timestamps FROM video frames, not from when audio was received.

**Why it works**: We're doing passthrough - the audio that generated video frame N should play back at the same time as frame N.

**The benefit**: Perfect sync guaranteed, no drift possible, simple code.

This is the **correct and only way** to do audio passthrough synchronization!
