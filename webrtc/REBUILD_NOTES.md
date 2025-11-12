# WebRTC Signaling Server Rebuild - V2

## Overview
The signaling server has been completely rebuilt from the ground up with clean architecture and clear separation of concerns. The original implementation (`signaling_server.py`) had accumulated too many patches and workarounds, making it difficult to debug sync issues.

## Problems with Original Implementation

### 1. **Tangled Responsibilities**
- Audio reception, stereo conversion, resampling, buffering, and sync all in one 250-line function
- No clear boundaries between components
- Difficult to test or debug individual pieces

### 2. **Complex Timestamp Management**
- Manual tracking of `accumulated_audio_duration`
- Separate timestamp systems for audio capture vs playback
- Error-prone manual calculations

### 3. **Multiple Audio Paths**
- 16kHz buffer for model
- 48kHz buffer for playback
- Complex splitting logic with potential for desync

### 4. **No Backpressure**
- Queues could grow unbounded if model was slow
- No mechanism to slow down audio input

### 5. **Overly Complex Stereo Handling**
- 100+ lines handling every possible stereo format
- Multiple edge cases in one place
- Difficult to verify correctness

## New Architecture

### Clean Separation of Concerns

```
AudioPassthrough → VideoGenerator → AVSynchronizer → WebRTC Tracks
      ↓                  ↓                ↓
  (16kHz + 48kHz)   (timestamps)    (sync pairs)
```

### Component Breakdown

#### 1. **AudioPassthrough** (80 lines)
**Single Responsibility**: Handle all audio conversion

```python
class AudioPassthrough:
    def add_frame(frame) -> bool  # Returns true when model chunk ready
    def get_16khz_chunk() -> np.ndarray  # For Ditto model
    def get_playback_chunk() -> (audio, timestamp)  # For WebRTC
```

**What it does**:
- Receives WebRTC audio frames (48kHz stereo)
- Converts stereo → mono (simple averaging)
- Downsamples to 16kHz for Ditto
- Stores 48kHz mono for playback
- Automatically timestamps chunks based on sample count

**Benefits**:
- Clean, testable interface
- All audio conversion in one place
- Automatic timestamp calculation (no manual tracking)
- Simple to verify: samples_in = samples_out

#### 2. **VideoGenerator** (40 lines)
**Single Responsibility**: Interface with Ditto SDK

```python
class VideoGenerator:
    def feed_audio(audio_16k)  # Send to model
    def on_frame(rgb, idx, timestamp)  # Callback from model
    async def get_next_frame() -> (rgb, idx, timestamp)  # For sync
```

**What it does**:
- Feeds 16kHz audio to `sdk.run_chunk()`
- Receives video frames via callback
- Queues frames with their timestamps

**Benefits**:
- Clean SDK interface
- No mixing of audio/video logic
- Timestamps come from model (single source of truth)

#### 3. **AVSynchronizer** (30 lines)
**Single Responsibility**: Match audio and video by timestamp

```python
class AVSynchronizer:
    async def get_next() -> (video, idx, timestamp, audio)
```

**What it does**:
- Gets next video frame from VideoGenerator
- Gets matching audio chunk from AudioPassthrough
- Returns synchronized pairs
- Logs sync differences for debugging

**Benefits**:
- Simple sync logic (just pair by timestamp)
- Easy to monitor sync quality
- No complex state management

#### 4. **WebRTC Tracks** (60 lines each)
**Single Responsibility**: Output to WebRTC

```python
class BufferedVideoTrack:
    async def recv() -> av.VideoFrame

class BufferedAudioTrack:
    async def recv() -> av.AudioFrame
```

**What they do**:
- Get synchronized pairs from AVSynchronizer
- Set correct PTS timestamps
- Let WebRTC handle timing

**Benefits**:
- No artificial delays
- WebRTC paces playback automatically
- Clean, simple implementation

## Key Design Decisions

### 1. **Single Source of Timestamps**
- Use Ditto model's timestamps (`frame_idx / 25`)
- Audio chunks timestamped when created (based on sample count)
- No manual accumulation or tracking

**Why**: Eliminates drift and calculation errors

### 2. **WebRTC Handles Timing**
- Set correct PTS on frames
- Remove all `asyncio.sleep()` calls
- No manual pacing

**Why**: WebRTC is designed to handle jitter buffers and timing

### 3. **Simplified Audio Path**
```
48kHz stereo → mono → {16kHz for model, 48kHz for playback}
```

**Why**: Clear, testable, hard to get wrong

### 4. **Bounded Queues**
- All queues have maxsize
- Drop frames/audio if queues fill
- Log when dropping occurs

**Why**: Prevents memory growth, provides backpressure

## Code Size Comparison

| Component | Original | V2 | Reduction |
|-----------|----------|----|----|
| Audio handling | ~250 lines | 80 lines | 68% |
| Video handling | ~150 lines | 40 lines | 73% |
| Sync logic | ~100 lines | 30 lines | 70% |
| WebRTC tracks | ~200 lines | 120 lines | 40% |
| **Total** | **~700 lines** | **~270 lines** | **61%** |

## Testing Strategy

### Unit Tests (Easy Now!)

```python
# AudioPassthrough
def test_stereo_to_mono():
    ap = AudioPassthrough()
    stereo_frame = create_test_frame(stereo=True)
    ap.add_frame(stereo_frame)
    # Verify mono output

def test_timestamp_accuracy():
    ap = AudioPassthrough()
    # Add 1 second of audio
    # Verify timestamp = 1.0s

# AVSynchronizer
def test_sync_matching():
    # Mock video at t=0.5s
    # Mock audio at t=0.5s
    # Verify they're paired correctly
```

### Integration Testing

1. **Audio passthrough**: Record input, compare output
2. **Timestamp accuracy**: Check video PTS matches audio PTS
3. **Sync quality**: Measure drift over time

## Migration Guide

### For Testing

1. **Run both servers side-by-side**:
```bash
# Terminal 1: Original
python webrtc/signaling_server.py --cfg_pkl ... --data_root ...

# Terminal 2: V2
python webrtc/signaling_server_v2.py --cfg_pkl ... --data_root ...
```

2. **Compare outputs**:
   - Check logs for sync differences
   - Monitor queue sizes
   - Verify frame rates

### For Production

1. Test V2 with your typical avatar sources
2. Verify audio quality (pitch, sync)
3. Check video smoothness
4. Monitor resource usage

### Rollback Plan

If issues occur, revert to original:
```bash
# Original is still at webrtc/signaling_server.py
python webrtc/signaling_server.py ...
```

## Expected Improvements

### 1. **Easier Debugging**
- Clear component boundaries
- Focused logging per component
- Easy to trace data flow

### 2. **Better Sync**
- Single timestamp source (no drift)
- WebRTC handles jitter
- No artificial delays

### 3. **More Reliable**
- Bounded queues prevent memory issues
- Clear error boundaries
- Simpler = fewer bugs

### 4. **Maintainable**
- Small, focused classes
- Easy to modify
- Easy to test

## Known Limitations

1. **Still uses deque for audio chunks**: Could switch to queue.Queue for thread safety
2. **No adaptive sync**: Doesn't adjust if model gets slow/fast
3. **Simple drop policy**: Drops frames when queue full (could be smarter)

## Future Enhancements

### Short Term
1. Add metrics (sync drift, queue sizes, frame drops)
2. Implement adaptive sync (adjust if drift detected)
3. Add recording mode for debugging

### Long Term
1. Support multiple sample rates (not just 48kHz)
2. Support audio generation (not just passthrough)
3. Add video quality adaptation based on network

## Questions?

If you encounter issues:
1. Check logs for component-level errors
2. Monitor sync differences (logged every 100 frames)
3. Verify queue sizes aren't maxing out
4. Compare with original implementation

The clean architecture makes it easy to swap out individual components for testing or enhancement.
