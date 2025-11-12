# Video Jitter Fix - LiveKit Frame Capture from Worker Thread

## Problem

**Symptom**: Video showing only a still frame or jittered frames on the client side, while audio works perfectly.

**Root Cause**: The `_on_frame_generated` callback is invoked from Ditto's worker thread (not the asyncio event loop), but was calling `self.video_source.capture_frame()` synchronously. LiveKit's `VideoSource.capture_frame()` expects to be called from the asyncio event loop context, not from arbitrary worker threads.

## Technical Details

### Why This Happens

1. **Ditto's Architecture**: StreamSDK uses worker threads for frame generation
   - Worker threads call `_on_frame_generated(frame_rgb, frame_idx, timestamp)`
   - These threads are NOT part of the asyncio event loop

2. **LiveKit's Expectations**: LiveKit's Python SDK is built on asyncio
   - `VideoSource.capture_frame()` must be called from event loop context
   - When called from a worker thread, frames may not be properly queued/sent
   - This causes frames to be dropped or not transmitted at all

3. **The Mismatch**:
   ```python
   # BEFORE (BROKEN):
   def _on_frame_generated(self, frame_rgb, frame_idx, timestamp):
       # This runs in worker thread!
       video_frame = rtc.VideoFrame(...)
       self.video_source.capture_frame(video_frame)  # ❌ Wrong context!
   ```

## Solution

Use `asyncio.run_coroutine_threadsafe()` to schedule frame capture on the event loop from the worker thread.

### Changes Made

#### 1. Store Event Loop Reference (webrtc/livekit_gemini_agent.py:240, 289)

```python
# In __init__
self._event_loop: Optional[asyncio.AbstractEventLoop] = None

# In initialize()
async def initialize(self):
    # Store event loop reference for frame capture from worker threads
    self._event_loop = asyncio.get_running_loop()
    logger.info("🔄 Event loop reference stored for frame capture")
```

#### 2. Schedule Frame Capture on Event Loop (webrtc/livekit_gemini_agent.py:413-418)

```python
def _on_frame_generated(self, frame_rgb, frame_idx, timestamp):
    """Callback when Ditto generates a frame (from worker thread)."""
    # ... frame processing ...

    # CRITICAL FIX: Schedule frame capture on the asyncio event loop
    # This callback runs from Ditto's worker thread, but LiveKit needs frames
    # to be captured from the asyncio event loop context
    asyncio.run_coroutine_threadsafe(
        self._capture_frame_async(video_frame, stage_start),
        self._event_loop
    )
```

#### 3. New Async Method for Frame Capture (webrtc/livekit_gemini_agent.py:431-442)

```python
async def _capture_frame_async(self, video_frame: rtc.VideoFrame, stage_start: float):
    """Capture frame in asyncio context (called from event loop)."""
    try:
        capture_start = time.perf_counter()
        self.video_source.capture_frame(video_frame)  # ✅ Now in correct context!
        capture_time = (time.perf_counter() - capture_start) * 1000
        self._profile_stage('frame_generation', capture_time, f"LiveKit capture")

        total_time = (time.perf_counter() - stage_start) * 1000
        self._profile_stage('frame_generation', total_time, f"Total (frame #{self._frames_generated})")
    except Exception as e:
        logger.error(f"Error capturing frame in async context: {e}")
```

## How It Works

### Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Ditto Worker Thread                                        │
├─────────────────────────────────────────────────────────────┤
│  1. Generate frame_rgb                                      │
│  2. Call _on_frame_generated(frame_rgb, ...)               │
│  3. Convert RGB → RGBA                                      │
│  4. Create rtc.VideoFrame                                   │
│  5. Schedule on event loop:                                 │
│     asyncio.run_coroutine_threadsafe(                       │
│         _capture_frame_async(video_frame),                  │
│         self._event_loop  # ← Event loop reference          │
│     )                                                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Cross-thread call
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Asyncio Event Loop (Main Thread)                          │
├─────────────────────────────────────────────────────────────┤
│  1. Receive scheduled coroutine                             │
│  2. Execute _capture_frame_async()                          │
│  3. Call video_source.capture_frame()  ✅ Correct context! │
│  4. Frame sent to LiveKit → WebRTC → Browser                │
└─────────────────────────────────────────────────────────────┘
```

### Key Functions

1. **`asyncio.run_coroutine_threadsafe(coro, loop)`**
   - Schedules a coroutine to run on a specific event loop
   - Thread-safe: Can be called from any thread
   - Returns a `concurrent.futures.Future`
   - Perfect for worker threads → event loop communication

2. **`asyncio.get_running_loop()`**
   - Gets the currently running event loop
   - Called during `initialize()` (which runs in event loop)
   - Stored for later use by worker threads

## Benefits

1. **Thread-Safe**: Properly bridges worker threads and asyncio event loop
2. **No Frame Loss**: Frames are properly queued and sent to LiveKit
3. **Smooth Video**: Browser receives continuous frame stream at 25 FPS
4. **Clean Architecture**: Separates frame generation (worker) from transmission (event loop)

## Testing

### Before Fix
- ❌ Video shows still frame or occasional jitter
- ✅ Audio works perfectly
- Logs show frames being generated but not visible on client

### After Fix
- ✅ Smooth video at 25 FPS
- ✅ Audio continues to work perfectly
- ✅ Both IDLE and SPEAKING animations visible

### How to Test

1. Start the agent:
   ```bash
   ./start_gemini_agent.sh
   ```

2. Open browser client:
   ```
   webrtc/client/livekit/index_simple.html
   ```

3. Connect and verify:
   - Video should show smooth animation
   - Audio should be in sync
   - No jitter or freezing

### Expected Logs

```
🔄 Event loop reference stored for frame capture
🎬 First frame generated from StreamSDK: (720, 1280, 3)
   Frame index: 0, Timestamp: 0.000s
📊 Frames: 100 sent, 2 dropped
📊 Frames: 200 sent, 5 dropped
```

## Related Issues

This is a common pattern when integrating:
- Multi-threaded libraries (like Ditto's StreamSDK)
- Asyncio-based libraries (like LiveKit Python SDK)

The key is to identify:
1. Which code runs in worker threads vs event loop
2. Which functions require event loop context
3. How to bridge between them safely

## Client Side (No Changes Needed)

The HTML client code is correct:

```javascript
// webrtc/client/livekit/index_simple.html:157-162
room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
    if (track.kind === Track.Kind.Video) {
        const video = document.getElementById('remoteVideo');
        track.attach(video);  // ✅ Correct
    }
});
```

The issue was entirely on the server side.

## Summary

**Problem**: Worker thread calling asyncio functions directly
**Solution**: Schedule asyncio functions on event loop using `run_coroutine_threadsafe()`
**Result**: Smooth 25 FPS video streaming to browser

---

**Date**: 2025-11-11
**Issue**: Video jitter/still frame despite audio working
**Fix**: Proper thread-safe frame capture scheduling
**Status**: ✅ Fixed and Tested
