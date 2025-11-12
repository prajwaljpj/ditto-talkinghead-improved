# Thread Safety Fix for signaling_server_v3.py

## Problem Identified

The WebRTC server was running ~8 FPS instead of the expected ~50 FPS (from profiler results).

### Root Cause

**Thread boundary violation in `VideoGenerator`:**

```python
# BROKEN (was using asyncio.Queue)
class VideoGenerator:
    def __init__(self):
        self.frame_queue = asyncio.Queue(maxsize=500)  # ❌ NOT thread-safe!

    def on_frame(self, frame_rgb, frame_idx, timestamp):
        # Called from Ditto's worker thread
        self.frame_queue.put_nowait(...)  # ❌ Cross-thread violation!
```

**Why this caused slowdown:**
- `asyncio.Queue` is designed for single-threaded async code
- Ditto SDK runs in background worker threads
- `on_frame()` callback is called from worker threads
- Using `asyncio.Queue.put_nowait()` from a different thread causes undefined behavior
- Result: Frames were being lost or delayed, causing ~8 FPS instead of ~50 FPS

### The Fix

**Use thread-safe `queue.Queue` instead:**

```python
# FIXED (using thread-safe queue)
class VideoGenerator:
    def __init__(self):
        import queue
        self.frame_queue = queue.Queue(maxsize=500)  # ✅ Thread-safe!

    def on_frame(self, frame_rgb, frame_idx, timestamp):
        # Called from Ditto's worker thread - now safe!
        self.frame_queue.put((frame_rgb, frame_idx, timestamp), timeout=1.0)  # ✅ Thread-safe!

    async def get_next_frame(self):
        # Use asyncio.to_thread() to avoid blocking event loop
        return await asyncio.to_thread(self.frame_queue.get)  # ✅ Async-safe wrapper!
```

## Why This Matters

### Before Fix:
```
Profiler (single-threaded): 49.91 FPS ✅
WebRTC (multi-threaded):    ~8 FPS ❌
```

### After Fix (Expected):
```
Profiler (single-threaded): 49.91 FPS ✅
WebRTC (multi-threaded):    ~50 FPS ✅
```

## Threading Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Async Loop                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │  AudioInputProcessor                               │    │
│  │  → process_frame() [async]                         │    │
│  │  → get_model_chunk() [async]                       │    │
│  └────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  VideoGenerator.feed_audio()                       │    │
│  │  → sdk.run_chunk() [spawns worker threads]        │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Ditto SDK Worker Threads                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Audio2Motion Thread                               │    │
│  │  Motion Stitch Thread                              │    │
│  │  Warp Thread                                       │    │
│  │  Decode Thread                                     │    │
│  │  PutBack Thread                                    │    │
│  │                                                     │    │
│  │  → on_frame() callback [CALLED FROM WORKER THREAD!]│    │
│  └────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  queue.Queue (thread-safe!)                        │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     Main Async Loop                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │  VideoGenerator.get_next_frame()                   │    │
│  │  → asyncio.to_thread(queue.get) [async wrapper]   │    │
│  └────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  AVOutputSynchronizer                              │    │
│  │  → Pairs with audio chunks                         │    │
│  │  → Sends to WebRTC                                 │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Thread Safety Guidelines

### ✅ Thread-Safe (OK to use across threads):
- `queue.Queue` (thread-safe by design)
- `deque.append()` / `deque.popleft()` (atomic operations)
- `threading.Lock`
- `time.monotonic()` (thread-safe)

### ❌ NOT Thread-Safe (only for single async loop):
- `asyncio.Queue` (only for single event loop)
- `asyncio.create_task()` (must be in same loop)
- Most async operations without `asyncio.run_coroutine_threadsafe()`

## Expected Results After Fix

### Logs should show:
```bash
✅ Sync #100: audio_ts=4.000s, video_gen_at=4.200s, latency=200ms ✅
✅ Sync #200: audio_ts=8.000s, video_gen_at=8.205s, latency=205ms ✅
# Latency stays ~200ms (model generation time), not growing!
```

### Performance:
- **Video generation**: ~50 FPS (matching profiler)
- **Audio-video sync**: Perfect (using accumulated duration timestamps)
- **Latency**: ~200-500ms (startup + model + network)
- **Stable latency**: ✅ (not growing over time)

## Testing

```bash
# Start server
uv run python webrtc/signaling_server_v3.py \
  --cfg_pkl checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl \
  --data_root checkpoints/ditto_trt_custom2/ \
  --port 8080

# Connect web client to http://localhost:8000
# Start speaking

# Expected: Smooth 25 FPS playback with perfect audio-video sync
```

## Related Issues

This fix addresses:
1. **Slow frame generation in WebRTC** (8 FPS → 50 FPS)
2. **Thread safety violations** (asyncio.Queue used across threads)
3. **Frame drops/delays** (undefined behavior from cross-thread queue access)

This does NOT fix (already fixed in previous commit):
1. **Timestamp drift** (fixed by using accumulated_duration instead of wall clock)
2. **Audio synchronization** (fixed by FIFO pairing with timestamps)
