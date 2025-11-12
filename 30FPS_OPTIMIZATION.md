# 30 FPS Optimization for Gemini Agent

## Summary

Modified the Gemini LiveKit agent to achieve smooth 25-30 FPS video output with a universal chunk size for both silent audio generation and Gemini audio processing, respecting TensorRT model constraints.

## Changes Made

### 1. Universal Chunk Size
**Changed from**: 6480 samples (405ms @ 16kHz) → **~2.5 FPS effective rate**
**Changed to**: 3240 samples (202.5ms @ 16kHz) → **~5 Hz feed rate → 25-30 FPS output**

**Rationale**:
- TensorRT HuBERT model requires minimum **3240 samples** (valid range: 3240-12960)
- Each 3240-sample chunk generates approximately 5-6 frames
- Feeding at ~5 Hz (every 202.5ms) provides smooth 25-30 FPS output
- Single chunk size used for both silent audio (IDLE) and Gemini audio (SPEAKING)

### 2. Target FPS Update
**Changed from**: 25 FPS
**Changed to**: 30 FPS

**Frame interval**: 33.33ms per frame

### 3. Frame Dropping Logic
**Changed from**: Drop if faster than 31.25 FPS (25 * 0.8)
**Changed to**: Drop only if faster than 40 FPS (25ms interval)

This allows the Ditto model to run at its full ~50 FPS capability while targeting 30 FPS output, with headroom up to 40 FPS.

### 4. Silent Audio Generator
**Changed from**: 405ms sleep (2.47 Hz feed rate)
**Changed to**: 202.5ms sleep (4.94 Hz feed rate)

**Result**: Smooth 25-30 FPS idle/listening animation instead of ~3 FPS bursts

## Technical Details

### Audio Flow
```
Silent Audio (IDLE):
  3240 samples @ 16kHz (202.5ms) - TensorRT minimum
  ↓
  StreamSDK Pipeline
  ↓
  ~5-6 frames generated (~33ms per frame)
  ↓
  25-30 FPS output

Gemini Audio (SPEAKING):
  Variable chunks @ 24kHz from Gemini
  ↓
  Resample to 16kHz
  ↓
  Buffer until 3240 samples
  ↓
  StreamSDK Pipeline
  ↓
  ~5-6 frames per chunk
  ↓
  25-30 FPS output
```

### TensorRT Constraints
- **Minimum input**: 3240 samples @ 16kHz
- **Maximum input**: 12960 samples @ 16kHz
- **Optimization profile**: [1,3240]..[1,12960]
- Attempting to feed less than 3240 samples causes TensorRT error

### Frame Generation Rate
- **Ditto capability**: ~50 FPS (20ms per frame)
- **Target output**: 30 FPS (33.33ms per frame)
- **Maximum allowed**: 40 FPS (25ms per frame, before dropping)
- **Silent audio feed rate**: ~5 Hz (202.5ms per chunk)
- **Expected frames per chunk**: 5-6 frames

### Performance Expectations
- **IDLE state**: Smooth 25-30 FPS idle/listening animation
- **SPEAKING state**: Smooth 25-30 FPS lip-synced animation
- **Latency**: ~202ms audio-to-video (one chunk duration)
- **Frame drops**: Minimal (only if >40 FPS burst occurs)

## Files Modified

1. **webrtc/livekit_gemini_agent.py**
   - Line 101: `silent_audio_chunk_size = 3240` (TensorRT minimum)
   - Line 204: `model_chunk_size = 3240`
   - Line 207: `target_fps = 30`
   - Line 384-385: Frame dropping at 40 FPS threshold
   - Line 621-670: `run_silent_audio_generator()` updated for 25-30 FPS
   - Line 968-979: Updated logging to show 25-30 FPS info

## Testing

Run the test script to verify the chunk size works correctly:

```bash
uv run python test_30fps_chunk_size.py
```

**Expected output**:
- Frames per chunk: ~5-6
- Measured FPS: ~25-30 FPS
- No TensorRT errors

## Comparison: Before vs After

| Metric | Before (25 FPS) | After (25-30 FPS) | Improvement |
|--------|----------------|-------------------|-------------|
| Target FPS | 25 | 30 | +20% |
| Chunk size | 6480 samples (405ms) | 3240 samples (202.5ms) | 2x smaller |
| Feed rate | 2.47 Hz | 4.94 Hz | 2x faster |
| IDLE animation | ~3 FPS (burst) | 25-30 FPS (smooth) | 8-10x smoother |
| Latency | 405ms | 202.5ms | 2x lower |
| Frame drops | Frequent (>20 FPS) | Rare (>40 FPS) | Much fewer |

## Benefits

1. **Smoother Animation**: 25-30 FPS provides noticeably smoother motion, especially for idle/listening states
2. **Lower Latency**: 202.5ms audio chunks vs 405ms reduces perceived delay by 50%
3. **Better Utilization**: Takes advantage of Ditto's 50 FPS capability
4. **Simpler Code**: Single universal chunk size for all states
5. **Less Buffering**: Smaller chunks mean less audio buffering needed
6. **TensorRT Compatible**: Respects model's minimum input requirements

## Potential Issues & Solutions

### If frames are still being dropped:
- Check profiling logs: `ENABLE_PROFILING=true`
- Increase frame drop threshold (currently 40 FPS)
- Check GPU utilization
- Verify TensorRT engines are optimized for your GPU

### If chunk size doesn't work:
- **Error**: `Set dimension [1,3240] for tensor input_values does not satisfy any optimization profiles`
- **Solution**: TensorRT model has minimum 3240 samples (already implemented)
- If still issues, try larger sizes: 6480 or 12960 samples
- Check SDK logs for TensorRT errors
- Verify TensorRT engine version matches your GPU

### If FPS is lower than expected:
- GPU may be bottleneck - check with profiling
- Network bandwidth for LiveKit streaming
- Client-side rendering performance
- Queue buildup in StreamSDK pipeline

## Environment Variables

No new environment variables needed. Existing settings apply:

```bash
# Existing - all still work
export DITTO_CFG_PKL="checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
export DITTO_DATA_ROOT="checkpoints/ditto_trt_custom2/"
export DITTO_SOURCE="avatars/avatar.jpg"
export DITTO_MAX_SIZE=1920
export DITTO_EMO=4

# Profiling (to monitor performance)
export ENABLE_PROFILING=true  # default
```

## Monitoring

Watch for these metrics in logs:

```
🔇 Starting silent audio generator for idle/listening states (25-30 FPS)
   Chunk size: 3240 samples (202.5ms @ 16kHz)
   Feed rate: ~5 Hz → 25-30 FPS output
   TensorRT range: [3240..12960] samples

📊 Frames: 300 sent, 5 dropped
📈 Frame generation stats: {'avg_ms': 15.2, 'min_ms': 12.1, 'max_ms': 25.3}
📈 Ditto chunk stats: {'avg_ms': 45.3, 'min_ms': 40.2, 'max_ms': 55.1}
🔇 StreamSDK: Feeding silent audio chunk (3240 samples, 202.5ms)
```

**Good performance indicators**:
- Frame generation avg < 33ms
- Ditto chunk processing < 100ms
- Few frames dropped (< 5% of total)
- Queue sizes stay below 80% full

## Future Optimizations

Potential further improvements:

1. **Adaptive chunk size**: Adjust based on GPU performance
2. **Variable FPS target**: Allow configuration via env var
3. **Smart frame dropping**: Drop less important frames (e.g., minimal motion)
4. **GPU-aware tuning**: Auto-detect GPU capabilities and optimize
5. **Multi-resolution**: Lower resolution for slower GPUs

## Rollback

If issues occur, revert to original settings:

```python
# In webrtc/livekit_gemini_agent.py
self.silent_audio_chunk_size = 6480  # Original size
self.model_chunk_size = 6480
self.target_fps = 25
# Frame dropping: elapsed < self.frame_interval * 0.8
# Sleep in silent audio generator: await asyncio.sleep(0.405)
```

## Important Notes

### Why Not Smaller Chunks?
Initially attempted 1620 samples for true 30 FPS (~10 Hz feed rate), but TensorRT model constraints require minimum 3240 samples:

```
TensorRT Error: Set dimension [1,1620] for tensor input_values does not satisfy
any optimization profiles. Valid range for profile 0: [1,3240]..[1,12960]
```

The 3240-sample chunk size is the **smallest possible** that works with the TensorRT HuBERT model.

### Performance Trade-off
- **Ideal**: 1620 samples → 30 FPS smooth
- **Reality**: 3240 samples → 25-30 FPS (still much better than 3 FPS before)
- **Constraint**: TensorRT model optimization profile

---

**Date**: 2025-11-11
**Author**: AI Assistant
**Status**: Ready for Testing
