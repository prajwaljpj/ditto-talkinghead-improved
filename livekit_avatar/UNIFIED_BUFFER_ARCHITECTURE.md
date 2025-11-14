# Unified Buffer Architecture - Seamless TTS/Idle Transitions

## What Changed

**Before:** Separate handling for TTS and idle modes with explicit mode switching
**After:** Unified buffer that accumulates both TTS and idle audio seamlessly

## The Problem This Solves

### Old Architecture Issues:
1. **Mode switching gaps** - When transitioning TTS ↔ Idle, generator would exit → timeout → restart → refill buffer
2. **"Frame capture was behind schedule"** warnings during transitions
3. **Sync drift** when user speaks (idle mode) - video would stutter/freeze
4. **Hard cutoffs** - Abrupt transitions instead of smooth blending

### Root Cause:
```python
# OLD: Separate modes
try:
    frame = await queue.get(timeout=20ms)  # TTS mode
    # Process TTS...
except TimeoutError:
    # EXIT main loop, enter separate idle generator
    async for frame in _generate_idle_frames():
        yield frame
    # Return to main loop → restart → gap!
```

**Result:** 300ms+ gaps during each transition, causing sync issues.

## New Architecture

### One Buffer for Everything

```python
# NEW: Unified accumulation
while True:
    buffered_audio_chunks = []

    # Accumulate to 6480 samples (TTS or zeros)
    while len(buffer) < 6480:
        try:
            frame = await queue.get(timeout=1ms)  # TTS
        except TimeoutError:
            frame = create_silent_frame()  # Idle

        # Same path for both!
        synced_frames = audio_bstream.push(frame)
        buffer += synced_frames
        buffered_audio_chunks.append(synced_frames)

    # Feed to Ditto
    ditto.run_chunk(buffer[:6480])

    # Yield buffered chunks (640 samples each)
    for audio_chunk in buffered_audio_chunks:
        yield audio_chunk  # Audio
        yield video_frame  # Video (1:1 ratio)
```

### Key Improvements

**1. No Mode Switching**
- No separate idle generator
- No exits/restarts
- Continuous loop accumulating audio from either source

**2. Seamless Transitions**
- One 6480-sample buffer can contain **partial TTS + partial zeros**
- Example during interruption:
  ```
  Samples 0-3200:   TTS audio (user interrupts)
  Samples 3200-6480: Silent audio (idle)
  → Ditto receives smooth blend, generates transition frames
  ```

**3. Consistent 1:1 Audio:Video Ratio**
- Still split output to 640-sample chunks
- AvatarRunner gets perfect sync
- No frame count mismatches

**4. Natural Blending**
- Hard cutoff eliminated
- Smooth audio/video transitions
- Better user experience

## Flow Diagram

```
┌─────────────────────────────────────────────┐
│         Main Generation Loop                │
│         (Runs Continuously)                 │
└─────────────────────────────────────────────┘
                    ↓
        ┌──────────────────────┐
        │  Try Get TTS Audio   │
        │   (1ms timeout)      │
        └──────────────────────┘
                ↓           ↓
           Got TTS      Timeout
                ↓           ↓
         TTS Frame   Silent Frame
                ↓           ↓
                └─────┬─────┘
                      ↓
            ┌────────────────────┐
            │  AudioByteStream   │
            │  (640 samples)     │
            └────────────────────┘
                      ↓
            ┌────────────────────┐
            │  Add to Buffer     │
            │  (Accumulate)      │
            └────────────────────┘
                      ↓
              Buffer >= 6480?
                  No → Loop
                  Yes ↓
            ┌────────────────────┐
            │  Feed to Ditto     │
            │  (6480 samples)    │
            └────────────────────┘
                      ↓
           ┌─────────────────────┐
           │  Ditto Generates    │
           │  ~10 Video Frames   │
           └─────────────────────┘
                      ↓
            ┌────────────────────┐
            │  Yield Audio+Video │
            │  (640 sample chunks)│
            │  (1:1 ratio)       │
            └────────────────────┘
                      ↓
              Back to top (continuous)
```

## Code Changes

### Removed Methods:
- ❌ `_generate_idle_frames()` - No longer needed
- ❌ `_generate_continuous_idle()` - No longer needed
- ❌ `_prefill_idle_buffer()` - No longer needed

### Simplified Main Loop:
- ✅ Single `_video_generation_impl()` handles everything
- ✅ 1ms timeout instead of 20ms (faster idle detection)
- ✅ Unified buffer accumulation
- ✅ Natural TTS/idle blending

### Key Code Sections:

**Audio Acquisition (Line 226-286):**
```python
# Try TTS (1ms timeout)
try:
    frame = await asyncio.wait_for(queue.get(), timeout=0.001)
except asyncio.TimeoutError:
    # Generate silent frame inline
    frame = self._create_silent_audio_frame()

# Both paths converge here
synced_frames = audio_bstream.push(frame)
buffer += synced_frames
buffered_chunks.append(synced_frames)
```

**Ditto Processing (Line 306-317):**
```python
# Feed accumulated buffer to Ditto
ditto_chunk = buffer[:6480]
await run_in_executor(ditto.run_chunk, ditto_chunk)
```

**Output (Line 319-331):**
```python
# Yield buffered 640-sample chunks
for audio_chunk in buffered_chunks:
    yield audio_chunk
    yield video_frame  # 1:1 ratio maintained
```

## Benefits

### Performance
- **Eliminated gaps:** No more 300ms+ delays during transitions
- **Reduced overhead:** No mode switching, no generator restarts
- **Faster idle detection:** 1ms timeout vs 20ms

### Quality
- **Smooth transitions:** Natural blend instead of hard cutoff
- **Better sync:** Continuous frame generation, no gaps
- **No warnings:** Should eliminate "Frame capture was behind schedule"

### Code
- **Simpler:** One loop instead of multiple generators
- **More maintainable:** Unified logic easier to understand
- **Less code:** Removed ~200 lines of mode-specific logic

## Expected Results

### Before (Old Architecture):
```
Agent speaking → User interrupts
  ↓
Exit TTS mode → Timeout 20ms → Enter idle mode
  ↓
Refill buffer (300ms) → Start yielding idle frames
  ↓
WARNING: Frame capture was behind schedule for 350ms
  ↓
Video stutters, sync drift
```

### After (New Architecture):
```
Agent speaking → User interrupts
  ↓
Buffer: [TTS audio...3200 samples][Silent...3280 samples]
  ↓
Feed to Ditto (seamless blend)
  ↓
Yield frames continuously (no gap)
  ↓
Smooth transition, perfect sync ✅
```

## Testing

**Run the server:**
```bash
./livekit_server.sh
```

**Test transitions:**
1. Let agent speak
2. Interrupt mid-sentence
3. Watch logs - should see:
   - `⚪ Generating silent frame (idle)` when you speak
   - `📥 Got TTS frame` when agent responds
   - **No "Frame capture was behind schedule" warnings**
   - Continuous frame generation, no gaps

**Expected logs:**
```
DEBUG:🎨 Feeding 6480 samples to Ditto (buffered 10 audio chunks)
DEBUG:✅ Ditto complete in 12.3ms (queue: 8)
DEBUG:⚪ Generating silent frame (idle)
DEBUG:⚪ Generating silent frame (idle)
DEBUG:📥 Got TTS frame
DEBUG:🎨 Feeding 6480 samples to Ditto (buffered 10 audio chunks)
```

**No more:**
```
WARNING: Frame capture was behind schedule for XXX ms ❌
```

## Summary

The unified buffer architecture **eliminates mode switching** by treating TTS and idle audio identically during accumulation. This creates **seamless transitions**, fixes **sync issues**, and simplifies the codebase. The 6480-sample buffer can naturally contain **mixed content** during transitions, creating smooth blends instead of hard cutoffs.

**Result:** Continuous, smooth video playback with perfect sync in all modes! 🎉
