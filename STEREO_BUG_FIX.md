# Stereo Audio Bug Fix - Root Cause Analysis

## The Bug

**Symptoms:**
- Audio plays at low pitch (sounds slow)
- Audio-video desync
- Timestamps show as "correct" in logs but playback is wrong

**Root Cause:** Incorrect stereo-to-mono conversion using `flatten()` instead of averaging channels.

## Technical Explanation

### What Was Happening (BUGGY CODE)

Browser sends stereo audio via WebRTC:
```
layout: stereo
samples: 960 (per channel)
sample_rate: 48000 Hz
channels: 2 (left + right)
duration: 960/48000 = 20ms
```

When converted to numpy array:
```python
audio_array = frame.to_ndarray()  # Shape: (2, 960)
# This is: [[L1, L2, L3, ..., L960],    # Left channel
#           [R1, R2, R3, ..., R960]]    # Right channel
```

**OLD CODE (WRONG):**
```python
audio_array = audio_array.flatten()
# Result: [L1, R1, L2, R2, L3, R3, ..., L960, R960]
# Length: 1920 values (interleaved stereo)
```

**The critical error:**
```python
# We treated these 1920 interleaved values as 1920 MONO samples
audio_48k = audio_float.copy()  # 1920 values

# Later, created frame:
frame = av.AudioFrame(samples=1920, sample_rate=48000)
# Browser interprets: "This is 1920 mono samples at 48kHz = 40ms"
# But ACTUALLY: It's 960 stereo samples (20ms) incorrectly flattened!
# Result: 20ms of audio stretched to 40ms = PLAYS AT HALF SPEED = LOW PITCH
```

### The Math

**Actual data:**
- 960 stereo samples = 20ms of audio
- At 48kHz, 20ms should be 960 samples

**What we sent:**
- 1920 samples claiming to be 48kHz mono
- Browser plays 1920/48000 = 40ms duration
- But we only have 20ms of actual audio
- **Playback speed: 20ms audio / 40ms duration = 0.5x speed = LOW PITCH**

**Timestamp error:**
```python
# Duration calculation:
chunk_duration = len(sub_chunk_48k) / 48000.0
# 1920 / 48000 = 0.04s (40ms)

# Actual duration: 0.02s (20ms)
# Accumulated timestamp: 2x too fast
# Result: Audio timestamps ahead of video timestamps = DESYNC
```

### Visual Representation

```
Browser sends (stereo, 20ms):
L: [====960 samples====]  20ms
R: [====960 samples====]  20ms

OLD CODE (flatten):
   [L1 R1 L2 R2 ... L960 R960]  ← 1920 values
   Treated as 1920 mono samples = 40ms
   Result: Plays at 0.5x speed (low pitch)

NEW CODE (average):
   [(L1+R1)/2, (L2+R2)/2, ... (L960+R960)/2]  ← 960 values
   Correctly: 960 mono samples = 20ms
   Result: Plays at 1.0x speed (correct pitch)
```

## The Fix

**NEW CODE (CORRECT):**
```python
# Convert stereo to mono by AVERAGING channels
if audio_array.ndim > 1:
    if audio_array.shape[0] == 2:  # (2, 960) - channels first
        audio_array = audio_array.mean(axis=0)  # → 960 mono samples
    elif audio_array.shape[1] == 2:  # (960, 2) - samples first
        audio_array = audio_array.mean(axis=1)  # → 960 mono samples
```

**Result:**
- 960 stereo samples → 960 mono samples
- Duration: 20ms (correct!)
- Playback speed: 1.0x (normal pitch!)
- Timestamps: accurate

## Why Logs Showed "Correct"

The logs showed:
```
✓ Correct: 1920 samples = 40ms @ 48kHz
```

This was technically true - **IF** the data was mono. But it wasn't mono, it was flattened stereo!

The bug was invisible in logs because:
1. Sample count was "correct" for the format we claimed (mono)
2. PTS timestamps were "aligned" because we used the same (wrong) duration for both audio and video
3. The actual problem was in the **interpretation** of what those samples represented

## Expected Behavior After Fix

### Before (Buggy):
```
Input: 960 stereo samples (20ms)
Flatten: 1920 interleaved values
Claim: 1920 mono samples (40ms)
Result: Audio plays at 0.5x speed, timestamps 2x too fast
```

### After (Fixed):
```
Input: 960 stereo samples (20ms)
Average: 960 mono samples
Claim: 960 mono samples (20ms)
Result: Audio plays at 1.0x speed, timestamps accurate
```

## Logs to Watch For

**New diagnostic logs:**
```
🔍 First audio array: dtype=int16, shape=(2, 960)
   Frame: layout=stereo, samples=960, rate=48000Hz
   Converting stereo to mono: (2, 960) → averaging across axis 0
   ✓ After stereo→mono conversion: (960,)
```

**Updated chunk logging:**
```
🎵 FIRST AUDIO CHUNK:
   Samples: 960 (expected 960 for 20ms@48kHz stereo→mono)
   Sample rate: 48000Hz
   Timestamp: 0.000s
   RMS: 0.0234
✓ Correct: 960 samples = 20ms @ 48kHz mono
```

**What changed:**
- Sample count: 1920 → 960 ✓
- Duration: 40ms → 20ms ✓
- Playback speed: 0.5x → 1.0x ✓

## Testing

1. **Audio pitch test:**
   - Before: Low pitch, sounds slow
   - After: Normal pitch, correct speed

2. **Sync test:**
   - Before: Audio ahead or behind video
   - After: Perfect lip sync

3. **Timestamp test:**
   ```bash
   grep "FIRST AUDIO CHUNK" server.log
   ```
   Should show: **960 samples** (not 1920)

## Why This Matters

This bug affected:
- **All stereo microphone inputs** (most modern devices)
- **WebRTC audio quality** (wrong sample count confused codec)
- **Playback timing** (wrong duration calculations)
- **Sync accuracy** (timestamp accumulation errors)

The fix ensures:
- Correct stereo-to-mono conversion
- Accurate audio duration
- Proper timestamp synchronization
- Normal playback speed

## Related Code Paths

This fix corrects:
1. **Audio reception** (line 465-482): Stereo → mono conversion
2. **Timestamp calculation** (line 607): Now uses correct sample count
3. **Audio frame creation** (line 101): Now has correct duration
4. **PTS calculation** (line 106): Now based on accurate timestamps

All downstream code now receives:
- Correct sample counts
- Accurate durations
- Proper timestamps
- Clean mono audio

No other changes needed - the fix at the source (stereo conversion) corrects everything downstream!
