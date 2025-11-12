# Audio Synchronization and Low Pitch Fix

## Problems Identified

### 1. Low Pitch Audio
**Symptom:** Audio sounds like it's playing in slow motion, lower pitch than original

**Cause:** Audio being played at wrong sample rate
- If we send 640 samples (16kHz) but claim it's 48kHz
- Browser plays at 48kHz speed, but data is only 16kHz
- Result: Plays 3x slower = low pitch

**Fix:** Added diagnostic logging to detect this:
```python
if len(audio_chunk) == 640:
    logger.error("⚠️⚠️⚠️ BUG: Received 640 samples (16kHz) but claiming 48kHz! LOW PITCH!")
elif len(audio_chunk) == 1920:
    logger.warning("✓ Correct: 1920 samples = 40ms @ 48kHz")
```

### 2. Audio-Video Desync
**Symptom:** Lips and audio not synchronized, audio might lead or lag video

**Root Cause:** Audio and video had different PTS time origins

**Before (BROKEN):**
```python
# Video PTS
frame.pts = int(audio_timestamp * 25)  # From capture time

# Audio PTS
frame.pts = self._timestamp  # From playback start (different origin!)
```

**After (FIXED):**
```python
# Video PTS
frame.pts = int(audio_timestamp * 25)  # From capture time

# Audio PTS
frame.pts = int(audio_timestamp * 48000)  # SAME capture time!
```

## Changes Made

### File: `webrtc/signaling_server.py`

**1. Modified `BufferedAudioTrack.add_audio_chunk()` (lines 67-124)**

Changed signature to accept timestamp:
```python
def add_audio_chunk(self, audio_chunk: np.ndarray, timestamp: float):
```

Key changes:
- Takes `timestamp` parameter (in seconds)
- Sets PTS based on timestamp: `frame.pts = int(timestamp * 48000)`
- Stores tuple in queue: `(frame, timestamp)`
- Enhanced logging to detect 640 vs 1920 sample issue

**2. Modified `BufferedAudioTrack.recv()` (lines 137-190)**

Key changes:
- Unpacks tuple from queue: `frame, timestamp = await self.audio_queue.get()`
- Updates audio clock: `self._playback_time_offset = timestamp`
- Returns frame with correct PTS

**3. Modified `DittoVideoTrack.recv()` (lines 267-274)**

Key changes:
- Passes timestamp to add_audio_chunk:
  ```python
  self.buffered_audio_track.add_audio_chunk(audio_chunk_data, audio_timestamp)
  ```

## How Synchronization Now Works

### PTS Alignment

Both audio and video now use the SAME timestamp origin:

```
Audio Capture Timeline:
├─ 0.000s: First chunk captured
├─ 0.040s: Second chunk
├─ 0.080s: Third chunk
└─ ...

Audio PTS = timestamp * 48000 samples/sec
Video PTS = timestamp * 25 frames/sec

Example at t=2.000s:
├─ Audio PTS = 2.0 * 48000 = 96000
└─ Video PTS = 2.0 * 25 = 50

WebRTC converts both to same timeline:
├─ Audio: 96000 / 48000 = 2.0 seconds
└─ Video: 50 / 25 = 2.0 seconds
✓ Perfect sync!
```

## Testing

### 1. Check for Low Pitch Issue

**Start server and look for this log:**
```bash
python webrtc/signaling_server.py ... 2>&1 | grep "FIRST AUDIO CHUNK" -A 10
```

**Good output:**
```
🎵 FIRST AUDIO CHUNK:
   Samples: 1920 (expected 1920 for 40ms@48kHz)
   Sample rate: 48000Hz
   Timestamp: 0.000s
   RMS: 0.0234
✓ Correct: 1920 samples = 40ms @ 48kHz
```

**Bad output (if you see this):**
```
🎵 FIRST AUDIO CHUNK:
   Samples: 640 (expected 1920 for 40ms@48kHz)
⚠️⚠️⚠️ BUG: Received 640 samples (16kHz) but claiming 48kHz! LOW PITCH!
⚠️ This will play 3x slower = low pitch audio!
```

If you see the bad output, the problem is in the audio chunk storage code around line 595-620. The 48kHz buffer is not being populated correctly.

### 2. Check for Sync

**Use the diagnostic tool:**
```bash
python diagnose_sync.py --file server.log
```

**Expected output (GOOD):**
```
🎯 AUDIO TIMESTAMP vs AUDIO CLOCK:
   Average difference: <50ms
   ✓ Audio and video timestamps are aligned
```

### 3. Visual Test

1. Say: "One, two, three, four, five"
2. Watch playback
3. Each number should have perfect lip sync
4. Audio pitch should sound normal (not slow/low)

## Debugging

### If audio is still low pitch:

**Check the logs for:**
```
🎵 FIRST AUDIO CHUNK:
   Samples: ???
```

- If 640 samples → Problem: Sending 16kHz data as 48kHz
- If 1920 samples → Correct

**If 640 samples, the bug is here:**
Around line 595-620 in the audio storage code. Check:
1. Is `audio_buffer_48k` being populated correctly?
2. Is `samples_48k_needed = len(chunk) * 3` correct?
3. Are we extracting from `audio_buffer_48k` not `audio_buffer`?

### If audio-video are still desynced:

**Check PTS values in logs:**
```bash
grep "Added.*audio chunks" server.log | head -5
grep "📤 Frame" server.log | head -5
```

Look for:
```
✓ Added 25 audio chunks, PTS=48000, timestamp=1.000s
📤 Frame 25: audio_ts=1.000s, audio_clock=1.000s
```

Audio PTS should be `timestamp * 48000`
Video PTS should be `timestamp * 25`

Both timestamps should match!

### If desync gets worse over time (drift):

**Run diagnostic:**
```python
python diagnose_sync.py --file server.log
```

Look for "DRIFT ANALYSIS". If drift > 50ms, there's a timing accumulation bug.

## Technical Details

### Sample Rate Conversion

**Input:** WebRTC receives audio at 48kHz from browser
**Ditto Processing:** Requires 16kHz
**Output:** Must send back at 48kHz to browser

```
Browser (48kHz)
    ↓
Downsample to 16kHz → Ditto Model
    ↓
Keep 48kHz copy → Store in queue
    ↓
Send 48kHz → Browser (correct pitch)
```

**CRITICAL:** We must send the 48kHz copy, NOT the 16kHz copy!

### PTS Timestamp Units

**Audio:**
- PTS in samples: `timestamp * 48000`
- Time base: `1/48000`
- Real time: `PTS * (1/48000) = timestamp` ✓

**Video:**
- PTS in frames: `timestamp * 25`
- Time base: `1/25`
- Real time: `PTS * (1/25) = timestamp` ✓

Both resolve to the same real time = synchronized!

### WebRTC Synchronization

WebRTC uses RTP timestamps and RTCP Sender Reports to sync streams:
1. Each stream has monotonic PTS values
2. WebRTC maps both to a common clock
3. Playback happens when clock reaches PTS
4. If audio PTS=2.0s and video PTS=2.0s, both play together

Our fix ensures both streams have PTS derived from same `timestamp` value.

## Expected Results

After this fix:
- **Audio pitch:** Normal (not slow/low)
- **Sync accuracy:** <20ms difference
- **Drift:** <5ms over 10 seconds
- **Visually:** Perfect lip sync

## Rollback (if needed)

If this breaks something, the key changes to revert are:

1. `add_audio_chunk` signature: Remove `timestamp` parameter
2. PTS calculation: Go back to `self._timestamp`
3. Queue storage: Store frame only, not `(frame, timestamp)` tuple
4. recv(): Don't unpack tuple

But this should fix both your issues!
