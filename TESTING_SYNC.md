# Testing Audio-Video Synchronization

## The Bug That Was Fixed

**Problem:** Audio was playing ahead of video by 10-50ms.

**Root Cause:**
```python
# OLD CODE (BUGGY):
audio_chunk = get_audio()
add_to_buffer(audio_chunk)  # ← Audio starts playing immediately
await asyncio.sleep(0.04)    # ← Video frame delayed by pacing
return video_frame           # ← Video sent 40ms after audio!
```

**Result:** Audio played before corresponding video, causing desync.

**Fix:**
```python
# NEW CODE (FIXED):
audio_chunk = get_audio()
add_to_buffer(audio_chunk)   # ← Audio added to buffer
return video_frame           # ← Video sent immediately
# WebRTC handles pacing based on PTS timestamps
```

**Result:** WebRTC synchronizes both streams based on PTS timestamps, maintaining perfect sync.

## Testing Tools

### 1. Quick Test: Check Server Logs

**Run server:**
```bash
python webrtc/signaling_server.py \
  --cfg_pkl checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl \
  --data_root checkpoints/ditto_trt_custom2/ \
  --host 0.0.0.0 \
  --port 8080 \
  2>&1 | tee server.log
```

**Connect with your web UI and record a few seconds.**

**Then analyze logs:**
```bash
python diagnose_sync.py --file server.log
```

**Expected output (GOOD):**
```
🎯 AUDIO TIMESTAMP vs AUDIO CLOCK:
   Average difference: 8.3ms
   ✓ Audio and video timestamps are aligned (within 100ms)

📈 DRIFT ANALYSIS:
   Early frames (first 5): 7.2ms difference
   Late frames (last 5): 9.1ms difference
   Drift: 1.9ms
   ✓ No significant drift detected
```

**Bad output:**
```
🎯 AUDIO TIMESTAMP vs AUDIO CLOCK:
   Average difference: 125.4ms
   ⚠️ PROBLEM DETECTED: Audio clock is ahead by 125.4ms
   This means: Audio is playing BEFORE the video it belongs to!
```

### 2. Full Test: WebRTC Test Client

**This test client acts like a browser and measures sync directly.**

**Install dependencies:**
```bash
uv pip install websockets librosa
```

**Run test:**
```bash
# Terminal 1: Start server
python webrtc/signaling_server.py \
  --cfg_pkl checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl \
  --data_root checkpoints/ditto_trt_custom2/ \
  --host 0.0.0.0 \
  --port 8080

# Terminal 2: Run test client
python test_webrtc_client.py \
  --server ws://localhost:8080 \
  --avatar avatars/Hyperrealistic_Indian_Woman_Professional_Presentation_static_5_loop.jpg \
  --audio openai-fm-coral-professional.wav \
  --output test_results
```

**Expected output:**
```
AUDIO-VIDEO SYNCHRONIZATION ANALYSIS
================================================================================
📊 RECEIVED FRAMES:
   Video frames: 250
   Audio frames: 250

⏱️ PRESENTATION TIMESTAMPS:
   Video PTS range: 0.000s - 10.000s
   Audio PTS range: 0.000s - 10.000s

🎯 SYNCHRONIZATION:
   Start time difference: 12.3ms
   End time difference: 18.7ms
   ✓ Timestamps aligned

📍 SYNC POINTS (every second):
   Time   | Video PTS  | Audio PTS  | Diff
   -------+------------+------------+---------
     0.0s |      0.000s |      0.000s |    0.0ms
     1.0s |      1.000s |      1.008s |   -8.0ms
     2.0s |      2.000s |      2.012s |  -12.0ms
   ...
```

### 3. Live Monitoring

**Monitor sync in real-time while using web UI:**

```bash
python webrtc/signaling_server.py ... 2>&1 | python diagnose_sync.py
```

This will show live analysis as frames are sent.

## What to Look For

### Good Sync
```
📤 Frame 50: audio_ts=2.000s, audio_clock=2.008s, queue_remaining=150
```
- `audio_clock - audio_ts` should be small (<50ms)
- Difference should stay relatively constant (no drift)

### Bad Sync (Audio Ahead)
```
📤 Frame 50: audio_ts=2.000s, audio_clock=2.150s, queue_remaining=150
```
- `audio_clock` is 150ms ahead of `audio_ts`
- Means audio played before video
- You'll hear words before seeing lips move

### Bad Sync (Video Ahead)
```
📤 Frame 50: audio_ts=2.000s, audio_clock=1.850s, queue_remaining=150
```
- `audio_ts` is ahead of `audio_clock`
- Means video played before audio
- You'll see lips move before hearing sound

## Troubleshooting

### If you still see desync:

**1. Check if it's browser buffering:**
- Different browsers have different jitter buffers
- Chrome: ~30-100ms audio buffer
- Firefox: ~50-150ms audio buffer
- Safari: Variable

**2. Check network latency:**
- WebRTC adds buffering for network jitter
- Local network: <10ms
- Internet: 50-200ms jitter buffer

**3. Verify PTS timestamps:**
```bash
# Look for this in logs:
📤 Frame 50: audio_ts=2.000s, audio_clock=2.040s
```
Should increment by ~0.04s per frame (25 FPS).

**4. Check model performance:**
```bash
python profile_inference.py ... --num_chunks 50
```
Should still show:
- Real-time factor: >1.0x
- Jitter: <20ms

### Known Issues

**Issue:** Audio pops/clicks during playback
**Cause:** Audio buffer underrun
**Fix:** Increase audio queue size in `BufferedAudioTrack` (line 60):
```python
self.audio_queue = asyncio.Queue(maxsize=200)  # Increase from 200 to 500
```

**Issue:** Video stutters but audio is smooth
**Cause:** Frame queue overflow (model generating too fast)
**Fix:** Increase frame queue size in `DittoVideoTrack` (line 172):
```python
self.frame_queue = asyncio.Queue(maxsize=50)  # Increase from 50 to 100
```

**Issue:** First second of video is out of sync, then corrects
**Cause:** Initial buffering phase
**Solution:** This is normal. WebRTC needs ~500ms to stabilize.

## Expected Performance

With the fix:
- **Sync accuracy:** <20ms difference between audio and video PTS
- **Drift:** <5ms per 10 seconds
- **Stability:** Sync should remain consistent throughout session

## Technical Details

**How WebRTC Synchronization Works:**

1. **PTS (Presentation Timestamp):**
   - Each audio frame has PTS = sample_count / sample_rate
   - Each video frame has PTS = audio_timestamp * 25 (frame units)

2. **WebRTC Playback:**
   - Browser maintains a clock
   - Plays audio frame when clock == audio PTS
   - Displays video frame when clock == video PTS
   - If both have same PTS origin, they stay synced

3. **Our Implementation:**
   - Audio timestamp = accumulated duration when captured
   - Video PTS = corresponding audio timestamp
   - Both referenced to same time origin = perfect sync

**Why Sleep-Based Pacing Failed:**
```
Timeline:
0ms:   Video frame ready
0ms:   Add audio to buffer ← Audio starts playing
40ms:  Sleep complete
40ms:  Return video frame ← Video displays now
Result: Audio is 40ms ahead!
```

**Why PTS-Based Sync Works:**
```
Timeline:
0ms:   Video frame ready, PTS=2.0s
0ms:   Add audio, PTS=2.0s
0ms:   Return both to WebRTC
2.0s:  WebRTC clock reaches 2.0s
2.0s:  Both audio and video play together
Result: Perfect sync!
```
