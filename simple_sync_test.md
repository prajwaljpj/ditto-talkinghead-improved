# Simple Sync Testing Guide

Since the WebRTC test client has compatibility issues, here's how to test sync using your existing web UI:

## Quick Sync Test

### 1. Start Server with Logging

```bash
python webrtc/signaling_server.py \
  --cfg_pkl checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl \
  --data_root checkpoints/ditto_trt_custom2/ \
  --host 0.0.0.0 \
  --port 8080 \
  2>&1 | tee sync_test.log
```

### 2. Use Your Web UI

1. Open your web UI in browser
2. Start a session
3. Speak for 10-15 seconds
4. Note if you see any sync issues

###3. Analyze the Logs

```bash
python diagnose_sync.py --file sync_test.log
```

## What to Look For

### Good Sync (Expected After Fix)

```
📤 Frame 50: audio_ts=2.000s, audio_clock=2.016s, queue_remaining=150
📤 Frame 100: audio_ts=4.000s, audio_clock=4.024s, queue_remaining=145
📤 Frame 150: audio_ts=6.000s, audio_clock=6.020s, queue_remaining=140
```

**Analysis:**
- Difference (`audio_clock - audio_ts`): ~16-24ms
- Consistent across all frames
- No drift

**Diagnosis Output:**
```
🎯 AUDIO TIMESTAMP vs AUDIO CLOCK:
   Average difference: 18.5ms
   ✓ Audio and video timestamps are aligned (within 100ms)

📈 DRIFT ANALYSIS:
   Drift: 2.3ms
   ✓ No significant drift detected
```

### Bad Sync (If Bug Still Exists)

```
📤 Frame 50: audio_ts=2.000s, audio_clock=2.200s, queue_remaining=150
📤 Frame 100: audio_ts=4.000s, audio_clock=4.385s, queue_remaining=145
📤 Frame 150: audio_ts=6.000s, audio_clock=6.570s, queue_remaining=140
```

**Analysis:**
- Difference: 200ms and growing!
- Getting worse over time (drift)
- Audio is way ahead of video

**Diagnosis Output:**
```
🎯 AUDIO TIMESTAMP vs AUDIO CLOCK:
   Average difference: 385.2ms
   ⚠️ PROBLEM: Audio clock is ahead by 385ms!
   This means: Audio is playing BEFORE the video!

📈 DRIFT ANALYSIS:
   Drift: 185.3ms
   ⚠️ DRIFT DETECTED: Sync is getting worse over time!
```

## Manual Observation

### What Audio-Ahead Looks Like:
- You hear words before lips move
- Audio seems to "lead" the video
- Delay is constant (e.g., always 100ms ahead)

### What Video-Ahead Looks Like:
- You see lips moving before hearing sound
- Video seems to "lead" the audio
- Delay is constant (e.g., always 100ms behind)

### What Drift Looks Like:
- Starts in sync, gets progressively worse
- By end of 10-second clip, might be 200+ms off
- Sync degradation over time

## Expected Results with Fix

After the fix I made:
- **Difference should be <50ms** (typically 10-30ms)
- **No drift** (difference stays constant)
- **Visually:** Lips and audio should be perfectly synchronized

The small 10-30ms difference is normal - it's the WebRTC jitter buffer and processing delay. As long as it's consistent and doesn't grow, sync is perfect.

## If Sync is Still Bad

### Check these in logs:

**1. Are frames being sent?**
```
📊 STATUS: Sent 100 frames to WebRTC
```
Should appear every 100 frames.

**2. Is audio being consumed?**
```
📤 Frame 50: audio_ts=2.000s, audio_clock=2.016s
```
Should appear every 50 frames. If you don't see these, audio isn't being matched with video.

**3. Are there errors?**
```
⚠️ No audio chunk for frame 50 (queue empty)
```
If you see this repeatedly, audio queue is being exhausted too quickly.

**4. Check model performance:**
```bash
python profile_inference.py \
  --cfg_pkl your_config.pkl \
  --data_root your_data \
  --source your_avatar.jpg \
  --audio test_audio.wav \
  --num_chunks 50
```

Should still show:
- Real-time factor: >1.5x
- Jitter: <20ms

If model slowed down, that could cause issues.

## Troubleshooting Commands

**Check if server is running:**
```bash
curl http://localhost:8080
# or
netstat -an | grep 8080
```

**Monitor logs in real-time:**
```bash
tail -f sync_test.log | grep "📤 Frame"
```

**Count frames sent:**
```bash
grep "📤 Frame" sync_test.log | wc -l
```

**Extract sync data:**
```bash
grep "📤 Frame" sync_test.log | awk '{print $8, $10}' > sync_data.txt
```

Then you can plot this in Excel/Python to visualize the difference over time.

## Alternative: Browser DevTools

Most browsers have WebRTC internals pages:

**Chrome:** `chrome://webrtc-internals`
**Firefox:** `about:webrtc`
**Edge:** `edge://webrtc-internals`

These show:
- Audio and video packet timestamps
- Jitter buffer stats
- Dropped frames/packets
- Playback timing

Look for the `googTimingFrameInfo` stat which shows A/V sync offset.

## Simple Manual Test

1. Speak a simple phrase: "One, two, three, four, five"
2. Count each number as you say it
3. Watch the video playback
4. Note if you hear the number before or after seeing lips move
5. Good sync: Sound and lips move together
6. Bad sync: Clear delay between sound and lips

If you can visually confirm sync is good, the fix worked even if we can't run the automated test!
