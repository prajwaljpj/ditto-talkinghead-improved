# Step-by-Step Debugging Guide

## Objective

Find the root cause of:
1. Low pitch audio
2. Audio-video desync

## Step 1: Check What Browser Sends

### Enable Detailed Logging

The server already has diagnostic logs. When you start a session, look for:

```
First audio frame: format=s16, layout=stereo, sample_rate=48000, samples=960
🔍 First audio array: dtype=int16, shape=(2, 960), range=[-55, 38]
   Frame: layout=stereo, samples=960, rate=48000Hz
```

**Questions to answer:**
1. Is it stereo or mono? → Look at `layout=`
2. How many samples? → Look at `samples=`
3. What's the array shape? → Look at `shape=`

**Expected for stereo:**
- layout=stereo
- samples=960 (per channel)
- shape=(2, 960) or (1, 1920)

**Expected for mono:**
- layout=mono
- samples=960
- shape=(960,)

## Step 2: Check Stereo-to-Mono Conversion

Look for this log:
```
Converting stereo to mono: (2, 960) → averaging across axis 0
✓ After stereo→mono conversion: (960,)
```

**Questions to answer:**
1. What shape before conversion? → `shape=(...)`
2. What shape after conversion? → Should be 1D like `(960,)`
3. How many samples after? → Should be 960, NOT 1920

**If you see:**
- Shape stays `(2, 960)` → Conversion didn't work
- Shape becomes `(1920,)` → Using flatten() (WRONG!)
- Shape becomes `(960,)` → Correct!

## Step 3: Check Audio Storage

Look for:
```
🎵 FIRST AUDIO CHUNK:
   Samples: ??? (expected ??? for ??ms@48kHz)
   Sample rate: 48000Hz
   Timestamp: 0.000s
```

**Questions to answer:**
1. How many samples? → Should match mono conversion (960)
2. What duration? → samples / 48000 (should be 0.02s for 960 samples)
3. Is timestamp starting at 0? → Should be 0.000s for first chunk

**Expected:**
- Samples: 960 (if browser sends 960 per channel, stereo)
- Duration: 960/48000 = 0.02s (20ms)
- Timestamp: 0.000s

**If you see:**
- Samples: 1920 → Flattened stereo (WRONG!)
- Samples: 960 → Correct mono conversion ✓

## Step 4: Check Audio Playback

Look for:
```
🎵 AUDIO PLAYBACK STARTED at wall time 1234567890.123
   First frame PTS: 0, timestamp: 0.000s
🔊 First audio frame: s16, 48000Hz, 960 samples, PTS=0
```

**Questions to answer:**
1. How many samples in playback frame? → `??? samples`
2. What's the PTS? → `PTS=???`
3. Does PTS match timestamp? → `PTS = timestamp * 48000`

**Expected:**
- samples: 960 (matches what we stored)
- PTS: 0 (for first frame)
- PTS calculation: `timestamp * 48000`

**Example:**
```
timestamp=0.020s → PTS = 0.020 * 48000 = 960
timestamp=2.000s → PTS = 2.000 * 48000 = 96000
```

## Step 5: Check Video-Audio Sync

Look for:
```
📤 Frame 0: audio_ts=0.000s, audio_clock=0.000s, queue_remaining=39
📤 Frame 50: audio_ts=2.000s, audio_clock=2.000s, queue_remaining=150
```

**Questions to answer:**
1. Are timestamps aligned? → `audio_ts` should equal `audio_clock`
2. Do they increment correctly? → Should increase by ~0.04s per frame
3. Is there drift? → Check if difference grows over time

**Expected:**
- Frame 0: audio_ts=0.000s, audio_clock=0.000s (diff = 0ms)
- Frame 25: audio_ts=1.000s, audio_clock=1.000s (diff = 0ms)
- Frame 50: audio_ts=2.000s, audio_clock=2.000s (diff = 0ms)

**Bad example (desync):**
- Frame 0: audio_ts=0.000s, audio_clock=0.000s (diff = 0ms)
- Frame 25: audio_ts=1.000s, audio_clock=1.200s (diff = 200ms!)
- Frame 50: audio_ts=2.000s, audio_clock=2.450s (diff = 450ms!)

## Step 6: Check Browser Playback (WebRTC Internals)

### Chrome/Edge

1. Open a new tab: `chrome://webrtc-internals`
2. Start your session in another tab
3. Look at the stats for your connection

**What to check:**

**Audio Stream (outbound - to server):**
```
Stats: ssrc_XXXXXXXXX_send (audio)
- codec: opus
- sampleRate: 48000
- channelCount: 1 or 2  ← Check this!
- bytesSent: increasing
```

**Audio Stream (inbound - from server):**
```
Stats: ssrc_XXXXXXXXX_recv (audio)
- codec: opus
- sampleRate: 48000
- packetsReceived: increasing
- jitterBufferDelay: ??? ms  ← Check this
```

**Video Stream (inbound):**
```
Stats: ssrc_XXXXXXXXX_recv (video)
- codec: VP8 or H264
- framesReceived: increasing (~25 per second)
- framesDecoded: should match framesReceived
```

**Sync offset:**
Look for `googTimingFrameInfo` or similar - shows A/V sync

### Firefox

1. Open: `about:webrtc`
2. Find your connection
3. Look at "RTP Stats"

**What to check:**
- Audio jitter
- Packet loss
- Frame rate

## Step 7: Check Actual Pitch

### Create Test Audio

Record yourself saying: "One, two, three, four, five" clearly

**Listen for:**
1. **Pitch:** Does it sound lower than your actual voice?
2. **Speed:** Does it sound slower than you spoke?
3. **Timing:** Do the numbers align with lip movements?

**If pitch is low:**
- Audio is playing slower than recorded
- Likely wrong sample count or sample rate

**If pitch is normal but sync is off:**
- Timestamps might be wrong
- Or WebRTC buffering issue

## Step 8: Measure Actual Duration

### Test with Known Audio

1. Speak for exactly 5 seconds (use a timer)
2. Stop recording
3. Check the logs

Look for the last timestamp:
```
📤 Frame 125: audio_ts=5.000s, audio_clock=5.000s
```

**Questions:**
1. Does `audio_ts` match your recording duration? (Should be ~5.0s)
2. Does `audio_clock` match? (Should also be ~5.0s)
3. Is the video also 5 seconds? (125 frames / 25 fps = 5.0s)

**If timestamps are wrong:**
- If audio_ts > actual time → Duration calculations too large
- If audio_ts < actual time → Duration calculations too small

## Common Issues and Symptoms

### Issue 1: Stereo Flattening

**Symptom:** Low pitch, sounds slow, timestamps 2x too fast

**Logs show:**
```
Samples: 1920 (expected 1920 for 40ms@48kHz)
```

**Cause:** Using `flatten()` instead of `mean()` for stereo

**Fix:** Convert with averaging, not flattening

### Issue 2: Wrong Sample Rate

**Symptom:** Low pitch, wrong speed

**Logs show:**
```
sample_rate=16000 but claiming 48000
```

**Cause:** Sending 16kHz audio but claiming 48kHz

**Fix:** Make sure we send 48kHz audio, not 16kHz

### Issue 3: PTS Misalignment

**Symptom:** Sync issues, might have normal pitch

**Logs show:**
```
Frame 50: audio_ts=2.000s, audio_clock=2.450s
```

**Cause:** Different time origins for audio and video PTS

**Fix:** Use same timestamp for both audio and video PTS

### Issue 4: WebRTC Buffering

**Symptom:** Constant delay but no drift

**Logs show perfect sync, but playback has delay:**
```
Frame 50: audio_ts=2.000s, audio_clock=2.000s  ← Perfect!
```

**Cause:** Browser jitter buffer (30-200ms is normal)

**Fix:** This is expected behavior, not a bug

## Diagnostic Checklist

Run through this checklist with your logs:

- [ ] Browser sends stereo audio (layout=stereo)
- [ ] Stereo correctly converted to mono (shape changes from (2, N) to (N,))
- [ ] Correct sample count after conversion (960, not 1920)
- [ ] Timestamp starts at 0.000s
- [ ] Timestamp increments by correct duration (0.02s for 960 samples)
- [ ] Audio frame has correct samples when sent back (960)
- [ ] Audio PTS = timestamp * 48000
- [ ] Video PTS = timestamp * 25
- [ ] audio_ts equals audio_clock (no drift)
- [ ] Actual playback duration matches recording duration

## Next Steps

After checking all these steps, share:

1. **Logs from Step 1-5** (what browser sends, what we process, what we send back)
2. **WebRTC internals screenshot** (Step 6)
3. **Your observation** (Step 7 - does it sound slow? Is it in sync?)

Then we can pinpoint exactly where the issue is!
