# WebRTC Monitoring Guide

## Overview

This guide explains how to monitor the health and performance of the WebRTC signaling server (`signaling_server_v2.py`) and diagnose audio-video synchronization issues.

## Server-Side Monitoring

### Key Log Messages

#### Sync Status Logs (Every 100 Frames)

```
✅ Sync #100: frame=99, ts=3.960s, video_queue=30/200, audio_queue=28/500, wait=5ms
```

**Fields**:
- **Status emoji**: ✅ (good) / ⚠️ (warning) / ❌ (bad)
- **Sync count**: Number of frames synchronized
- **Frame index**: Current frame number
- **Timestamp**: Playback position in seconds
- **video_queue**: Buffered video frames (current/max)
- **audio_queue**: Buffered audio chunks (current/max)
- **wait**: Time waited for audio to be available (ms)

**Interpretation**:

| Metric | Good | Warning | Bad | Action |
|--------|------|---------|-----|--------|
| wait | <100ms | 100-500ms | >500ms | Check model speed or network |
| video_queue | 20-50/200 | 50-180/200 | 180+/200 | May drop frames, increase queue |
| audio_queue | 20-50/500 | 5-20/500 | <5/500 | Client not sending fast enough |

#### Audio Reception Logs

```
📥 First audio frame: 48000Hz, stereo, 960 samples
   Range: [-0.002, 0.002], RMS: 0.0003
```

**Check**:
- Sample rate should be 48000Hz (standard WebRTC)
- Format should be stereo (browser default)
- RMS > 0 indicates audio has signal (not silence)
- Range should be within [-1, 1] for normalized audio

#### Video Generation Logs

```
🎬 First video frame queued: idx=0, timestamp=0.000s
📊 Generated 100 frames, queue size: 30/200
```

**Check**:
- First frame should appear within ~1-2s of first audio
- Queue size growing indicates model faster than consumption (good!)
- Queue size shrinking or at 0 indicates model slower than consumption (bad)

#### Warning Signs

```
⚠️ Video queue near full (185/200) - model generating faster than network can transmit
⚠️ Audio queue low (3) - client may not be sending audio fast enough
❌ Video queue full! This breaks audio sync.
❌ Timeout waiting for audio after 10.5s!
```

**Actions**:
1. **Video queue near full**: Normal for fast model, but watch for actual drops
2. **Audio queue low**: Check client network, verify continuous recording
3. **Video queue full** (drop): Increase maxsize or investigate network bottleneck
4. **Audio timeout**: Check client is sending audio, verify network connectivity

### Performance Metrics

#### Model Speed

Track frame generation rate:

```python
# In logs every 100 frames
frames_generated = 100
time_elapsed = 2.0  # seconds
actual_fps = frames_generated / time_elapsed  # e.g., 50 FPS

# Compare to target
real_time_factor = actual_fps / 25  # e.g., 2.0x real-time
```

**Interpretation**:
- **>1.5x**: Excellent (model faster than real-time, buffering ahead)
- **1.0-1.5x**: Good (model keeping up)
- **0.8-1.0x**: Warning (model barely keeping up, may stutter)
- **<0.8x**: Bad (model too slow, will definitely stutter)

#### Network Throughput

Estimate bandwidth usage:

```python
# Video: 1280x720 @ 25fps, typical H.264 encoding
video_bitrate = 1500  # kbps (varies with encoder settings)

# Audio: 48kHz mono, Opus codec
audio_bitrate = 64   # kbps

total_bandwidth = 1564  # kbps ≈ 1.5 Mbps
```

**Check**:
- Client upload speed should be >500 kbps (for audio input)
- Client download speed should be >2 Mbps (for video+audio output)
- Server upload speed should be >2 Mbps per client

#### Latency Breakdown

```
T0: Client speaks
T1: Server receives audio        (network latency: 10-50ms)
T2: Model generates video         (processing latency: 50-200ms)
T3: Client receives video         (network latency: 10-50ms)
T4: Client plays video            (jitter buffer: 30-200ms)

Total latency: 100-500ms (acceptable for real-time avatar)
```

**Monitor** in logs:
```
⏱️ TIMING: First audio chunk sent to Ditto at t=21.000
🎬 TIMING: First video frame at t=22.500, model latency: 1.500s
```

**First chunk latency** >2s indicates slow model startup.

## Client-Side Monitoring (chrome://webrtc-internals)

### Accessing WebRTC Stats

1. Open Chrome browser
2. Navigate to `chrome://webrtc-internals`
3. Select your PeerConnection
4. Click "Download the PeerConnection updates and stats data"

### Key Metrics to Monitor

#### Audio Track Stats (Inbound - from server)

```
ssrc_XXXXXX_recv (audio)
├─ packetsReceived: 5000      # Should increase steadily
├─ packetsLost: 5             # Should be <1% of received
├─ jitter: 0.015              # <50ms is good
├─ jitterBufferDelay: 0.045   # 30-200ms is normal
└─ audioLevel: 0.05           # >0 means audio has signal
```

**Check**:
- **packetsLost**: <1% (network is stable)
- **jitter**: <0.050 (50ms - network timing variance low)
- **jitterBufferDelay**: 0.03-0.20 (30-200ms - browser buffering)
- **audioLevel**: >0.01 (audio has content, not silence)

#### Video Track Stats (Inbound - from server)

```
ssrc_YYYYYY_recv (video)
├─ packetsReceived: 10000
├─ packetsLost: 10
├─ framesDecoded: 250         # Should match expected FPS * time
├─ framesDropped: 0           # Should be 0 or very low
├─ framesPerSecond: 25        # Should match target (25)
└─ googTimingFrameInfo: 45ms  # A/V offset
```

**Check**:
- **framesPerSecond**: Should be stable at 25 (not fluctuating)
- **framesDropped**: Should be 0 (or <1% of decoded)
- **googTimingFrameInfo**: <50ms (audio-video offset)

#### RTCPeerConnection Stats

```
RTCIceCandidate (selected pair)
├─ googRtt: 25ms              # Round-trip time
├─ bytesReceived: 5000000     # Total bytes
└─ bytesSent: 500000
```

**Check**:
- **googRtt**: <100ms (network latency good)
- **bytesReceived/Sent**: Should increase steadily (no stalls)

### Interpreting A/V Sync Offset

**googTimingFrameInfo** shows actual A/V offset measured by browser:

| Offset | Status | Action |
|--------|--------|--------|
| <50ms | ✅ Excellent | None - perfect sync |
| 50-100ms | ⚠️ Noticeable | Acceptable for most use cases |
| 100-200ms | ⚠️ Bad | Check server logs for wait times |
| >200ms | ❌ Unacceptable | Investigate FIFO pairing logic |

**Note**: Our server uses FIFO pairing with shared timestamps, so this should ALWAYS be <50ms. If not, there's a bug!

## Diagnostic Scenarios

### Scenario 1: Audio Ahead of Video

**Symptoms**:
- Voice finishes before lips stop moving
- googTimingFrameInfo negative (e.g., -150ms)

**Diagnosis**:
```bash
# Check server logs
grep "Sync #" server.log | tail -20

# Look for:
❌ Sync #500: wait=800ms  # High wait = model too slow
```

**Root Cause**: Video generation slower than audio arrival

**Fix**:
1. Optimize model (reduce quality, use TensorRT, etc.)
2. Accept higher latency (buffer more audio before starting)

### Scenario 2: Video Ahead of Audio

**Symptoms**:
- Lips move before voice heard
- googTimingFrameInfo positive (e.g., +150ms)

**Diagnosis**:
```bash
# Check server logs
grep "audio_queue=" server.log | tail -20

# Look for:
⚠️ Audio queue low (2) - client may not be sending audio fast enough
```

**Root Cause**: Audio not arriving fast enough from client

**Fix**:
1. Check client network connection
2. Verify microphone permissions
3. Check browser console for errors

### Scenario 3: Intermittent Desync

**Symptoms**:
- Sync good initially, then drifts
- googTimingFrameInfo fluctuates wildly

**Diagnosis**:
```bash
# Check for frame drops
grep "dropped" server.log

# Look for:
❌ Video queue full! This breaks audio sync.
```

**Root Cause**: Frame dropping breaks FIFO pairing

**Fix**:
1. Increase video_queue maxsize (default: 200)
2. Improve network bandwidth
3. Reduce video quality

### Scenario 4: Stuttering Playback

**Symptoms**:
- Video freezes intermittently
- framesDropped increases
- framesPerSecond fluctuates

**Diagnosis**:
```bash
# Check video generation rate
grep "Generated.*frames" server.log | tail -20

# Calculate FPS:
# If 100 frames generated in 5 seconds → 20 FPS (too slow!)
```

**Root Cause**: Model not generating fast enough

**Fix**:
1. Optimize model (faster GPU, lower quality)
2. Reduce target FPS (change from 25 to 15?)
3. Accept stuttering (model is too slow)

## Automated Monitoring Script

Create `monitor_webrtc.sh`:

```bash
#!/bin/bash
# Monitor WebRTC signaling server health

LOG_FILE="server.log"
ALERT_THRESHOLD_WAIT=500  # ms
ALERT_THRESHOLD_QUEUE=180 # frames

echo "Monitoring WebRTC server..."
tail -f "$LOG_FILE" | while read line; do
    # Check for high wait times
    if echo "$line" | grep -q "wait=[5-9][0-9][0-9]ms\|wait=[0-9]\{4,\}ms"; then
        echo "⚠️ ALERT: High wait time detected!"
        echo "$line"
    fi

    # Check for queue near full
    if echo "$line" | grep -q "video_queue=1[8-9][0-9]\|video_queue=200"; then
        echo "⚠️ ALERT: Video queue near full!"
        echo "$line"
    fi

    # Check for errors
    if echo "$line" | grep -q "❌"; then
        echo "🚨 ERROR detected!"
        echo "$line"
    fi
done
```

Run with: `./monitor_webrtc.sh`

## Health Check Checklist

Use this checklist every hour during production:

### Server Health
- [ ] Sync wait time <100ms (check recent logs)
- [ ] Video queue 20-100/200 (healthy buffer)
- [ ] Audio queue 20-100/500 (healthy buffer)
- [ ] No "❌" errors in logs (check last 1000 lines)
- [ ] Model generating >25 FPS (check generation stats)

### Client Health (chrome://webrtc-internals)
- [ ] framesPerSecond = 25 ±2 (stable frame rate)
- [ ] packetsLost <1% (network stable)
- [ ] framesDropped = 0 (no client-side drops)
- [ ] googTimingFrameInfo <50ms (perfect sync)
- [ ] jitterBufferDelay 30-200ms (normal buffering)

### User Experience
- [ ] Lip-sync appears correct (manual check)
- [ ] No audio crackling or dropouts
- [ ] No video freezes or stuttering
- [ ] Latency feels acceptable (<500ms end-to-end)

## Tuning Parameters

If experiencing issues, adjust these in `signaling_server_v2.py`:

### Queue Sizes

```python
# VideoGenerator.__init__()
self.frame_queue = asyncio.Queue(maxsize=200)  # Default: 200

# AudioPassthrough.__init__()
self.playback_chunks = deque(maxlen=500)  # Default: 500
```

**Guidelines**:
- Model 2x real-time: 200/500 is good
- Model 3x real-time: Increase to 300/750
- Model 1x real-time: Can reduce to 100/250
- Slow network: Increase both

### Timeout Values

```python
# AVSynchronizer.get_next()
timeout_seconds = 10.0  # Default: 10s
```

**Guidelines**:
- Fast model (<1s latency): 10s is safe
- Slow model (>2s latency): Increase to 20s
- Production (want fast failure): Reduce to 5s

### Logging Frequency

```python
# Log every N frames
if self._sync_count % 100 == 0:  # Default: every 100

# For debugging: every 10 frames
if self._sync_count % 10 == 0:

# For production: every 1000 frames
if self._sync_count % 1000 == 0:
```

## Conclusion

Monitor these key indicators:

✅ **Sync wait time** <100ms (audio arriving fast enough)
✅ **Queue sizes** 20-100 (healthy buffering)
✅ **googTimingFrameInfo** <50ms (perfect sync)
✅ **No frame drops** (FIFO ordering preserved)

The FIFO pairing architecture makes monitoring simple - if timestamps are shared correctly, sync will be perfect. Most issues come from model speed or network problems, not the sync logic itself.
