# Audio-Video Synchronization Fix

This document explains the audio-video desync fix implemented in `webrtc/signaling_server.py` and provides tools for profiling and debugging.

## Problem

The original implementation had video driving audio synchronization, which caused desync issues:
1. **Wrong driver**: Video frames pulled audio chunks when ready, instead of audio clock driving video playback
2. **No model latency compensation**: Ditto model has ~1-3s delay between audio input and video output
3. **No adaptive sync**: Frames sent immediately without waiting for audio playback clock
4. **No frame dropping**: Late frames not dropped, causing accumulated drift

## Solution

### Audio-Driven Synchronization (Implemented)

The fix implements proper audio-driven synchronization following these principles:

**1. Audio is the Master Clock**
- `BufferedAudioTrack` maintains `audio_clock` that advances as audio is played
- Video synchronization is driven by this audio clock
- Human ear is more sensitive to audio timing than video timing

**2. Timestamp Everything**
- Each 48kHz audio chunk is tagged with its capture timestamp when it arrives
- Formula: `chunk_timestamp = accumulated_audio_duration`
- Stored as `(audio_chunk, timestamp)` tuples in FIFO queue

**3. Model Latency Compensation**
- **Phase 1 (first 5 frames)**: Measure model latency
  - `latency = audio_clock - audio_timestamp`
  - Average latency from 5 samples
  - Pre-fill audio buffer by `latency * 25 fps` frames

- **Phase 2 (subsequent frames)**: Synchronized playback
  - Target playback time = `audio_timestamp + model_latency`
  - Calculate: `time_until_frame = target_playback_time - audio_clock`

  - **If frame is early** (`time_until_frame > 10ms`): Wait for audio clock
  - **If frame is on time** (`-100ms < time_until_frame < 10ms`): Send immediately
  - **If frame is late** (`time_until_frame < -100ms`): **Drop frame**

**4. Adaptive Frame Dropping**
- Frames more than 100ms late are dropped
- Audio chunk still added to maintain sync
- Recursive call to get next frame

## Implementation Details

### Key Changes in `signaling_server.py`

**BufferedAudioTrack (lines 49-192)**:
- Added `_playback_time_offset` to track audio playback time
- Added `get_audio_clock_time()` method that returns current audio time
- Audio clock updates automatically as frames are played

**DittoVideoTrack (lines 194-378)**:
- Added `_model_latency`, `_latency_samples`, `_sync_enabled` for latency compensation
- `recv()` method implements two-phase synchronization:
  1. Latency measurement (first 5 frames)
  2. Synchronized playback with compensation

**DittoWebRTCSession (lines 380+)**:
- Modified audio chunk storage to include timestamps (line 610)
- Timestamp calculation: `chunk_timestamp = accumulated_audio_duration` (line 606)
- Each chunk duration: `len(chunk) / 48000.0` seconds (line 607)

## Profiling Tools

### 1. `profile_inference.py` - Offline Model Profiling

Measures Ditto model's frame generation characteristics without WebRTC overhead.

**Usage:**
```bash
python profile_inference.py \
  --cfg_pkl /path/to/config.pkl \
  --data_root /path/to/data \
  --source /path/to/avatar.jpg \
  --audio /path/to/audio.wav \
  --num_chunks 20
```

**Metrics Measured:**
- Average FPS and real-time factor
- Frame interval statistics (mean, median, std dev)
- Jitter (timing consistency)
- Startup latency (audio → first frame)
- Frame interval distribution histogram
- Identification of slow frames (>60ms)

**Example Output:**
```
📈 OVERALL METRICS:
   Total frames generated: 250
   Total duration: 10.234s
   Average FPS: 24.43
   Real-time factor: 0.977x
   ⚠️ WARNING: Real-time factor < 1.0 means model is TOO SLOW for real-time!

📊 FRAME INTERVAL STATISTICS:
   Expected interval (25 FPS): 40.0ms
   Mean interval: 40.9ms
   Std deviation: 12.3ms

📉 JITTER (timing consistency):
   Frame timing jitter: 12.3ms
   ⚠️ HIGH JITTER: Frame generation is inconsistent!
```

### 2. `monitor_sync.py` - Real-time WebRTC Monitoring

Analyzes log output during WebRTC streaming to detect sync issues in real-time.

**Usage:**
```bash
python webrtc/signaling_server.py ... 2>&1 | python monitor_sync.py
```

**Tracks:**
- Audio chunks stored vs retrieved
- Frames sent, waiting, and dropped
- Frame drop rate and average lateness
- Wait time statistics
- Audio clock progression

**Example Output:**
```
⚠️ DROPPED FRAME 45: late by 125ms (audio_clock=1.840s, frame_time=1.715s)

📊 AUDIO CHUNKS:
   Stored: 250
   Retrieved: 245
   Remaining: 5

🎬 VIDEO FRAMES:
   Sent: 238
   Dropped: 12
   Drop rate: 4.8%
```

## Debugging

### Log Messages to Watch

**Normal Operation:**
```
🎤 AUDIO CAPTURE STARTED at wall time 123456.789
🎵 AUDIO CLOCK STARTED at wall time 123457.234
📥 STORE: queue_idx=0, timestamp=0.000s, 48kHz samples=1920, RMS=0.1234
🔬 MEASURING LATENCY: Frame 0 - audio_ts=0.000s, audio_clock=0.040s, latency=0.040s
📊 MODEL LATENCY MEASURED: 1.234s (avg of 5 samples)
📥 PRE-FILLING AUDIO BUFFER: target=35 chunks
✓ BUFFER PRE-FILLED: 35 chunks, enabling sync on next frame
🎯 SYNC: frame 10, audio_ts=0.400s, target=1.634s, audio_clock=1.640s, diff=6.0ms
✓ ON TIME: frame 10, diff=6.0ms
```

**Problems:**
```
⚠️ DROP FRAME 25: too late by 125.3ms (audio_clock=1.840s, target=1.715s)
⚠️ HIGH JITTER: Frame generation is inconsistent!
⚠️ WARNING: Real-time factor < 1.0 means model is TOO SLOW for real-time!
⚠️ No audio chunk for frame 50, queue empty
```

### Troubleshooting

**High Frame Drop Rate (>5%):**
- Run `profile_inference.py` to check model FPS
- If real-time factor < 1.0, model is too slow:
  - Reduce `max_size` parameter
  - Use GPU acceleration if not already enabled
  - Consider model optimization

**High Jitter (>20ms std dev):**
- Ditto model has inconsistent frame generation
- May need larger buffer (increase `target_prefill` in line 292)
- Consider using fixed-rate interpolation

**Frames Always Waiting:**
- Model latency underestimated
- Increase prefill buffer size
- Check if audio is arriving too fast

**Audio Queue Empty:**
- Audio input stopped or too slow
- Check WebRTC audio track status
- Verify microphone permissions

## Testing

Run the profiler first to establish baseline:
```bash
python profile_inference.py \
  --cfg_pkl config/ditto.pkl \
  --data_root /data/ditto \
  --source examples/avatar.jpg \
  --audio examples/test.wav \
  --num_chunks 50
```

Then test WebRTC streaming with monitoring:
```bash
python webrtc/signaling_server.py \
  --cfg_pkl config/ditto.pkl \
  --data_root /data/ditto \
  --host 0.0.0.0 \
  --port 8080 \
  2>&1 | python monitor_sync.py
```

## Performance Expectations

**Good Performance:**
- Real-time factor: > 1.0x
- Frame jitter: < 10ms std dev
- Frame drop rate: < 1%
- Model latency: < 2.0s

**Acceptable Performance:**
- Real-time factor: 0.95-1.0x
- Frame jitter: 10-20ms std dev
- Frame drop rate: 1-5%
- Model latency: 2.0-3.0s

**Poor Performance (needs optimization):**
- Real-time factor: < 0.95x
- Frame jitter: > 20ms std dev
- Frame drop rate: > 5%
- Model latency: > 3.0s

## Technical References

The implementation follows audio-video synchronization best practices:

1. **Audio Master Clock**: Audio is less tolerant to timing errors than video
2. **Presentation Timestamps (PTS)**: Adjusted by `model_latency` for proper WebRTC timing
3. **Adaptive Buffering**: Pre-fill buffer based on measured latency
4. **Late Frame Dropping**: Frames >100ms late are dropped to prevent drift accumulation
5. **Jitter Tolerance**: 10ms tolerance for "on time" classification (±5ms)

## Future Improvements

Possible enhancements:
1. **Dynamic latency adjustment**: Re-measure latency periodically to adapt to changing conditions
2. **Interpolation**: Generate intermediate frames for smoother playback during slow model periods
3. **Predictive scheduling**: Use model's known chunksize to predict frame timing
4. **Quality adaptation**: Reduce model quality when real-time factor drops below 1.0
