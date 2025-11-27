# Troubleshooting Guide

This guide covers common issues and their solutions when working with the LiveKit Avatar system.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Connection Issues](#connection-issues)
3. [Video Generation Issues](#video-generation-issues)
4. [Audio Issues](#audio-issues)
5. [Performance Issues](#performance-issues)
6. [GPU and Memory Issues](#gpu-and-memory-issues)
7. [LiveKit Issues](#livekit-issues)
8. [Google Cloud Issues](#google-cloud-issues)

---

## Quick Diagnostics

### Health Check Script

Run this to check all components:

```bash
# Check CUDA
nvidia-smi

# Check Python imports
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

import tensorrt
print(f'TensorRT: {tensorrt.__version__}')

from livekit import rtc
print('LiveKit: OK')

from stream_pipeline_online import StreamSDK
print('StreamSDK: OK')
"
```

### Log Levels

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or via environment:

```bash
export LOGLEVEL=DEBUG
python livekit_avatar/agent_worker.py dev
```

---

## Connection Issues

### Token Server Not Running

**Symptoms:**
- Browser shows "Token failed. Start token server!"
- Network error in browser console

**Solution:**
```bash
# Start token server (using uv)
uv run python livekit_client/token_server.py

# Verify it's running
curl http://localhost:8000/token -X POST \
  -H "Content-Type: application/json" \
  -d '{"room":"test","identity":"user"}'
```

### LiveKit Server Not Reachable

**Symptoms:**
- "Connection failed" in browser
- Agent worker fails to connect

**Solution:**
```bash
# Check if LiveKit is running
curl http://localhost:7880/rtc

# Start LiveKit server using Docker (recommended)
sudo docker run --rm \
    -p 7880:7880 \
    -p 7881:7881 \
    -p 7882:7882/udp \
    -e LIVEKIT_KEYS="devkey: devsecret" \
    livekit/livekit-server:latest

# Or using native binary
livekit-server --dev

# Or check LiveKit Cloud status
curl https://your-project.livekit.cloud/rtc
```

### CORS Errors

**Symptoms:**
- Browser console shows CORS errors
- Token request blocked

**Solution:**
The token server includes CORS headers. If still having issues:

```python
# In token_server.py, verify headers are set:
self.send_header("Access-Control-Allow-Origin", "*")
self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
self.send_header("Access-Control-Allow-Headers", "Content-Type")
```

### WebSocket Connection Failed

**Symptoms:**
- "WebSocket connection failed" error
- "Connection closed unexpectedly"

**Solutions:**
1. Check URL protocol (`ws://` for local, `wss://` for secure)
2. Verify firewall allows WebSocket connections
3. Check LiveKit server logs for connection errors

```bash
# Local development
export LIVEKIT_URL="ws://localhost:7880"

# LiveKit Cloud (must be wss://)
export LIVEKIT_URL="wss://your-project.livekit.cloud"
```

---

## Video Generation Issues

### No Video Output

**Symptoms:**
- Audio works but no video appears
- Avatar worker running but no frames

**Diagnostic Steps:**

```python
# Add logging to video generator
logger.info(f"Video queue: {len(self._video_queue_internal)}")
logger.info(f"Audio queue: {len(self._audio_queue_internal)}")
logger.info(f"Paired frames: {len(self._paired_frames)}")
```

**Common Causes:**

1. **Avatar worker not connected:**
   ```bash
   # Check room participants
   # Should see both agent_worker and avatar_worker
   ```

2. **Source image issue:**
   ```bash
   # Verify source image exists and is readable
   file $SOURCE_PATH
   # Should be: JPEG image data or PNG image data
   ```

3. **TensorRT engine mismatch:**
   ```bash
   # Rebuild engines for your GPU
   python scripts/cvt_onnx_to_trt.py \
     --onnx_dir checkpoints/ditto_onnx \
     --trt_dir checkpoints/ditto_trt_custom
   ```

### Video Freezes

**Symptoms:**
- Video starts then freezes
- Audio continues but video stuck

**Solutions:**

1. **Check GPU memory:**
   ```bash
   nvidia-smi
   # Should show < 80% memory usage
   ```

2. **Increase queue size:**
   ```python
   # In avatar_worker.py
   runner = AvatarRunner(
       ...,
       _queue_size_ms=1000,  # Increase from 500
   )
   ```

3. **Check for exceptions:**
   ```python
   # Enable exception logging
   @utils.log_exceptions(logger=logger)
   async def main(...):
   ```

### Poor Video Quality

**Symptoms:**
- Blurry or distorted video
- Artifacts in face region

**Solutions:**

1. **Use higher quality source image:**
   - Minimum 512x512 resolution
   - Clear frontal face
   - Good lighting

2. **Check crop parameters:**
   ```python
   # In setup()
   crop_scale=2.3,  # Increase for more context
   crop_vx_ratio=0,
   crop_vy_ratio=-0.125,
   ```

3. **Increase sampling steps:**
   ```python
   sampling_timesteps=100,  # Default: 50
   ```

---

## Audio Issues

### No Audio Output

**Symptoms:**
- Video plays but no audio
- Agent responds visually but silently

**Diagnostic Steps:**

```javascript
// In browser console
document.querySelectorAll('audio').forEach(a => {
  console.log('Audio element:', a.srcObject, 'Volume:', a.volume, 'Muted:', a.muted);
});
```

**Solutions:**

1. **Check audio track subscription:**
   ```javascript
   // Verify in simple_client.html
   if (participant.identity === 'avatar_worker') {
     console.log('Playing audio from avatar');
     const audioElement = track.attach();
     audioElement.volume = 1.0;
     document.body.appendChild(audioElement);
   }
   ```

2. **Verify DataStream setup:**
   ```python
   # In agent_worker.py
   session.output.audio = DataStreamAudioOutput(
       ctx.room,
       destination_identity=AVATAR_IDENTITY,
   )
   ```

### Audio Desync (Lip Sync Issues)

**Symptoms:**
- Audio and video out of sync
- Lips move before/after audio

**Solutions:**

1. **Check pairing order:**
   ```python
   # Audio must be yielded BEFORE video
   yield audio
   yield video
   ```

2. **Reduce processing latency:**
   ```python
   # Decrease chunk size for lower latency (but lower quality)
   self.chunksize = (2, 3, 1)  # Default: (3, 5, 2)
   ```

3. **Check queue drift:**
   ```python
   # Monitor in _main_loop
   logger.info(f"Audio queue: {len(self._audio_queue_internal)}, "
               f"Video queue: {len(self._video_queue_internal)}")
   ```

### Echo or Feedback

**Symptoms:**
- Agent hears its own output
- Audio feedback loop

**Solutions:**

1. **Ensure agent doesn't publish audio:**
   ```python
   # In agent_worker.py
   await session.start(
       ...,
       room_output_options=RoomOutputOptions(
           audio_enabled=False,  # Must be False
       ),
   )
   ```

2. **Client-side echo cancellation:**
   ```javascript
   const localAudioTrack = await createLocalAudioTrack({
     echoCancellation: true,
     noiseSuppression: true,
   });
   ```

---

## Performance Issues

### High Latency

**Symptoms:**
- Noticeable delay between speaking and response
- > 500ms end-to-end latency

**Diagnostic:**
```python
# Enable timing in video generator
generator._log_wait_statistics()

# Output shows:
# - Queue wait times
# - Ditto processing times
# - Pair event wait times (KEY METRIC)
```

**Solutions:**

1. **Reduce diffusion steps:**
   ```python
   sampling_timesteps=25,  # Default: 50
   ```

2. **Use faster GPU:**
   - RTX 4090 > RTX 4080 > RTX 3090 > RTX 3080

3. **Optimize TensorRT engines:**
   ```bash
   # Build with FP16
   python scripts/cvt_onnx_to_trt.py --fp16
   ```

### Low Frame Rate

**Symptoms:**
- Choppy video
- Frames dropping

**Diagnostic:**
```bash
# Monitor GPU
watch -n 0.5 nvidia-smi

# Check frame rate in logs
logger.info(f"FPS: {self._frames_yielded / elapsed:.1f}")
```

**Solutions:**

1. **Reduce resolution:**
   ```python
   avatar_options = AvatarOptions(
       video_width=640,   # Reduce from 1280
       video_height=360,  # Reduce from 720
   )
   ```

2. **Check GPU thermal throttling:**
   ```bash
   nvidia-smi -q -d TEMPERATURE
   ```

3. **Close other GPU applications**

### Memory Leak

**Symptoms:**
- Memory usage grows over time
- Eventually crashes

**Solutions:**

1. **Clear diagnostic arrays:**
   ```python
   # Already implemented in _log_wait_statistics()
   self._audio_queue_wait_times.clear()
   # ... other arrays
   ```

2. **Check for zombie threads:**
   ```python
   import threading
   print(f"Active threads: {threading.active_count()}")
   ```

---

## GPU and Memory Issues

### CUDA Out of Memory

**Symptoms:**
- "CUDA out of memory" error
- Process killed

**Solutions:**

1. **Check memory usage:**
   ```bash
   nvidia-smi
   ```

2. **Reduce batch size / resolution:**
   ```python
   avatar_options = AvatarOptions(
       video_width=640,
       video_height=360,
   )
   ```

3. **Clear CUDA cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. **Use memory-efficient model:**
   ```bash
   export DATA_ROOT="checkpoints/ditto_trt_custom"  # FP16 version
   ```

### TensorRT Engine Load Failed

**Symptoms:**
- "TensorRT engine not compatible"
- "Serialization error"

**Solutions:**

1. **Rebuild for your GPU:**
   ```bash
   python scripts/cvt_onnx_to_trt.py \
     --onnx_dir checkpoints/ditto_onnx \
     --trt_dir checkpoints/ditto_trt_$(hostname)
   ```

2. **Check TensorRT version:**
   ```python
   import tensorrt
   print(tensorrt.__version__)
   # Should match version used to build engines
   ```

3. **Verify CUDA compute capability:**
   ```python
   import torch
   print(torch.cuda.get_device_capability())
   # Should be >= (7, 5) for RTX cards
   ```

### cuDNN Errors

**Symptoms:**
- "cuDNN error: CUDNN_STATUS_*"

**Solutions:**

1. **Check cuDNN version:**
   ```bash
   cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR
   ```

2. **Reinstall cuDNN:**
   ```bash
   # Match CUDA version
   pip install nvidia-cudnn-cu12
   ```

---

## LiveKit Issues

### Avatar Worker Not Joining Room

**Symptoms:**
- Agent starts but avatar doesn't connect
- "participant_disconnected" not firing

**Solutions:**

1. **Check subprocess launch:**
   ```python
   # Add logging
   logger.info(f"Avatar command: {' '.join(cmd)}")
   logger.info(f"Avatar env: LIVEKIT_URL={env['LIVEKIT_URL']}")
   ```

2. **Check avatar worker logs separately:**
   ```bash
   # Run avatar worker directly
   LIVEKIT_URL=ws://localhost:7880 \
   LIVEKIT_TOKEN="..." \
   python livekit_avatar/avatar_worker.py
   ```

3. **Verify token grants:**
   ```python
   # Token must have:
   # - room_join=True
   # - room=<room_name>
   # - kind="agent"
   ```

### DataStream Not Working

**Symptoms:**
- Audio not reaching avatar
- "data_received" not firing

**Solutions:**

1. **Verify DataStream setup:**
   ```python
   # Agent side
   session.output.audio = DataStreamAudioOutput(
       ctx.room,
       destination_identity="avatar_worker",  # Must match exactly
   )
   
   # Avatar side
   audio_recv = DataStreamAudioReceiver(room)
   ```

2. **Check room permissions:**
   - Both participants need data channel permissions

### Room Disconnection

**Symptoms:**
- Sudden disconnection
- "disconnected" event fires unexpectedly

**Solutions:**

1. **Check server timeouts:**
   ```bash
   # LiveKit server config
   livekit-server --dev --room-empty-timeout 300
   ```

2. **Implement reconnection:**
   ```python
   @room.on("reconnecting")
   def on_reconnecting():
       logger.warning("Reconnecting...")
   
   @room.on("reconnected")
   def on_reconnected():
       logger.info("Reconnected!")
   ```

---

## Google Cloud Issues

### Vertex AI Authentication Failed

**Symptoms:**
- "Permission denied" errors
- "Could not authenticate"

**Solutions:**

1. **Verify credentials:**
   ```bash
   gcloud auth application-default print-access-token
   ```

2. **Check service account permissions:**
   - Must have `Vertex AI User` role

3. **Set environment correctly:**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/creds.json"
   ```

### Gemini Model Not Available

**Symptoms:**
- "Model not found" error
- "Invalid model name"

**Solutions:**

1. **Check model availability:**
   ```python
   # List available models
   from google.cloud import aiplatform
   aiplatform.init(project="your-project", location="us-central1")
   # Check Vertex AI console for available models
   ```

2. **Use correct model name:**
   ```bash
   export GEMINI_MODEL="gemini-live-2.5-flash-preview-native-audio-09-2025"
   ```

3. **Check region availability:**
   - Not all models available in all regions
   - Try `us-central1` first

### Quota Exceeded

**Symptoms:**
- "Quota exceeded" error
- Rate limiting

**Solutions:**

1. **Check quotas in Cloud Console**
2. **Request quota increase**
3. **Implement rate limiting:**
   ```python
   import asyncio
   await asyncio.sleep(0.1)  # Between requests
   ```

---

## Getting Help

If you've tried the above solutions and still have issues:

1. **Collect diagnostic info:**
   ```bash
   nvidia-smi > diagnostics.txt
   python --version >> diagnostics.txt
   pip list >> diagnostics.txt
   ```

2. **Enable verbose logging:**
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Check GitHub issues** for similar problems

4. **Include in bug reports:**
   - OS version
   - GPU model
   - Python version
   - Full error traceback
   - Steps to reproduce

