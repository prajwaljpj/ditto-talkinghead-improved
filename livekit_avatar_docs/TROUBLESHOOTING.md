# Troubleshooting Guide

Common issues and their solutions for the LiveKit Avatar Agent.

## Table of Contents

1. [Agent Startup Issues](#agent-startup-issues)
2. [Audio Problems](#audio-problems)
3. [Video Problems](#video-problems)
4. [Network and Connection Issues](#network-and-connection-issues)
5. [Performance Issues](#performance-issues)
6. [Google Cloud / Vertex AI Issues](#google-cloud--vertex-ai-issues)
7. [Ditto Model Issues](#ditto-model-issues)
8. [Client Browser Issues](#client-browser-issues)

---

## Agent Startup Issues

### "GOOGLE_APPLICATION_CREDENTIALS must be set"

**Symptom:** Agent fails to start with environment variable error.

**Cause:** Google Cloud credentials not configured.

**Solution:**
```bash
# Set the environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Verify it's set
echo $GOOGLE_APPLICATION_CREDENTIALS

# Verify file exists
ls -l $GOOGLE_APPLICATION_CREDENTIALS
```

**Permanent Fix:** Add to `~/.bashrc` or `~/.zshrc`:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/avatar-service-account.json"
```

---

### "No module named 'stream_pipeline_online'"

**Symptom:** Import error when starting agent.

**Cause:** Running from wrong directory or missing file.

**Solution:**
```bash
# Ensure you're in the project root
cd /path/to/ditto-talkinghead

# Verify file exists
ls stream_pipeline_online.py

# Run from this directory
./livekit_server.sh
```

---

### "Failed to connect to LiveKit server"

**Symptom:** Agent can't connect to LiveKit.

**Cause:** LiveKit server not running or wrong URL.

**Solution:**
```bash
# Check if LiveKit server is running
docker ps | grep livekit

# If not running, start it
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
  livekit/livekit-server --dev

# Verify you can reach it
curl http://localhost:7880
```

For cloud LiveKit:
```bash
# Verify URL format
export LIVEKIT_URL="wss://your-project.livekit.cloud"

# Test connectivity
curl https://your-project.livekit.cloud
```

---

### "Vertex AI authentication failed"

**Symptom:** Error about invalid credentials or permissions.

**Cause:** Service account lacks permissions or wrong project.

**Solution:**
```bash
# Verify credentials file is valid JSON
cat $GOOGLE_APPLICATION_CREDENTIALS | python -m json.tool

# Check service account has correct permissions
gcloud projects get-iam-policy $VERTEX_PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:*"

# Grant Vertex AI User role
gcloud projects add-iam-policy-binding $VERTEX_PROJECT_ID \
  --member="serviceAccount:your-sa@project.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

---

## Audio Problems

### "No audio from agent"

**Symptom:** Avatar video plays but no sound.

**Causes and Solutions:**

**1. Browser Autoplay Policy**
```javascript
// In browser console, check:
console.log(document.autoplay);

// Solution: User must interact with page first
// Add a "Click to enable audio" button
```

**2. Audio Element Not Created**
```javascript
// Check if audio track is being attached
room.on(RoomEvent.TrackSubscribed, (track) => {
  if (track.kind === Track.Kind.Audio) {
    console.log('Audio track received');
    const audioElement = track.attach();
    audioElement.play(); // Explicitly play
  }
});
```

**3. Volume Muted**
```bash
# Check system volume
amixer get Master

# Check browser isn't muted (see browser tab icon)
```

---

### "Audio stuttering or choppy"

**Symptom:** Avatar audio playback is interrupted.

**Causes:**
- Network bandwidth insufficient
- CPU overloaded
- Audio buffer underrun

**Solutions:**
```python
# Increase audio buffer size in custom_avatar_worker.py
self._audio_queue = asyncio.Queue(maxsize=200)  # Increase from 100

# Reduce video quality to free bandwidth
export AVATAR_WIDTH=640
export AVATAR_HEIGHT=360
```

```javascript
// In client, adjust audio jitter buffer
const track = await createLocalAudioTrack({
  echoCancellation: true,
  noiseSuppression: true,
  // Add jitter buffer
  latencyHint: 'playback',
});
```

---

### "Audio out of sync with video"

**Symptom:** Lip movements don't match audio.

**Cause:** Latency mismatch between audio and video pipelines.

**Solution:**
```python
# Check frame generation latency in logs
logger.info(f"Frame gen time: {frame_time_ms}ms")

# If > 50ms, GPU may be overloaded
# Reduce resolution or FPS

# In custom_avatar_worker.py
FPS = 30  # Reduce from 50
```

---

## Video Problems

### "No video appearing"

**Symptom:** Black screen or no video element.

**Diagnostic Steps:**
```javascript
// 1. Check if track is being published (agent logs)
// Should see: "✅ Published video track: 1280x720"

// 2. Check if track is being received (browser console)
room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
  console.log('Track received:', {
    kind: track.kind,
    name: publication.trackName,
    participant: participant.identity
  });
});

// 3. Check if track is being attached
if (track.kind === Track.Kind.Video) {
  const element = track.attach();
  console.log('Attached to:', element);
}
```

**Solutions:**
- Ensure agent is publishing: Check agent logs for "Published video track"
- Verify track subscription: Check browser console for TrackSubscribed event
- Check video element: Inspect DOM for `<video>` element and src

---

### "Video is frozen"

**Symptom:** First frame shows but no updates.

**Cause:** Frame generation stopped or GPU issue.

**Solution:**
```bash
# Check GPU status
nvidia-smi

# Check GPU memory
# If "Out of Memory", reduce resolution

# Check agent logs for errors
# Look for frame generation errors

# Restart agent
./livekit_server.sh
```

---

### "Video quality is poor"

**Symptom:** Blocky, pixelated, or blurry video.

**Causes and Solutions:**

**1. Insufficient Bitrate**
```python
# In main_agent.py, increase bitrate
options = rtc.TrackPublishOptions(
    source=rtc.TrackSource.SOURCE_CAMERA,
    video_encoding=rtc.VideoEncoding(
        max_framerate=50,
        max_bitrate=8_000_000,  # Increase to 8 Mbps
    ),
)
```

**2. Resolution Too Low**
```bash
export AVATAR_WIDTH=1920
export AVATAR_HEIGHT=1080
```

**3. Network Bandwidth Limited**
- Check network speed: Use speedtest
- Reduce resolution if bandwidth < 5 Mbps

---

### "Frame rate is low / choppy video"

**Symptom:** Video plays at < 30 FPS.

**Diagnostic:**
```python
# Add FPS logging in custom_avatar_worker.py
import time

last_frame_time = time.time()
frame_count = 0

def _handle_generated_frame(self, frame_rgb, frame_idx, timestamp):
    global frame_count, last_frame_time
    frame_count += 1

    if time.time() - last_frame_time >= 1.0:
        logger.info(f"FPS: {frame_count}")
        frame_count = 0
        last_frame_time = time.time()

    # ... rest of method
```

**Solutions:**
- **GPU Overloaded:** Reduce resolution or FPS
- **CPU Bottleneck:** Check CPU usage with `top`
- **I/O Bottleneck:** Use faster storage for model files

---

## Network and Connection Issues

### "Connection drops frequently"

**Symptom:** Client disconnects and reconnects often.

**Causes:**
- Unstable network
- Firewall blocking UDP
- LiveKit server issues

**Solutions:**
```bash
# Enable TCP fallback (if using cloud LiveKit)
# Check LiveKit dashboard for connection stats

# Test network stability
ping -c 100 google.com

# Check for packet loss
mtr google.com

# Ensure UDP ports are open
sudo ufw allow 7882/udp  # For local server
```

---

### "High latency (> 1 second)"

**Symptom:** Delay between speaking and avatar response.

**Diagnostic:**
```python
# Add latency tracking in main_agent.py
import time

@main_agent_session.on("user_turn_completed")
def on_user_stopped(data):
    global user_stop_time
    user_stop_time = time.time()

@main_agent_session.on("agent_started_speaking")
def on_agent_speaking(data):
    latency = time.time() - user_stop_time
    logger.info(f"Response latency: {latency:.2f}s")
```

**Solutions:**
- **Network Latency:** Use cloud LiveKit closer to users
- **LLM Latency:** Use faster Gemini model or optimize prompt
- **Avatar Generation:** Reduce resolution/FPS

---

## Performance Issues

### "High GPU memory usage"

**Symptom:** `nvidia-smi` shows > 8GB VRAM usage.

**Solution:**
```bash
# Reduce resolution
export AVATAR_WIDTH=640
export AVATAR_HEIGHT=360

# Check for memory leaks
watch -n 1 nvidia-smi

# Restart agent to free memory
# If persistent, may need different TensorRT engine
```

---

### "High CPU usage"

**Symptom:** CPU at 100%, system slow.

**Causes:**
- Audio resampling overhead
- Too many concurrent threads
- Inefficient video encoding

**Solutions:**
```python
# Reduce FPS
FPS = 25  # in custom_avatar_worker.py

# Use hardware video encoding (if available)
# Check if ffmpeg has GPU support

# Limit concurrent operations
# Ensure only one avatar instance per machine
```

---

### "Slow avatar generation (> 100ms per frame)"

**Symptom:** Frame generation takes too long.

**Diagnostic:**
```python
# Time each pipeline stage
import time

start = time.time()
self.sdk.run_chunk(audio_for_sdk, self.chunksize)
logger.info(f"Generation time: {(time.time() - start)*1000:.1f}ms")
```

**Solutions:**
- **GPU Too Slow:** Use faster GPU (RTX 4000+ series)
- **Model Not Optimized:** Rebuild TensorRT engines for your GPU
- **CPU Bottleneck:** Check preprocessing isn't using CPU

---

### "Frame capture was behind schedule" warning at startup

**Symptom:** Warning like "Frame capture was behind schedule for 348.39 ms" appears during agent startup.

**Cause:** Ditto model cold start - CUDA and TensorRT engines need initialization time on first run.

**Impact:**
- Only occurs during initial model warmup
- Does NOT affect runtime performance
- After initialization, model produces frames in real-time

**Solution:**
The current implementation includes a warmup phase that pre-initializes the model:
```python
# In custom_avatar_worker.py __init__
logger.info("Warming up Ditto model...")
self._warmup_model()  # Generates 3 dummy frames
logger.info("✅ Model warmup complete")
```

This warmup:
- Initializes CUDA memory allocations
- Loads TensorRT engines into GPU
- Prepares model for real-time inference
- Minimizes lag warnings during actual operation

**Note:** First-time model loading may still show brief warnings. This is expected behavior and does not indicate a problem.

---

## Google Cloud / Vertex AI Issues

### "Quota exceeded" error

**Symptom:** Agent fails with quota/rate limit error.

**Cause:** Vertex AI API quota exceeded.

**Solution:**
```bash
# Check quotas in GCP console
# Navigate to: IAM & Admin > Quotas

# Request quota increase
# Or wait for quota reset (usually daily)

# Temporarily use lower request rate
# Add rate limiting in agent code
```

---

### "Model not found" error

**Symptom:** Gemini model name not recognized.

**Cause:** Model not available in your region or project.

**Solution:**
```bash
# Check available models
gcloud ai models list --region=$VERTEX_LOCATION

# Use a different model
export GEMINI_MODEL="gemini-1.5-flash-002"

# Or try different region
export VERTEX_LOCATION="us-central1"
```

---

## Ditto Model Issues

### "Ditto checkpoint files not found"

**Symptom:** Error loading model files.

**Solution:**
```bash
# Verify paths
ls -la checkpoints/ditto_trt_custom2/
ls -la checkpoints/ditto_cfg/

# Check environment variables
echo $DATA_ROOT
echo $CFG_PKL

# Ensure paths are absolute or relative to where you run from
export DATA_ROOT="$(pwd)/checkpoints/ditto_trt_custom2/"
```

---

### "TensorRT engine incompatible with GPU"

**Symptom:** Error about CUDA/TensorRT version mismatch.

**Cause:** TensorRT engines built for different GPU architecture.

**Solution:**
```bash
# Rebuild TensorRT engines for your GPU
# See Ditto documentation for rebuild instructions

# Or use generic (slower) engines
# Check if alternative checkpoints available
```

---

### "Avatar looks distorted"

**Symptom:** Generated avatar has visual artifacts.

**Causes:**
- Poor source image quality
- Mismatched resolution
- Model inference issues

**Solutions:**
```bash
# Use high-quality source image
# - Clear frontal face
# - Good lighting
# - Neutral expression
# - At least 512x512

# Verify source image
file avatars/my_avatar.jpg
# Should show: JPEG, dimensions, etc.

# Try different source image
export SOURCE_PATH="avatars/alternative_avatar.jpg"
```

---

## Client Browser Issues

### "LiveKit SDK failed to load"

**Symptom:** JavaScript error about LiveKit undefined.

**Cause:** CDN failed or wrong SDK version.

**Solution:**
```html
<!-- Try different CDN -->
<script src="https://unpkg.com/livekit-client@latest/dist/livekit-client.umd.min.js"></script>

<!-- Or specific version -->
<script src="https://cdn.jsdelivr.net/npm/livekit-client@2.0.0/dist/livekit-client.umd.min.js"></script>

<!-- Or local copy -->
<!-- Download SDK and serve locally -->
```

---

### "Microphone permission denied"

**Symptom:** getUserMedia error.

**Cause:** Browser permission denied or insecure context.

**Solutions:**
- **HTTPS Required:** Use HTTPS or localhost
- **Permission Denied:** User must allow in browser settings
- **No Microphone:** Check hardware connection

```javascript
// Better error handling
try {
  const track = await createLocalAudioTrack();
} catch (err) {
  if (err.name === 'NotAllowedError') {
    alert('Microphone access denied. Please allow in browser settings.');
  } else if (err.name === 'NotFoundError') {
    alert('No microphone found. Please connect a microphone.');
  } else {
    alert(`Microphone error: ${err.message}`);
  }
}
```

---

### "Video won't play (browser autoplay policy)"

**Symptom:** Video element exists but doesn't play.

**Cause:** Browser autoplay restrictions.

**Solution:**
```javascript
// Add user gesture requirement
<button id="startButton">Start Video</button>

document.getElementById('startButton').onclick = async () => {
  await room.connect(url, token);
  // Video will now play
};

// Or programmatically trigger play
track.attach(videoElement);
await videoElement.play().catch(err => {
  console.log('Autoplay prevented:', err);
  // Show play button
});
```

---

## Debugging Tips

### Enable Verbose Logging

```python
# In agent code
import logging
logging.basicConfig(level=logging.DEBUG)

# For LiveKit SDK
os.environ['LIVEKIT_LOG_LEVEL'] = 'debug'
```

```javascript
// In browser console
localStorage.debug = 'livekit:*';
// Reload page
```

### Monitor Network Traffic

```bash
# Check bandwidth usage
iftop -i eth0

# Monitor WebRTC stats in browser
# Open chrome://webrtc-internals
# Or firefox about:webrtc
```

### GPU Monitoring

```bash
# Real-time GPU stats
watch -n 1 nvidia-smi

# Log GPU usage
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
  --format=csv -l 1 > gpu_usage.log
```

### Memory Profiling

```python
# Add memory profiling
import tracemalloc
tracemalloc.start()

# ... run code ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

---

## Getting Help

If issues persist:

1. **Check Logs:**
   - Agent logs: Console output from `./livekit_server.sh`
   - LiveKit server logs: `docker logs livekit-server`
   - Browser console: F12 → Console tab

2. **Gather System Info:**
   ```bash
   # Python version
   python --version

   # GPU info
   nvidia-smi

   # LiveKit version
   uv run python -c "import livekit; print(livekit.__version__)"
   ```

3. **Create Minimal Reproduction:**
   - Isolate the issue
   - Test with minimal configuration
   - Document steps to reproduce

4. **Consult Documentation:**
   - LiveKit: https://docs.livekit.io
   - Gemini: https://cloud.google.com/vertex-ai/docs
   - Ditto: See project documentation
