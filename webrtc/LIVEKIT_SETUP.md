# LiveKit + Ditto Avatar Setup Guide

## Why LiveKit?

LiveKit solves all the Python WebRTC issues:
- ✅ **No threading problems** - LiveKit handles WebRTC in native code
- ✅ **Production-ready** - Used by companies at scale
- ✅ **Simple Python API** - Just publish/subscribe tracks
- ✅ **Auto-scaling** - Handles 1000s of concurrent users
- ✅ **Cloud or self-hosted** - Your choice

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   LiveKit Cloud/Server                    │
│     (Handles ALL WebRTC: NAT, ICE, codecs, etc.)        │
└──────────┬────────────────────────────┬──────────────────┘
           │                            │
     (WebRTC)                      (WebRTC)
           │                            │
┌──────────▼───────────┐    ┌───────────▼────────────────┐
│   Browser Client     │    │  Ditto Agent (Python)      │
│   - Sends audio      │    │                            │
│   - Receives video   │    │  LiveKit SDK               │
│   - Simple HTML/JS   │    │    ↓                       │
└──────────────────────┘    │  Ditto Model               │
                            │    ↓                       │
                            │  Video frames → LiveKit    │
                            └────────────────────────────┘
```

## Quick Start (Development)

### 1. Start LiveKit Server (Docker)

```bash
# Run LiveKit server locally
docker run --rm \
  -p 7880:7880 \
  -p 7881:7881 \
  -p 7882:7882/udp \
  -e LIVEKIT_KEYS="devkey: devsecret" \
  livekit/livekit-server:latest
```

**What this does:**
- Port 7880: WebSocket/HTTP (for signaling)
- Port 7881: HTTP (for health checks)
- Port 7882/UDP: DTLS/SRTP (for media)
- Dev credentials: `devkey` / `devsecret` (CHANGE IN PRODUCTION!)

### 2. Install Python Dependencies

```bash
# Already added to pyproject.toml
uv sync
```

### 3. Start Ditto Agent

**Option A: Use the startup script (easiest):**

```bash
./start_livekit_agent.sh
```

**Option B: Manual configuration:**

```bash
# Set LiveKit credentials
export LIVEKIT_URL=ws://localhost:7880
export LIVEKIT_API_KEY=devkey
export LIVEKIT_API_SECRET=devsecret

# Set Ditto configuration
export DITTO_CFG_PKL=checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl
export DITTO_DATA_ROOT=checkpoints/ditto_trt_custom2/
export DITTO_SOURCE=avatars/Hyperrealistic_Indian_Woman_Professional_Presentation_static_5_loop.jpg
export DITTO_MAX_SIZE=1920
export DITTO_EMO=4

# Run agent (dev mode)
uv run python webrtc/livekit_ditto_agent.py dev
```

**Note**: The `dev` command tells LiveKit to run in development mode (connects to local server).

### 4. Open Web Client

```bash
# Serve the client
cd webrtc/client/livekit
python -m http.server 8000

# Open browser to:
# http://localhost:8000
```

### 5. Connect!

1. Click "Connect to Avatar"
2. Allow microphone access
3. Start speaking
4. Watch the avatar respond!

## How It Works

### Audio Flow

```
Browser Microphone
  → LiveKit (automatic WebRTC encoding)
  → Agent subscribes to audio track
  → Audio resampled 48kHz → 16kHz
  → Ditto model processes audio
  → Generates video frames
  → Agent publishes video track
  → LiveKit (automatic WebRTC encoding)
  → Browser displays video
```

### No More Thread Issues!

**Old approach (signaling_server_v3.py):**
```python
# ❌ Complex: Manual thread coordination
asyncio.Queue vs queue.Queue
Thread-safe callbacks
Manual PTS calculation
Custom audio-video sync
```

**New approach (LiveKit):**
```python
# ✅ Simple: LiveKit handles everything
video_source = rtc.VideoSource(1280, 720)
track = rtc.LocalVideoTrack.create_video_track("avatar", video_source)
await room.local_participant.publish_track(track)

# In Ditto callback (called from worker thread):
video_source.capture_frame(frame)  # Thread-safe!
```

## Configuration

### Agent Options

```bash
python livekit_ditto_agent.py \
  --cfg_pkl <path/to/config.pkl> \
  --data_root <path/to/model/data> \
  --source <path/to/avatar.jpg> \
  --max_size 1920 \
  --emo 4
```

**Parameters:**
- `cfg_pkl`: Ditto model configuration
- `data_root`: Ditto model weights
- `source`: Avatar source image
- `max_size`: Max image dimension (lower = faster)
- `emo`: Emotion (0=happy, 1=angry, 2=sad, 3=fear, 4=neutral, 5=surprised)

### LiveKit Configuration

Set via environment variables:

```bash
# Required
export LIVEKIT_URL=ws://your-livekit-server:7880
export LIVEKIT_API_KEY=your-api-key
export LIVEKIT_API_SECRET=your-api-secret

# Optional
export LIVEKIT_LOG_LEVEL=info  # debug, info, warn, error
```

## Production Deployment

### Option 1: LiveKit Cloud (Easiest)

1. Sign up at https://livekit.io
2. Get your credentials from dashboard
3. Update environment variables:

```bash
export LIVEKIT_URL=wss://your-project.livekit.cloud
export LIVEKIT_API_KEY=<from-dashboard>
export LIVEKIT_API_SECRET=<from-dashboard>
```

4. Deploy agent:

```bash
# Option A: Run on server with GPU
screen -S ditto-agent
uv run python webrtc/livekit_ditto_agent.py

# Option B: Docker (create Dockerfile)
# See below
```

**Costs:**
- Free tier: 10,000 participant minutes/month
- Pay-as-you-go: $0.004/min per participant
- Example: 100 users × 10 min = $4

### Option 2: Self-Hosted LiveKit

```bash
# Production LiveKit server
docker run -d \
  --name livekit \
  -p 7880:7880 \
  -p 7881:7881 \
  -p 7882:7882/udp \
  -v $(pwd)/livekit-config.yaml:/config.yaml \
  livekit/livekit-server:latest \
  --config /config.yaml
```

**livekit-config.yaml:**
```yaml
port: 7880
rtc:
  port_range_start: 50000
  port_range_end: 60000
  use_external_ip: true

keys:
  your-api-key: your-api-secret

# TURN server (for NAT traversal)
rtc:
  turn_servers:
    - host: your-turn-server.com
      port: 3478
      protocol: udp
      username: user
      credential: pass
```

### Docker Deployment for Agent

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

WORKDIR /app

# Copy project
COPY . .

# Install dependencies
RUN uv sync

# Run agent
CMD ["uv", "run", "python", "webrtc/livekit_ditto_agent.py"]
```

**Build and run:**
```bash
docker build -t ditto-agent .

docker run -d \
  --name ditto-agent \
  --gpus all \
  -e LIVEKIT_URL=$LIVEKIT_URL \
  -e LIVEKIT_API_KEY=$LIVEKIT_API_KEY \
  -e LIVEKIT_API_SECRET=$LIVEKIT_API_SECRET \
  ditto-agent
```

## Scaling

### Multiple Agents

LiveKit automatically load-balances across multiple agents:

```bash
# Start 3 agents on different machines
# Agent 1:
LIVEKIT_URL=... python livekit_ditto_agent.py

# Agent 2:
LIVEKIT_URL=... python livekit_ditto_agent.py

# Agent 3:
LIVEKIT_URL=... python livekit_ditto_agent.py

# LiveKit will distribute rooms across agents
```

### GPU Requirements

- **Development**: 1 GPU (RTX 3060+)
- **Production (10 users)**: 2-4 GPUs
- **Production (100 users)**: 10-20 GPUs

Each Ditto instance uses ~4GB VRAM and can handle 1-2 concurrent users.

## Troubleshooting

### Agent not connecting

```bash
# Check LiveKit server is running
curl http://localhost:7881/

# Check credentials
echo $LIVEKIT_API_KEY
echo $LIVEKIT_API_SECRET

# Check logs
uv run python livekit_ditto_agent.py --log-level debug
```

### Browser can't connect

```bash
# Check browser console for errors
# Common issues:
# 1. HTTPS required for getUserMedia (use localhost or HTTPS)
# 2. Firewall blocking UDP port 7882
# 3. TURN server needed for NAT traversal
```

### Poor video quality

```bash
# Reduce resolution
python livekit_ditto_agent.py --max_size 1280

# Adjust LiveKit bitrate
# In browser console:
room.options.videoCaptureDefaults = {
  resolution: { width: 1280, height: 720 }
}
```

### High latency

```bash
# Profile model performance
uv run python profile_inference.py

# Expected:
# - Model latency: ~200ms
# - Network latency: ~50-100ms
# - Total: ~300-400ms

# If higher:
# 1. Check GPU utilization
# 2. Reduce max_size
# 3. Check network bandwidth
```

## Monitoring

### LiveKit Dashboard

Access at: `http://localhost:7880` (self-hosted) or cloud dashboard

Shows:
- Active rooms
- Participants
- Track stats
- Bandwidth usage

### Agent Logs

```bash
# View logs
tail -f agent.log

# Key metrics to watch:
# - Frames generated per second
# - Audio chunks processed
# - Queue sizes
```

### Python Profiling

```bash
# Profile Ditto performance
uv run python profile_inference.py \
  --cfg_pkl <path> \
  --data_root <path> \
  --source <avatar> \
  --audio <test-audio.wav>

# Expected: ~50 FPS generation
```

## Comparison: LiveKit vs Custom WebRTC

| Feature | Custom (v3) | LiveKit |
|---------|-------------|---------|
| **Setup complexity** | ❌ High | ✅ Low |
| **Thread safety** | ⚠️ Manual | ✅ Automatic |
| **NAT traversal** | ⚠️ Manual STUN/TURN | ✅ Built-in |
| **Scalability** | ❌ Single server | ✅ Auto-scale |
| **Production ready** | ⚠️ Needs work | ✅ Yes |
| **Latency** | ⚠️ Variable | ✅ Optimized |
| **Multi-platform** | ⚠️ Browser only | ✅ iOS/Android/Desktop |
| **Recording** | ❌ Manual | ✅ Built-in |
| **Analytics** | ❌ Manual | ✅ Dashboard |
| **Cost** | Free (DIY) | Free tier + pay-as-you-go |

## Next Steps

1. ✅ **Test locally** - Follow Quick Start above
2. 🔄 **Integrate Gemini** - Add Gemini Live API for conversations
3. 🚀 **Deploy to production** - Use LiveKit Cloud or self-host
4. 📊 **Monitor performance** - Use LiveKit dashboard
5. 🎨 **Customize UI** - Build your own client

## Resources

- **LiveKit Docs**: https://docs.livekit.io
- **Python SDK**: https://docs.livekit.io/agents/quickstart
- **Examples**: https://github.com/livekit/agents
- **Community**: https://livekit.io/discord

## Support

For issues:
1. Check logs (agent + LiveKit server)
2. Test with LiveKit example clients
3. Profile Ditto performance separately
4. Ask in LiveKit Discord or file issue

**Key insight**: LiveKit separates WebRTC (hard) from ML (your strength), letting Python do what it's good at!
