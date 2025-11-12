# 🚀 Quick Start: Ditto Avatar with LiveKit

## Prerequisites

- Docker installed (for LiveKit server)
- Python 3.10 with `uv` installed
- NVIDIA GPU with TensorRT models downloaded

## 3-Step Setup

### Step 1: Start LiveKit Server

```bash
docker run --rm \
  -p 7880:7880 \
  -p 7881:7881 \
  -p 7882:7882/udp \
  -e LIVEKIT_KEYS="devkey: devsecret" \
  livekit/livekit-server:latest
```

Keep this running in a terminal.

### Step 2: Start Ditto Agent

In a new terminal:

```bash
./start_livekit_agent.sh
```

You should see:
```
🚀 Agent starting...
🎭 Initializing Ditto SDK...
✅ Ditto SDK initialized
✅ Agent ready and waiting for participants
```

### Step 3: Open Web Client

In a third terminal:

```bash
cd webrtc/client/livekit
python -m http.server 8000
```

Open browser to: **http://localhost:8000**

1. Click "Connect to Avatar"
2. Allow microphone access
3. Start speaking!

## What You Should See

- **Browser**: Avatar video animating as you speak
- **Agent logs**: Frame generation stats
- **LiveKit server**: Connection status

## Troubleshooting

### "Connection failed"

Check LiveKit server is running:
```bash
curl http://localhost:7881/
# Should return: OK
```

### "Agent not starting"

Check environment variables:
```bash
echo $LIVEKIT_URL
echo $LIVEKIT_API_KEY
# Should show: ws://localhost:7880 and devkey
```

### "No video"

Check agent logs for errors:
- GPU available?
- Model files exist?
- Correct paths in environment variables?

### "Poor performance"

Profile the model:
```bash
uv run python profile_inference.py \
  --cfg_pkl checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl \
  --data_root checkpoints/ditto_trt_custom2/ \
  --source avatars/Hyperrealistic_Indian_Woman_Professional_Presentation_static_5_loop.jpg \
  --audio openai-fm-coral-professional.wav
```

Expected: ~50 FPS generation

## Next Steps

- See `LIVEKIT_SETUP.md` for production deployment
- See `webrtc/gemini_server.py` for Gemini integration
- Customize emotion with `DITTO_EMO` env var (0-7)

## Architecture

```
Browser (you) → LiveKit Server → Ditto Agent
  - Sends audio      ↓              ↓
  - Receives video   Handles WebRTC Generates video
                     (no Python!)   (Ditto model)
```

**Key insight**: LiveKit handles ALL the WebRTC complexity (NAT, codecs, threading), letting Python focus on ML!
