# Ditto Talking Head - WebRTC Streaming

Real-time talking head video generation with WebRTC streaming support.

## Overview

This WebRTC implementation allows you to stream real-time talking head video generation directly to web browsers and mobile clients. It uses:

- **WebRTC** for low-latency audio/video streaming
- **WebSockets** for signaling (SDP/ICE exchange)
- **aiortc** for Python WebRTC stack
- **Pipecat** (optional) for pipeline orchestration
- **Ditto StreamSDK** for talking head generation

## Architecture

```
┌─────────────┐         ┌──────────────────┐         ┌─────────────┐
│ Web Client  │◄────────┤ WebSocket Signal ├────────►│   Server    │
│             │         │      Server      │         │             │
│ - Mic Audio │─────────┤                  ├─────────┤ StreamSDK   │
│ - Video Out │◄────────┤   WebRTC Data    ├────────►│ - Audio2Motion
│             │         │   Channel        │         │ - Rendering │
└─────────────┘         └──────────────────┘         └─────────────┘
```

### Pipeline Flow

1. **Client** captures audio from microphone
2. **WebRTC** streams audio to server
3. **StreamSDK** processes audio → generates motion → renders frames
4. **WebRTC** streams video frames back to client
5. **Client** displays avatar video in real-time

## Directory Structure

```
webrtc/
├── README.md                    # This file
├── webrtc_server.py             # Main server (Pipecat-based)
├── signaling_server.py          # WebSocket signaling server (aiortc-based)
├── processors/                  # Custom Pipecat processors
│   ├── ditto_processor.py       # Ditto avatar processor
│   ├── idle_animator.py         # Idle animation generator
│   └── h264_encoder.py          # Video encoder
└── client/
    └── web/
        ├── index.html           # Web client UI
        └── app.js               # WebRTC client logic
```

## Installation

Dependencies are already added to `pyproject.toml`. Install with:

```bash
uv sync
```

Or manually:

```bash
pip install pipecat-ai aiortc av websockets
```

## Quick Start

### Option 1: Using Signaling Server (Recommended)

The signaling server provides a simple WebSocket-based WebRTC setup without requiring Daily.co.

**1. Start the signaling server:**

```bash
uv run python webrtc/signaling_server.py \
    --cfg_pkl /path/to/config.pkl \
    --data_root /path/to/model/data \
    --host 0.0.0.0 \
    --port 8080
```

**2. Open web client:**

Open `webrtc/client/web/index.html` in a browser (or serve with a local HTTP server):

```bash
# Using Python's built-in HTTP server
cd webrtc/client/web
python3 -m http.server 8000
```

Then open: http://localhost:8000

**3. Connect:**

- Enter server URL: `ws://localhost:8080`
- Enter avatar source path (relative to server): `assets/examples/portraits/1.jpg`
- Click "Connect to Avatar"
- Allow microphone access
- Click "Start Speaking"

### Option 2: Using Pipecat Server

If you have Daily.co API credentials:

```bash
uv run python webrtc/webrtc_server.py \
    --cfg_pkl /path/to/config.pkl \
    --data_root /path/to/model/data \
    --source /path/to/avatar.jpg \
    --transport daily \
    --room_url https://your-domain.daily.co/room-name \
    --token your_daily_token
```

## Configuration

### Server Options

#### Signaling Server (`signaling_server.py`)

```bash
--cfg_pkl       # Path to Ditto config pickle (required)
--data_root     # Path to model data root (required)
--host          # Server host (default: 0.0.0.0)
--port          # WebSocket port (default: 8080)
--max_size      # Max image dimension (default: 1920)
--emo           # Emotion 0-7 (default: 4=neutral)
```

#### Pipecat Server (`webrtc_server.py`)

```bash
--cfg_pkl       # Path to Ditto config pickle (required)
--data_root     # Path to model data root (required)
--source        # Path to avatar source image/video (required)
--transport     # Transport type: none, daily (default: none)
--room_url      # Daily.co room URL (if transport=daily)
--token         # Daily.co token (if transport=daily)
--max_size      # Max image dimension (default: 1920)
--crop_scale    # Face crop scale (default: 2.3)
--emo           # Emotion 0-7 (default: 4=neutral)
```

### Ditto Pipeline Options

The WebRTC server supports all standard Ditto configuration options:

- `max_size`: Maximum image dimension (default: 1920)
- `crop_scale`: Face crop scale factor (default: 2.3)
- `crop_vx_ratio`: Horizontal crop offset (default: 0)
- `crop_vy_ratio`: Vertical crop offset (default: -0.125)
- `emo`: Emotion (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Neutral, 5=Sad, 6=Surprise, 7=Contempt)
- `online_mode`: Enable streaming mode (automatically set to True)

## Implementation Details

### Video Resolution Pipeline

**Important**: The output video resolution matches your **source image/video dimensions**, not a fixed 512x512!

Resolution flow:
1. **Input source**: Your image/video (e.g., 1920x1080, 1280x720, etc.)
2. **Face crop**: Extracted face region scaled to 256x256 for processing
3. **Internal rendering**: Face rendered at 512x512 resolution
4. **PutBack compositing**: Rendered face composited back onto original frame
5. **Final output**: Same dimensions as source (e.g., 1920x1080)

The `max_size` parameter (default: 1920) limits the maximum dimension. If your source is 4K (3840x2160), it will be downscaled to 1920x1080 for processing.

**Example**:
- Source: 1920x1080 portrait photo → Output: 1920x1080 video
- Source: 512x512 avatar → Output: 512x512 video
- Source: 1280x720 image → Output: 1280x720 video

### Modified Components

#### 1. `stream_pipeline_online.py`

Modified `StreamSDK.setup()` to support frame callbacks:

```python
sdk.setup(
    source_path,
    output_path=None,
    frame_callback=on_frame_ready,  # New parameter
    online_mode=True
)

def on_frame_ready(frame_rgb, frame_idx, timestamp):
    # Called for each generated frame
    # frame_rgb: numpy array (H, W, 3) uint8
    pass
```

When `frame_callback` is provided:
- No video file is written
- Frames are pushed to callback in real-time
- Threading pipeline continues working as normal

### Custom Processors

#### 1. `DittoAvatarProcessor`

Wraps the Ditto `StreamSDK` as a Pipecat processor:

- **Input**: `AudioRawFrame` (16kHz, mono)
- **Output**: `OutputImageRawFrame` (RGB)
- **Initialization**: Runs avatar registration on first frame
- **Processing**: Feeds audio chunks to SDK, outputs video frames

#### 2. `DittoIdleAnimationProcessor`

Generates idle animations when no speech is detected:

- **VAD**: Simple volume-based voice activity detection
- **Idle trigger**: After 0.5s of silence
- **Animation**: Breathing effect (subtle brightness variation)
- **TODO**: Proper motion generation through Ditto pipeline

#### 3. `H264EncoderProcessor`

Encodes RGB frames to H.264 (optional, WebRTC handles this):

- **Codec**: libx264
- **Preset**: ultrafast (for low latency)
- **Profile**: baseline (WebRTC compatible)

### WebRTC Setup

#### Signaling Protocol

**Client → Server:**

```json
{
    "type": "connect",
    "source": "path/to/avatar.jpg"
}

{
    "type": "offer",
    "sdp": "v=0\r\no=- ..."
}

{
    "type": "ice-candidate",
    "candidate": {
        "candidate": "...",
        "sdpMid": "0",
        "sdpMLineIndex": 0
    }
}
```

**Server → Client:**

```json
{
    "type": "ready",
    "message": "Server ready to receive offer"
}

{
    "type": "answer",
    "sdp": "v=0\r\no=- ..."
}

{
    "type": "ice-candidate",
    "candidate": {...}
}

{
    "type": "error",
    "message": "Error description"
}
```

#### STUN/TURN Servers

The client is configured with Google's public STUN servers:

```javascript
{
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' }
    ]
}
```

For production, add TURN servers for NAT traversal:

```javascript
{
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        {
            urls: 'turn:your-turn-server.com:3478',
            username: 'user',
            credential: 'pass'
        }
    ]
}
```

## Latency Analysis

### Expected Latency (Unoptimized)

| Component                  | Latency   |
|----------------------------|-----------|
| Network (client → server)  | 30-50ms   |
| Audio buffering            | 40-60ms   |
| HuBERT processing          | 40ms      |
| LMDM (amortized)          | ~7ms      |
| MotionStitch              | 5ms       |
| WarpF3D                   | 20ms      |
| DecodeF3D                 | 25ms      |
| PutBack (compositing)     | 5-10ms    |
| Network (server → client) | 30-50ms   |
| Client decode/render      | 20-30ms   |
| **Total**                 | **~280ms** |

Note: PutBack latency increases with higher output resolution (e.g., 1080p vs 720p).

### Cold Start Latency

First frame generation:
- Audio buffering until batch: 200-400ms
- LMDM full batch: ~500ms
- Rendering: ~55ms
- **Total first frame: ~900-1000ms**

This is acceptable as users expect a slight delay before the avatar starts responding.

## Troubleshooting

### Connection Issues

**Problem**: WebSocket connection fails

- Check server is running: `netstat -an | grep 8080`
- Check firewall allows port 8080
- Use correct URL: `ws://` (not `wss://` unless using SSL)

**Problem**: WebRTC connection stuck in "checking" state

- Check STUN server is reachable
- May need TURN server for NAT traversal
- Check browser console for errors

### Audio Issues

**Problem**: No audio captured

- Allow microphone permission in browser
- Check correct microphone selected
- Verify audio level indicator shows activity

**Problem**: Avatar doesn't respond to audio

- Check WebRTC connection state is "connected"
- Verify audio is being sent (check stats)
- Look at server logs for errors

### Video Issues

**Problem**: No video stream

- Check video track is added to peer connection
- Verify server is generating frames (check logs)
- Check browser console for errors

**Problem**: Low frame rate

- Check server GPU usage (should be ~100%)
- Verify network bandwidth is sufficient
- Check stats for dropped frames

### Server Issues

**Problem**: High GPU memory usage

- Each connection loads full Ditto pipeline
- Limit concurrent connections
- Consider model optimization (quantization, pruning)

**Problem**: High latency

- Check LMDM inference time (should be ~500ms per batch)
- Verify TensorRT engines are being used
- Check network latency between client/server

## Next Steps

### Immediate Improvements

1. **Proper Idle Animation**: Generate motion through Ditto pipeline instead of simple brightness variation
2. **Audio Resampling**: Properly resample audio to 16kHz if input is different
3. **Error Handling**: More robust error handling and recovery
4. **Connection Management**: Support multiple concurrent sessions efficiently

### Latency Optimization

See main README for detailed optimization strategies:

- LMDM optimization (FP16, reduced sampling steps)
- Hardware H.264 encoding (NVENC)
- Pipeline parallelization
- Reduced audio buffering

### Production Deployment

1. **HTTPS/WSS**: Use secure WebSocket (wss://) and HTTPS
2. **TURN Server**: Setup coturn or use commercial TURN service
3. **Load Balancing**: Distribute sessions across multiple GPU servers
4. **Monitoring**: Add metrics, logging, and alerting
5. **Authentication**: Add user authentication and session management

## Examples

### Example 1: Single User Avatar

```bash
# Start server
uv run python webrtc/signaling_server.py \
    --cfg_pkl outputs/final_ckpt/cfg.pkl \
    --data_root outputs/final_ckpt \
    --port 8080

# Open client
open http://localhost:8000 (after starting HTTP server)
```

### Example 2: Multiple Avatar Sources

Clients can specify different avatar sources per connection:

```javascript
// In web client
ws.send(JSON.stringify({
    type: 'connect',
    source: 'avatars/person1.jpg'  // Different per user
}));
```

### Example 3: Custom Emotion

```bash
# Start with happy emotion
uv run python webrtc/signaling_server.py \
    --cfg_pkl outputs/final_ckpt/cfg.pkl \
    --data_root outputs/final_ckpt \
    --emo 3  # 3 = Happy
```

## FAQ

**Q: Can I use this without Pipecat?**

A: Yes! The `signaling_server.py` uses pure aiortc without Pipecat dependency.

**Q: What's the difference between the two servers?**

A:
- `signaling_server.py`: Simple WebSocket + aiortc, self-contained
- `webrtc_server.py`: Uses Pipecat framework, supports Daily.co

**Q: Can this run on CPU?**

A: Technically yes, but expect 5-10x slower inference (~5 seconds latency). GPU strongly recommended.

**Q: How many concurrent users can one server handle?**

A: Depends on GPU. With RTX 4090:
- 1 user: ~280ms latency
- 2-3 users: acceptable with optimizations
- 4+: need multiple GPUs or model optimization

**Q: Does this support mobile clients?**

A: Yes! WebRTC works in mobile browsers (Chrome, Safari). Can also use native SDKs (Swift, Kotlin).

## License

Same as main Ditto project.

## Support

For issues or questions:
1. Check the main Ditto README
2. Review server logs for errors
3. Check browser console for client-side errors
4. Open an issue on GitHub

---

**Built with**:
- [Ditto](https://github.com/...) - Talking head synthesis
- [Pipecat](https://github.com/pipecat-ai/pipecat) - Real-time AI pipeline framework
- [aiortc](https://github.com/aiortc/aiortc) - Python WebRTC
- [WebRTC](https://webrtc.org/) - Real-time communication
