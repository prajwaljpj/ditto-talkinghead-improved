# LiveKit Avatar System Documentation

> Real-time AI-powered talking head avatar using Ditto TalkingHead and LiveKit

## Overview

This documentation covers the complete LiveKit Avatar system - a real-time conversational AI avatar that combines:

- **Google Gemini** for natural language understanding and speech synthesis
- **Ditto TalkingHead** for photorealistic lip-synced video generation
- **LiveKit** for real-time WebRTC communication

The system enables users to have natural conversations with an AI avatar that responds with synchronized audio and video in real-time.

## Quick Links

| Document | Description |
|----------|-------------|
| [Architecture](./ARCHITECTURE.md) | System design, data flow, and component interaction |
| [Setup Guide](./SETUP_GUIDE.md) | Step-by-step installation and configuration |
| [API Reference](./API_REFERENCE.md) | Detailed API documentation for all components |
| [Troubleshooting](./TROUBLESHOOTING.md) | Common issues and solutions |
| [Pipeline Deep Dive](./PIPELINE_DEEP_DIVE.md) | Internals of the StreamSDK pipeline |

## System Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LiveKit Server                                  │
│                            (WebRTC Signaling)                               │
└─────────────────────────────────────────────────────────────────────────────┘
        │                           │                           │
        │ WebRTC                    │ DataStream               │ WebRTC
        │                           │ (audio)                   │
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│    Client     │           │  Agent Worker │           │ Avatar Worker │
│  (Browser)    │◀──────────│   (Gemini)    │──────────▶│   (Ditto)     │
│               │   audio   │               │   TTS     │               │
│ • Captures    │   +video  │ • STT/LLM/TTS │   audio   │ • Video Gen   │
│   user audio  │           │ • Conversation│           │ • Lip Sync    │
│ • Displays    │           │   management  │           │ • Publishes   │
│   avatar      │           │               │           │   audio+video │
└───────────────┘           └───────────────┘           └───────────────┘
```

## Key Features

### 🎭 Photorealistic Avatar
- High-quality lip-synced video generation using Ditto TalkingHead
- Support for custom avatar images (any portrait photo)
- Configurable resolution (up to 1920x1080)
- Real-time 25fps video output

### 🗣️ Natural Conversation
- Powered by Google Gemini's native audio model
- Real-time speech-to-text and text-to-speech
- Natural conversation flow with interruption handling
- Configurable voice and personality

### ⚡ Low Latency Architecture
- Decoupled two-worker design for optimal performance
- Streaming audio processing with chunked inference
- Parallel pipeline stages for maximum throughput
- WebRTC for minimal network latency

### 🔧 Flexible Deployment
- Local development with LiveKit's dev server
- Production-ready with LiveKit Cloud
- GPU-accelerated inference (TensorRT optimized)
- Docker-friendly architecture

## Prerequisites

Before setting up the system, ensure you have:

- **Hardware**: NVIDIA GPU with 8GB+ VRAM (RTX 3070+ recommended)
- **Software**:
  - Python 3.10+
  - CUDA 12.x
  - TensorRT 10.x
  - Node.js 18+ (for client development)
- **Services**:
  - Google Cloud account with Vertex AI enabled
  - LiveKit server (local or cloud)

## Quick Start

### 1. Install Dependencies

```bash
# Clone and setup
cd ditto-talkinghead
./setup_uv.sh

# Activate environment
source .venv/bin/activate
```

### 2. Configure Credentials

```bash
# Google Cloud (for Gemini)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export VERTEX_PROJECT_ID="your-project-id"
export VERTEX_LOCATION="us-central1"

# LiveKit credentials are set automatically by livekit_server.sh
```

### 3. Start the System (3 Terminals)

```bash
# Terminal 1: Start LiveKit server (Docker)
sudo docker run --rm \
    -p 7880:7880 \
    -p 7881:7881 \
    -p 7882:7882/udp \
    -e LIVEKIT_KEYS="devkey: devsecret" \
    livekit/livekit-server:latest

# Terminal 2: Start the agent (auto-launches avatar worker)
./livekit_server.sh

# Terminal 3: Start token server (for client)
uv run python livekit_client/token_server.py
```

### 4. Connect Client

Open `http://localhost:8000/simple_client.html` in your browser and click "Connect".

## Directory Structure

```
livekit_avatar/
├── __init__.py                      # Package init
├── agent_worker.py                  # Main agent (Gemini conversation)
├── avatar_worker.py                 # Avatar subprocess (video generation)
└── ditto_video_generator_decoupled.py  # LiveKit VideoGenerator implementation

livekit_client/
├── simple_client.html               # Browser test client
└── token_server.py                  # Development token server

stream_pipeline_online.py            # Core Ditto streaming SDK
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Video Resolution | 1280x720 (configurable) |
| Frame Rate | 25 fps |
| Audio Sample Rate | 16 kHz |
| Chunk Processing Time | ~80-100ms |
| End-to-End Latency | ~200-400ms |
| GPU Memory Usage | ~4-6 GB |

## Next Steps

1. Read the [Architecture](./ARCHITECTURE.md) document to understand the system design
2. Follow the [Setup Guide](./SETUP_GUIDE.md) for detailed installation instructions
3. Review the [API Reference](./API_REFERENCE.md) for customization options
4. Check [Troubleshooting](./TROUBLESHOOTING.md) if you encounter issues

## License

This project uses the Ditto TalkingHead model. Please review the checkpoint license at `checkpoints/LICENSE` before deployment.

