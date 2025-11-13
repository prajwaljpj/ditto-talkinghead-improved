# LiveKit Conversational Avatar Agent

A real-time conversational AI avatar powered by:
- **LiveKit** - Real-time video/audio streaming
- **Google Gemini** - Large Language Model for conversation
- **Ditto** - Audio-driven avatar animation

## What This Does

This agent creates a conversational AI avatar that:
1. **Listens** to users speaking via WebRTC
2. **Understands** speech using Gemini's real-time API
3. **Responds** with natural language
4. **Animates** a photorealistic avatar with lip-sync

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (for Ditto avatar generation)
- LiveKit server (local or cloud)
- Google Cloud Platform account with Vertex AI enabled

### 1. Environment Setup

```bash
# Set required environment variables
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export VERTEX_PROJECT_ID="your-gcp-project-id"
export VERTEX_LOCATION="us-central1"

# Optional: Custom avatar settings
export SOURCE_PATH="avatars/your_avatar_image.jpg"
export AVATAR_WIDTH="1280"
export AVATAR_HEIGHT="720"
```

### 2. Start LiveKit Server

If running locally:
```bash
# LiveKit will use default dev credentials
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
  livekit/livekit-server --dev
```

### 3. Start the Avatar Agent

```bash
# Make the script executable
chmod +x livekit_server.sh

# Run the agent
./livekit_server.sh
```

### 4. Start the Token Server (for client access)

```bash
cd livekit_client
python token_server.py
```

### 5. Open the Client

Navigate to: `http://localhost:8000/simple_client.html`

## Project Structure

```
ditto-talkinghead/
├── livekit_avatar/              # Avatar agent implementation
│   ├── main_agent.py           # Main LiveKit agent entrypoint
│   ├── custom_avatar_worker.py # Avatar generation worker
│   └── __init__.py
├── livekit_client/              # Web client
│   ├── simple_client.html      # Browser-based client
│   └── token_server.py         # Token generation server
├── livekit_avatar_docs/         # Documentation (this folder)
├── stream_pipeline_online.py    # Ditto avatar SDK wrapper
├── livekit_server.sh            # Server startup script
└── checkpoints/                 # Ditto model weights
```

## Key Features

### Real-Time Streaming
- Sub-second latency for natural conversations
- Adaptive bitrate for varying network conditions
- WebRTC-based peer-to-peer communication

### Lip-Sync Animation
- Audio-driven facial animation using Ditto model
- 50 FPS video output for smooth motion
- HD video quality (1280x720 default)

### Conversation Management
- Voice Activity Detection (VAD)
- Turn-taking between user and agent
- Context-aware responses from Gemini

### State Management
- **Idle**: Subtle breathing/blinking animations
- **Listening**: Active listening pose when user speaks
- **Thinking**: Contemplative expression during processing
- **Speaking**: Synchronized lip movements with TTS

## Documentation

- [Architecture Overview](ARCHITECTURE.md) - System design and data flow
- [Setup Guide](SETUP_GUIDE.md) - Detailed installation instructions
- [API Reference](API_REFERENCE.md) - Code API documentation
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (RTX 3000+ recommended)
- 8GB+ GPU VRAM
- 16GB+ system RAM

### Software
- Python 3.10+
- CUDA 11.8+ / TensorRT 8.6+
- FFmpeg (for video encoding)
- Modern web browser with WebRTC support

## License

See project root for license information.

## Support

For issues and questions:
- Check [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review LiveKit documentation: https://docs.livekit.io
- Review Ditto documentation in the project root
