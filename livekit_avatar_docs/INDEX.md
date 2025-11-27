# Documentation Index

Complete documentation for the LiveKit Avatar system - a real-time AI-powered talking head using Ditto TalkingHead and LiveKit.

---

## 📚 Documentation Overview

| Document | Description | Audience |
|----------|-------------|----------|
| [README](./README.md) | Overview and quick start | Everyone |
| [Architecture](./ARCHITECTURE.md) | System design and data flow | Developers |
| [Setup Guide](./SETUP_GUIDE.md) | Installation instructions | DevOps/Developers |
| [API Reference](./API_REFERENCE.md) | Complete API documentation | Developers |
| [Pipeline Deep Dive](./PIPELINE_DEEP_DIVE.md) | StreamSDK internals | Advanced Developers |
| [Troubleshooting](./TROUBLESHOOTING.md) | Problem solving guide | Everyone |

---

## 🚀 Quick Navigation

### Getting Started

1. **New to the project?** → Start with [README](./README.md)
2. **Setting up development?** → Follow [Setup Guide](./SETUP_GUIDE.md)
3. **Having issues?** → Check [Troubleshooting](./TROUBLESHOOTING.md)

### Understanding the System

1. **How does it work?** → Read [Architecture](./ARCHITECTURE.md)
2. **What are the components?** → See [API Reference](./API_REFERENCE.md)
3. **How does video generation work?** → Dive into [Pipeline Deep Dive](./PIPELINE_DEEP_DIVE.md)

### Development Tasks

| Task | Document | Section |
|------|----------|---------|
| Change avatar image | [Setup Guide](./SETUP_GUIDE.md) | Configuration |
| Customize video settings | [API Reference](./API_REFERENCE.md) | AvatarOptions |
| Add new emotions | [Pipeline Deep Dive](./PIPELINE_DEEP_DIVE.md) | Condition Handler |
| Debug latency issues | [Troubleshooting](./TROUBLESHOOTING.md) | Performance |
| Deploy to production | [Setup Guide](./SETUP_GUIDE.md) | Production Deployment |

---

## 📁 Code Structure Reference

### Core Components

```
livekit_avatar/
├── __init__.py                          # Package initialization
├── agent_worker.py                      # Conversation agent (Gemini)
├── avatar_worker.py                     # Video generation worker
└── ditto_video_generator_decoupled.py   # VideoGenerator implementation
```

### Supporting Files

```
livekit_client/
├── simple_client.html    # Browser test client
└── token_server.py       # Development token server

stream_pipeline_online.py # Core streaming SDK
livekit_server.sh         # Startup script
```

### Configuration Files

```
checkpoints/
├── ditto_cfg/
│   └── v0.4_hubert_cfg_trt_online.pkl   # Online streaming config
├── ditto_trt_Ampere_Plus/               # TensorRT engines
└── ...
```

---

## 🚀 Quick Start Commands

```bash
# Terminal 1: LiveKit Server (Docker)
sudo docker run --rm \
    -p 7880:7880 \
    -p 7881:7881 \
    -p 7882:7882/udp \
    -e LIVEKIT_KEYS="devkey: devsecret" \
    livekit/livekit-server:latest

# Terminal 2: Agent & Avatar Workers
./livekit_server.sh

# Terminal 3: Token Server (for web client)
uv run python livekit_client/token_server.py

# Then open: http://localhost:8000/simple_client.html
```

## 🔧 Configuration Quick Reference

### Environment Variables

```bash
# Required - LiveKit (set automatically by livekit_server.sh)
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=devsecret

# Required - Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json
VERTEX_PROJECT_ID=your-project
VERTEX_LOCATION=us-central1

# Required - Ditto (set automatically by livekit_server.sh)
DATA_ROOT=checkpoints/ditto_trt_Ampere_Plus
CFG_PKL=checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl
SOURCE_PATH=avatars/your_avatar.jpg

# Optional - Avatar Settings
AVATAR_WIDTH=1280
AVATAR_HEIGHT=720
AVATAR_FPS=25
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `video_fps` | 25 | Frame rate |
| `split_len` | 6480 | Audio chunk size |
| `chunksize` | (3,5,2) | Frame processing |
| `sampling_timesteps` | 50 | Diffusion steps |

---

## 📊 Performance Reference

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 8GB | RTX 4080 16GB |
| RAM | 16 GB | 32 GB |
| CPU | 8 cores | 12+ cores |

### Expected Performance

| Metric | Value |
|--------|-------|
| Video Resolution | 1280×720 |
| Frame Rate | 25 fps |
| End-to-End Latency | 300-500ms |
| GPU Memory | 4-6 GB |

---

## 🐛 Common Issues Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| No video | Check avatar worker logs |
| No audio | Verify DataStream setup |
| High latency | Reduce `sampling_timesteps` |
| CUDA OOM | Reduce resolution |
| Token failed | Run `uv run python livekit_client/token_server.py` |
| LiveKit not running | Start Docker: `sudo docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp -e LIVEKIT_KEYS="devkey: devsecret" livekit/livekit-server:latest` |

See [Troubleshooting](./TROUBLESHOOTING.md) for detailed solutions.

---

## 📝 Document Conventions

### Code Examples

```python
# Python code blocks show implementation examples
def example():
    pass
```

```bash
# Bash blocks show terminal commands
export VAR=value
```

### Diagrams

```
ASCII diagrams show architecture and flow
┌─────────┐
│  Box    │──▶ Arrow
└─────────┘
```

### Tables

| Column | Description |
|--------|-------------|
| Data | Information |

---

## 🔄 Document Updates

Last updated: 2025-01-15

### Version History

- **v1.0** - Initial documentation release
  - Complete architecture documentation
  - Setup guide for development and production
  - API reference for all components
  - Troubleshooting guide
  - Pipeline deep dive

---

## 📬 Feedback

Found an issue or have suggestions? Please:

1. Check existing documentation first
2. Create an issue with specific details
3. Include:
   - Document name
   - Section reference
   - Suggested improvement

---

*This documentation is part of the Ditto TalkingHead + LiveKit Avatar integration project.*

