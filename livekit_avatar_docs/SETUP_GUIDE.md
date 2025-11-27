# Setup Guide

This guide walks you through setting up the complete LiveKit Avatar system from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Model Checkpoints](#model-checkpoints)
4. [LiveKit Configuration](#livekit-configuration)
5. [Google Cloud Setup](#google-cloud-setup)
6. [Running the System](#running-the-system)
7. [Testing the Setup](#testing-the-setup)
8. [Production Deployment](#production-deployment)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX 3060 (8GB) | NVIDIA RTX 4080+ (16GB) |
| VRAM | 6 GB | 12+ GB |
| RAM | 16 GB | 32 GB |
| CPU | 8 cores | 12+ cores |
| Storage | 20 GB | 50 GB (for models) |

### Software Requirements

```bash
# Check CUDA version (need 12.x)
nvcc --version

# Check TensorRT (need 10.x)
python -c "import tensorrt; print(tensorrt.__version__)"

# Check Python version (need 3.10+)
python --version
```

### Required Software

- **Operating System**: Ubuntu 20.04/22.04 or Windows 11 with WSL2
- **Python**: 3.10 or 3.11
- **CUDA**: 12.0 or higher
- **TensorRT**: 10.x (installed via the setup scripts)
- **Node.js**: 18+ (optional, for client development)

---

## Environment Setup

### Option 1: Using UV (Recommended)

```bash
# Navigate to project
cd ditto-talkinghead

# Run setup script
./setup_uv.sh

# Activate environment
source .venv/bin/activate
```

### Option 2: Using Conda

```bash
# Create environment
conda env create -f environment.yaml

# Activate
conda activate ditto

# Install additional dependencies
pip install livekit livekit-agents livekit-plugins-google
```

### Verify Installation

```bash
# Run verification script
python verify_installation.py

# Expected output:
# ✅ CUDA available
# ✅ TensorRT available
# ✅ Ditto components loaded
# ✅ LiveKit SDK available
```

---

## Model Checkpoints

### Download Checkpoints

The Ditto model requires several checkpoint files. Download and organize them as follows:

```bash
checkpoints/
├── ditto_cfg/
│   ├── v0.4_hubert_cfg_trt.pkl         # TensorRT config
│   └── v0.4_hubert_cfg_trt_online.pkl  # Online streaming config
├── ditto_trt_Ampere_Plus/              # TensorRT engines (RTX 30xx+)
│   ├── appearance_extractor.trt
│   ├── motion_extractor.trt
│   ├── warping_spade.trt
│   ├── decoder.trt
│   ├── lmdm.trt
│   └── hubert.trt
├── ditto_trt_custom/                   # Custom TRT engines (optional)
├── LICENSE
└── README.md
```

### Build Custom TensorRT Engines (If Needed)

If you're using a different GPU architecture:

```bash
# Convert ONNX to TensorRT
python scripts/cvt_onnx_to_trt.py \
    --onnx_dir checkpoints/ditto_onnx \
    --trt_dir checkpoints/ditto_trt_custom
```

### Configure Checkpoint Paths

Set the appropriate paths in your environment:

```bash
# For RTX 30xx/40xx (Ampere+)
export DATA_ROOT="checkpoints/ditto_trt_Ampere_Plus"
export CFG_PKL="checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"

# For custom builds
export DATA_ROOT="checkpoints/ditto_trt_custom"
export CFG_PKL="checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
```

---

## LiveKit Configuration

### Option 1: Local Development Server (Docker - Recommended)

The easiest way to get started is with LiveKit's Docker image:

```bash
# Start LiveKit server using Docker
sudo docker run --rm \
    -p 7880:7880 \
    -p 7881:7881 \
    -p 7882:7882/udp \
    -e LIVEKIT_KEYS="devkey: devsecret" \
    livekit/livekit-server:latest

# Default credentials:
# URL: ws://localhost:7880
# API Key: devkey
# API Secret: devsecret
```

**Ports:**
- `7880`: WebSocket signaling (HTTP/WS)
- `7881`: RTC (TCP)
- `7882`: RTC (UDP)

### Option 1b: Local Development Server (Native Binary)

Alternatively, use LiveKit's native binary:

```bash
# Install LiveKit CLI
curl -sSL https://get.livekit.io/cli | bash

# Start development server
livekit-server --dev
```

### Option 2: LiveKit Cloud

For production, use [LiveKit Cloud](https://cloud.livekit.io):

1. Create an account at https://cloud.livekit.io
2. Create a new project
3. Copy your credentials:

```bash
export LIVEKIT_URL="wss://your-project.livekit.cloud"
export LIVEKIT_API_KEY="your-api-key"
export LIVEKIT_API_SECRET="your-api-secret"
```

### Option 3: Self-Hosted Server

For production self-hosting, see [LiveKit Deployment Guide](https://docs.livekit.io/deploying/).

---

## Google Cloud Setup

### Create Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing
3. Enable the **Vertex AI API**
4. Create a service account:
   - Go to IAM & Admin → Service Accounts
   - Create service account
   - Grant role: `Vertex AI User`
   - Create JSON key and download

### Configure Credentials

```bash
# Set credentials file path
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Set project configuration
export VERTEX_PROJECT_ID="your-gcp-project-id"
export VERTEX_LOCATION="us-central1"  # or your preferred region

# Set model (default is latest Gemini Live)
export GEMINI_MODEL="gemini-live-2.5-flash-preview-native-audio-09-2025"
```

### Verify Google Cloud Setup

```bash
# Test authentication
gcloud auth application-default print-access-token

# Test Vertex AI access
python -c "
from google.cloud import aiplatform
aiplatform.init(project='$VERTEX_PROJECT_ID', location='$VERTEX_LOCATION')
print('✅ Vertex AI connection successful')
"
```

---

## Running the System

### Quick Start (3 Terminals)

You need three terminal windows to run the complete system:

#### Terminal 1: LiveKit Server (Docker)

```bash
sudo docker run --rm \
    -p 7880:7880 \
    -p 7881:7881 \
    -p 7882:7882/udp \
    -e LIVEKIT_KEYS="devkey: devsecret" \
    livekit/livekit-server:latest
```

#### Terminal 2: Agent & Avatar Workers

```bash
./livekit_server.sh
```

This script:
1. Validates Google Cloud credentials
2. Sets all environment variables (LiveKit, Vertex AI, Ditto)
3. Checks and installs missing dependencies
4. Starts the agent worker (which auto-launches avatar worker as subprocess)

#### Terminal 3: Token Server (for web client)

```bash
uv run python livekit_client/token_server.py

# Output:
# 🚀 Token server running on http://localhost:8000
# 🎫 Token endpoint: http://localhost:8000/token
# 🌐 Client page: http://localhost:8000/simple_client.html
```

### Manual Start (For Development)

If you want more control over each component:

#### Terminal 1: LiveKit Server

```bash
# Option A: Docker (recommended)
sudo docker run --rm \
    -p 7880:7880 \
    -p 7881:7881 \
    -p 7882:7882/udp \
    -e LIVEKIT_KEYS="devkey: devsecret" \
    livekit/livekit-server:latest

# Option B: Native binary
livekit-server --dev
```

#### Terminal 2: Token Server

```bash
uv run python livekit_client/token_server.py
```

#### Terminal 3: Agent Worker

```bash
# Set environment variables
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/creds.json"
export VERTEX_PROJECT_ID="your-project-id"
export VERTEX_LOCATION="us-central1"
export DATA_ROOT="checkpoints/ditto_trt_Ampere_Plus"
export CFG_PKL="checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
export SOURCE_PATH="avatars/your_avatar.jpg"

# Start agent (launches avatar worker automatically)
uv run python livekit_avatar/agent_worker.py dev
```

---

## Testing the Setup

### 1. Open the Web Client

Navigate to: `http://localhost:8000/simple_client.html`

### 2. Connect to Room

1. Enter your name/ID in the input field
2. Click "Connect"
3. Allow microphone access when prompted

### 3. Test Conversation

1. Speak into your microphone
2. The avatar should respond with synchronized audio and video
3. Check console for any errors

### Expected Logs

**Agent Worker:**
```
INFO:livekit.agents - Starting agent in room: my-avatar-test-room
INFO:livekit.agents - ✅ Connected to room
INFO:livekit.agents - Launching avatar worker subprocess...
INFO:livekit.agents - ✅ Avatar worker launched (PID: 12345)
INFO:livekit.agents - ✅ Agent session started
```

**Avatar Worker:**
```
INFO:avatar-worker - Avatar worker starting...
INFO:avatar-worker - Initializing Ditto video generator...
INFO:avatar-worker - ✅ Ready
INFO:avatar-worker - ✅ Connected to room: my-avatar-test-room
INFO:avatar-worker - ✅ Avatar runner started
```

### Troubleshooting Quick Checks

| Issue | Check |
|-------|-------|
| No video | Is avatar worker running? Check GPU memory |
| No audio | Is microphone allowed? Check browser permissions |
| High latency | Check GPU utilization, network latency |
| Crashes | Check CUDA/TensorRT versions, memory |

See [Troubleshooting Guide](./TROUBLESHOOTING.md) for detailed solutions.

---

## Production Deployment

### Environment Variables Summary

```bash
# Required - LiveKit
LIVEKIT_URL="wss://your-project.livekit.cloud"
LIVEKIT_API_KEY="your-api-key"
LIVEKIT_API_SECRET="your-api-secret"

# Required - Google Cloud
GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
VERTEX_PROJECT_ID="your-gcp-project-id"
VERTEX_LOCATION="us-central1"
GEMINI_MODEL="gemini-live-2.5-flash-preview-native-audio-09-2025"

# Required - Ditto
DATA_ROOT="checkpoints/ditto_trt_Ampere_Plus"
CFG_PKL="checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
SOURCE_PATH="avatars/your_avatar.jpg"

# Optional - Avatar Settings
AVATAR_WIDTH="1280"
AVATAR_HEIGHT="720"
AVATAR_FPS="25"
```

### Docker Deployment (Example)

```dockerfile
FROM nvidia/cuda:12.2-cudnn8-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy application
WORKDIR /app
COPY . .

# Install Python packages
RUN pip install -r requirements.txt

# Set entrypoint
CMD ["python", "livekit_avatar/agent_worker.py", "start"]
```

### Scaling Considerations

1. **GPU per Avatar Worker**: Each avatar worker needs dedicated GPU
2. **Agent Workers**: Can be scaled horizontally (CPU-bound)
3. **Room Routing**: Use LiveKit's room routing for multi-room support
4. **Health Checks**: Monitor GPU memory, frame rate, latency

---

## Next Steps

- [Architecture Overview](./ARCHITECTURE.md) - Understand system design
- [API Reference](./API_REFERENCE.md) - Customize components
- [Troubleshooting](./TROUBLESHOOTING.md) - Solve common issues

