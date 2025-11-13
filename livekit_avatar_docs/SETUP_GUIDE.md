# Setup Guide

Complete step-by-step instructions for setting up the LiveKit Avatar Agent.

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 or better recommended)
- **VRAM**: Minimum 6GB, 8GB+ recommended
- **RAM**: 16GB system memory minimum
- **CPU**: Multi-core processor (4+ cores recommended)

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended) or Windows with WSL2
- **Python**: 3.10 or 3.11 (3.12 may have compatibility issues)
- **CUDA**: 11.8 or newer
- **TensorRT**: 8.6+ (should match Ditto requirements)
- **Docker**: For running LiveKit server locally (optional)

## Step 1: Clone and Setup Project

```bash
# Navigate to your project directory
cd /path/to/ditto-talkinghead

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

## Step 2: Verify Ditto Models

Ensure your Ditto checkpoint files are in place:

```bash
# Check for required files
ls checkpoints/ditto_trt_custom2/
# Should contain: TensorRT engine files, feature extractors, etc.

ls checkpoints/ditto_cfg/
# Should contain: v0.4_hubert_cfg_trt_online.pkl or similar
```

If missing, follow the Ditto setup instructions in the main project README.

## Step 3: Google Cloud Platform Setup

### 3.1 Create GCP Project

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select an existing one
3. Note your Project ID (e.g., `my-avatar-project-123`)

### 3.2 Enable Required APIs

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT_ID
```

Or via the console:
- Navigate to "APIs & Services" → "Library"
- Search for "Vertex AI API"
- Click "Enable"

### 3.3 Create Service Account

```bash
# Create service account
gcloud iam service-accounts create avatar-agent \
    --display-name="Avatar Agent Service Account" \
    --project=YOUR_PROJECT_ID

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:avatar-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create and download key
gcloud iam service-accounts keys create ~/avatar-service-account.json \
    --iam-account=avatar-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 3.4 Set Environment Variables

```bash
# Add to your ~/.bashrc or ~/.zshrc
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/avatar-service-account.json"
export VERTEX_PROJECT_ID="YOUR_PROJECT_ID"
export VERTEX_LOCATION="us-central1"  # or your preferred region

# Reload your shell
source ~/.bashrc
```

## Step 4: LiveKit Server Setup

You have two options: local development server or cloud deployment.

### Option A: Local Development Server

```bash
# Pull and run LiveKit server (dev mode)
docker run --rm \
    --name livekit-server \
    -p 7880:7880 \
    -p 7881:7881 \
    -p 7882:7882/udp \
    livekit/livekit-server \
    --dev

# Server will use default dev credentials:
# API Key: devkey
# API Secret: devsecret
# WebSocket URL: ws://localhost:7880
```

### Option B: Cloud Deployment

For production, use [LiveKit Cloud](https://cloud.livekit.io) or self-host:

1. Sign up at LiveKit Cloud
2. Create a new project
3. Note your:
   - WebSocket URL (e.g., `wss://your-project.livekit.cloud`)
   - API Key
   - API Secret

Update environment variables:
```bash
export LIVEKIT_URL="wss://your-project.livekit.cloud"
export LIVEKIT_API_KEY="your-api-key"
export LIVEKIT_API_SECRET="your-api-secret"
```

## Step 5: Configure Avatar Settings

### 5.1 Prepare Avatar Source

Choose or create an avatar source image:
- **Format**: JPG or PNG
- **Resolution**: 512x512 or higher
- **Content**: Clear frontal face photo
- **Recommended**: Professional headshot with neutral expression

```bash
# Create avatars directory (if not exists)
mkdir -p avatars

# Copy your avatar image
cp /path/to/your/image.jpg avatars/my_avatar.jpg
```

### 5.2 Set Avatar Configuration

```bash
# Add to environment variables
export SOURCE_PATH="avatars/my_avatar.jpg"
export AVATAR_WIDTH="1280"
export AVATAR_HEIGHT="720"

# Optional: Adjust Ditto paths if using custom models
export DATA_ROOT="checkpoints/ditto_trt_custom2/"
export CFG_PKL="checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
```

## Step 6: Install Additional Dependencies

The main dependencies should be installed via `uv sync`, but verify:

```bash
# Check for scipy (needed for audio resampling)
uv run python -c "import scipy.signal"

# Check for google-genai
uv run python -c "import google.genai"

# If missing, add them
uv add scipy google-genai
```

## Step 7: Test Ditto Model

Before running the full agent, test that Ditto works:

```bash
# Run a simple Ditto test (if you have a test script)
uv run python test_ditto_inference.py

# Should generate a test video without errors
```

## Step 8: Update Startup Script

The `livekit_server.sh` script should already be configured, but verify:

```bash
# Check the script
cat livekit_server.sh

# Ensure it has:
# - GOOGLE_APPLICATION_CREDENTIALS check
# - Correct environment variable exports
# - Correct paths to your config files
```

## Step 9: Start the Agent

```bash
# Make script executable
chmod +x livekit_server.sh

# Start the agent
./livekit_server.sh

# You should see:
# ✅ Published video track: 1280x720
# ✅ Avatar worker started (idle mode)
# ✅ Agent session started
```

### Troubleshooting Agent Startup

**Error: "GOOGLE_APPLICATION_CREDENTIALS must be set"**
- Verify the environment variable is set: `echo $GOOGLE_APPLICATION_CREDENTIALS`
- Ensure the file exists: `ls -l $GOOGLE_APPLICATION_CREDENTIALS`

**Error: "No module named 'stream_pipeline_online'"**
- Verify you're running from the project root directory
- Check that `stream_pipeline_online.py` exists in the current directory

**Error: "CUDA out of memory"**
- Reduce avatar resolution: `export AVATAR_WIDTH=640 AVATAR_HEIGHT=360`
- Close other GPU-intensive applications
- Check GPU memory: `nvidia-smi`

## Step 10: Start the Token Server

In a new terminal:

```bash
cd livekit_client
uv run python token_server.py

# You should see:
# 🚀 Token server running on http://localhost:8000
# 🎫 Token endpoint: http://localhost:8000/token
# 🌐 Client page: http://localhost:8000/simple_client.html
```

## Step 11: Test the Client

1. Open a web browser
2. Navigate to: `http://localhost:8000/simple_client.html`
3. Enter your name (e.g., "Test User")
4. Click "Connect"
5. Allow microphone access when prompted
6. You should see:
   - Status: "Connected - Viewing avatar_video"
   - Avatar video playing
   - The agent greeting you

### Troubleshooting Client Connection

**"Token server not responding"**
- Verify token server is running: `curl http://localhost:8000/token`
- Check firewall settings

**"Failed to connect to LiveKit"**
- Verify LiveKit server is running: `docker ps | grep livekit`
- Check WebSocket URL in `simple_client.html` matches your server

**"No video appearing"**
- Open browser console (F12) and check for errors
- Verify agent is publishing tracks (check agent logs)
- Try refreshing the page

**"No audio from agent"**
- Check browser console for autoplay restrictions
- Click on the page to enable audio playback
- Verify volume is not muted

## Step 12: Verify End-to-End Flow

1. **Speak into your microphone**: "Hello, can you hear me?"
2. **Observe agent logs**: Should show "User started speaking"
3. **Wait for response**: Agent should reply verbally
4. **Check avatar animation**: Lips should move in sync with speech

## Advanced Configuration

### Custom Gemini Instructions

Edit `livekit_avatar/main_agent.py`:

```python
llm_model = google.beta.realtime.RealtimeModel(
    # ... other params ...
    instructions=(
        "You are a helpful customer service agent. "
        "Always be polite and professional. "
        "Keep responses under 30 seconds."
    ),
)
```

### Adjust Video Quality

```bash
# Higher quality (more bandwidth)
export AVATAR_WIDTH="1920"
export AVATAR_HEIGHT="1080"

# Lower quality (less bandwidth)
export AVATAR_WIDTH="640"
export AVATAR_HEIGHT="360"
```

### Change Frame Rate

Edit `livekit_avatar/custom_avatar_worker.py`:

```python
FPS = 30  # Change from 50 to 30 for lower GPU usage
```

### Multiple Avatar Images

Create multiple avatar configurations:

```bash
# Create avatar config files
export SOURCE_PATH_1="avatars/avatar1.jpg"
export SOURCE_PATH_2="avatars/avatar2.jpg"

# Switch between them by changing SOURCE_PATH before starting
```

## Production Deployment

### Security Checklist

- [ ] Use cloud-hosted LiveKit (not dev server)
- [ ] Implement proper token expiration (1-4 hours)
- [ ] Enable TLS for all connections
- [ ] Use service account with minimal permissions
- [ ] Rotate API keys regularly
- [ ] Implement rate limiting on token server
- [ ] Add authentication to token endpoint

### Monitoring

```bash
# View agent logs
tail -f /path/to/agent.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor LiveKit server
docker logs -f livekit-server
```

### Backup and Recovery

```bash
# Backup configuration
tar -czf avatar-config-backup.tar.gz \
    checkpoints/ \
    avatars/ \
    livekit_avatar/ \
    livekit_client/

# Backup service account key (keep secure!)
cp $GOOGLE_APPLICATION_CREDENTIALS ~/backups/avatar-sa-key.json
```

## Next Steps

- Review [Architecture Documentation](ARCHITECTURE.md) for system understanding
- Check [API Reference](API_REFERENCE.md) for customization options
- Read [Troubleshooting Guide](TROUBLESHOOTING.md) for common issues
- Test with multiple concurrent users
- Implement custom conversation flows
- Add analytics and logging
