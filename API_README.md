# Ditto Talking Head API

Real-time talking head synthesis API using Motion-Space Diffusion.

## Features

- **WebSocket Streaming**: Real-time audio streaming for online inference
- **Session Management**: Concurrent user support with configurable limits
- **Dual Backend**: Support for both PyTorch and TensorRT backends
- **Structured Logging**: Rich console output + JSON file logging with correlation IDs
- **Health Monitoring**: Comprehensive health checks and GPU monitoring
- **Auto-cleanup**: Automatic cleanup of expired sessions

## Architecture

```
api/
├── main.py              # FastAPI application entry point
├── config.py            # Configuration management
├── models/              # Pydantic request/response models
├── routers/             # API endpoints
│   ├── streaming.py     # WebSocket streaming endpoint
│   └── health.py        # Health check endpoints
├── services/            # Business logic
│   ├── session_manager.py    # Session management
│   └── inference_service.py  # Inference wrapper
└── middleware/          # Custom middleware
    ├── cors.py          # CORS configuration
    └── logging.py       # Request logging
```

## Quick Start

### 1. Install Dependencies

```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 3. Start the API Server

```bash
# Using the startup script
python start_api.py

# With auto-reload for development
python start_api.py --reload

# Custom host/port
python start_api.py --host 0.0.0.0 --port 8080
```

### 4. Test the API

```bash
# Run the test suite
python test_api.py

# Test specific URL
python test_api.py --url http://localhost:8000
```

### 5. Access API Documentation

- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

## API Endpoints

### Health & Status

- `GET /api/v1/health` - Service health check
- `GET /api/v1/models` - List available models
- `GET /api/v1/gpu-info` - GPU information
- `GET /api/v1/ready` - Readiness probe
- `GET /api/v1/live` - Liveness probe

### Streaming

- `WS /api/v1/stream/ws` - WebSocket streaming endpoint
- `GET /api/v1/stream/sessions` - List active sessions
- `GET /api/v1/stream/stats` - Streaming statistics

## WebSocket Protocol

### Client → Server Messages

**Initialize Session:**
```json
{
  "type": "init",
  "source": "<base64-encoded image/video>",
  "config": {
    "max_size": 1920,
    "crop_scale": 2.3,
    "emo": 4,
    "sampling_timesteps": 50
  }
}
```

**Send Audio Chunk:**
```json
{
  "type": "audio_chunk",
  "data": "<base64-encoded PCM audio (16kHz, int16)>",
  "timestamp": 123.45
}
```

**Stop Streaming:**
```json
{
  "type": "stop"
}
```

### Server → Client Messages

**Status Update:**
```json
{
  "type": "status",
  "message": "Session initialized. Ready for audio chunks.",
  "session_id": "abc-123"
}
```

**Generated Frame:**
```json
{
  "type": "frame",
  "data": "<base64-encoded JPEG>",
  "frame_id": 42,
  "timestamp": 1.68
}
```

**Progress Update:**
```json
{
  "type": "progress",
  "processed": 100,
  "fps": 25.5,
  "latency_ms": 45.2
}
```

**Error:**
```json
{
  "type": "error",
  "message": "Error description",
  "code": "ERROR_CODE"
}
```

**Completion:**
```json
{
  "type": "complete",
  "total_frames": 500,
  "duration_seconds": 20.5
}
```

## Configuration

Configuration is managed via environment variables with the `DITTO_` prefix:

```bash
# Backend selection
DITTO_INFERENCE_BACKEND=pytorch  # or tensorrt

# Server
DITTO_HOST=0.0.0.0
DITTO_PORT=8000

# GPU & Sessions
DITTO_GPU_ID=0
DITTO_MAX_CONCURRENT_SESSIONS=2

# CORS
DITTO_CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Logging
DITTO_LOG_LEVEL=INFO
DITTO_ENABLE_CONSOLE_LOGGING=true
DITTO_ENABLE_FILE_LOGGING=true
```

See `.env.example` for all available options.

## Example Client

### Python WebSocket Client

```python
import asyncio
import websockets
import json
import base64
from PIL import Image
import io

async def stream_inference():
    async with websockets.connect("ws://localhost:8000/api/v1/stream/ws") as ws:
        # Receive initial status
        msg = await ws.recv()
        print(json.loads(msg))

        # Send init with source image
        with open("source.jpg", "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        await ws.send(json.dumps({
            "type": "init",
            "source": img_b64,
            "config": {"emo": 4}
        }))

        # Wait for ready
        msg = await ws.recv()
        print(json.loads(msg))

        # Send audio chunks
        # (Your audio streaming logic here)

        # Stop
        await ws.send(json.dumps({"type": "stop"}))

asyncio.run(stream_inference())
```

### JavaScript WebSocket Client

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/stream/ws');

ws.onopen = () => {
  console.log('Connected');

  // Send init message
  const img = document.getElementById('sourceImage');
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  const imageData = canvas.toDataURL('image/jpeg').split(',')[1];

  ws.send(JSON.stringify({
    type: 'init',
    source: imageData,
    config: { emo: 4 }
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log('Received:', msg.type);

  if (msg.type === 'frame') {
    // Display frame
    const img = new Image();
    img.src = 'data:image/jpeg;base64,' + msg.data;
    document.getElementById('output').appendChild(img);
  }
};
```

## Logging

Logs are written to:
- **Console**: Rich-formatted colored output
- **Files**: JSON-formatted logs in `logs/` directory
  - Current log: `logs/ditto_YYYYMMDD_HHMMSS.json`
  - Symlink to latest: `logs/latest.json`

Each request gets a unique correlation ID for tracking across logs.

## Performance

### Throughput
- PyTorch backend: ~15-20 FPS on RTX 3090
- TensorRT backend: ~40-50 FPS on RTX 3090

### Latency
- WebSocket message latency: <10ms
- Frame generation latency: 40-60ms (PyTorch), 20-25ms (TensorRT)

### Concurrency
- Recommended: 1-2 concurrent sessions per GPU
- Configurable via `DITTO_MAX_CONCURRENT_SESSIONS`

## Troubleshooting

### API won't start
```bash
# Check if models are downloaded
ls checkpoints/ditto_cfg/

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check logs
tail -f logs/latest.json
```

### WebSocket connection fails
```bash
# Check CORS settings in .env
DITTO_CORS_ORIGINS=http://your-frontend-url

# Test with curl
curl http://localhost:8000/api/v1/health
```

### Out of memory errors
```bash
# Reduce concurrent sessions
DITTO_MAX_CONCURRENT_SESSIONS=1

# Use smaller max_size
# In config: "max_size": 512
```

## Development

### Run with auto-reload
```bash
python start_api.py --reload
```

### Run tests
```bash
python test_api.py
```

### View logs
```bash
# Follow logs in real-time
tail -f logs/latest.json | jq .

# Search logs
jq 'select(.level == "ERROR")' logs/latest.json
```

## Production Deployment

### Using Docker
```bash
# Build image
docker build -t ditto-api .

# Run container
docker run -p 8000:8000 --gpus all ditto-api
```

### Using systemd
```bash
# Create service file: /etc/systemd/system/ditto-api.service
sudo systemctl enable ditto-api
sudo systemctl start ditto-api
```

### Behind Nginx
```nginx
location /api/ {
    proxy_pass http://localhost:8000/api/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
}
```

## License

See main repository LICENSE file.
