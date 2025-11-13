# Architecture Overview

## System Components

The LiveKit Avatar Agent consists of several interconnected components:

```
┌─────────────┐
│   Browser   │ (Web Client)
│   Client    │
└──────┬──────┘
       │ WebRTC (Audio/Video)
       │
┌──────▼──────────────────────────────────────────┐
│           LiveKit Server                        │
│  (Signaling, Media Routing, Recording)          │
└──────┬──────────────────────────────────────────┘
       │
       │ LiveKit SDK
       │
┌──────▼──────────────────────────────────────────┐
│         Avatar Agent (Python)                   │
│  ┌──────────────────────────────────────────┐  │
│  │  main_agent.py                           │  │
│  │  - AgentSession (VAD, STT, TTS)          │  │
│  │  - Gemini LLM integration                │  │
│  │  - Audio/Video track management          │  │
│  └────┬─────────────────────────┬────────────┘  │
│       │ TTS Audio              │ Video Frames   │
│  ┌────▼─────────────────────────▼────────────┐  │
│  │  custom_avatar_worker.py                  │  │
│  │  - Audio buffering & processing           │  │
│  │  - State management (idle/speaking/etc)   │  │
│  │  - Frame callback handling                │  │
│  └────┬──────────────────────────────────────┘  │
│       │ Audio chunks                             │
│  ┌────▼──────────────────────────────────────┐  │
│  │  stream_pipeline_online.py (Ditto SDK)    │  │
│  │  - Audio → Motion generation (LMDM)       │  │
│  │  - Motion → Video rendering (TensorRT)    │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Data Flow

### 1. User Speech → Agent Response

```
User speaks
    ↓
LiveKit captures audio (WebRTC)
    ↓
AgentSession VAD detects speech
    ↓
Speech-to-Text (Gemini)
    ↓
LLM generates response (Gemini)
    ↓
Text-to-Speech audio (Gemini)
    ↓
Audio fed to CustomAvatarWorker
    ↓
Ditto generates lip-synced video frames
    ↓
Frames published to LiveKit
    ↓
Client receives and displays video
```

### 2. Audio Processing Pipeline

```
TTS Audio Frame (24kHz PCM)
    ↓
Convert to float32 [-1.0, 1.0]
    ↓
Resample to 16kHz (if needed)
    ↓
Buffer accumulation (6480 samples)
    ↓
Feed to Ditto StreamSDK
    ↓
Audio → Facial Motion (LMDM model)
    ↓
Motion → 3D Warping
    ↓
3D → Rendered Frame (Decoder)
    ↓
RGB → I420 conversion
    ↓
Push to LiveKit VideoSource
```

## Component Details

### main_agent.py

**Purpose**: Main entrypoint for the LiveKit agent

**Key Responsibilities**:
- Initialize and configure Gemini LLM model
- Create `AgentSession` for conversation management
- Publish video track to LiveKit room
- Capture TTS audio from agent's audio output
- Feed TTS audio to CustomAvatarWorker
- Handle conversation state changes
- Graceful shutdown and cleanup

**Key Events Handled**:
- `user_turn_started` - User begins speaking
- `user_turn_completed` - User finishes speaking
- `agent_started_speaking` - Agent begins TTS
- `agent_stopped_speaking` - Agent finishes TTS

### custom_avatar_worker.py

**Purpose**: Bridge between LiveKit audio and Ditto avatar generation

**Key Responsibilities**:
- Initialize Ditto StreamSDK with avatar configuration
- Manage audio buffering and chunking
- Handle different animation states (idle/listening/speaking)
- Convert generated RGB frames to I420 for LiveKit
- Coordinate between async (LiveKit) and thread-based (Ditto) execution

**Audio Processing**:
- Accepts audio frames via `feed_audio()` method
- Buffers audio in an async queue
- Chunks audio into Ditto-compatible sizes (6480 samples)
- Generates silent audio for idle states
- Uses ThreadPoolExecutor for blocking Ditto SDK calls

**State Machine**:
```
┌────────┐
│  Idle  │ ←──────────────┐
└───┬────┘                 │
    │ User speaks          │ Agent finishes
┌───▼────────┐             │
│ Listening  │             │
└───┬────────┘             │
    │ User stops           │
┌───▼────────┐             │
│ Thinking   │             │
└───┬────────┘             │
    │ Agent speaks         │
┌───▼────────┐             │
│ Speaking   │─────────────┘
└────────────┘
```

### stream_pipeline_online.py

**Purpose**: Ditto avatar generation SDK wrapper

**Key Components**:
1. **Audio2Motion (LMDM)**: Converts audio features to facial motion parameters
2. **MotionStitch**: Blends generated motion with source avatar characteristics
3. **WarpF3D**: Applies 3D warping to source image based on motion
4. **DecodeF3D**: Renders final RGB frame from 3D representation

**Threading Model**:
- Multiple worker threads for parallel processing
- Queue-based pipeline for frame generation
- Frame callback executed in worker thread (not main thread)

### Web Client (simple_client.html)

**Purpose**: Browser-based interface for users to interact with the avatar

**Key Features**:
- Token-based authentication
- Microphone access for user audio
- Video rendering for avatar display
- Audio playback for agent speech
- Connection state management

**LiveKit Client SDK Usage**:
- `Room` - Main connection object
- `createLocalAudioTrack()` - Capture user's microphone
- `track.attach()` - Display video/audio from agent
- Event handlers for track subscriptions

## Technical Considerations

### Latency Optimization

**Total Pipeline Latency**: ~200-500ms
- VAD detection: 50-100ms
- STT processing: 100-200ms
- LLM generation: 100-300ms (streaming)
- TTS generation: 50-100ms (streaming)
- Avatar rendering: 20ms per frame
- Network latency: 20-100ms

**Optimization Strategies**:
1. Streaming TTS (audio arrives incrementally)
2. Pipelined video generation (multiple frames in flight)
3. Pre-buffering for smooth playback
4. GPU acceleration for avatar rendering

### Resource Usage

**GPU**:
- ~4-6 GB VRAM for Ditto models (TensorRT optimized)
- ~100-300ms per frame generation (depends on GPU)

**CPU**:
- Audio resampling and buffering
- Video format conversion (RGB → I420)
- Network I/O and WebRTC encoding

**Network**:
- Upstream (client → server): ~50-100 kbps (audio only)
- Downstream (server → client): 1-3 Mbps (HD video + audio)

### Concurrency Model

```python
# Main Thread
- asyncio event loop
- LiveKit async operations
- AgentSession coordination

# Worker Threads (Ditto SDK)
- Audio2Motion inference
- Motion stitching
- Video rendering
- Frame callbacks

# ThreadPoolExecutor
- Bridges async and sync code
- Prevents blocking event loop
```

### Error Handling

**Audio Stream Interruption**:
- Falls back to silent audio generation
- Maintains continuous video output

**Network Issues**:
- LiveKit automatic reconnection
- Buffering to handle jitter

**GPU Errors**:
- Graceful shutdown with cleanup
- Error logging for debugging

## Security Considerations

### Authentication
- Token-based access control (JWT)
- Short-lived tokens (recommended: 1-4 hours)
- Room-specific permissions

### Network
- WebRTC encryption (DTLS-SRTP)
- TLS for signaling (recommended)
- No P2P in production (use SFU)

### API Keys
- Service account credentials for Vertex AI
- API keys for LiveKit server
- Never expose credentials to client

## Scalability

### Single Agent Capacity
- 1 concurrent conversation per agent instance
- Limited by GPU capacity (1 avatar render per instance)

### Multi-Agent Deployment
- Deploy multiple agent instances
- LiveKit agent dispatcher for load balancing
- Each instance handles one avatar session

### Horizontal Scaling
```
Multiple Agent Instances
        ↓
LiveKit Agent Dispatcher
        ↓
    SFU Router
        ↓
   Many Clients
```

## Extension Points

### Custom Avatar Sources
- Modify `SOURCE_PATH` environment variable
- Provide different images/videos as avatar base
- Ditto automatically extracts facial features

### Custom LLM Models
- Replace Gemini with other LLM providers
- Implement custom `RealtimeModel` interface
- Must support streaming TTS output

### Custom Animation States
- Extend `set_state()` method
- Add emotion-specific animations
- Modify Ditto control parameters

### Recording and Analytics
- LiveKit Egress for recording sessions
- Webhook integration for events
- Custom analytics via event handlers
