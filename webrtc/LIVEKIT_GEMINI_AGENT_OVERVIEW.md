# LiveKit Gemini Agent - Detailed Architecture Overview

## Overview

The `livekit_gemini_agent.py` implements a conversational avatar system that combines:
- **LiveKit** for WebRTC audio/video streaming
- **Google Gemini Live API** for natural conversation (ASR + LLM + TTS)
- **Ditto** for real-time avatar video generation with lip-sync

## Key Design Principles

1. **Gemini Handles VAD & Turn Detection**: No custom VAD needed - Gemini's built-in VAD automatically detects when the user starts/stops speaking and manages turn-taking.

2. **Avatar Always Visible**: The avatar continuously generates frames at 25 FPS, ensuring it's always visible with natural animations.

3. **State-Based Audio Routing**: Simple two-state system (IDLE/SPEAKING) determines whether to feed real audio or silent audio to Ditto.

## Architecture Flow

### 1. Initialization Phase

```
Entrypoint
  ├─> Create GeminiDittoAgent
  ├─> Initialize Ditto SDK (loads models, sets up pipeline)
  ├─> Initialize Gemini Live API client (Vertex AI or API key)
  ├─> Connect to LiveKit room
  ├─> Create video track (avatar output)
  ├─> Create audio track (avatar speech output)
  └─> Start background tasks
```

**Key Components Initialized:**
- `StreamSDK`: Ditto model pipeline (always running)
- `gemini_client`: Google Gemini Live API client
- `state_manager`: Simple state tracker (IDLE/SPEAKING)
- `video_source`: LiveKit video track for avatar frames
- `audio_source`: LiveKit audio track for avatar speech

### 2. User Audio Flow

```
User speaks in browser
  │
  ├─> LiveKit captures audio (48kHz, mono/stereo)
  │
  ├─> AudioStream receives AudioFrame
  │
  ├─> send_user_audio_to_gemini()
  │   ├─> Resample 48kHz → 16kHz (if needed)
  │   ├─> Convert stereo → mono (if needed)
  │   └─> Forward directly to Gemini Live API
  │
  └─> Gemini Live API
      ├─> Built-in VAD detects speech start/stop
      ├─> Processes audio when user stops speaking
      ├─> Generates LLM response
      └─> Converts to TTS audio (24kHz PCM16)
```

**Important**: No manual VAD, no audio accumulation, no state management for user speech - Gemini handles everything!

### 3. Gemini Response Flow

```
Gemini generates response
  │
  ├─> _handle_gemini_response()
  │   ├─> Detects audio in response
  │   ├─> Transitions to SPEAKING state
  │   └─> Calls _process_gemini_audio()
  │
  ├─> _process_gemini_audio()
  │   ├─> Branch 1: Send to browser (user hears avatar)
  │   │   └─> Resample 24kHz → 48kHz
  │   │   └─> Send to LiveKit audio_source
  │   │
  │   └─> Branch 2: Feed to Ditto (avatar lip-sync)
  │       ├─> Only if state == SPEAKING
  │       ├─> Resample 24kHz → 16kHz
  │       ├─> Buffer to 6480-sample chunks
  │       └─> Feed to SDK.run_chunk()
  │
  └─> When response ends
      └─> Transition to IDLE state
```

### 4. Silent Audio Generator (Background Task)

```
run_silent_audio_generator() [runs continuously]
  │
  ├─> Check state every 405ms
  │
  ├─> If state == IDLE:
  │   ├─> Generate 6480 samples of silence
  │   └─> Feed to SDK.run_chunk()
  │       └─> Keeps avatar animated with idle pose
  │
  └─> If state == SPEAKING:
      └─> Skip (real audio is being fed)
```

**Purpose**: Ensures avatar is always generating frames, even when idle.

### 5. Frame Generation Flow

```
Ditto SDK processes audio
  │
  ├─> run_chunk() called with audio (real or silent)
  │   ├─> wav2feat: Extract audio features
  │   ├─> motion_extractor: Generate motion
  │   ├─> warp_network: Generate video frames
  │   └─> Frame callback triggered
  │
  ├─> _on_frame_generated() callback
  │   ├─> Frame pacing (25 FPS target)
  │   ├─> Convert RGB → RGBA
  │   └─> Send to LiveKit video_source
  │
  └─> Browser receives video frames
      └─> Avatar visible to user
```

## State Management

### Simplified Two-State System

```
IDLE
  ├─> Avatar: Idle animation (silent audio → Ditto)
  ├─> User: Can speak (audio forwarded to Gemini)
  └─> Gemini: Listening, processing, or waiting

SPEAKING
  ├─> Avatar: Lip-syncing to Gemini TTS
  ├─> User: Can interrupt (Gemini handles this)
  └─> Gemini: Generating and streaming TTS audio
```

**State Transitions:**
- `IDLE → SPEAKING`: When Gemini starts sending audio
- `SPEAKING → IDLE`: When Gemini finishes sending audio

**Why Simplified?**
- Gemini handles user speech detection (VAD)
- Gemini handles turn-taking automatically
- Gemini handles interruptions
- We only need to track if AI is speaking to route audio correctly

## Audio Processing Details

### Chunk Sizes

All audio chunks use **6480 samples** (405ms @ 16kHz), matching the SDK's expected size:
```
Formula: sum(chunksize) * 0.04 * 16000 + 80
For chunksize=(3,5,2): 10 * 0.04 * 16000 + 80 = 6480
```

This satisfies:
- SDK requirements (exact match)
- TensorRT minimum (3240 samples)

### Sample Rate Conversions

```
User Audio:
  48kHz (LiveKit) → 16kHz (Gemini) → 16kHz (Ditto)

Gemini Audio:
  24kHz (Gemini TTS) → 48kHz (LiveKit output) → 16kHz (Ditto)
```

### Audio Buffering

- **Gemini audio buffer**: Accumulates 24kHz audio, processes in 6480-sample chunks at 16kHz
- **No user audio buffer**: Audio forwarded directly to Gemini (no accumulation needed)

## Key Features

### 1. Automatic Turn Detection
- Gemini's built-in VAD detects when user starts/stops speaking
- No manual speech detection needed
- Handles background noise and silence automatically

### 2. Natural Interruptions
- User can speak while AI is speaking
- Gemini handles interruptions gracefully
- State management adapts automatically

### 3. Always-On Avatar
- Avatar generates frames continuously (25 FPS)
- Silent audio keeps avatar animated during idle
- Smooth transitions between idle and speaking states

### 4. Low Latency
- Direct audio forwarding (no buffering delays)
- Real-time frame generation
- WebRTC streaming for minimal latency

## Error Handling

### Gemini Session
- Automatic reconnection on connection drops
- Handles authentication errors gracefully
- Logs errors but continues operation

### Ditto SDK
- Frame generation continues even if errors occur
- Silent audio generator handles errors gracefully
- State management is resilient to failures

## Configuration

### Environment Variables

**LiveKit:**
- `LIVEKIT_URL`: WebSocket URL (e.g., `ws://localhost:7880`)
- `LIVEKIT_API_KEY`: API key
- `LIVEKIT_API_SECRET`: API secret

**Gemini Authentication (choose one):**
- Vertex AI: `GOOGLE_APPLICATION_CREDENTIALS`, `VERTEX_PROJECT_ID`, `VERTEX_LOCATION`
- API Key: `GEMINI_API_KEY`

**Ditto:**
- `DITTO_CFG_PKL`: Config pickle path
- `DITTO_DATA_ROOT`: Model directory
- `DITTO_SOURCE`: Avatar image/video path
- `DITTO_MAX_SIZE`: Max resolution
- `DITTO_EMO`: Emotion ID

**Gemini:**
- `GEMINI_MODEL`: Model name (default: `gemini-live-2.5-flash-preview-native-audio-09-2025`)
- `GEMINI_VOICE`: Voice name (default: `Puck`)
- `GEMINI_INSTRUCTION`: System instruction

## Performance Characteristics

### Frame Generation
- Target: 25 FPS
- Frame pacing: Drops frames if generating too fast
- Frame callback: Async, non-blocking

### Audio Processing
- Silent audio: ~3 FPS (405ms intervals)
- Real audio: Continuous (as Gemini streams)
- Chunk size: 6480 samples (405ms @ 16kHz)

### Latency
- User speech → Gemini: < 50ms (direct forwarding)
- Gemini response → Avatar: < 200ms (audio processing + frame generation)
- Frame → Browser: < 50ms (WebRTC streaming)

## Comparison: Before vs After Simplification

### Before (Custom VAD)
- 4 states: IDLE, LISTENING, THINKING, SPEAKING
- Manual VAD with silero
- Audio accumulation during LISTENING
- Manual state transitions based on VAD events
- ~200 lines of VAD/state management code

### After (Gemini VAD)
- 2 states: IDLE, SPEAKING
- No custom VAD needed
- Direct audio forwarding
- State transitions based on Gemini responses only
- ~50 lines of state management code

**Benefits:**
- Simpler codebase
- More reliable (Gemini's VAD is production-tested)
- Better turn detection (Gemini understands context)
- Supports interruptions naturally
- Less code to maintain

## Future Enhancements

Potential improvements:
1. Emotion detection from Gemini responses
2. Gesture control based on conversation context
3. Multi-avatar support
4. Custom voice cloning
5. Background music/noise handling
6. Conversation history management

