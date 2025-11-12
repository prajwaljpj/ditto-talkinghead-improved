# Turn-Based Conversation Flow Diagrams

## Complete System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER BROWSER (WebRTC Client)                  │
│                                                                   │
│  🎤 Microphone ──────────────────────────────> 🔊 Speakers       │
│                                                                   │
│  📹 Video Display <─────────────────────────────────────────┐   │
└────────────┬────────────────────────────────────────────────┼───┘
             │ Audio Out (48kHz)                              │
             │                                        Video In │
             ▼                                                 │
┌─────────────────────────────────────────────────────────────┼───┐
│                      LIVEKIT SERVER (WebRTC)                 │   │
│                                                              │   │
│  Audio Track <──────────> Agent Audio In/Out                │   │
│  Video Track <──────────> Agent Video Out                   │   │
└────────────┬────────────────────────────────────────────────┼───┘
             │ Audio (user speech)                Video Frames│
             ▼                                                 │
┌────────────────────────────────────────────────────────────────┐│
│              GEMINI DITTO AGENT (Main Process)                 ││
│                                                                ││
│  ┌──────────────────────────────────────────────────────┐    ││
│  │          CONVERSATION STATE MANAGER                   │    ││
│  │                                                        │    ││
│  │  Current State: [IDLE|LISTENING|THINKING|SPEAKING]   │    ││
│  │  User Audio Buffer: [accumulated audio chunks]       │    ││
│  │  Silent Audio Generator: 1600 samples @ 16kHz        │    ││
│  └──────────────────────────────────────────────────────┘    ││
│                           │                                    ││
│  ┌────────────────────────┼────────────────────────────┐     ││
│  │         VAD (Voice Activity Detection)               │     ││
│  │                        │                             │     ││
│  │  Analyzes: 16kHz mono audio                         │     ││
│  │  Detects: START_OF_SPEECH, END_OF_SPEECH           │     ││
│  └──────────────────────────────────────────────────────┘     ││
│                           │                                    ││
│  ┌────────────────────────┼────────────────────────────┐     ││
│  │    USER AUDIO PROCESSOR (send_user_audio_to_gemini) │     ││
│  │                        │                             │     ││
│  │  1. Resample 48kHz → 16kHz                          │     ││
│  │  2. Run VAD analysis                                │     ││
│  │  3. Manage state transitions                        │     ││
│  │  4. Accumulate audio in LISTENING                   │     ││
│  │  5. Send to Gemini on END_OF_SPEECH                 │     ││
│  └──────────────────────────────────────────────────────┘     ││
│                           │                                    ││
│                           ▼                                    ││
│  ┌─────────────────────────────────────────────────────┐     ││
│  │      GEMINI LIVE API SESSION (WebSocket)            │     ││
│  │                                                      │     ││
│  │  → Send: User audio (PCM16, 16kHz)                 │     ││
│  │  ← Receive: AI audio (PCM16, 24kHz)                │     ││
│  │            + Text transcription                     │     ││
│  └──────────────────────────────────────────────────────┘     ││
│                           │                                    ││
│                           ▼                                    ││
│  ┌─────────────────────────────────────────────────────┐     ││
│  │    GEMINI RESPONSE HANDLER (_handle_gemini_response)│     ││
│  │                                                      │     ││
│  │  1. Detect first audio → THINKING to SPEAKING      │     ││
│  │  2. Process audio chunks                           │     ││
│  │  3. Detect end → SPEAKING to IDLE                  │     ││
│  └──────────────────────────────────────────────────────┘     ││
│                           │                                    ││
│                           ▼                                    ││
│  ┌─────────────────────────────────────────────────────┐     ││
│  │   AUDIO ROUTER (_process_gemini_audio)              │     ││
│  │                                                      │     ││
│  │  Branch 1: User Playback (always)                  │     ││
│  │    → Resample 24kHz → 48kHz                        │     ││
│  │    → Send to LiveKit audio source                  │     ││
│  │                                                      │     ││
│  │  Branch 2: Ditto (only in SPEAKING state)          │     ││
│  │    → Resample 24kHz → 16kHz                        │     ││
│  │    → Buffer to 400ms chunks                        │     ││
│  │    → Feed to Ditto                                 │     ││
│  └──────────────────────────────────────────────────────┘     ││
│                           │                                    ││
│  ┌────────────────────────┼────────────────────────────┐     ││
│  │  SILENT AUDIO GENERATOR (run_silent_audio_generator)│     ││
│  │                        │                             │     ││
│  │  Loop every 100ms:                                  │     ││
│  │    If state != SPEAKING:                            │     ││
│  │      Generate 1600 samples (100ms @ 16kHz)         │     ││
│  │      Feed to Ditto                                  │     ││
│  └──────────────────────────────────────────────────────┘     ││
│                           │                                    ││
│                           ▼                                    ││
│  ┌─────────────────────────────────────────────────────┐     ││
│  │         DITTO SDK (stream_pipeline_online)          │     ││
│  │                                                      │     ││
│  │  Input: Audio chunks (16kHz, float32)              │     ││
│  │  Process: Lip-sync video generation                │     ││
│  │  Output: Video frames (25 FPS, RGB)               │     ││
│  │  Callback: _on_frame_generated                     │     ││
│  └──────────────────────────────────────────────────────┘     ││
│                           │                                    ││
└───────────────────────────┼────────────────────────────────────┘│
                            │ Video Frames (RGB, 25 FPS)          │
                            └─────────────────────────────────────┘
```

## State Machine Diagram

```
                    ┌──────────────┐
                    │     IDLE     │
                    │              │
                    │  - Silent    │
                    │  - Waiting   │
                    └──────┬───────┘
                           │
            VAD: START_OF_SPEECH (user speaks)
                           │
                           ▼
                    ┌──────────────┐
                    │  LISTENING   │
                    │              │
                    │  - Silent    │
             ┌─────>│  - Buffer    │
             │      │    Audio     │
             │      └──────┬───────┘
             │             │
             │  VAD: END_OF_SPEECH (user stops)
             │             │
             │             ▼
             │      ┌──────────────┐
             │      │   THINKING   │
             │      │              │
             │      │  - Silent    │
             │      │  - Waiting   │
             │      │    Gemini    │
             │      └──────┬───────┘
             │             │
             │  Gemini: FIRST_AUDIO_CHUNK
             │             │
             │             ▼
             │      ┌──────────────┐
             │      │   SPEAKING   │
             │      │              │
             │      │  - Real      │
             │      │    Audio     │
             │      │  - Lip-sync  │
             │      └──────┬───────┘
             │             │
             │  Gemini: NO_MORE_AUDIO
             │             │
             └─────────────┘
```

## Audio Flow by State

### IDLE State

```
User: Silent
        ↓
    [BLOCKED] ──> VAD detects nothing

Avatar:
    Silent Audio Generator (100ms chunks)
        ↓
    Ditto SDK
        ↓
    Idle Animation Video (25 FPS)
        ↓
    LiveKit Video Track
        ↓
    User sees: Neutral expression, subtle animation
```

### LISTENING State

```
User: Speaking
        ↓
    LiveKit Audio In (48kHz)
        ↓
    Resample to 16kHz
        ↓
    VAD Analysis ──> Detects SPEECH
        ↓
    Accumulate in Buffer

Avatar:
    Silent Audio Generator (100ms chunks)
        ↓
    Ditto SDK
        ↓
    Listening Animation Video (25 FPS)
        ↓
    LiveKit Video Track
        ↓
    User sees: Listening expression
```

### THINKING State

```
User: Silent (buffered audio sent to Gemini)
        ↓
    [BLOCKED] ──> Waiting for Gemini

Gemini:
    Processing user audio
        ↓
    Generating response

Avatar:
    Silent Audio Generator (100ms chunks)
        ↓
    Ditto SDK
        ↓
    Thinking Animation Video (25 FPS)
        ↓
    LiveKit Video Track
        ↓
    User sees: Thinking expression
```

### SPEAKING State

```
User: Silent
        ↓
    [BLOCKED] ──> Cannot interrupt

Gemini:
    Audio Response (24kHz PCM16)
        ↓
        ├──> Branch 1: User Playback
        │       ↓
        │    Resample 24kHz → 48kHz
        │       ↓
        │    LiveKit Audio Out
        │       ↓
        │    User hears: AI voice
        │
        └──> Branch 2: Avatar Lip-sync
                ↓
             Resample 24kHz → 16kHz
                ↓
             Buffer to 400ms (6400 samples)
                ↓
             Ditto SDK
                ↓
             Speaking Animation Video (25 FPS)
                ↓
             LiveKit Video Track
                ↓
             User sees: Lip-synced speech

Note: Silent Audio Generator is PAUSED during SPEAKING
```

## Timeline Example: One Conversation Turn

```
Time      State         User Action       System Action              Ditto Input
────────────────────────────────────────────────────────────────────────────────
0.0s      IDLE         Silent            Silent audio generator      Silent (100ms)
0.1s      IDLE         Silent            Silent audio generator      Silent (100ms)
0.2s      IDLE         Silent            Silent audio generator      Silent (100ms)

0.3s      IDLE         Starts speaking   VAD: START_OF_SPEECH       Silent (100ms)
0.3s      LISTENING    Speaking          Accumulate audio           Silent (100ms)
0.4s      LISTENING    Speaking          Accumulate audio           Silent (100ms)
0.5s      LISTENING    Speaking          Accumulate audio           Silent (100ms)
...       ...          ...               ...                        ...
2.0s      LISTENING    Speaking          Accumulate audio           Silent (100ms)

2.1s      LISTENING    Stops speaking    VAD: END_OF_SPEECH        Silent (100ms)
2.1s      THINKING     Silent            Send audio to Gemini       Silent (100ms)
2.2s      THINKING     Silent            Waiting for response       Silent (100ms)
2.3s      THINKING     Silent            Waiting for response       Silent (100ms)
2.4s      THINKING     Silent            Waiting for response       Silent (100ms)

2.5s      THINKING     Silent            Gemini: First audio        Silent (100ms)
2.5s      SPEAKING     Silent            Process audio chunk        Real audio (400ms)
2.6s      SPEAKING     Silent            Process audio chunk        Real audio (400ms)
2.7s      SPEAKING     Silent            Process audio chunk        Real audio (400ms)
...       ...          ...               ...                        ...
5.0s      SPEAKING     Silent            Process audio chunk        Real audio (400ms)

5.1s      SPEAKING     Silent            Gemini: No more audio      Real audio (400ms)
5.1s      IDLE         Silent            Resume silent audio        Silent (100ms)
5.2s      IDLE         Silent            Silent audio generator     Silent (100ms)
...       ...          ...               ...                        ...
```

## Audio Sample Rate Conversions

```
┌──────────────────────────────────────────────────────────┐
│                USER AUDIO PIPELINE                        │
└──────────────────────────────────────────────────────────┘

Client Microphone
    ↓
48000 Hz Stereo (LiveKit standard)
    ↓
Convert stereo → mono (average channels)
    ↓
48000 Hz Mono
    ↓
Resample 48kHz → 16kHz (scipy.signal.resample_poly, up=1, down=3)
    ↓
16000 Hz Mono (VAD and Gemini input format)
    ↓
Buffer during LISTENING state
    ↓
Send to Gemini on END_OF_SPEECH

┌──────────────────────────────────────────────────────────┐
│                GEMINI AUDIO PIPELINE                      │
└──────────────────────────────────────────────────────────┘

Gemini Live API Response
    ↓
24000 Hz Mono PCM16 (Gemini's native format)
    ↓
Convert PCM16 → float32 (÷ 32768.0)
    ↓
    ├──> BRANCH 1: User Playback
    │       ↓
    │    Resample 24kHz → 48kHz (scipy.signal.resample_poly, up=2, down=1)
    │       ↓
    │    48000 Hz Mono
    │       ↓
    │    Convert float32 → PCM16 (× 32768.0)
    │       ↓
    │    LiveKit Audio Source (user hears this)
    │
    └──> BRANCH 2: Ditto Lip-sync (only in SPEAKING state)
            ↓
         Resample 24kHz → 16kHz (scipy.signal.resample_poly, up=2, down=3)
            ↓
         16000 Hz Mono float32
            ↓
         Buffer to 6400 samples (400ms @ 16kHz)
            ↓
         Ditto SDK (generates lip-synced video)

┌──────────────────────────────────────────────────────────┐
│            SILENT AUDIO PIPELINE                          │
└──────────────────────────────────────────────────────────┘

Silent Audio Generator (runs every 100ms)
    ↓
Generate zeros: np.zeros(1600, dtype=float32)
    ↓
1600 samples = 100ms @ 16kHz
    ↓
Only when state != SPEAKING
    ↓
Ditto SDK (generates idle/listening animation)
```

## Frame Generation Flow

```
┌──────────────────────────────────────────────────────────┐
│                CONTINUOUS FRAME GENERATION                │
└──────────────────────────────────────────────────────────┘

Audio Input (from Silent Generator OR Gemini)
    ↓
Ditto SDK (stream_pipeline_online.py)
    ↓
TensorRT Inference (GPU)
    │
    ├─> Audio Processing (Hubert features)
    ├─> Face Generation
    ├─> Emotion Application
    └─> Lip-sync Application
    ↓
RGB Frame (H×W×3, uint8)
    ↓
Callback: _on_frame_generated(frame, frame_idx, timestamp)
    ↓
Frame Pacing Check
    │
    ├─> Too fast? (< 32ms since last) → DROP
    └─> OK timing → Continue
    ↓
Convert RGB → RGBA (add alpha channel)
    ↓
Create rtc.VideoFrame
    ↓
video_source.capture_frame(frame)
    ↓
LiveKit Video Track
    ↓
WebRTC Stream to Client
    ↓
User Browser (sees avatar at 25 FPS)

Target: 25 FPS = 40ms per frame
Actual: 32ms pacing = ~31 FPS max, drops to 25 FPS
Drops: < 5% acceptable
```

## Concurrency Model

```
┌────────────────────────────────────────────────────────┐
│            ASYNCIO EVENT LOOP (Main Thread)             │
└────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Gemini Task │   │  Silent Task │   │ Audio Stream │
│              │   │              │   │              │
│  async loop  │   │  async loop  │   │  async loop  │
│  receive()   │   │  while True  │   │  for frame   │
│              │   │              │   │              │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────────────────────────────────────────────┐
│          CONVERSATION STATE MANAGER                   │
│          (Thread-safe with asyncio.Lock)              │
└──────────────────────────────────────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────────────────────────────────────────────┐
│         DITTO SDK (Worker Thread Pool)                │
│         (via asyncio.to_thread)                       │
└──────────────────────────────────────────────────────┘
       │
       └──> Callback: _on_frame_generated (from worker thread)
                │
                └──> Captured by event loop
                        │
                        └──> video_source.capture_frame()
```

## Logging Flow

```
┌────────────────────────────────────────────────────────┐
│                   INITIALIZATION                        │
└────────────────────────────────────────────────────────┘
🎭 Initializing Ditto SDK...
✅ Ditto SDK initialized
🎤 Initializing VAD...
✅ VAD initialized
🤖 Initializing Gemini Live API...
✅ Gemini client initialized
✅ Background tasks started
🔇 Starting silent audio generator
✅ Agent ready - speak to start conversation!

┌────────────────────────────────────────────────────────┐
│                CONVERSATION TURN                        │
└────────────────────────────────────────────────────────┘
                    [User speaks]
👂 User started speaking
🔄 State transition: idle → listening
                    [User stops]
🤔 User stopped speaking - sending to Gemini
🔄 State transition: listening → thinking
                    [Gemini responds]
🔄 State transition: thinking → speaking
💬 Gemini: [transcription text]
🎤 Processed 10 Gemini audio chunks
🎤 Processed 20 Gemini audio chunks
                    [Gemini finishes]
🔄 State transition: speaking → idle

┌────────────────────────────────────────────────────────┐
│              CONTINUOUS MONITORING                      │
└────────────────────────────────────────────────────────┘
🎬 First frame generated: (720, 1280, 3)
📊 Frames: 100 sent, 2 dropped
📊 Frames: 200 sent, 5 dropped
📊 Frames: 300 sent, 8 dropped
...

┌────────────────────────────────────────────────────────┐
│                  ERROR SCENARIOS                        │
└────────────────────────────────────────────────────────┘
⚠️  Failed to send audio to Gemini: [error]
⚠️  Connection closed: [error] - reconnecting in 2s...
❌ Error in silent audio generator: [error]
❌ Error sending frame: [error]
```

## Summary

The turn-based conversational avatar uses a carefully orchestrated system of:

1. **State Machine**: 4 states managing conversation flow
2. **VAD Integration**: Automatic speech detection
3. **Dual Audio Paths**: Silent audio (idle) + real audio (speaking)
4. **Continuous Frame Generation**: 25 FPS regardless of state
5. **Sample Rate Conversions**: Proper resampling for all components
6. **Concurrent Tasks**: Gemini + silent audio generators
7. **Thread-Safe State**: Locked access to shared state

All working together to create a natural, turn-based conversational experience with an always-visible avatar.
