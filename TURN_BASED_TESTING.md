# Turn-Based Conversational Avatar Testing Guide

## Overview

The LiveKit + Gemini + Ditto agent now implements a **turn-based conversational avatar** with the following architecture:

### Conversation States

1. **IDLE**: Avatar in idle animation (silent audio to Ditto)
2. **LISTENING**: User speaking (silent audio to Ditto, accumulate user audio)
3. **THINKING**: Processing user input (silent audio to Ditto)
4. **SPEAKING**: AI responding (Gemini TTS audio to Ditto)

### Key Features

✅ **Always-Visible Avatar**: Ditto continuously generates frames at 25 FPS regardless of state
✅ **Silent Audio Generation**: Maintains avatar animation during idle/listening/thinking states
✅ **VAD-Based Turn Detection**: Silero VAD detects when user starts/stops speaking
✅ **Audio Buffer Accumulation**: User audio is accumulated during LISTENING state
✅ **Turn-Based Coordination**: No overlapping speech - one speaker at a time
✅ **State-Based Audio Routing**: Audio routed to Ditto based on conversation state

## State Transitions

```
IDLE → LISTENING          (VAD detects user speech start)
LISTENING → THINKING      (VAD detects user speech end, audio sent to Gemini)
THINKING → SPEAKING       (Gemini starts responding with audio)
SPEAKING → IDLE           (Gemini finishes responding)
```

## Testing Setup

### 1. Environment Variables

Make sure you have the required environment variables set:

```bash
# Authentication (Vertex AI - Recommended)
export GOOGLE_APPLICATION_CREDENTIALS="gnani-video-ai-c3b9b902d4d8.json"
export VERTEX_PROJECT_ID="gnani-video-ai"
export VERTEX_LOCATION="us-central1"

# LiveKit Configuration
export LIVEKIT_URL="ws://localhost:7880"
export LIVEKIT_API_KEY="devkey"
export LIVEKIT_API_SECRET="devsecret"

# Ditto Configuration
export DITTO_CFG_PKL="checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
export DITTO_DATA_ROOT="checkpoints/ditto_trt_custom2/"
export DITTO_SOURCE="avatars/Hyperrealistic_Indian_Woman_Professional_Presentation_static_5_loop.jpg"
export DITTO_MAX_SIZE="1920"
export DITTO_EMO="4"

# Gemini Configuration
export GEMINI_MODEL="gemini-live-2.5-flash-preview-native-audio-09-2025"
export GEMINI_VOICE="Puck"
export GEMINI_INSTRUCTION="You are a helpful AI assistant. Keep responses concise and natural, as this is a voice conversation."
```

### 2. Start LiveKit Server

```bash
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
  -e LIVEKIT_KEYS="devkey: devsecret" \
  livekit/livekit-server:latest
```

### 3. Start the Agent

```bash
./start_gemini_agent.sh
```

Expected output:
```
🎭 Initializing Ditto SDK...
✅ Ditto SDK initialized
🎤 Initializing VAD (Voice Activity Detection)...
✅ VAD initialized
🤖 Initializing Gemini Live API...
✅ Gemini client initialized
✅ Video track published
✅ Audio track published (you'll hear the avatar speak!)
✅ Background tasks started (Gemini session + silent audio generator)
🔇 Starting silent audio generator for idle/listening/thinking states
✅ Agent ready - speak to start conversation!
💡 Turn-based conversation enabled:
   - Avatar always visible with idle/listening animation
   - Speak to start (VAD detects your speech)
   - AI responds when you finish speaking
   - No overlapping speech - one speaker at a time
```

### 4. Connect Client

Open a web browser to:
```
http://localhost:8000/index_simple.html
```

## Testing Scenarios

### Scenario 1: Basic Turn-Based Conversation

**Test Steps:**
1. Connect to the room
2. Wait for avatar to appear (should show idle animation)
3. Speak: "Hello, can you introduce yourself?"
4. Stop speaking and wait
5. Observe avatar lip-sync during AI response

**Expected Behavior:**
- State transitions: `IDLE → LISTENING → THINKING → SPEAKING → IDLE`
- Avatar visible throughout entire conversation
- No overlapping speech
- Avatar lip-syncs during AI response
- Avatar shows idle animation when not speaking

**Logs to Check:**
```
👂 User started speaking
🔄 State transition: idle → listening
🤔 User stopped speaking - sending to Gemini
🔄 State transition: listening → thinking
🔄 State transition: thinking → speaking
💬 Gemini: [response text]
🔄 State transition: speaking → idle
```

### Scenario 2: Multiple Turns

**Test Steps:**
1. Have a multi-turn conversation
2. Ask 3-4 questions in sequence
3. Wait for complete AI response before asking next question

**Expected Behavior:**
- Each turn follows the state machine correctly
- Avatar maintains visibility throughout
- No audio drops or visual glitches
- Smooth transitions between states

### Scenario 3: Silent Periods

**Test Steps:**
1. Connect and wait 30 seconds without speaking
2. Observe avatar during silence
3. Start speaking after silence

**Expected Behavior:**
- Avatar continuously displays idle animation
- Frames generated at 25 FPS (check logs)
- VAD correctly detects speech start after silence
- No freezing or stuttering

**Logs to Check:**
```
📊 Frames: 100 sent, X dropped
📊 Frames: 200 sent, X dropped
...
```

### Scenario 4: Interruption Prevention

**Test Steps:**
1. Ask a question
2. Try to speak while AI is responding

**Expected Behavior:**
- User audio is blocked during SPEAKING state
- No state transition while AI is speaking
- Avatar continues lip-syncing AI response
- User must wait for AI to finish

**Logs to Check:**
```
# No "👂 User started speaking" logs while in SPEAKING state
```

### Scenario 5: VAD Sensitivity

**Test Steps:**
1. Speak very quietly
2. Speak with background noise
3. Make short utterances (< 1 second)
4. Make long utterances (> 10 seconds)

**Expected Behavior:**
- VAD correctly detects speech start/end
- Short utterances handled correctly
- Long utterances don't time out
- Background noise doesn't trigger false starts

### Scenario 6: Frame Rate Consistency

**Test Steps:**
1. Run conversation for 5 minutes
2. Monitor frame statistics in logs
3. Check for dropped frames

**Expected Behavior:**
- Consistent frame generation at ~25 FPS
- Dropped frame rate < 5%
- No memory leaks
- No performance degradation over time

**Logs to Check:**
```
📊 Frames: 100 sent, 3 dropped  # ~3% drop rate is acceptable
📊 Frames: 200 sent, 7 dropped
...
```

## Debugging

### Issue: Avatar Not Visible

**Check:**
- Silent audio generator is running (look for "🔇 Starting silent audio generator")
- Ditto SDK initialized correctly
- Video track published successfully

### Issue: VAD Not Detecting Speech

**Check:**
- VAD initialized (look for "✅ VAD initialized")
- Audio stream is being received from client
- Correct sample rate (48kHz from client, resampled to 16kHz for VAD)

**Debug Commands:**
```bash
# Check VAD is working
grep "👂 User started speaking" logs/agent.log
grep "🤔 User stopped speaking" logs/agent.log
```

### Issue: State Not Transitioning

**Check:**
- Look for "🔄 State transition" logs
- Check if Gemini session is connected
- Verify audio is being sent to Gemini

**Debug Commands:**
```bash
# Check state transitions
grep "🔄 State transition" logs/agent.log

# Check Gemini connection
grep "✅ Gemini Live session started" logs/agent.log
```

### Issue: Overlapping Speech

**Check:**
- `can_accept_user_audio()` is blocking correctly
- State transitions happening properly
- VAD events being processed

### Issue: Frame Drops

**Check:**
- CPU/GPU utilization
- Frame pacing in `_on_frame_generated`
- Silent audio generator not running too fast

## Performance Metrics

Track these metrics during testing:

- **Frame Rate**: Target 25 FPS, acceptable 23-25 FPS
- **Frame Drops**: Target < 5%
- **Latency**: User speech end → AI response start
- **State Transition Time**: Time between state changes
- **VAD Accuracy**: False positives/negatives

## Expected Logs

Normal operation should show:

```
[Initialization]
🎭 Initializing Ditto SDK...
✅ Ditto SDK initialized
🎤 Initializing VAD...
✅ VAD initialized
🤖 Initializing Gemini Live API...
✅ Gemini client initialized

[Startup]
✅ Video track published
✅ Audio track published
✅ Background tasks started
🔇 Starting silent audio generator
✅ Agent ready

[Conversation - User Turn]
👂 User started speaking
🔄 State transition: idle → listening
🤔 User stopped speaking
🔄 State transition: listening → thinking

[Conversation - AI Turn]
🔄 State transition: thinking → speaking
💬 Gemini: [response]
🎤 Processed X Gemini audio chunks
🔄 State transition: speaking → idle

[Continuous Operation]
📊 Frames: 100 sent, 2 dropped
📊 Frames: 200 sent, 5 dropped
```

## Success Criteria

✅ Avatar visible 100% of the time (25 FPS)
✅ Turn-based conversation works smoothly
✅ No overlapping speech
✅ VAD correctly detects speech start/stop
✅ State transitions happen correctly
✅ Lip-sync accurate during AI speech
✅ Idle animation during silence
✅ No memory leaks over extended operation
✅ < 5% frame drop rate
✅ Latency < 2 seconds for typical queries

## Known Limitations

1. **No Interruption Support**: User cannot interrupt AI mid-response (by design)
2. **VAD Sensitivity**: May need tuning for different environments
3. **Frame Drops**: Some drops expected on lower-end hardware
4. **Gemini Session Reconnection**: Brief gaps during reconnection

## Next Steps

After basic testing passes:

1. **Tune VAD Parameters**: Adjust sensitivity for production environment
2. **Add Interruption Support**: Allow user to interrupt AI if needed
3. **Optimize Frame Rate**: Reduce drops on lower-end hardware
4. **Add Emotion Detection**: Use Gemini response analysis for expression control
5. **Add Activity Indicators**: Visual feedback for state transitions
6. **Performance Profiling**: Identify bottlenecks for optimization
