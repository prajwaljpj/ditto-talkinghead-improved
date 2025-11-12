# Quick Start: Turn-Based Conversational Avatar

## Prerequisites

- LiveKit server running on localhost:7880
- Vertex AI credentials configured
- Ditto model checkpoints in place

## Setup (One-Time)

```bash
# Set Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="gnani-video-ai-c3b9b902d4d8.json"
export VERTEX_PROJECT_ID="gnani-video-ai"
export VERTEX_LOCATION="us-central1"

# Optional: Set custom Ditto/Gemini config
export DITTO_SOURCE="avatars/Hyperrealistic_Indian_Woman_Professional_Presentation_static_5_loop.jpg"
export GEMINI_VOICE="Puck"  # Options: Puck, Charon, Kore, Fenrir, Aoede
```

## Start LiveKit Server

Terminal 1:
```bash
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
  -e LIVEKIT_KEYS="devkey: devsecret" \
  livekit/livekit-server:latest
```

## Start Agent

Terminal 2:
```bash
./start_gemini_agent.sh
```

Wait for:
```
✅ Agent ready - speak to start conversation!
💡 Turn-based conversation enabled:
   - Avatar always visible with idle/listening animation
   - Speak to start (VAD detects your speech)
   - AI responds when you finish speaking
   - No overlapping speech - one speaker at a time
```

## Connect Client

Open browser:
```
http://localhost:8000/index_simple.html
```

## Usage

1. **Wait for avatar to appear** (idle animation with neutral expression)
2. **Start speaking** (VAD automatically detects when you start)
3. **Stop speaking** (VAD detects end, sends your audio to Gemini)
4. **Wait for response** (avatar lip-syncs AI's speech)
5. **Repeat** (avatar returns to idle, ready for next turn)

## State Indicators (in logs)

Watch the terminal for state transitions:

```
👂 User started speaking          # You started talking
🔄 State transition: idle → listening

🤔 User stopped speaking          # You finished talking
🔄 State transition: listening → thinking

🔄 State transition: thinking → speaking    # AI responding
💬 Gemini: [AI response text]

🔄 State transition: speaking → idle        # AI finished
```

## Troubleshooting

### Avatar not visible
Check:
```bash
grep "🔇 Starting silent audio generator" logs/agent.log
grep "🎬 First frame generated" logs/agent.log
```

### VAD not detecting speech
Increase microphone volume or check:
```bash
grep "👂 User started speaking" logs/agent.log
```

### Gemini not responding
Check authentication:
```bash
grep "✅ Gemini Live session started" logs/agent.log
grep "❌" logs/agent.log  # Look for errors
```

### Frame drops
Check frame statistics:
```bash
grep "📊 Frames:" logs/agent.log
```

## Configuration Options

### Change Avatar

```bash
export DITTO_SOURCE="path/to/your/avatar.jpg"
```

### Change Voice

```bash
export GEMINI_VOICE="Charon"  # Deep, authoritative voice
# Options: Puck (default), Charon, Kore, Fenrir, Aoede
```

### Change System Instruction

```bash
export GEMINI_INSTRUCTION="You are a professional customer service agent. Be polite and helpful."
```

## Testing

Run the comprehensive test suite:

```bash
# See TURN_BASED_TESTING.md for detailed test scenarios
```

## Architecture

For detailed architecture information, see:
- `TURN_BASED_ARCHITECTURE.md` - Complete technical architecture
- `TURN_BASED_TESTING.md` - Testing guide and scenarios

## Key Features

✅ **Always-Visible Avatar**: Continuous 25 FPS frame generation
✅ **Automatic Turn Detection**: VAD-based speech start/stop detection
✅ **No Overlapping Speech**: Clean turn-based conversation
✅ **Natural Lip-Sync**: Accurate lip movement during AI speech
✅ **Idle Animation**: Subtle animation when not speaking
✅ **Auto-Reconnect**: Resilient to network issues

## Next Steps

1. Try different avatars with `DITTO_SOURCE`
2. Experiment with different voices using `GEMINI_VOICE`
3. Customize system instructions for different use cases
4. Test in different network conditions
5. Monitor performance metrics (frame rate, latency)

## Support

For issues or questions, check:
- Logs in `logs/` directory
- `DEBUG_GUIDE.md` for debugging tips
- Architecture docs for technical details
