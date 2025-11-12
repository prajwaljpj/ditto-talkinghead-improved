# Vertex AI Cascade Agent (STT->LLM->TTS) Setup

This setup uses separate Vertex AI services for each component, allowing you to measure latency at each step and optimize accordingly.

## Architecture

```
User Speech → Google STT → Text
                              ↓
                          Gemini LLM → Response Text
                                          ↓
                                   Google TTS → Audio
                                                  ↓
                                              Ditto → Video + Audio → Browser
```

## Latency Measurement

The agent tracks latency for each step:
- **STT Latency**: Time to transcribe speech to text
- **LLM Latency**: Time for Gemini to generate response
- **TTS Latency**: Time to synthesize speech from text
- **Total Latency**: End-to-end response time

## Setup

### 1. Prerequisites

Make sure LiveKit server is running:
```bash
docker run --rm \
  -p 7880:7880 \
  -p 7881:7881 \
  -p 7882:7882/udp \
  -e LIVEKIT_KEYS="devkey: devsecret" \
  livekit/livekit-server:latest
```

### 2. Configure Vertex AI Credentials

```bash
export GOOGLE_APPLICATION_CREDENTIALS="gnani-video-ai-c3b9b902d4d8.json"
export VERTEX_PROJECT_ID="gnani-video-ai"
export VERTEX_LOCATION="us-central1"  # optional
```

### 3. Start the Cascade Agent

```bash
./start_vertex_cascade_agent.sh
```

### 4. Open Web Client

In another terminal:
```bash
cd webrtc/client/livekit
python3 token_server.py  # or use existing server on port 8000
```

Open browser: http://localhost:8000/index_simple.html

### 5. Connect and Test

1. Click "Connect"
2. Allow microphone access
3. Speak to the avatar
4. Watch the console for latency metrics!

## Configuration Options

### Change TTS Voice

```bash
# Female voices
export TTS_VOICE="en-US-Neural2-F"  # Warm, professional (default)
export TTS_VOICE="en-US-Neural2-C"  # Young, energetic

# Male voices
export TTS_VOICE="en-US-Neural2-A"  # Deep, authoritative
export TTS_VOICE="en-US-Neural2-D"  # Young, friendly
export TTS_VOICE="en-US-Neural2-J"  # Casual, conversational

./start_vertex_cascade_agent.sh
```

### Change LLM Model

```bash
export GEMINI_MODEL="gemini-2.0-flash-exp"  # Faster
./start_vertex_cascade_agent.sh
```

### Change System Instruction

```bash
export SYSTEM_INSTRUCTION="You are a friendly teacher who explains concepts simply and concisely."
./start_vertex_cascade_agent.sh
```

### Optimize for Lower Latency

```bash
# Smaller video resolution = faster rendering
export DITTO_MAX_SIZE=1280

# Use faster model if available
export GEMINI_MODEL="gemini-2.0-flash-exp"

./start_vertex_cascade_agent.sh
```

## Expected Latency

Based on typical performance:

| Component | Expected Latency | Notes |
|-----------|------------------|-------|
| **STT** | 200-500ms | Depends on speech duration |
| **LLM** | 500-1500ms | Depends on response length |
| **TTS** | 200-400ms | Depends on text length |
| **Ditto** | ~200ms | Per chunk, parallel with TTS |
| **TOTAL** | 1000-2500ms | Acceptable for conversational AI |

## Viewing Statistics

The agent prints statistics when it closes or periodically during operation:

```
📊 Performance Statistics
═══════════════════════════════════════
STT: 10 calls, avg 350ms
LLM: 10 calls, avg 1200ms
TTS: 10 calls, avg 300ms
TOTAL: avg 1850ms, min 1200ms, max 2500ms
═══════════════════════════════════════
```

## Comparing with Gemini Live API

| Approach | Latency | Pros | Cons |
|----------|---------|------|------|
| **Cascade** (this) | 1-2.5s | Flexible, easy to debug, swap components | Higher latency, more API calls |
| **Gemini Live** | 0.8-1s | Lower latency, streaming | Single vendor, less flexibility |

## Troubleshooting

### High STT Latency
- Audio chunks are too long
- Reduce `min_audio_duration` in code (currently 2s)

### High LLM Latency
- Response is too long
- Adjust system instruction to request shorter responses
- Try: "Keep all responses under 2 sentences."

### High TTS Latency
- Text is too long
- Instruct LLM to be more concise
- Consider streaming TTS (not implemented yet)

### Missing Libraries
The script will auto-install:
- `google-cloud-speech`
- `google-cloud-texttospeech`
- `google-genai`
- `scipy`

## Next Steps

If latency is acceptable:
1. ✅ Use this cascade approach
2. Consider optimizations:
   - Streaming STT (real-time transcription)
   - Streaming TTS (start playback before full synthesis)
   - Parallel processing where possible

If latency is too high:
1. Try the Gemini Live API agent instead (`start_gemini_agent.sh`)
2. Or optimize each component:
   - Use shorter audio chunks
   - Request shorter LLM responses
   - Use faster TTS voices

## API Costs (Approximate)

Per conversation turn:
- **STT**: $0.006 per minute of audio
- **Gemini 2.0 Flash**: $0.000075 per 1K characters
- **TTS**: $0.000016 per character (Neural2 voices)

Example: 10-turn conversation (~5 min audio, 5K chars)
- STT: $0.03
- LLM: $0.00038
- TTS: $0.08
- **Total: ~$0.11**

Much cheaper than using a proprietary all-in-one solution!

## Resources

- [Google STT Documentation](https://cloud.google.com/speech-to-text)
- [Google TTS Documentation](https://cloud.google.com/text-to-speech)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [LiveKit Documentation](https://docs.livekit.io)
