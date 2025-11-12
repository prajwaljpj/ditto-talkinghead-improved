# Conversational Avatar with Gemini Live API - Quick Start

## What You Get

A real-time conversational avatar that listens, thinks, and responds naturally with lip-synced video.

## Setup (5 minutes)

### 1. Get Gemini API Key

Visit: https://aistudio.google.com/apikey

### 2. Start Services

**Terminal 1 - LiveKit Server:**
```bash
docker run --rm \
  -p 7880:7880 \
  -p 7881:7881 \
  -p 7882:7882/udp \
  -e LIVEKIT_KEYS="devkey: devsecret" \
  livekit/livekit-server:latest
```

**Terminal 2 - Gemini Agent:**
```bash
export GEMINI_API_KEY="your-api-key-here"
./start_gemini_agent.sh
```

**Terminal 3 - Web Client:**
```bash
cd webrtc/client/livekit
uv run python token_server.py
```

### 3. Open Browser

http://localhost:8000/index_simple.html

1. Click "Connect"
2. Allow microphone
3. Start speaking!
4. Avatar will respond

## What's Different from Basic Agent?

| Feature | Basic Agent | Gemini Agent |
|---------|-------------|--------------|
| Your speech animates avatar | ✅ | ✅ |
| Avatar speaks back | ❌ | ✅ |
| AI conversation | ❌ | ✅ |
| Response time | Instant | 0.8-1.0s |

## Configuration

### Change Voice

```bash
export GEMINI_VOICE=Charon  # Deep voice
# Or: Puck, Kore, Fenrir, Aoede
./start_gemini_agent.sh
```

### Change Personality

```bash
export GEMINI_INSTRUCTION="You are a friendly teacher who explains concepts simply."
./start_gemini_agent.sh
```

### Change Avatar

```bash
export DITTO_SOURCE=avatars/your_avatar.jpg
./start_gemini_agent.sh
```

## Troubleshooting

**No audio response?**
- Check agent logs for "🎤 Processed N Gemini audio chunks"
- Verify `GEMINI_API_KEY` is set correctly

**Slow responses?**
```bash
export GEMINI_MODEL=gemini-live-2.5-flash-preview  # Faster
export DITTO_MAX_SIZE=1280  # Smaller video = faster
```

**Video jittery?**
- Already fixed with frame pacing
- If still jittery, reduce `DITTO_MAX_SIZE`

## How It Works

```
You speak → Gemini (ASR + LLM + TTS) → Ditto (video) → You see & hear response
```

Total latency: 800-1000ms (natural conversation speed!)

## Next Steps

See `GEMINI_SETUP.md` for:
- Advanced configuration
- Production deployment
- Cost estimation
- Emotion mapping

**Enjoy your conversational avatar!** 🎭
