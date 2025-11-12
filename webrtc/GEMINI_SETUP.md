# Gemini + Ditto Conversational Avatar Setup

This guide explains how to set up and run the Gemini Live API integration with Ditto talking head for full-duplex conversational AI avatar.

## Features

✨ **Full-Duplex Conversation**: Natural, interruptible conversations with the avatar
🎭 **Emotion-Aware Expressions**: Avatar facial expressions match conversation sentiment
💬 **Conversation Context**: Multi-turn context tracking for coherent conversations
🎙️ **ASR + LLM + TTS**: Single Gemini API call handles speech recognition, response generation, and text-to-speech
🎨 **Beautiful Web UI**: Real-time transcript, audio levels, and emotion indicators

## Architecture

> 📊 **For detailed flow diagrams and visual explanations, see [ARCHITECTURE_FLOW.md](./ARCHITECTURE_FLOW.md)**
>
> This includes:
> - Mermaid sequence diagrams
> - State machine diagrams
> - Audio flow decision trees
> - Frame-by-frame processing flows

### Quick Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser (Client)                         │
│                                                                  │
│  Microphone Input ──────────────► [WebRTC Audio Out]            │
│                                                                  │
│  [WebRTC Video In] ◄──────────── Video Display                  │
└─────────────────────────────────────────────────────────────────┘
                            │                    ▲
                            │ User Audio         │ Avatar Video
                            ▼                    │
┌─────────────────────────────────────────────────────────────────┐
│                    GeminiProcessor                               │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Process User Audio:                                       │  │
│  │  • Send to Gemini API (ASR + LLM)                        │  │
│  │  • Detect if user is speaking (VAD)                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Audio Output Decision:                                    │  │
│  │                                                            │  │
│  │  IF Avatar is Speaking:                                   │  │
│  │    └──► Push Gemini TTS Audio ──────┐                    │  │
│  │                                       │                    │  │
│  │  ELSE (User speaking or silence):     │                    │  │
│  │    └──► Push SILENCE frames ─────────┤                    │  │
│  │                                       │                    │  │
│  │  Result: Ditto ALWAYS receives audio │                    │  │
│  └───────────────────────────────────────┼────────────────────┘  │
└───────────────────────────────────────────┼──────────────────────┘
                                            │ Audio (TTS or Silence)
                                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DittoProcessor                                │
│                                                                  │
│  Audio Input ──► StreamSDK (Audio2Motion → MotionStitch →       │
│                              Warp3D → Decode → PutBack)          │
│                                                                  │
│  Emotion Updates ──► update_emotion() ──► Facial Expression     │
│                                                                  │
│  Output: Video Frames (avatar lip-synced to audio OR idle)      │
└─────────────────────────────────────────────────────────────────┘
                                            │ Video Frames
                                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                IdleAnimationProcessor                            │
│  Adds subtle movements when receiving silence                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- ✅ User audio → Gemini (for conversation)
- ✅ Gemini TTS audio → Ditto (avatar speaks)
- ✅ Silence → Ditto (when user speaks or idle - keeps avatar animated)
- ✅ Ditto NEVER freezes (always receives audio)

## Prerequisites

1. **Python Environment**: Python 3.10 (as specified in project)
2. **Gemini API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
3. **Ditto Models**: Already set up (cfg_pkl and data_root)
4. **Avatar Source**: Image or video of the person/character

## Installation

### 1. Install Dependencies

Dependencies were already added during setup. If needed, run:

```bash
uv sync
```

This installs:
- `google-genai>=0.2.0` - Gemini Live API SDK
- `protobuf>=4.0.0` - Protocol buffers for Gemini

### 2. Set Gemini API Key

Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```bash
echo "GEMINI_API_KEY=your-api-key-here" > .env
```

## Usage

### Option 1: Test Pipeline (No WebRTC)

Test the Gemini + Ditto pipeline without WebRTC:

```bash
python webrtc/gemini_server.py \
  --cfg_pkl outputs/cfg_f_model.pkl \
  --data_root ./ \
  --source examples/avatar.jpg
```

This runs the Pipecat pipeline but doesn't create WebRTC connections. Good for testing Gemini integration.

### Option 2: Full WebRTC Server (Recommended)

For actual browser-based conversations, you need to integrate with the signaling server.

**Step 1: Start the Signaling Server**

The existing signaling_server.py handles WebRTC connections. We need to modify it to use the Gemini pipeline.

```bash
# For now, use the test mode above
# Full integration requires modifying signaling_server.py
```

**Future Enhancement**: Modify `webrtc/signaling_server.py` to instantiate `GeminiLiveProcessor` instead of directly processing audio.

### Option 3: Using the Web UI

1. Open `webrtc/client/web/gemini_chat.html` in a browser
2. Click "Connect"
3. Allow microphone access
4. Start talking! The avatar will respond

**Default Settings:**
- Server: `ws://localhost:8080`
- Interruptions: Enabled
- Emotion Sensitivity: Medium

## Configuration

### Gemini Options

```bash
python webrtc/gemini_server.py \
  --cfg_pkl outputs/cfg_f_model.pkl \
  --data_root ./ \
  --source examples/avatar.jpg \
  --model "models/gemini-2.0-flash-exp" \  # Gemini model
  --voice "Aoede" \                         # Voice (Puck, Charon, Kore, Fenrir, Aoede)
  --no-interruptions                        # Disable interruptions (turn-based)
```

### Ditto Options

```bash
python webrtc/gemini_server.py \
  --cfg_pkl outputs/cfg_f_model.pkl \
  --data_root ./ \
  --source examples/avatar.jpg \
  --emo 0 \           # Initial emotion (0=happy, 4=neutral)
  --max_size 1920 \   # Max image dimension
  --crop_scale 2.3    # Face crop scale
```

### Emotion Codes

The system automatically maps detected emotions to Ditto codes:

| Emotion | Ditto Code | Number |
|---------|------------|--------|
| Happy | hap | 0 |
| Angry | ang | 1 |
| Sad | sad | 2 |
| Fear | - | 3 |
| Neutral | neu | 4 |
| Surprised | sur | 5 |
| Disgusted | - | 6 |
| Contemptuous | - | 7 |

## Project Structure

```
webrtc/
├── gemini_server.py              # Main server with Gemini integration
├── conversation_manager.py        # Conversation context & emotion tracking
├── processors/
│   ├── gemini_processor.py       # Gemini Live API Pipecat processor
│   ├── ditto_processor.py        # Ditto avatar processor (updated)
│   ├── idle_animator.py          # Idle animation
│   └── h264_encoder.py           # Video encoding
├── client/web/
│   ├── gemini_chat.html          # Enhanced UI
│   └── gemini_chat.js            # WebRTC client logic
└── GEMINI_SETUP.md               # This file
```

## How It Works

### 1. GeminiLiveProcessor

Located in `webrtc/processors/gemini_processor.py`

**Key Features:**
- Bidirectional audio streaming with Gemini API
- Automatic ASR (user speech → text)
- LLM response generation with context
- TTS (text → audio) in real-time
- Emotion detection from responses
- Full-duplex support with VAD-based interruption

**Audio Flow:**
1. Browser captures microphone (48kHz)
2. Downsampled to 16kHz for Gemini
3. Sent to Gemini Live API
4. **Silence frames** sent to DittoProcessor (keeps avatar animated while listening)
5. Gemini returns audio response (16kHz)
6. **Gemini's TTS audio** forwarded to DittoProcessor (avatar lip-syncs to Gemini's speech)

**Three Audio States:**
- **User Speaking + Avatar Silent**: Silence → Ditto (avatar in listening/idle pose)
- **Avatar Speaking**: Gemini TTS audio → Ditto (avatar lip-syncs to Gemini)
- **Both Silent**: Silence → Ditto (avatar in idle pose, handled by IdleAnimationProcessor)

### 2. ConversationManager

Located in `webrtc/conversation_manager.py`

**Key Features:**
- Multi-turn conversation history (up to 20 turns)
- Sliding context window (10 recent turns)
- Emotion tracking and smoothing
- Session timeout (5 minutes inactivity)
- Simple keyword-based emotion detection

**Emotion Detection:**
- Analyzes Gemini response text
- Uses keyword matching (e.g., "happy", "sad", "angry")
- Can be extended with proper sentiment analysis models

### 3. DittoAvatarProcessor (Updated)

Located in `webrtc/processors/ditto_processor.py`

**New Feature:**
- `update_emotion(emotion_code)` method for dynamic emotion updates
- Maps emotion strings to Ditto codes
- Updates SDK emotion in real-time

### 4. Enhanced Web UI

Located in `webrtc/client/web/gemini_chat.html` and `gemini_chat.js`

**Features:**
- Live video stream from avatar
- Real-time conversation transcript
- Audio level indicators (microphone + avatar)
- Emotion badge showing current expression
- Session statistics (turns, duration, latency, interruptions)
- Settings panel (server URL, interruption threshold, emotion sensitivity)
- Connection controls (connect, disconnect, reset, mute)

## Troubleshooting

### Gemini API Issues

**Error: "API key not set"**
```bash
export GEMINI_API_KEY="your-key"
```

**Error: "Model not found"**
- Ensure you're using `models/gemini-2.0-flash-exp`
- Check API access at [Google AI Studio](https://makersuite.google.com/)

**High latency (>1s)**
- Gemini API adds ~200-500ms naturally
- Check your internet connection
- Consider using faster model if available

### Audio Issues

**No audio captured**
- Check browser microphone permissions
- Ensure HTTPS or localhost (WebRTC requirement)
- Try different browser (Chrome/Firefox recommended)

**Audio choppy/garbled**
- Check CPU usage (Ditto is computationally intensive)
- Reduce `max_size` parameter
- Use TensorRT acceleration (if available)

**Echo/feedback**
- Enable echo cancellation in browser settings
- Use headphones
- Check `echoCancellation: true` in getUserMedia

### Video Issues

**No video stream**
- Check WebRTC connection status
- Verify avatar source image exists
- Check console for errors

**Low FPS (<20fps)**
- Normal for CPU-only Ditto
- Use TensorRT for better performance
- Reduce video resolution

### Connection Issues

**WebSocket connection fails**
- Ensure server is running
- Check firewall settings
- Verify port 8080 is available
- Try `ws://127.0.0.1:8080` instead of `ws://localhost:8080`

**WebRTC negotiation fails**
- Check STUN/TURN server configuration
- Verify ICE candidates are exchanged
- Check browser console for ICE errors

## Performance Tips

1. **Use TensorRT**: Significantly faster inference (if available)
2. **Reduce Resolution**: Lower `max_size` for faster processing
3. **Adjust Chunk Size**: Smaller chunks = lower latency, higher CPU
4. **Local Deployment**: Run on same machine as browser for lowest latency
5. **GPU Acceleration**: Ensure CUDA is available for Ditto models

## API Costs

Gemini Live API pricing (as of 2024):
- Input audio: ~$0.02 per minute
- Output audio: ~$0.04 per minute
- Total: ~$0.06 per minute of conversation

**Cost Optimization:**
- Use shorter responses (configure in system prompt)
- Implement turn-based mode (disable interruptions)
- Set session timeouts to avoid idle API usage

## Next Steps

### 1. Integrate with Signaling Server

Modify `webrtc/signaling_server.py` to use `GeminiLiveProcessor`:

```python
from webrtc.processors.gemini_processor import GeminiLiveProcessor
from webrtc.processors.ditto_processor import DittoAvatarProcessor

# In on_track handler:
gemini = GeminiLiveProcessor(api_key=os.getenv("GEMINI_API_KEY"))
ditto = DittoAvatarProcessor(cfg_pkl, data_root, source_path)

# Process: user audio → gemini → ditto → video
```

### 2. Add Advanced Emotion Detection

Replace simple keyword matching with:
- Transformer-based sentiment analysis (e.g., DistilBERT)
- Emotion classification model (e.g., GoEmotions)
- Gemini's built-in sentiment extraction

### 3. Improve Interruption Handling

- Implement proper VAD (Voice Activity Detection)
- Add turn-taking prediction
- Buffer management for smoother transitions

### 4. Add Conversation Features

- Save/load conversation history
- Multiple avatar personalities (system prompts)
- Custom voices and speaking styles
- Multi-language support

### 5. Production Deployment

- HTTPS with valid certificates
- Proper STUN/TURN servers
- Load balancing for multiple users
- Session management and cleanup
- Monitoring and logging

## Example Commands

### Basic Conversation

```bash
# Start server
export GEMINI_API_KEY="your-key"
python webrtc/gemini_server.py \
  --cfg_pkl outputs/cfg_f_model.pkl \
  --data_root ./ \
  --source examples/avatar.jpg

# Open browser to webrtc/client/web/gemini_chat.html
# Click Connect and start talking!
```

### Custom Voice

```bash
python webrtc/gemini_server.py \
  --cfg_pkl outputs/cfg_f_model.pkl \
  --data_root ./ \
  --source examples/avatar.jpg \
  --voice Charon  # Deeper, more serious voice
```

### Turn-Based (No Interruptions)

```bash
python webrtc/gemini_server.py \
  --cfg_pkl outputs/cfg_f_model.pkl \
  --data_root ./ \
  --source examples/avatar.jpg \
  --no-interruptions
```

### Happy Avatar

```bash
python webrtc/gemini_server.py \
  --cfg_pkl outputs/cfg_f_model.pkl \
  --data_root ./ \
  --source examples/avatar.jpg \
  --emo 0  # Start with happy emotion
```

## Resources

- **Gemini Live API**: https://ai.google.dev/gemini-api/docs/live
- **Ditto Paper**: https://arxiv.org/abs/2406.01217
- **Pipecat Framework**: https://github.com/pipecat-ai/pipecat
- **WebRTC**: https://webrtc.org/

## License

Same as parent project (Ditto Talking Head).

## Support

For issues specific to Gemini integration:
1. Check this documentation
2. Review console logs (browser and server)
3. Verify Gemini API key and quota
4. Test with simple examples first

For Ditto-specific issues:
- See main project README.md
- Check webrtc/README.md

---

**Happy conversing! 🎉**
