# Gemini + Ditto Quick Reference Card

## 🚀 Quick Start

```bash
# 1. Set API key
export GEMINI_API_KEY="your-api-key-here"

# 2. Run server
python webrtc/gemini_server.py \
  --cfg_pkl outputs/cfg_f_model.pkl \
  --data_root ./ \
  --source examples/avatar.jpg

# 3. Open browser
open webrtc/client/web/gemini_chat.html
```

## 📋 Audio Flow Cheat Sheet

| State | User Audio | Audio to Gemini | Audio to Ditto | Avatar Behavior |
|-------|------------|-----------------|----------------|-----------------|
| **Idle** | None | ❌ None | ✅ Silence | Subtle idle movements |
| **User Speaking** | Active | ✅ User audio | ✅ Silence | Listening pose |
| **Avatar Speaking** | Any | ❌ None | ✅ Gemini TTS | Lip-synced speech |

## 🎭 Emotion Codes

| Emotion | String Code | Integer Code | Description |
|---------|-------------|--------------|-------------|
| Happy | `hap` | 0 | Joyful, excited |
| Angry | `ang` | 1 | Frustrated, upset |
| Sad | `sad` | 2 | Unhappy, disappointed |
| Fear | - | 3 | Scared, worried |
| Neutral | `neu` | 4 | Default, calm |
| Surprised | `sur` | 5 | Amazed, shocked |
| Disgusted | - | 6 | Repulsed |
| Contemptuous | - | 7 | Disdainful |

## 🎤 Gemini Voice Options

- `Puck` - Default, balanced voice
- `Charon` - Deep, serious voice
- `Kore` - Warm, friendly voice
- `Fenrir` - Strong, authoritative voice
- `Aoede` - Soft, melodic voice

## 🔧 Key Configuration Parameters

### Gemini Settings
```bash
--model "models/gemini-2.0-flash-exp"  # Model name
--voice "Puck"                          # TTS voice
--api_key "your-key"                    # API key (or env var)
--no-interruptions                      # Disable interruptions
```

### Ditto Settings
```bash
--emo 4                                 # Initial emotion (0-7)
--max_size 1920                         # Max image dimension
--crop_scale 2.3                        # Face crop scale
--cfg_pkl outputs/cfg_f_model.pkl      # Config file
--data_root ./                          # Model data root
--source examples/avatar.jpg            # Avatar image
```

### Server Settings
```bash
--host "0.0.0.0"                        # Server host
--port 8080                             # Server port
```

## 🐛 Common Issues & Fixes

| Issue | Cause | Solution |
|-------|-------|----------|
| Avatar freezes | No audio to Ditto | ✅ Fixed: Silence frames sent |
| No Gemini response | API key not set | `export GEMINI_API_KEY="..."` |
| Audio choppy | CPU overload | Reduce `max_size`, use TensorRT |
| Echo/feedback | No echo cancellation | Use headphones, enable browser EC |
| High latency | Network/API delay | Normal (200-500ms from Gemini) |
| Connection fails | Port blocked | Check firewall, try port 8081 |

## 📁 Key Files

| File | Purpose |
|------|---------|
| `gemini_processor.py` | Gemini API integration |
| `ditto_processor.py` | Talking head generation |
| `conversation_manager.py` | Context & emotion tracking |
| `gemini_server.py` | Main server |
| `gemini_chat.html` | Web UI |
| `GEMINI_SETUP.md` | Full documentation |
| `ARCHITECTURE_FLOW.md` | Visual diagrams |

## 🔄 Pipeline Flow (Simplified)

```
User Speaks
    ↓
GeminiProcessor
    ├─→ To Gemini API (ASR + LLM)
    └─→ Silence to Ditto
         ↓
    Idle Avatar Video
         ↓
Gemini Responds (TTS)
    ↓
GeminiProcessor
    └─→ TTS Audio to Ditto
         ↓
    Lip-Synced Avatar Video
```

## 🎯 Audio State Logic

```python
if avatar_is_speaking:
    # Push Gemini TTS audio to Ditto
    audio = gemini_tts_output
else:
    # Push silence to Ditto
    audio = b'\x00' * frame_length

ditto_processor.process(audio)  # Always has audio!
```

## 📊 Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| **Audio Latency** | < 100ms | 50-80ms |
| **Gemini API Latency** | < 500ms | 200-400ms |
| **Video FPS** | 25 fps | 20-25 fps |
| **Total E2E Latency** | < 800ms | 500-700ms |
| **CPU Usage (no TRT)** | - | 60-80% |
| **CPU Usage (with TRT)** | - | 30-50% |

## 🔐 Environment Variables

```bash
# Required
export GEMINI_API_KEY="your-gemini-api-key"

# Optional
export SERVER_HOST="0.0.0.0"
export SERVER_PORT="8080"
export GEMINI_MODEL="models/gemini-2.0-flash-exp"
export GEMINI_VOICE="Puck"
```

## 📝 Example Commands

### Basic Usage
```bash
python webrtc/gemini_server.py \
  --cfg_pkl outputs/cfg_f_model.pkl \
  --data_root ./ \
  --source examples/avatar.jpg
```

### With Custom Voice
```bash
python webrtc/gemini_server.py \
  --cfg_pkl outputs/cfg_f_model.pkl \
  --data_root ./ \
  --source examples/avatar.jpg \
  --voice Aoede
```

### Turn-Based Mode
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
  --emo 0
```

## 🧪 Testing Checklist

- [ ] Microphone access granted
- [ ] WebRTC connection established
- [ ] Video stream visible
- [ ] User can speak and see transcript
- [ ] Avatar responds with lip-sync
- [ ] Avatar shows idle animation when silent
- [ ] Avatar shows listening pose when user speaks
- [ ] Emotion changes visible
- [ ] Interruption works (if enabled)
- [ ] Audio levels show activity
- [ ] Stats update correctly

## 💡 Pro Tips

1. **Use headphones** to avoid echo/feedback
2. **Speak clearly** for better ASR accuracy
3. **Wait for response** in turn-based mode
4. **Reduce max_size** if FPS is low
5. **Use TensorRT** for best performance
6. **Check logs** if something fails
7. **Test with simple prompts** first
8. **Monitor CPU usage** during testing

## 📚 Documentation Links

- Full Setup: [GEMINI_SETUP.md](./GEMINI_SETUP.md)
- Architecture: [ARCHITECTURE_FLOW.md](./ARCHITECTURE_FLOW.md)
- WebRTC Guide: [README.md](./README.md)
- Testing Guide: [TESTING.md](./TESTING.md)

## 🆘 Getting Help

1. Check console logs (browser and server)
2. Verify API key and quota
3. Test with simple example first
4. Check [GEMINI_SETUP.md](./GEMINI_SETUP.md) troubleshooting section
5. Review [ARCHITECTURE_FLOW.md](./ARCHITECTURE_FLOW.md) for flow understanding

---

**Last Updated**: 2025-11-06
**Version**: 1.0
