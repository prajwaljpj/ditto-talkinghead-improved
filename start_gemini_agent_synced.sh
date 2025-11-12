#!/bin/bash

# LiveKit connection (local dev server)
export LIVEKIT_URL=ws://localhost:7880
export LIVEKIT_API_KEY=devkey
export LIVEKIT_API_SECRET=devsecret

# Gemini authentication (choose one)
# Option 1: API Key
# export GEMINI_API_KEY=your-api-key

# Option 2: Vertex AI (recommended for production)
export GOOGLE_APPLICATION_CREDENTIALS="gnani-video-ai-c3b9b902d4d8.json"
export VERTEX_PROJECT_ID="gnani-video-ai"
export VERTEX_LOCATION="us-central1"

# Ditto configuration
export DITTO_CFG_PKL="checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
export DITTO_DATA_ROOT="checkpoints/ditto_trt_custom2/"
export DITTO_SOURCE="avatars/Hyperrealistic_Indian_Woman_Professional_Presentation_static_5_loop.jpg"
export DITTO_MAX_SIZE=1920
export DITTO_EMO=4

# Gemini configuration
export GEMINI_MODEL="gemini-live-2.5-flash-preview-native-audio-09-2025"
export GEMINI_VOICE="Puck"
export GEMINI_INSTRUCTION="You are a helpful AI assistant. Keep responses concise and natural."

# Profiling (optional)
export ENABLE_PROFILING=false
# export WEBSOCKETS_LOG_LEVEL=INFO
export WEBSOCKETS_LOG_LEVEL=ERROR

# Run synced agent
echo "🚀 Starting SYNCED Gemini agent (audio-video synchronized + proactive transitions)..."
echo ""
echo "✨ IMPROVEMENTS:"
echo "   - Audio-video sync: <50ms (was 605ms)"
echo "   - No frame gaps on transitions (was 0-200ms)"
echo "   - Complete audio processing (no loss at speech end)"
echo "   - Proactive state transitions (300ms timeout)"
echo ""
python webrtc/livekit_gemini_agent_synced.py dev
