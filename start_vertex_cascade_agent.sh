#!/bin/bash
# Start LiveKit + Vertex AI Cascade (STT->LLM->TTS) + Ditto Avatar Agent
#
# This uses separate Vertex AI services for each step:
# - Speech-to-Text for transcription
# - Gemini for LLM responses
# - Text-to-Speech for audio generation
#
# Usage:
#   export GOOGLE_APPLICATION_CREDENTIALS="gnani-video-ai-c3b9b902d4d8.json"
#   export VERTEX_PROJECT_ID="gnani-video-ai"
#   ./start_vertex_cascade_agent.sh

# Check for Vertex AI credentials
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "❌ ERROR: GOOGLE_APPLICATION_CREDENTIALS must be set"
    echo ""
    echo "Example:"
    echo "  export GOOGLE_APPLICATION_CREDENTIALS=\"gnani-video-ai-c3b9b902d4d8.json\""
    echo "  export VERTEX_PROJECT_ID=\"gnani-video-ai\""
    echo "  ./start_vertex_cascade_agent.sh"
    exit 1
fi

if [ -z "$VERTEX_PROJECT_ID" ]; then
    echo "❌ ERROR: VERTEX_PROJECT_ID must be set"
    echo ""
    echo "Example:"
    echo "  export VERTEX_PROJECT_ID=\"gnani-video-ai\""
    exit 1
fi

# LiveKit configuration
export LIVEKIT_URL=${LIVEKIT_URL:-ws://localhost:7880}
export LIVEKIT_API_KEY=${LIVEKIT_API_KEY:-devkey}
export LIVEKIT_API_SECRET=${LIVEKIT_API_SECRET:-devsecret}

# Ditto configuration (with defaults)
export DITTO_CFG_PKL=${DITTO_CFG_PKL:-checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl}
export DITTO_DATA_ROOT=${DITTO_DATA_ROOT:-checkpoints/ditto_trt_custom2/}
export DITTO_SOURCE=${DITTO_SOURCE:-avatars/Hyperrealistic_Indian_Woman_Professional_Presentation_static_5_loop.jpg}
export DITTO_MAX_SIZE=${DITTO_MAX_SIZE:-1920}
export DITTO_EMO=${DITTO_EMO:-4}

# Vertex AI configuration (with defaults)
export VERTEX_LOCATION=${VERTEX_LOCATION:-us-central1}
export GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.0-flash-exp}
export TTS_VOICE=${TTS_VOICE:-en-US-Neural2-F}
export SYSTEM_INSTRUCTION=${SYSTEM_INSTRUCTION:-"You are a helpful AI assistant. Keep responses concise and natural, as this is a voice conversation."}

echo "==================================================="
echo "Starting LiveKit + Vertex AI Cascade + Ditto Agent"
echo "==================================================="
echo "LiveKit Server:  $LIVEKIT_URL"
echo ""
echo "Vertex AI Config:"
echo "  Project ID:    $VERTEX_PROJECT_ID"
echo "  Location:      $VERTEX_LOCATION"
echo "  Credentials:   $GOOGLE_APPLICATION_CREDENTIALS"
echo ""
echo "Pipeline:"
echo "  STT:           Google Speech-to-Text (streaming)"
echo "  LLM:           $GEMINI_MODEL"
echo "  TTS:           Google Text-to-Speech ($TTS_VOICE)"
echo ""
echo "Ditto Config:"
echo "  Source:        $DITTO_SOURCE"
echo "  Max Size:      $DITTO_MAX_SIZE"
echo "  Emotion:       $DITTO_EMO (4=neutral)"
echo "==================================================="
echo ""
echo "Available TTS voices:"
echo "  en-US-Neural2-F  - Female, warm"
echo "  en-US-Neural2-C  - Female, young"
echo "  en-US-Neural2-A  - Male, deep"
echo "  en-US-Neural2-D  - Male, young"
echo "  en-US-Neural2-J  - Male, casual"
echo ""
echo "To change voice: export TTS_VOICE=en-US-Neural2-C"
echo "==================================================="
echo ""

# Check if scipy is installed (needed for audio resampling)
if ! uv run python -c "import scipy.signal" 2>/dev/null; then
    echo "⚠️  Installing scipy for audio resampling..."
    uv add scipy
fi

# Check if Google Cloud libraries are installed
if ! uv run python -c "import google.cloud.speech" 2>/dev/null; then
    echo "⚠️  Installing google-cloud-speech..."
    uv add google-cloud-speech
fi

if ! uv run python -c "import google.cloud.texttospeech" 2>/dev/null; then
    echo "⚠️  Installing google-cloud-texttospeech..."
    uv add google-cloud-texttospeech
fi

if ! uv run python -c "import google.genai" 2>/dev/null; then
    echo "⚠️  Installing google-genai..."
    uv add google-genai
fi

# Run agent with LiveKit CLI
echo "Starting agent..."
echo ""
echo "💡 This agent will measure latency at each step:"
echo "   - STT: Speech-to-Text transcription time"
echo "   - LLM: Gemini response generation time"
echo "   - TTS: Text-to-Speech synthesis time"
echo "   - TOTAL: End-to-end latency"
echo ""
uv run python webrtc/livekit_vertex_cascade_agent.py dev
