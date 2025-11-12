#!/bin/bash
# Start LiveKit + Gemini + Ditto Conversational Avatar Agent
#
# Usage:
#   export GEMINI_API_KEY="your-api-key"
#   ./start_gemini_agent.sh
#
# First, make sure LiveKit server is running:
#   docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
#     -e LIVEKIT_KEYS="devkey: devsecret" \
#     livekit/livekit-server:latest

# Check for Gemini API key OR Vertex AI credentials
if [ -z "$GEMINI_API_KEY" ] && [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "❌ ERROR: Either GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS must be set"
    echo ""
    echo "Option 1 - Gemini API Key:"
    echo "  Get your API key from: https://aistudio.google.com/apikey"
    echo "  export GEMINI_API_KEY=\"your-api-key\""
    echo ""
    echo "Option 2 - Vertex AI (Recommended for production):"
    echo "  export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/service-account.json\""
    echo "  export VERTEX_PROJECT_ID=\"your-gcp-project-id\""
    echo "  export VERTEX_LOCATION=\"us-central1\"  # optional, default: us-central1"
    echo ""
    echo "Then run: ./start_gemini_agent.sh"
    exit 1
fi

# Display which auth method is being used
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Using Vertex AI authentication"
    echo "Credentials: $GOOGLE_APPLICATION_CREDENTIALS"
    if [ -z "$VERTEX_PROJECT_ID" ]; then
        echo "⚠️  WARNING: VERTEX_PROJECT_ID not set, agent may fail"
    fi
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

# Gemini configuration (with defaults)
export GEMINI_MODEL=${GEMINI_MODEL:-gemini-live-2.5-flash-preview-native-audio-09-2025}
export GEMINI_VOICE=${GEMINI_VOICE:-Kore}
export GEMINI_INSTRUCTION=${GEMINI_INSTRUCTION:-"You are a helpful AI assistant. Keep responses concise and natural, as this is a voice conversation."}

echo "==================================================="
echo "Starting LiveKit + Gemini + Ditto Avatar Agent"
echo "==================================================="
echo "LiveKit Server:  $LIVEKIT_URL"
echo ""
echo "Ditto Config:"
echo "  Source:        $DITTO_SOURCE"
echo "  Max Size:      $DITTO_MAX_SIZE"
echo "  Emotion:       $DITTO_EMO (4=neutral)"
echo ""
echo "Gemini Config:"
echo "  Model:         $GEMINI_MODEL"
echo "  Voice:         $GEMINI_VOICE"
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "  Auth:          Vertex AI (${GOOGLE_APPLICATION_CREDENTIALS})"
    echo "  Project ID:    ${VERTEX_PROJECT_ID:-NOT SET ⚠️}"
    echo "  Location:      ${VERTEX_LOCATION:-us-central1}"
else
    echo "  Auth:          API Key (${GEMINI_API_KEY:0:10}...)"
fi
echo "==================================================="
echo ""
echo "Available Gemini voices:"
echo "  Puck     - Conversational, natural"
echo "  Charon   - Deep, authoritative"
echo "  Kore     - Warm, friendly"
echo "  Fenrir   - Strong, confident"
echo "  Aoede    - Melodic, pleasant"
echo ""
echo "To change voice: export GEMINI_VOICE=Charon"
echo "==================================================="
echo ""

# Check if scipy is installed (needed for audio resampling)
if ! uv run python -c "import scipy.signal" 2>/dev/null; then
    echo "⚠️  Installing scipy for audio resampling..."
    uv add scipy
fi

# Check if google-genai is installed
if ! uv run python -c "import google.genai" 2>/dev/null; then
    echo "⚠️  Installing google-genai..."
    uv add google-genai
fi

# Run agent with LiveKit CLI
echo "Starting agent..."
echo ""
uv run python webrtc/livekit_gemini_agent.py dev
