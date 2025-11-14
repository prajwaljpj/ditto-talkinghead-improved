#!/bin/bash
# Start LiveKit + Gemini + Custom Talking Head Agent
#
# This script configures environment variables needed by main_agent.py
# and custom_avatar_worker.py, then starts the main agent process.

# --- CHECK FOR REQUIRED CREDENTIALS ---
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "❌ ERROR: GOOGLE_APPLICATION_CREDENTIALS must be set for Vertex AI."
    echo ""
    echo "Please set:"
    echo "  export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/service-account.json\""
    echo "  export VERTEX_PROJECT_ID=\"your-gcp-project-id\""
    echo "  export VERTEX_LOCATION=\"us-central1\""
    echo ""
    echo "Then run: ./start_main_agent.sh"
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

# --- 1. LIVEKIT CONFIGURATION ---
# Default to local development server credentials if not set
export LIVEKIT_URL=${LIVEKIT_URL:-ws://localhost:7880}
export LIVEKIT_API_KEY=${LIVEKIT_API_KEY:-devkey}
export LIVEKIT_API_SECRET=${LIVEKIT_API_SECRET:-devsecret}

# --- 2. VERTEX AI CONFIGURATION ---
# Map the script's variables to the names used in main_agent.py
export GCP_PROJECT_ID=${VERTEX_PROJECT_ID:-$GCP_PROJECT_ID}
export GCP_REGION=${VERTEX_LOCATION:-us-central1}

# Set the specific Gemini model you are using
export GEMINI_MODEL=${GEMINI_MODEL:-gemini-live-2.5-flash-preview-native-audio-09-2025}

# --- 3. CUSTOM TALKING HEAD (DITTO) CONFIGURATION ---
# Map the script's variables to the names used in main_agent.py
export DATA_ROOT=${DITTO_DATA_ROOT:-checkpoints/ditto_trt_custom2/}
export CFG_PKL=${DITTO_CFG_PKL:-checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl}
export SOURCE_PATH=${DITTO_SOURCE:-avatars/Hyperrealistic_Indian_Woman_Professional_Presentation_static_5_loop.jpg}

echo "==================================================="
echo "Starting LiveKit + Vertex AI + Custom Avatar Agent"
echo "==================================================="
echo "Architecture: Two-Worker System"
echo "  1. Agent Worker: Conversation (STT, LLM, TTS)"
echo "  2. Avatar Worker: Video Generation (Ditto)"
echo "  Audio sent via DataStream (agent → avatar)"
echo "==================================================="
echo "LiveKit Server:  $LIVEKIT_URL"
echo ""
echo "Avatar Config:"
echo "  Source Path:   $SOURCE_PATH"
echo "  Data Root:     $DATA_ROOT"
echo ""
echo "Vertex AI Config:"
echo "  Model:         $GEMINI_MODEL"
echo "  Project ID:    $GCP_PROJECT_ID"
echo "  Region:        $GCP_REGION"
echo "==================================================="
echo ""

# --- 4. CHECK DEPENDENCIES (using your script's style) ---
# Check for scipy
if ! uv run python -c "import scipy.signal" 2>/dev/null; then
    echo "⚠️  Installing scipy for audio resampling..."
    uv add scipy
fi

# Check for google-genai
# Note: The 'google' LiveKit plugin relies on google-genai or google-cloud-aiplatform.
# We'll use the check for google-genai as per the original script.
if ! uv run python -c "import google.genai" 2>/dev/null; then
    echo "⚠️  Installing google-genai..."
    uv add google-genai
fi

# --- 5. RUN THE AGENT WORKER ---
echo "Starting agent worker..."
echo "(Avatar worker will be launched automatically as subprocess)"
echo ""
# Execute the agent worker (which launches avatar worker)
uv run python livekit_avatar/agent_worker.py dev
