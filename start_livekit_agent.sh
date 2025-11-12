#!/bin/bash
# Start LiveKit Ditto Agent
#
# Usage:
#   ./start_livekit_agent.sh
#
# First, make sure LiveKit server is running:
#   docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
#     -e LIVEKIT_KEYS="devkey: devsecret" \
#     livekit/livekit-server:latest

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

echo "=================================================="
echo "Starting LiveKit Ditto Avatar Agent"
echo "=================================================="
echo "LiveKit Server: $LIVEKIT_URL"
echo "Ditto Config:   $DITTO_CFG_PKL"
echo "Ditto Data:     $DITTO_DATA_ROOT"
echo "Avatar Source:  $DITTO_SOURCE"
echo "Max Size:       $DITTO_MAX_SIZE"
echo "Emotion:        $DITTO_EMO"
echo "=================================================="
echo ""

# Run agent with LiveKit CLI
uv run python webrtc/livekit_ditto_agent.py dev
