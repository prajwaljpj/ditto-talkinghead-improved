#!/bin/bash
# Test LiveKit Setup
# This script checks if all components are ready

echo "=================================================="
echo "🔍 Testing LiveKit + Ditto Setup"
echo "=================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track status
ALL_GOOD=true

# Test 1: Check if required files exist
echo "📁 Checking required files..."

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1 (missing)"
        ALL_GOOD=false
    fi
}

check_file "webrtc/livekit_ditto_agent.py"
check_file "start_livekit_agent.sh"
check_file "webrtc/client/livekit/index_simple.html"
check_file "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
check_file "avatars/Hyperrealistic_Indian_Woman_Professional_Presentation_static_5_loop.jpg"

# Check if data_root exists
if [ -d "checkpoints/ditto_trt_custom2/" ]; then
    echo -e "${GREEN}✓${NC} checkpoints/ditto_trt_custom2/ (directory)"
else
    echo -e "${RED}✗${NC} checkpoints/ditto_trt_custom2/ (missing)"
    ALL_GOOD=false
fi

echo ""

# Test 2: Check Python dependencies
echo "🐍 Checking Python dependencies..."

check_import() {
    if uv run python -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1 (not installed)"
        ALL_GOOD=false
    fi
}

check_import "livekit"
check_import "livekit.agents"
check_import "numpy"
check_import "torch"
check_import "librosa"

echo ""

# Test 3: Check GPU
echo "🎮 Checking GPU..."

if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
        echo -e "${GREEN}✓${NC} GPU detected: $GPU_NAME ($GPU_MEM)"
    else
        echo -e "${RED}✗${NC} nvidia-smi failed"
        ALL_GOOD=false
    fi
else
    echo -e "${RED}✗${NC} nvidia-smi not found"
    ALL_GOOD=false
fi

echo ""

# Test 4: Check Docker
echo "🐋 Checking Docker..."

if command -v docker &> /dev/null; then
    if docker ps &> /dev/null; then
        echo -e "${GREEN}✓${NC} Docker is running"
    else
        echo -e "${YELLOW}⚠${NC} Docker not running (start with: sudo systemctl start docker)"
        ALL_GOOD=false
    fi
else
    echo -e "${RED}✗${NC} Docker not installed"
    ALL_GOOD=false
fi

echo ""

# Test 5: Check if LiveKit server is running
echo "🔌 Checking LiveKit server..."

if curl -s http://localhost:7881/ > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} LiveKit server is running on port 7881"
else
    echo -e "${YELLOW}⚠${NC} LiveKit server not running"
    echo "  Start with:"
    echo "  docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \\"
    echo "    -e LIVEKIT_KEYS=\"devkey: devsecret\" \\"
    echo "    livekit/livekit-server:latest"
fi

echo ""

# Summary
echo "=================================================="
if [ "$ALL_GOOD" = true ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo ""
    echo "Ready to start the agent:"
    echo "  ./start_livekit_agent.sh"
    echo ""
    echo "Then open browser to:"
    echo "  http://localhost:8000/index_simple.html"
    echo ""
    echo "(Start client server with: cd webrtc/client/livekit && python -m http.server 8000)"
else
    echo -e "${RED}❌ Some checks failed${NC}"
    echo ""
    echo "Fix the issues above before starting."
    echo "See webrtc/README_LIVEKIT.md for setup instructions."
fi
echo "=================================================="
