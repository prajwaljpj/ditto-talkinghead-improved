#!/bin/bash
# Start all LiveKit services
# This script helps coordinate starting all components

set -e  # Exit on error

echo "=================================================="
echo "🚀 Starting LiveKit + Ditto Avatar"
echo "=================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Check prerequisites
echo -e "${BLUE}Step 1: Checking prerequisites...${NC}"
./webrtc/test_setup.sh
echo ""

# Check if test passed
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Prerequisites check failed!${NC}"
    echo "Fix the issues above before continuing."
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"
echo ""

# Step 2: Check if LiveKit server is already running
echo -e "${BLUE}Step 2: Checking LiveKit server...${NC}"
if curl -s http://localhost:7881/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅ LiveKit server already running${NC}"
else
    echo -e "${YELLOW}⚠ LiveKit server not running${NC}"
    echo ""
    echo "Please start LiveKit server in a separate terminal:"
    echo ""
    echo -e "${GREEN}docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \\${NC}"
    echo -e "${GREEN}  -e LIVEKIT_KEYS=\"devkey: devsecret\" \\${NC}"
    echo -e "${GREEN}  livekit/livekit-server:latest${NC}"
    echo ""
    echo "Press Enter when LiveKit server is running..."
    read -r

    # Check again
    if curl -s http://localhost:7881/ > /dev/null 2>&1; then
        echo -e "${GREEN}✅ LiveKit server detected!${NC}"
    else
        echo -e "${RED}❌ LiveKit server still not running${NC}"
        exit 1
    fi
fi
echo ""

# Step 3: Offer to start web client server
echo -e "${BLUE}Step 3: Web client server${NC}"
echo "Do you want to start the web client server? (y/n)"
read -r start_web

if [ "$start_web" = "y" ]; then
    echo "Starting web client server in background..."
    cd webrtc/client/livekit
    python -m http.server 8000 > /dev/null 2>&1 &
    WEB_PID=$!
    cd ../../..
    echo -e "${GREEN}✅ Web server started (PID: $WEB_PID)${NC}"
    echo "   URL: http://localhost:8000/index_simple.html"
    echo ""

    # Save PID for cleanup
    echo $WEB_PID > /tmp/livekit_web_server.pid
else
    echo "Skipping web server."
    echo "Start manually with: cd webrtc/client/livekit && python -m http.server 8000"
    echo ""
fi

# Step 4: Start Ditto agent
echo -e "${BLUE}Step 4: Starting Ditto agent...${NC}"
echo ""
echo "=================================================="
echo "Starting agent with configuration:"
echo "=================================================="
echo "LiveKit URL:  ${LIVEKIT_URL:-ws://localhost:7880}"
echo "Avatar:       ${DITTO_SOURCE:-avatars/Hyperrealistic_Indian_Woman_Professional_Presentation_static_5_loop.jpg}"
echo "Emotion:      ${DITTO_EMO:-4} (neutral)"
echo "=================================================="
echo ""
echo "Press Ctrl+C to stop the agent"
echo ""

# Trap Ctrl+C to clean up web server
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -f /tmp/livekit_web_server.pid ]; then
        WEB_PID=$(cat /tmp/livekit_web_server.pid)
        if ps -p $WEB_PID > /dev/null 2>&1; then
            echo "Stopping web server (PID: $WEB_PID)..."
            kill $WEB_PID
        fi
        rm /tmp/livekit_web_server.pid
    fi
    echo "Done!"
}

trap cleanup EXIT INT TERM

# Start the agent
./start_livekit_agent.sh
