#!/bin/bash

# UV Environment Setup Script for Ditto TalkingHead
# This script automates the setup process using UV package manager

set -e  # Exit on error

echo "======================================================================"
echo "Ditto TalkingHead - UV Environment Setup"
echo "======================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if UV is installed
echo ""
echo "Step 1: Checking UV installation..."
if ! command -v uv &> /dev/null; then
    echo -e "${RED}✗ UV is not installed${NC}"
    echo ""
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add UV to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    # Check again
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}✗ UV installation failed${NC}"
        echo "Please install UV manually: https://github.com/astral-sh/uv"
        exit 1
    fi
    echo -e "${GREEN}✓ UV installed successfully${NC}"
else
    echo -e "${GREEN}✓ UV is already installed${NC}"
    uv --version
fi

# Check Python 3.10
echo ""
echo "Step 2: Checking Python 3.10..."
if command -v python3.10 &> /dev/null; then
    echo -e "${GREEN}✓ Python 3.10 found${NC}"
    python3.10 --version
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$PYTHON_VERSION" = "3.10" ]; then
        echo -e "${GREEN}✓ Python 3.10 found (as python3)${NC}"
        python3 --version
        PYTHON_CMD="python3"
    else
        echo -e "${RED}✗ Python 3.10 not found (found $PYTHON_VERSION)${NC}"
        echo ""
        echo "Please install Python 3.10:"
        echo "  Ubuntu/Debian: sudo apt install python3.10 python3.10-venv python3.10-dev"
        echo "  Or use pyenv: pyenv install 3.10.13"
        exit 1
    fi
else
    echo -e "${RED}✗ Python 3.10 not found${NC}"
    echo ""
    echo "Please install Python 3.10:"
    echo "  Ubuntu/Debian: sudo apt install python3.10 python3.10-venv python3.10-dev"
    echo "  Or use pyenv: pyenv install 3.10.13"
    exit 1
fi

# Check CUDA installation
echo ""
echo "Step 3: Checking CUDA installation..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA driver installed${NC}"
    nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader | head -n 1

    # Get CUDA version
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "  CUDA Version: $CUDA_VERSION"

    # Check if CUDA version is compatible
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    if [ "$CUDA_MAJOR" -ge 12 ] || [ "$CUDA_MAJOR" -eq 11 ]; then
        echo -e "${GREEN}✓ CUDA version is compatible${NC}"
    else
        echo -e "${YELLOW}! Warning: CUDA version may not be compatible${NC}"
        echo "  This project is tested with CUDA 12.x"
    fi
else
    echo -e "${YELLOW}! Warning: nvidia-smi not found${NC}"
    echo "  CUDA installation not detected"
    echo "  GPU acceleration may not work"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check ffmpeg
echo ""
echo "Step 4: Checking ffmpeg installation..."
if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}✓ ffmpeg installed${NC}"
    ffmpeg -version | head -n 1
else
    echo -e "${YELLOW}! Warning: ffmpeg not found${NC}"
    echo "  ffmpeg is required for audio/video processing"
    echo ""
    echo "Install ffmpeg:"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo ""
echo "Step 5: Creating virtual environment..."
if [ -d ".venv" ]; then
    echo -e "${YELLOW}! .venv directory already exists${NC}"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing .venv..."
        rm -rf .venv
    else
        echo "Using existing .venv"
    fi
fi

if [ ! -d ".venv" ]; then
    echo "Creating new virtual environment with Python 3.10..."
    uv venv --python $PYTHON_CMD
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Using existing virtual environment${NC}"
fi

# Activate virtual environment
echo ""
echo "Step 6: Activating virtual environment..."
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Verify Python version in venv
VENV_PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$VENV_PYTHON_VERSION" != "3.10" ]; then
    echo -e "${RED}✗ Virtual environment is using Python $VENV_PYTHON_VERSION, not 3.10${NC}"
    exit 1
fi
echo "  Using Python $VENV_PYTHON_VERSION"

# Install dependencies
echo ""
echo "Step 7: Installing dependencies..."
echo "This may take 5-10 minutes depending on your internet connection..."
echo ""

# Install with UV
if uv pip install -e .; then
    echo ""
    echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
else
    echo ""
    echo -e "${RED}✗ Installation failed${NC}"
    echo "Check the error messages above"
    exit 1
fi

# Run verification
echo ""
echo "Step 8: Verifying installation..."
echo ""
if python verify_installation.py; then
    echo ""
    echo -e "${GREEN}======================================================================"
    echo "✓ SETUP COMPLETE!"
    echo "======================================================================${NC}"
    echo ""
    echo "Your environment is ready to use."
    echo ""
    echo "To activate the environment in the future:"
    echo "  source .venv/bin/activate"
    echo ""
    echo "Next steps:"
    echo "  1. Download model checkpoints (see README.md)"
    echo "  2. Run inference: python inference_video.py --help"
    echo "  3. Read documentation: personal_docs/README.md"
    echo ""
    echo -e "${GREEN}======================================================================${NC}"
else
    echo ""
    echo -e "${YELLOW}======================================================================"
    echo "! SETUP COMPLETE WITH WARNINGS"
    echo "======================================================================${NC}"
    echo ""
    echo "Some checks failed. Review the output above."
    echo "You may still be able to use the system with limited functionality."
    echo ""
    echo "See SETUP_UV.md for troubleshooting help."
    echo ""
fi

# Show activation command
echo ""
echo "To activate this environment in a new terminal:"
echo -e "${GREEN}  cd $(pwd)${NC}"
echo -e "${GREEN}  source .venv/bin/activate${NC}"
echo ""
