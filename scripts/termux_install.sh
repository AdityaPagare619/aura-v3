#!/bin/bash
# =============================================================================
# AURA v3 - Termux Quick Install
# =============================================================================
# One-liner install for Termux:
#   curl -sL https://raw.githubusercontent.com/AdityaPagare619/aura-v3/main/scripts/termux_install.sh | bash
#
# Or manual:
#   bash scripts/termux_install.sh
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
BOLD='\033[1m'

echo ""
echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}   AURA v3 - Termux Quick Install${NC}"
echo -e "${BOLD}========================================${NC}"
echo ""

# -----------------------------------------------------------------------------
# Check if running on Termux
# -----------------------------------------------------------------------------
if [ -z "$TERMUX_VERSION" ] && [ ! -d "/data/data/com.termux" ]; then
    echo -e "${RED}[ERROR]${NC} This script is for Termux only!"
    echo "For other environments, use: bash scripts/install.sh"
    exit 1
fi

echo -e "${BLUE}[INFO]${NC} Detected Termux environment"

# -----------------------------------------------------------------------------
# Update packages
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}[1/6]${NC} Updating packages..."
# Using -o Dpkg::Options::="--force-confnew" to automatically handle config file prompts
# Using -y to automatically answer yes to all prompts
pkg update -y -o Dpkg::Options::="--force-confnew" 2>/dev/null || true
pkg upgrade -y -o Dpkg::Options::="--force-confnew" 2>/dev/null || true

# -----------------------------------------------------------------------------
# Install required packages
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}[2/6]${NC} Installing required packages..."
pkg install -y python git cmake clang 2>/dev/null || {
    echo -e "${YELLOW}[WARN]${NC} Some packages may have failed, continuing..."
}

# Verify Python
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python installation failed!"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)
echo -e "${GREEN}[OK]${NC} Python installed: $($PYTHON_CMD --version)"

# -----------------------------------------------------------------------------
# Clone or update repository
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}[3/6]${NC} Getting AURA v3..."

cd ~

if [ -d "aura-v3" ]; then
    echo -e "${BLUE}[INFO]${NC} AURA v3 directory exists, updating..."
    cd aura-v3
    git pull origin main 2>/dev/null || {
        echo -e "${YELLOW}[WARN]${NC} Could not update, using existing version"
    }
else
    echo -e "${BLUE}[INFO]${NC} Cloning AURA v3..."
    git clone https://github.com/AdityaPagare619/aura-v3.git
    cd aura-v3
fi

echo -e "${GREEN}[OK]${NC} AURA v3 ready at ~/aura-v3"

# -----------------------------------------------------------------------------
# Install Python dependencies
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}[4/6]${NC} Installing Python dependencies..."

$PYTHON_CMD -m pip install --upgrade pip --quiet 2>/dev/null || true
$PYTHON_CMD -m pip install -r requirements.txt --quiet 2>/dev/null || {
    echo -e "${YELLOW}[WARN]${NC} Some dependencies may have failed"
    # Install core deps one by one
    $PYTHON_CMD -m pip install aiofiles pyyaml cryptography python-telegram-bot --quiet 2>/dev/null || true
}

echo -e "${GREEN}[OK]${NC} Dependencies installed"

# -----------------------------------------------------------------------------
# Create directories
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}[5/6]${NC} Creating directories..."

mkdir -p logs data/memories data/sessions data/patterns/intents data/patterns/strategies models cache

# Copy config if not exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        echo "TELEGRAM_TOKEN=" > .env
        echo "AURA_ENV=production" >> .env
    fi
fi

echo -e "${GREEN}[OK]${NC} Directories created"

# -----------------------------------------------------------------------------
# Install LLM (optional, try pre-built first)
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}[6/6]${NC} Installing LLM support (optional)..."

if $PYTHON_CMD -m pip install llama-cpp-python --only-binary :all: --quiet 2>/dev/null; then
    echo -e "${GREEN}[OK]${NC} LLM support installed (pre-built)"
else
    echo -e "${YELLOW}[WARN]${NC} LLM support not installed (AURA will use MOCK mode)"
    echo -e "${BLUE}[INFO]${NC} You can install manually later: pip install llama-cpp-python"
fi

# -----------------------------------------------------------------------------
# Verify installation
# -----------------------------------------------------------------------------
echo ""
echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}   Verifying Installation${NC}"
echo -e "${BOLD}========================================${NC}"
echo ""

# Test imports
if $PYTHON_CMD -c "import aiofiles, yaml, cryptography, telegram; print('OK')" 2>/dev/null; then
    echo -e "${GREEN}[OK]${NC} Core imports working"
else
    echo -e "${YELLOW}[WARN]${NC} Some imports failed - run manually to debug"
fi

# Test AURA modules
if $PYTHON_CMD -c "import sys; sys.path.insert(0, 'src'); from main import AuraProduction; print('OK')" 2>/dev/null; then
    echo -e "${GREEN}[OK]${NC} AURA modules working"
else
    echo -e "${YELLOW}[WARN]${NC} AURA module test failed - may need debugging"
fi

# -----------------------------------------------------------------------------
# Print success message
# -----------------------------------------------------------------------------
echo ""
echo -e "${BOLD}========================================${NC}"
echo -e "${GREEN}${BOLD}   Installation Complete!${NC}"
echo -e "${BOLD}========================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. Set your Telegram token:"
echo "     nano ~/aura-v3/.env"
echo "     # Add: TELEGRAM_TOKEN=your-bot-token"
echo ""
echo "  2. Start AURA:"
echo "     cd ~/aura-v3"
echo "     bash run_aura.sh telegram"
echo ""
echo "  3. Or use CLI mode:"
echo "     bash run_aura.sh cli"
echo ""
echo "For background running:"
echo "  nohup python main.py --mode telegram > aura.log 2>&1 &"
echo ""
echo "For help:"
echo "  cat docs/TERMUX_SETUP.md"
echo ""
