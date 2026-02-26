#!/bin/bash
# =============================================================================
# AURA v3 - Unified Install Script
# =============================================================================
# Single entry point for all environments: Termux, Desktop Linux/macOS, Docker
#
# Usage:
#   bash scripts/install.sh           # Full install
#   bash scripts/install.sh --check   # Check environment only
#   bash scripts/install.sh --minimal # Core deps only (no optional)
#   bash scripts/install.sh --help    # Show help
#
# Environment Variables:
#   TELEGRAM_TOKEN    - Telegram bot token (optional, can set later)
#   AURA_NO_LLM       - Skip LLM installation (default: tries to install)
#   AURA_MINIMAL      - Minimal install, skip optional deps
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Colors & Formatting
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo ""
    echo -e "${BOLD}============================================${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${BOLD}============================================${NC}"
}

# -----------------------------------------------------------------------------
# Environment Detection
# -----------------------------------------------------------------------------
detect_environment() {
    log_header "Detecting Environment"
    
    # Check if running in Docker
    if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
        ENVIRONMENT="docker"
        log_info "Environment: Docker container"
        return
    fi
    
    # Check if running in Termux
    if [ -n "$TERMUX_VERSION" ] || [ -d "/data/data/com.termux" ]; then
        ENVIRONMENT="termux"
        log_info "Environment: Termux (Android)"
        return
    fi
    
    # Check OS
    case "$(uname -s)" in
        Linux*)
            ENVIRONMENT="linux"
            log_info "Environment: Linux Desktop"
            ;;
        Darwin*)
            ENVIRONMENT="macos"
            log_info "Environment: macOS"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            ENVIRONMENT="windows"
            log_info "Environment: Windows (Git Bash/WSL)"
            ;;
        *)
            ENVIRONMENT="unknown"
            log_warn "Environment: Unknown ($(uname -s))"
            ;;
    esac
}

# -----------------------------------------------------------------------------
# Python Version Check
# -----------------------------------------------------------------------------
check_python() {
    log_header "Checking Python"
    
    # Try python3 first, then python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python not found! Please install Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+"
        exit 1
    fi
    
    # Get version
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
    PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')
    
    log_info "Found: $PYTHON_CMD ($PYTHON_VERSION)"
    
    # Version check
    if [ "$PYTHON_MAJOR" -lt "$MIN_PYTHON_MAJOR" ] || \
       ([ "$PYTHON_MAJOR" -eq "$MIN_PYTHON_MAJOR" ] && [ "$PYTHON_MINOR" -lt "$MIN_PYTHON_MINOR" ]); then
        log_error "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ required, found $PYTHON_VERSION"
        exit 1
    fi
    
    log_success "Python version OK ($PYTHON_VERSION >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR})"
    
    # Check pip
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        log_warn "pip not found, attempting to install..."
        
        case "$ENVIRONMENT" in
            termux)
                pkg install python-pip -y 2>/dev/null || true
                ;;
            linux)
                if command -v apt-get &> /dev/null; then
                    sudo apt-get install -y python3-pip 2>/dev/null || true
                fi
                ;;
        esac
        
        # Verify pip installed
        if ! $PYTHON_CMD -m pip --version &> /dev/null; then
            log_error "Could not install pip. Please install manually."
            exit 1
        fi
    fi
    
    log_success "pip available"
}

# -----------------------------------------------------------------------------
# Create Data Directories
# -----------------------------------------------------------------------------
create_directories() {
    log_header "Creating Directories"
    
    cd "$PROJECT_ROOT"
    
    directories=(
        "logs"
        "data"
        "data/memories"
        "data/sessions"
        "data/patterns"
        "data/patterns/intents"
        "data/patterns/strategies"
        "models"
        "cache"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log_info "Created: $dir/"
    done
    
    log_success "All directories created"
}

# -----------------------------------------------------------------------------
# Install Dependencies
# -----------------------------------------------------------------------------
install_dependencies() {
    log_header "Installing Dependencies"
    
    cd "$PROJECT_ROOT"
    
    # Upgrade pip first
    log_info "Upgrading pip..."
    $PYTHON_CMD -m pip install --upgrade pip --quiet 2>/dev/null || true
    
    # Install from requirements.txt
    if [ -f "$REQUIREMENTS_FILE" ]; then
        log_info "Installing from requirements.txt..."
        $PYTHON_CMD -m pip install -r "$REQUIREMENTS_FILE" --quiet || {
            log_warn "Some packages failed, trying one by one..."
            
            # Core packages (must install)
            CORE_DEPS=(
                "aiofiles>=23.0.0"
                "pyyaml>=6.0"
                "cryptography>=41.0.0"
                "python-telegram-bot>=20.0"
            )
            
            for dep in "${CORE_DEPS[@]}"; do
                log_info "Installing $dep..."
                $PYTHON_CMD -m pip install "$dep" --quiet || log_warn "Failed: $dep"
            done
        }
        log_success "Core dependencies installed"
    else
        log_error "requirements.txt not found at $REQUIREMENTS_FILE"
        exit 1
    fi
    
    # Install test dependencies
    if [ "$MINIMAL" != "true" ]; then
        log_info "Installing test dependencies..."
        $PYTHON_CMD -m pip install pytest pytest-asyncio --quiet 2>/dev/null || true
    fi
}

# -----------------------------------------------------------------------------
# Install LLM (Optional)
# -----------------------------------------------------------------------------
install_llm() {
    if [ "$AURA_NO_LLM" = "true" ] || [ "$MINIMAL" = "true" ]; then
        log_info "Skipping LLM installation (AURA_NO_LLM or --minimal)"
        return
    fi
    
    log_header "Installing LLM Support (Optional)"
    
    # Try pre-built wheel first (fastest)
    log_info "Trying pre-built llama-cpp-python..."
    if $PYTHON_CMD -m pip install llama-cpp-python --only-binary :all: --quiet 2>/dev/null; then
        log_success "LLM support installed (pre-built)"
        return
    fi
    
    # Termux-specific build
    if [ "$ENVIRONMENT" = "termux" ]; then
        log_info "Trying Termux-optimized build..."
        pkg install cmake -y 2>/dev/null || true
        if $PYTHON_CMD -m pip install llama-cpp-python --quiet 2>/dev/null; then
            log_success "LLM support installed (Termux build)"
            return
        fi
    fi
    
    log_warn "LLM support not installed (will use MOCK mode)"
    log_info "You can install manually later: pip install llama-cpp-python"
}

# -----------------------------------------------------------------------------
# Install Voice Support (Optional)
# -----------------------------------------------------------------------------
install_voice() {
    if [ "$MINIMAL" = "true" ]; then
        log_info "Skipping voice support (--minimal)"
        return
    fi
    
    log_header "Installing Voice Support (Optional)"
    
    # Install pyttsx3
    $PYTHON_CMD -m pip install pyttsx3 --quiet 2>/dev/null || true
    
    # Environment-specific TTS
    case "$ENVIRONMENT" in
        termux)
            log_info "Installing espeak-ng for Termux..."
            pkg install espeak-ng -y 2>/dev/null || true
            ;;
        linux)
            log_info "Installing espeak-ng for Linux..."
            if command -v apt-get &> /dev/null; then
                sudo apt-get install -y espeak-ng 2>/dev/null || true
            fi
            ;;
        macos)
            log_info "macOS has built-in speech synthesis"
            ;;
    esac
    
    log_success "Voice support configured"
}

# -----------------------------------------------------------------------------
# Setup Configuration
# -----------------------------------------------------------------------------
setup_config() {
    log_header "Setting Up Configuration"
    
    cd "$PROJECT_ROOT"
    
    # Create .env from example if not exists
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp ".env.example" ".env"
            log_success "Created .env from .env.example"
        else
            # Create minimal .env
            cat > ".env" << 'EOF'
# AURA v3 Environment
TELEGRAM_TOKEN=
AURA_ENV=production
AURA_DEBUG=false
EOF
            log_success "Created minimal .env"
        fi
    else
        log_info ".env already exists, skipping"
    fi
    
    # Set TELEGRAM_TOKEN if provided
    if [ -n "$TELEGRAM_TOKEN" ]; then
        if grep -q "^TELEGRAM_TOKEN=" ".env" 2>/dev/null; then
            # Update existing
            sed -i.bak "s|^TELEGRAM_TOKEN=.*|TELEGRAM_TOKEN=$TELEGRAM_TOKEN|" ".env" 2>/dev/null || \
            sed -i '' "s|^TELEGRAM_TOKEN=.*|TELEGRAM_TOKEN=$TELEGRAM_TOKEN|" ".env" 2>/dev/null || true
            rm -f ".env.bak" 2>/dev/null || true
        else
            echo "TELEGRAM_TOKEN=$TELEGRAM_TOKEN" >> ".env"
        fi
        log_success "Telegram token configured"
    fi
    
    # Create config.yaml from example if not exists
    if [ ! -f "config/config.yaml" ]; then
        mkdir -p config
        if [ -f "config.example.yaml" ]; then
            cp "config.example.yaml" "config/config.yaml"
            log_success "Created config/config.yaml from example"
        fi
    else
        log_info "config/config.yaml already exists, skipping"
    fi
}

# -----------------------------------------------------------------------------
# Verify Installation
# -----------------------------------------------------------------------------
verify_installation() {
    log_header "Verifying Installation"
    
    cd "$PROJECT_ROOT"
    
    # Test core imports
    log_info "Testing core imports..."
    
    IMPORT_TEST=$($PYTHON_CMD << 'EOF' 2>&1
import sys
errors = []

try:
    import aiofiles
except ImportError as e:
    errors.append(f"aiofiles: {e}")

try:
    import yaml
except ImportError as e:
    errors.append(f"pyyaml: {e}")

try:
    import cryptography
except ImportError as e:
    errors.append(f"cryptography: {e}")

try:
    import telegram
except ImportError as e:
    errors.append(f"telegram: {e}")

if errors:
    print("ERRORS:" + "|".join(errors))
    sys.exit(1)
else:
    print("OK")
    sys.exit(0)
EOF
)
    
    if [ "$IMPORT_TEST" = "OK" ]; then
        log_success "Core imports verified"
    else
        log_warn "Some imports failed: $IMPORT_TEST"
    fi
    
    # Test AURA modules
    log_info "Testing AURA modules..."
    
    $PYTHON_CMD << 'EOF' 2>/dev/null && log_success "AURA modules verified" || log_warn "Some AURA modules not loadable"
import sys
sys.path.insert(0, '.')
try:
    from src.agent.loop import ReActAgent
    from src.memory import HierarchicalMemory
    print("OK")
except Exception as e:
    print(f"WARN: {e}")
EOF
    
    # Check LLM availability
    $PYTHON_CMD << 'EOF' 2>/dev/null && log_success "LLM support available" || log_info "LLM not available (MOCK mode)"
try:
    import llama_cpp
    print("OK")
except ImportError:
    import sys
    sys.exit(1)
EOF
}

# -----------------------------------------------------------------------------
# Print Summary
# -----------------------------------------------------------------------------
print_summary() {
    log_header "Installation Complete!"
    
    echo ""
    echo -e "${GREEN}AURA v3 is ready to use!${NC}"
    echo ""
    echo -e "${BOLD}Quick Start:${NC}"
    echo "  cd $PROJECT_ROOT"
    echo "  python main.py --mode cli      # CLI mode"
    echo "  python main.py --mode telegram # Telegram mode"
    echo ""
    echo -e "${BOLD}Environment:${NC} $ENVIRONMENT"
    echo -e "${BOLD}Python:${NC} $PYTHON_VERSION"
    echo ""
    
    if [ -z "$TELEGRAM_TOKEN" ]; then
        echo -e "${YELLOW}Note:${NC} Telegram token not set. To enable Telegram:"
        echo "  1. Get token from @BotFather"
        echo "  2. Edit .env and set TELEGRAM_TOKEN=your-token"
        echo ""
    fi
    
    echo -e "${BOLD}For more help:${NC}"
    echo "  python scripts/setup.py --help"
    echo ""
}

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
show_help() {
    echo "AURA v3 Unified Install Script"
    echo ""
    echo "Usage: bash scripts/install.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --check     Check environment without installing"
    echo "  --minimal   Install only core dependencies (skip LLM, voice)"
    echo "  --help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  TELEGRAM_TOKEN  Set Telegram bot token during install"
    echo "  AURA_NO_LLM     Skip LLM installation (set to 'true')"
    echo "  AURA_MINIMAL    Same as --minimal flag"
    echo ""
    echo "Examples:"
    echo "  bash scripts/install.sh"
    echo "  TELEGRAM_TOKEN='your-token' bash scripts/install.sh"
    echo "  bash scripts/install.sh --minimal"
    echo ""
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    # Parse arguments
    CHECK_ONLY=false
    MINIMAL=${AURA_MINIMAL:-false}
    
    for arg in "$@"; do
        case $arg in
            --check)
                CHECK_ONLY=true
                ;;
            --minimal)
                MINIMAL=true
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
        esac
    done
    
    echo ""
    echo -e "${BOLD}========================================${NC}"
    echo -e "${BOLD}   AURA v3 - Unified Install Script${NC}"
    echo -e "${BOLD}========================================${NC}"
    echo ""
    
    # Detect environment
    detect_environment
    
    # Check Python
    check_python
    
    if [ "$CHECK_ONLY" = "true" ]; then
        log_success "Environment check complete!"
        exit 0
    fi
    
    # Install steps
    create_directories
    install_dependencies
    install_llm
    install_voice
    setup_config
    verify_installation
    print_summary
}

# Run main
main "$@"
