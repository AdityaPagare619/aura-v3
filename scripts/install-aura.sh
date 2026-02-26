#!/bin/bash
#
# AURA v3 - One-Curl Installer
# ============================
# Install with: curl -sL https://raw.githubusercontent.com/AdityaPagare619/aura-v3/main/scripts/install-aura.sh | bash
#
# Options:
#   --check   Check installation status only
#   --update  Update existing installation
#   --help    Show this help message

set -e

AURA_DIR="${AURA_DIR:-$HOME/aura-v3}"
REPO_URL="https://github.com/AdityaPagare619/aura-v3.git"
CHECK_MODE=false
UPDATE_MODE=false

log() {
    echo "[AURA] $1"
}

warn() {
    echo "[AURA WARN] $1" >&2
}

error() {
    echo "[AURA ERROR] $1" >&2
    exit 1
}

success() {
    echo "[AURA OK] $1"
}

show_help() {
    cat << EOF
AURA v3 - One-Curl Installer
============================

Usage:
  curl -sL https://raw.githubusercontent.com/AdityaPagare619/aura-v3/main/scripts/install-aura.sh | bash
  curl -sL https://raw.githubusercontent.com/AdityaPagare619/aura-v3/main/scripts/install-aura.sh | bash -s -- --check
  curl -sL https://raw.githubusercontent.com/AdityaPagare619/aura-v3/main/scripts/install-aura.sh | bash -s -- --update

Options:
  --check   Check installation status only (no installation)
  --update  Update existing installation
  --help    Show this help message

Supports:
  - Termux (Android)
  - Linux VPS
  - Linux Desktop
EOF
}

detect_platform() {
    if [ -f /data/data/com.termux/files/usr/bin/pkg ]; then
        echo "termux"
    elif [ "$(uname)" = "Linux" ]; then
        echo "linux"
    elif [ "$(uname)" = "Darwin" ]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

check_dependencies() {
    local missing=()
    
    for cmd in git python3; do
        if ! command -v $cmd &> /dev/null; then
            missing+=("$cmd")
        fi
    done
    
    if [ ${#missing[@]} -gt 0 ]; then
        error "Missing dependencies: ${missing[*]}. Please install them first."
    fi
}

check_python_version() {
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        error "Python 3.8+ required, found $PYTHON_VERSION"
    fi
    
    success "Python version: $PYTHON_VERSION"
}

install_system_deps() {
    local platform=$(detect_platform)
    log "Detected platform: $platform"
    
    case $platform in
        termux)
            log "Installing dependencies for Termux..."
            pkg update -y
            pkg install -y git python python-dev clang make ffmpeg
            ;;
        linux)
            log "Installing dependencies for Linux..."
            if command -v apt-get &> /dev/null; then
                sudo apt-get update -qq
                sudo apt-get install -y -qq python3 python3-pip git curl wget ffmpeg build-essential
            elif command -v yum &> /dev/null; then
                sudo yum install -y python3 python3-pip git curl wget ffmpeg gcc gcc-c++
            elif command -v apk &> /dev/null; then
                sudo apk add --no-cache python3 py3-pip git curl wget ffmpeg build-base
            fi
            ;;
        macos)
            log "Installing dependencies for macOS..."
            if command -v brew &> /dev/null; then
                brew install python3 git ffmpeg
            fi
            ;;
        *)
            error "Unsupported platform: $(uname)"
            ;;
    esac
    
    success "System dependencies installed"
}

clone_or_update() {
    if [ -d "$AURA_DIR/.git" ]; then
        if [ "$UPDATE_MODE" = true ]; then
            log "Updating existing installation..."
            cd "$AURA_DIR"
            git fetch origin
            git pull origin main
            success "AURA updated"
        else
            log "AURA already installed at $AURA_DIR"
            log "Use --update to update"
        fi
    else
        log "Cloning AURA v3 repository..."
        git clone "$REPO_URL" "$AURA_DIR"
        success "AURA cloned"
    fi
}

install_python_deps() {
    log "Installing Python dependencies..."
    
    cd "$AURA_DIR"
    
    python3 -m pip install --upgrade pip --user 2>/dev/null || true
    
    if [ -f requirements.txt ]; then
        pip install --user -r requirements.txt
    fi
    
    if [ -f scripts/requirements.txt ]; then
        pip install --user -r scripts/requirements.txt
    fi
    
    success "Python dependencies installed"
}

setup_env() {
    cd "$AURA_DIR"
    
    if [ -f .env ]; then
        log ".env already exists"
    elif [ -f .env.example ]; then
        cp .env.example .env
        success ".env created from example"
        warn "Please configure .env with your API keys and settings"
    else
        warn ".env.example not found, skipping"
    fi
}

check_status() {
    local platform=$(detect_platform)
    echo ""
    echo "========================================"
    echo "  AURA v3 Status Check"
    echo "========================================"
    echo ""
    echo "Platform: $platform"
    echo "User: $USER"
    echo "Home: $HOME"
    echo ""
    
    if [ -d "$AURA_DIR/.git" ]; then
        echo "Installation: EXISTS"
        echo "Location: $AURA_DIR"
        
        cd "$AURA_DIR"
        local branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
        local commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        echo "Branch: $branch"
        echo "Commit: $commit"
        
        if [ -f .env ]; then
            echo "Config: Configured"
        else
            echo "Config: NOT CONFIGURED"
        fi
    else
        echo "Installation: NOT INSTALLED"
        echo ""
        echo "To install, run:"
        echo "  curl -sL https://raw.githubusercontent.com/AdityaPagare619/aura-v3/main/scripts/install-aura.sh | bash"
    fi
    
    echo ""
    echo "Python: $(python3 --version 2>&1)"
    echo " pip: $(python3 -m pip --version 2>&1 | awk '{print $2}' || echo 'not found')"
    
    echo ""
    echo "AURA dependencies:"
    for pkg in aiofiles pyyaml cryptography psutil; do
        if python3 -c "import ${pkg%%:*}" 2>/dev/null; then
            echo "  $pkg: INSTALLED"
        else
            echo "  $pkg: NOT INSTALLED"
        fi
    done
    
    echo ""
}

print_next_steps() {
    echo ""
    echo "========================================"
    echo "  Installation Complete!"
    echo "========================================"
    echo ""
    echo "Location: $AURA_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Configure .env:"
    echo "     cd $AURA_DIR"
    echo "     nano .env"
    echo ""
    echo "  2. Start AURA:"
    echo "     cd $AURA_DIR"
    echo "     python3 main.py"
    echo ""
    echo "  Or use the convenience scripts:"
    echo "     ./run_aura.sh"
    echo ""
    echo "Documentation: $AURA_DIR/README.md"
    echo ""
}

main() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check)
                CHECK_MODE=true
                shift
                ;;
            --update)
                UPDATE_MODE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1. Use --help for usage."
                ;;
        esac
    done
    
    if [ "$CHECK_MODE" = true ]; then
        check_status
        exit 0
    fi
    
    echo ""
    echo "AURA v3 Installer"
    echo "================"
    echo ""
    
    check_dependencies
    check_python_version
    install_system_deps
    clone_or_update
    install_python_deps
    setup_env
    print_next_steps
}

main "$@"
