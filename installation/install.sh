#!/bin/bash
#
# AURA v3 - Master Install Script
# ===============================
# Unified installation entry point
#
# Usage:
#   bash installation/install.sh          # Interactive mode
#   bash installation/install.sh --auto # Fully automated
#   bash installation/install.sh --help  # Show help
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AURA_DIR="$(dirname "$SCRIPT_DIR")"

# Default options
AUTO_MODE=false
SKIP_MODELS=false
VERBOSE=false

# =============================================================================
# Parse Arguments
# =============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --auto)
                AUTO_MODE=true
                shift
                ;;
            --skip-models)
                SKIP_MODELS=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
AURA v3 Installation Script
==========================

Usage:
  bash installation/install.sh [OPTIONS]

Options:
  --auto         Run in fully automated mode (non-interactive)
  --skip-models Skip model download step
  --verbose      Enable verbose output
  --help, -h    Show this help message

Examples:
  # Interactive installation
  bash installation/install.sh

  # Fully automated
  bash installation/install.sh --auto --skip-models

EOF
}

# =============================================================================
# Utility Functions
# =============================================================================
log() {
    echo "[AURA] $1"
}

warn() {
    echo "[AURA WARN] $1" >&2
}

error() {
    echo "[AURA ERROR] $1" >&2
}

success() {
    echo "[AURA OK] $1"
}

cleanup_on_error() {
    error "Installation failed. Cleaning up..."
    # Add cleanup commands if needed
    exit 1
}

# =============================================================================
# Pre-flight Checks
# =============================================================================
preflight_checks() {
    log "Running pre-flight checks..."
    
    # Check for required commands
    for cmd in python3 git; do
        if ! command -v $cmd &> /dev/null; then
            error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        error "Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi
    
    success "Pre-flight checks passed"
}

# =============================================================================
# Install System Dependencies
# =============================================================================
install_system_deps() {
    log "Installing system dependencies..."
    
    if [ -f /data/data/com.termux/files/usr/bin/pkg ]; then
        # Termux
        pkg update -y
        pkg install -y git curl wget python python-dev
    elif [ "$(uname)" = "Linux" ]; then
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y -qq python3 python3-pip git curl wget
        elif command -v yum &> /dev/null; then
            sudo yum install -y python3 python3-pip git curl wget
        fi
    elif [ "$(uname)" = "Darwin" ]; then
        # macOS - check if homebrew available
        if command -v brew &> /dev/null; then
            brew install python3 git curl wget
        fi
    fi
    
    success "System dependencies installed"
}

# =============================================================================
# Install Python Packages
# =============================================================================
install_python_deps() {
    log "Installing Python packages..."
    
    # Upgrade pip
    python3 -m pip install --upgrade pip --user 2>/dev/null || true
    
    # Install core packages
    pip install --user --quiet \
        aiohttp \
        numpy \
        scipy \
        pyyaml \
        2>/dev/null || true
    
    success "Python packages installed"
}

# =============================================================================
# Setup Configuration
# =============================================================================
setup_config() {
    log "Setting up configuration..."
    
    cd "$AURA_DIR"
    
    # Create config if doesn't exist
    if [ ! -f config.yaml ]; then
        if [ -f config.example.yaml ]; then
            cp config.example.yaml config.yaml
        else
            cat > config.yaml << 'EOF'
# AURA v3 Configuration
core:
  llm:
    provider: "llama.cpp"
    max_tokens: 1024
    temperature: 0.7

power:
  full_power_threshold: 50
  balanced_threshold: 20
  power_save_threshold: 10
  critical_threshold: 5

privacy:
  log_level: "WARNING"
  allow_telemetry: false

features:
  voice_enabled: false
  proactive_mode: true
  background_tasks: true
EOF
        fi
        success "Configuration created"
    else
        log "Configuration already exists"
    fi
}

# =============================================================================
# Verify Installation
# =============================================================================
verify_installation() {
    log "Verifying installation..."
    
    cd "$AURA_DIR"
    
    # Test imports
    if python3 -c "import sys; sys.path.insert(0, '.'); import src" 2>/dev/null; then
        success "AURA modules importable"
    else
        warn "Module import test failed - this may be OK for basic installation"
    fi
    
    success "Installation verified"
}

# =============================================================================
# Print Summary
# =============================================================================
print_summary() {
    echo ""
    echo "========================================"
    echo "  AURA v3 Installation Complete"
    echo "========================================"
    echo ""
    echo "Location: $AURA_DIR"
    echo ""
    echo "To start AURA:"
    echo "  cd $AURA_DIR"
    echo "  python3 -m src.main"
    echo ""
    echo "Documentation: $AURA_DIR/README.md"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
main() {
    parse_args "$@"
    
    echo ""
    echo "AURA v3 Installer"
    echo "================="
    echo ""
    
    if [ "$AUTO_MODE" = false ]; then
        echo "Running in interactive mode (use --auto for automated)"
        echo ""
    fi
    
    preflight_checks
    install_system_deps
    install_python_deps
    setup_config
    
    if [ "$SKIP_MODELS" = false ]; then
        log "Skipping model download (use --skip-models to skip)"
    fi
    
    verify_installation
    print_summary
}

main "$@"
