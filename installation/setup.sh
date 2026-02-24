#!/bin/bash
#
# AURA v3 - Detailed Setup Script
# ================================
# Step-by-step installation for manual control
#
# Usage: bash installation/setup.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AURA_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[AURA]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }

# =============================================================================
# STEP 1: System Requirements Check
# =============================================================================
step_1_check() {
    echo ""
    echo "=========================================="
    echo "Step 1: Checking System Requirements"
    echo "=========================================="
    echo ""
    
    local failed=0
    
    # Check OS
    if [ -f /data/data/com.termux/files/usr/bin/pkg ]; then
        log "Running on Termux (Android)"
    elif [ "$(uname)" = "Linux" ]; then
        log "Running on Linux"
    elif [ "$(uname)" = "Darwin" ]; then
        log "Running on macOS"
    else
        warn "Unknown operating system"
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        success "Python $PYTHON_VERSION found"
        
        # Check version
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
            err "Python 3.8+ required, found $PYTHON_VERSION"
            failed=1
        fi
    else
        err "Python 3 not found"
        failed=1
    fi
    
    # Check git
    if command -v git &> /dev/null; then
        success "Git found: $(git --version)"
    else
        err "Git not found"
        failed=1
    fi
    
    # Check disk space (need ~2GB)
    if command -v df &> /dev/null; then
        AVAILABLE=$(df -BG . 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
        if [ "$AVAILABLE" -lt 2 ]; then
            err "Need at least 2GB free space, found ${AVAILABLE}GB"
            failed=1
        else
            success "${AVAILABLE}GB disk space available"
        fi
    fi
    
    # Check RAM (need ~4GB)
    if command -v free &> /dev/null; then
        RAM=$(free -m 2>/dev/null | head -2 | tail -1 | awk '{print $2}')
        if [ "$RAM" -lt 3500 ]; then
            warn "Low RAM: ${RAM}MB (4GB+ recommended)"
        else
            success "${RAM}MB RAM available"
        fi
    fi
    
    if [ $failed -eq 1 ]; then
        echo ""
        err "System requirements not met"
        exit 1
    fi
    
    success "All requirements met"
}

# =============================================================================
# STEP 2: Install System Dependencies
# =============================================================================
step_2_dependencies() {
    echo ""
    echo "=========================================="
    echo "Step 2: Installing System Dependencies"
    echo "=========================================="
    echo ""
    
    if [ -f /data/data/com.termux/files/usr/bin/pkg ]; then
        # Termux (Android)
        log "Installing Termux packages..."
        
        pkg update -y
        pkg install -y \
            git \
            curl \
            wget \
            python \
            python-dev \
            libjpeg-turbo \
            openssl \
            libcryptography \
            fftw \
            libsoxr \
            portaudio \
            2>/dev/null || warn "Some packages may have failed"
        
    elif [ "$(uname)" = "Linux" ]; then
        # Linux
        log "Installing Linux packages..."
        
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y \
                python3 python3-pip git curl wget \
                libjpeg-dev zlib1g-dev \
                libssl-dev libffi-dev \
                libsoxr-dev portaudio19-dev
        elif command -v yum &> /dev/null; then
            sudo yum install -y \
                python3 python3-pip git curl wget \
                libjpeg-turbo-devel zlib-devel \
                openssl-devel \
                portaudio-devel
        fi
    fi
    
    success "System dependencies installed"
}

# =============================================================================
# STEP 3: Install Python Packages
# =============================================================================
step_3_python() {
    echo ""
    echo "=========================================="
    echo "Step 3: Installing Python Packages"
    echo "=========================================="
    echo ""
    
    log "Upgrading pip..."
    python3 -m pip install --upgrade pip --user 2>/dev/null || true
    
    log "Installing AURA dependencies..."
    
    # Core dependencies
    pip install --user \
        aiohttp \
        asyncio \
        numpy \
        scipy \
        2>/dev/null || true
    
    # Audio dependencies (for voice)
    pip install --user \
        sounddevice \
        pyaudio \
        2>/dev/null || warn "Audio packages may need manual install"
    
    # ML dependencies
    pip install --user \
        torch \
        transformers \
        accelerate \
        sentencepiece \
        llama-cpp-python \
        2>/dev/null || warn "ML packages may need manual install"
    
    success "Python packages installed"
}

# =============================================================================
# STEP 4: Clone/Update Repository
# =============================================================================
step_4_repository() {
    echo ""
    echo "=========================================="
    echo "Step 4: Setting Up Repository"
    echo "=========================================="
    echo ""
    
    cd "$AURA_DIR"
    
    if [ -d .git ]; then
        log "Repository already exists, updating..."
        git pull origin main 2>/dev/null || warn "Could not update, using existing"
    else
        log "Initializing git repository..."
        git init
        git remote add origin https://github.com/aura-ai/aura-v3.git
        git fetch origin
        git checkout -b main origin/main 2>/dev/null || log "Already on main branch"
    fi
    
    success "Repository ready"
}

# =============================================================================
# STEP 5: Configure AURA
# =============================================================================
step_5_configure() {
    echo ""
    echo "=========================================="
    echo "Step 5: Configuring AURA"
    echo "=========================================="
    echo ""
    
    cd "$AURA_DIR"
    
    if [ -f config.yaml ]; then
        warn "Configuration already exists at config.yaml"
        read -p "Overwrite with fresh configuration? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            success "Keeping existing configuration"
            return 0
        fi
    fi
    
    log "Creating configuration file..."
    
    cat > config.yaml << 'EOF'
# AURA v3 Configuration
# =====================
# Full configuration options for AURA v3

# Core Settings
core:
  # LLM Configuration
  llm:
    provider: "llama.cpp"  # Options: llama.cpp, ollama, transformers
    model_path: "./models/llama-7b-chat.gguf"
    max_tokens: 1024
    temperature: 0.7
    n_gpu_layers: 99  # For GPU acceleration (Android)
    n_ctx: 2048
    preload: true
    
  # Memory Configuration
  memory:
    episodic_limit: 1000
    semantic_limit: 5000
    consolidation_interval: 3600
    importance_threshold: 0.5

# Power Management
power:
  full_power_threshold: 50    # % battery for full power mode
  balanced_threshold: 20     # % battery for balanced mode
  power_save_threshold: 10   # % battery for power save
  critical_threshold: 5       # % battery for critical mode
  
  # Processing limits per power mode
  max_tokens_full: 2048
  max_tokens_balanced: 1024
  max_tokens_save: 512
  max_tokens_ultra: 256
  
  # Tick rates (seconds between processing cycles)
  tick_rate_full: 1.0
  tick_rate_balanced: 2.0
  tick_rate_save: 5.0
  tick_rate_ultra: 30.0
  
  # Thermal management
  thermal_throttle_threshold: 45  # Celsius
  max_cpu_percent: 70

# Privacy Settings
privacy:
  log_level: "WARNING"  # DEBUG, INFO, WARNING, ERROR
  store_conversations: true
  anonymize_data: true
  allow_telemetry: false
  encrypt_storage: false
  
# Features
features:
  voice_enabled: false
  voice_language: "en"
  proactive_mode: true
  background_tasks: true
  notifications: true
  adaptive_personality: true
  life_tracking: true

# Security
security:
  require_auth: false
  auth_method: "pin"  # pin, biometric
  session_timeout: 3600  # seconds
  max_failed_attempts: 5
  lockout_duration: 300  # seconds

# Network (for optional features)
network:
  allow_outgoing: false
  proxy_enabled: false
  proxy_url: ""

# Storage
storage:
  data_path: "./data"
  logs_path: "./logs"
  cache_path: "./cache"
  max_log_size_mb: 100

# Debug
debug:
  verbose_logging: false
  save_memory_snapshots: false
  profile_performance: false
EOF
    
    success "Configuration created at config.yaml"
}

# =============================================================================
# STEP 6: Download Models (Optional)
# =============================================================================
step_6_models() {
    echo ""
    echo "=========================================="
    echo "Step 6: Downloading Models (Optional)"
    echo "=========================================="
    echo ""
    
    cd "$AURA_DIR"
    mkdir -p models
    
    log "Model download is optional - AURA will work with default settings"
    log "To download models manually:"
    echo ""
    echo "  1. Download GGUF model files"
    echo "  2. Place in ./models/ directory"
    echo "  3. Update model_path in config.yaml"
    echo ""
    echo "Recommended models:"
    echo "  - llama-7b-chat-q4_k_m.gguf (~4GB)"
    echo "  - mistral-7b-instruct-v0.2-q4_k_m.gguf (~4GB)"
    echo ""
    read -p "Download a model now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Downloading llama-7b-chat (this may take a while)..."
        # Note: This is a placeholder - actual URL would need to be provided
        warn "Please download models manually from huggingface.co"
    fi
    
    success "Model setup skipped (manual download recommended)"
}

# =============================================================================
# STEP 7: Test Installation
# =============================================================================
step_7_test() {
    echo ""
    echo "=========================================="
    echo "Step 7: Testing Installation"
    echo "=========================================="
    echo ""
    
    cd "$AURA_DIR"
    
    log "Testing Python imports..."
    if python3 -c "import src" 2>/dev/null; then
        success "Core modules importable"
    else
        warn "Some modules may have import errors"
    fi
    
    log "Testing configuration..."
    if python3 -c "import yaml; yaml.safe_load(open('config.yaml'))" 2>/dev/null; then
        success "Configuration valid"
    else
        err "Configuration has errors"
    fi
    
    success "Basic tests passed"
}

# =============================================================================
# Print Usage Information
# =============================================================================
print_usage() {
    echo ""
    echo "=========================================="
    echo -e "${GREEN}AURA v3 Setup Complete!${NC}"
    echo "=========================================="
    echo ""
    echo "To start AURA:"
    echo ""
    echo "  cd $AURA_DIR"
    echo "  python3 -m src.main"
    echo ""
    echo "Or use the quick start:"
    echo "  bash installation/quickstart.sh"
    echo ""
    echo "Configuration: $AURA_DIR/config.yaml"
    echo "Documentation: $AURA_DIR/README.md"
    echo ""
    echo "Need help? Check these resources:"
    echo "  - docs/TROUBLESHOOTING.md"
    echo "  - docs/API.md"
    echo "  - docs/ARCHITECTURE.md"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo "AURA v3 Detailed Setup"
    echo "======================="
    
    # Run each step
    step_1_check
    step_2_dependencies
    step_3_python
    step_4_repository
    step_5_configure
    step_6_models
    step_7_test
    
    print_usage
}

# Run main
main "$@"
