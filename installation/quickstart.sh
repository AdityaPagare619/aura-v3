#!/bin/bash
#
# AURA v3 - Quick Install Script
# =============================
# One-command installation for Termux/Android
#
# Usage: curl -sSL https://raw.githubusercontent.com/aura-ai/aura-v3/main/installation/quickstart.sh | bash

set -euo pipefail

AURA_DIR="${AURA_DIR:-$HOME/aura-v3}"
REPO_URL="https://github.com/aura-ai/aura-v3.git"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_step()   { echo -e "${BLUE}[STEP]${NC} $1"; }
log_ok()     { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()  { echo -e "${RED}[ERROR]${NC} $1"; }
log_info()   { echo -e "${CYAN}[INFO]${NC} $1"; }

is_termux() {
    [ -f /data/data/com.termux/files/usr/bin/pkg ]
}

detect_os() {
    if is_termux; then
        echo "termux"
    elif [ "$(uname)" = "Darwin" ]; then
        echo "macos"
    elif [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    else
        echo "unknown"
    fi
}

check_prerequisites() {
    log_step "Checking prerequisites..."
    
    OS=$(detect_os)
    log_info "Detected OS: $OS"
    
    if ! is_termux; then
        log_warn "Not running in Termux. This script is optimized for Termux/Android."
        echo ""
        echo "For other systems, please install manually:"
        echo "  1. Install Python 3.8+: python3 --version"
        echo "  2. Install git: apt install git (or brew install git)"
        echo "  3. Clone: git clone $REPO_URL"
        echo "  4. Install deps: pip install -r requirements.txt"
        echo ""
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Installation cancelled."
            exit 0
        fi
    fi
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        log_ok "Python $PYTHON_VERSION found"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
        log_ok "Python $PYTHON_VERSION found (python command)"
    else
        log_error "Python not found."
        if is_termux; then
            echo ""
            echo "Install Python in Termux with:"
            echo "  pkg install python"
        fi
        exit 1
    fi
    
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        log_error "Python 3.8+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    log_ok "Prerequisites OK"
}

install_dependencies() {
    log_step "Installing system dependencies..."
    
    if is_termux; then
        pkg update -y 2>/dev/null || log_warn "pkg update failed, continuing..."
        
        pkg install -y git curl wget python libjpeg-turbo openssl 2>/dev/null || true
        
        pkg install -y libcryptography 2>/dev/null || true
        
        log_ok "Termux packages installed"
        
    elif [ "$(detect_os)" = "ubuntu" ] || [ "$(detect_os)" = "debian" ]; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3 python3-pip git curl wget libssl-dev libffi-dev
        log_ok "Debian/Ubuntu packages installed"
        
    elif [ "$(detect_os)" = "macos" ]; then
        if command -v brew &> /dev/null; then
            brew install python3 git curl wget
        else
            log_error "Homebrew not found. Install from https://brew.sh"
            exit 1
        fi
        log_ok "macOS packages installed"
    fi
}

install_python_packages() {
    log_step "Installing Python packages..."
    
    python3 -m pip install --upgrade pip --user 2>/dev/null || true
    
    CORE_PACKAGES="aiohttp numpy scipy pyyaml requests sounddevice"
    
    for pkg in $CORE_PACKAGES; do
        if python3 -c "import $pkg" 2>/dev/null; then
            log_ok "$pkg already installed"
        else
            python3 -m pip install --user "$pkg" 2>/dev/null || log_warn "Failed to install $pkg"
        fi
    done
    
    log_ok "Python packages ready"
}

clone_or_update() {
    log_step "Setting up AURA repository..."
    
    if [ -d "$AURA_DIR/.git" ]; then
        log_info "AURA already exists at $AURA_DIR"
        read -p "Pull latest changes? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd "$AURA_DIR"
            git pull origin main 2>/dev/null || log_warn "Git pull failed, using existing version"
            log_ok "Repository updated"
        fi
    else
        if [ -d "$AURA_DIR" ]; then
            log_warn "Directory $AURA_DIR exists but is not a git repo"
            read -p "Remove and re-clone? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$AURA_DIR"
            else
                log_info "Using existing directory"
                return 0
            fi
        fi
        
        log_info "Cloning AURA v3..."
        git clone --depth 1 "$REPO_URL" "$AURA_DIR" 2>/dev/null || {
            log_error "Failed to clone repository"
            log_info "Check your internet connection and try again"
            exit 1
        }
        log_ok "Repository cloned"
    fi
}

setup_config() {
    log_step "Setting up configuration..."
    
    cd "$AURA_DIR"
    
    if [ -f config.yaml ]; then
        log_ok "Configuration already exists"
        return 0
    fi
    
    if [ -f config.example.yaml ]; then
        cp config.example.yaml config.yaml
        log_ok "Configuration created from template"
    else
        cat > config.yaml << 'EOF'
core:
  llm:
    provider: "llama.cpp"
    model_path: "./models/llama-7b-chat.gguf"
    max_tokens: 1024
    temperature: 0.7

power:
  full_power_threshold: 50
  balanced_threshold: 20
  power_save_threshold: 10
  critical_threshold: 5
  max_tokens_full: 2048
  max_tokens_balanced: 1024
  max_tokens_save: 512
  max_tokens_ultra: 256

privacy:
  log_level: "WARNING"
  store_conversations: true
  anonymize_data: true
  allow_telemetry: false
  encrypt_storage: false

features:
  voice_enabled: false
  proactive_mode: true
  background_tasks: true
  notifications: true
EOF
        log_ok "Default configuration created"
    fi
}

prompt_telegram() {
    echo ""
    log_step "Telegram Bot Setup (optional)"
    echo "==============================="
    echo "AURA can integrate with Telegram for remote control."
    echo "You need a Bot Token from @BotFather on Telegram."
    echo ""
    read -p "Set up Telegram bot now? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Enter your Telegram Bot Token (from @BotFather):"
        read -p "Token: " TELEGRAM_TOKEN
        
        if [ -n "$TELEGRAM_TOKEN" ]; then
            cd "$AURA_DIR"
            
            if ! grep -q "telegram:" config.yaml; then
                cat >> config.yaml << EOF

telegram:
  enabled: true
  bot_token: "$TELEGRAM_TOKEN"
  allowed_users: []
EOF
                log_ok "Telegram bot configured"
                log_info "Start a chat with your bot and send /start to authorize"
            fi
        fi
    else
        log_info "Telegram setup skipped"
    fi
}

prompt_llm_model() {
    echo ""
    log_step "LLM Model Selection"
    echo "==================="
    echo "Choose your LLM provider:"
    echo ""
    echo "  1) llama.cpp  - Local GGUF models (recommended for privacy)"
    echo "  2) ollama     - Ollama API (local or remote)"
    echo "  3) transformers - Hugging Face transformers"
    echo "  4) api        - External API (OpenAI, Anthropic, etc.)"
    echo ""
    read -p "Choice (1-4, default 1): " LLM_CHOICE
    
    LLM_CHOICE="${LLM_CHOICE:-1}"
    
    cd "$AURA_DIR"
    
    case $LLM_CHOICE in
        1)
            sed -i 's/provider: ".*"/provider: "llama.cpp"/' config.yaml 2>/dev/null || true
            log_ok "LLM provider set to llama.cpp"
            echo ""
            echo "Place GGUF model files in: $AURA_DIR/models/"
            echo "Download from: https://huggingface.co/TheBloke"
            ;;
        2)
            sed -i 's/provider: ".*"/provider: "ollama"/' config.yaml 2>/dev/null || true
            log_ok "LLM provider set to ollama"
            echo ""
            read -p "Ollama URL (default http://localhost:11434): " OLLAMA_URL
            OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
            
            if ! grep -q "ollama_url" config.yaml; then
                sed -i "/provider: \"ollama\"/a\\    ollama_url: \"$OLLAMA_URL\"" config.yaml 2>/dev/null || true
            fi
            ;;
        3)
            sed -i 's/provider: ".*"/provider: "transformers"/' config.yaml 2>/dev/null || true
            log_ok "LLM provider set to transformers"
            ;;
        4)
            sed -i 's/provider: ".*"/provider: "api"/' config.yaml 2>/dev/null || true
            log_ok "LLM provider set to external API"
            echo ""
            echo "Configure your API key in config.yaml:"
            echo "  core.llm.api_key: YOUR_API_KEY"
            ;;
    esac
}

prompt_privacy() {
    echo ""
    log_step "Privacy Preferences"
    echo "===================="
    echo ""
    echo "Configure privacy settings:"
    echo ""
    
    cd "$AURA_DIR"
    
    read -p "Allow telemetry/anonymous usage data? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sed -i 's/allow_telemetry: false/allow_telemetry: true/' config.yaml 2>/dev/null || true
        log_ok "Telemetry enabled"
    else
        sed -i 's/allow_telemetry: true/allow_telemetry: false/' config.yaml 2>/dev/null || true
        log_info "Telemetry disabled (recommended)"
    fi
    
    read -p "Store conversations for memory? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        sed -i 's/store_conversations: true/store_conversations: false/' config.yaml 2>/dev/null || true
        log_info "Conversation storage disabled"
    else
        log_ok "Conversation storage enabled"
    fi
    
    read -p "Anonymize stored data? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        sed -i 's/anonymize_data: true/anonymize_data: false/' config.yaml 2>/dev/null || true
    else
        log_ok "Data anonymization enabled"
    fi
}

interactive_setup() {
    log_step "Running interactive setup..."
    
    prompt_telegram
    prompt_llm_model
    prompt_privacy
    
    log_ok "Interactive setup complete"
}

create_launcher() {
    log_step "Creating launcher script..."
    
    cat > "$AURA_DIR/aura.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
exec python3 -m src.main "$@"
EOF
    chmod +x "$AURA_DIR/aura.sh"
    
    log_ok "Launcher script created"
}

create_alias() {
    log_step "Setting up command alias..."
    
    ALIAS_LINE="alias aura='$AURA_DIR/aura.sh'"
    RC_FILE="$HOME/.bashrc"
    
    if [ -n "${TERMUX_VERSION:-}" ] || is_termux; then
        RC_FILE="$HOME/.bashrc"
    fi
    
    if [ -f "$RC_FILE" ]; then
        if ! grep -q "alias aura=" "$RC_FILE" 2>/dev/null; then
            echo "" >> "$RC_FILE"
            echo "# AURA v3" >> "$RC_FILE"
            echo "$ALIAS_LINE" >> "$RC_FILE"
            log_ok "Alias added to $RC_FILE"
            log_info "Run 'source $RC_FILE' or restart shell to use 'aura' command"
        else
            log_ok "Alias already exists"
        fi
    else
        echo "# AURA v3" > "$RC_FILE"
        echo "$ALIAS_LINE" >> "$RC_FILE"
        log_ok "Created $RC_FILE with aura alias"
    fi
}

verify_installation() {
    log_step "Verifying installation..."
    
    cd "$AURA_DIR"
    
    if python3 -c "import sys; sys.path.insert(0, '.'); import src" 2>/dev/null; then
        log_ok "AURA modules importable"
    else
        log_warn "Some modules may need attention"
        log_info "Try: cd $AURA_DIR && pip install -r requirements.txt"
    fi
    
    if [ -f config.yaml ]; then
        log_ok "Configuration valid"
    fi
}

print_summary() {
    echo ""
    echo "========================================"
    echo -e "  ${GREEN}AURA v3 Installation Complete!${NC}"
    echo "========================================"
    echo ""
    echo "Location: $AURA_DIR"
    echo ""
    echo "To start AURA:"
    echo "  cd $AURA_DIR"
    echo "  ./aura.sh"
    echo ""
    echo "Or use the alias (restart shell first):"
    echo "  aura"
    echo ""
    echo "Configuration: $AURA_DIR/config.yaml"
    echo ""
    echo "Need help? Check $AURA_DIR/README.md"
    echo ""
}

main() {
    echo ""
    echo "========================================"
    echo -e "  ${BLUE}AURA v3 Quick Installer${NC}"
    echo "========================================"
    echo ""
    
    check_prerequisites
    install_dependencies
    install_python_packages
    clone_or_update
    setup_config
    
    if [ -t 0 ]; then
        interactive_setup
    else
        log_info "Non-interactive mode - skipping prompts"
    fi
    
    create_launcher
    create_alias
    verify_installation
    print_summary
}

main "$@"
