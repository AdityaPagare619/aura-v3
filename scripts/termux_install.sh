#!/data/data/com.termux/files/usr/bin/bash
# =============================================================================
# AURA v3 — Termux Production Installer (v2.2 — curl|bash safe)
#
# DESIGN: No set -e, no set -o pipefail, no heredocs.
# These all cause issues when piped via curl | bash.
# Exit codes are checked explicitly where needed.
# =============================================================================

# ─── Establish known CWD (user may have rm -rf'd it) ─────────────────────────
cd "$HOME" 2>/dev/null || cd /data/data/com.termux/files/home 2>/dev/null || true

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# ─── Globals ─────────────────────────────────────────────────────────────────
AURA_DIR="$HOME/aura-v3"
AURA_BRANCH="production-hardening"
LOG_FILE=""
ERRORS_FOUND=0
PYTHON_CMD=""

# ─── Parse arguments ─────────────────────────────────────────────────────────
REPAIR_MODE=false
CHECK_MODE=false
VERBOSE=false

while [ $# -gt 0 ]; do
    case $1 in
        --repair)  REPAIR_MODE=true; shift ;;
        --check)   CHECK_MODE=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        *)         echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# ─── Logging (two-phase: stdout only before clone, file after) ───────────────
log_info() {
    echo -e "${BLUE}[INFO]${NC}  $1"
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        echo "[INFO]  $(date +%H:%M:%S) $1" >> "$LOG_FILE"
    fi
    return 0
}

log_ok() {
    echo -e "${GREEN}[OK]${NC}    $1"
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        echo "[OK]    $(date +%H:%M:%S) $1" >> "$LOG_FILE"
    fi
    return 0
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC}  $1"
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        echo "[WARN]  $(date +%H:%M:%S) $1" >> "$LOG_FILE"
    fi
    return 0
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        echo "[ERROR] $(date +%H:%M:%S) $1" >> "$LOG_FILE"
    fi
    ERRORS_FOUND=$((ERRORS_FOUND + 1))
    return 0
}

enable_file_logging() {
    mkdir -p "$AURA_DIR/logs" 2>/dev/null
    LOG_FILE="$AURA_DIR/logs/install.log"
    echo "=== AURA Install Log $(date) ===" > "$LOG_FILE"
    log_info "File logging enabled"
    return 0
}

# ─── Platform Check ──────────────────────────────────────────────────────────
check_platform() {
    if [ -z "$TERMUX_VERSION" ] && [ ! -d "/data/data/com.termux" ]; then
        log_error "This script is for Termux only!"
        exit 1
    fi
    log_ok "Termux detected (version: ${TERMUX_VERSION:-unknown})"

    if command -v python3 > /dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python > /dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        log_error "Python not found! Run: pkg install python"
        exit 1
    fi

    PYTHON_VERSION="$($PYTHON_CMD --version 2>&1 | awk '{print $2}')"
    PYTHON_MAJOR="$(echo "$PYTHON_VERSION" | cut -d. -f1)"
    PYTHON_MINOR="$(echo "$PYTHON_VERSION" | cut -d. -f2)"

    if [ "$PYTHON_MAJOR" -lt 3 ] 2>/dev/null || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; } 2>/dev/null; then
        log_error "Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi

    log_ok "Python $PYTHON_VERSION"
    log_info "Architecture: $(uname -m)"
    return 0
}

# ─── Clone or Update ─────────────────────────────────────────────────────────
clone_or_update() {
    log_info "Getting AURA v3..."
    cd "$HOME"

    if [ -d "$AURA_DIR/.git" ]; then
        log_info "Updating existing installation..."
        cd "$AURA_DIR"
        git fetch origin "$AURA_BRANCH" 2>/dev/null || true
        git checkout "$AURA_BRANCH" 2>/dev/null || git checkout -b "$AURA_BRANCH" "origin/$AURA_BRANCH" 2>/dev/null || true
        git pull origin "$AURA_BRANCH" 2>/dev/null || log_warn "Could not update"
        log_ok "AURA updated (branch: $AURA_BRANCH)"
    elif [ -d "$AURA_DIR" ]; then
        log_warn "Found leftover $AURA_DIR without .git — cleaning..."
        rm -rf "$AURA_DIR"
        git clone --branch "$AURA_BRANCH" --single-branch https://github.com/AdityaPagare619/aura-v3.git "$AURA_DIR"
        cd "$AURA_DIR"
        log_ok "AURA cloned fresh (branch: $AURA_BRANCH)"
    else
        log_info "Cloning AURA v3 (branch: $AURA_BRANCH)..."
        git clone --branch "$AURA_BRANCH" --single-branch https://github.com/AdityaPagare619/aura-v3.git "$AURA_DIR"
        cd "$AURA_DIR"
        log_ok "AURA cloned to $AURA_DIR"
    fi
    return 0
}

# ─── Install Termux-Native Packages ──────────────────────────────────────────
install_termux_native_deps() {
    log_info "Installing Termux-native packages..."

    pkg update -y -o Dpkg::Options::="--force-confnew" 2>/dev/null || true
    pkg upgrade -y -o Dpkg::Options::="--force-confnew" 2>/dev/null || true

    for pkg_name in git python clang cmake ninja; do
        if command -v "$pkg_name" > /dev/null 2>&1; then
            log_ok "$pkg_name already installed"
        else
            pkg install -y "$pkg_name" 2>/dev/null && log_ok "$pkg_name installed" || log_error "Failed to install $pkg_name"
        fi
    done

    pkg install -y build-essential libffi openssl 2>/dev/null || true

    # cryptography
    if $PYTHON_CMD -c "import cryptography" 2>/dev/null; then
        log_ok "cryptography already importable"
    else
        log_info "Installing cryptography via Termux pkg..."
        pkg install -y python-cryptography 2>/dev/null || true
        if $PYTHON_CMD -c "import cryptography" 2>/dev/null; then
            log_ok "cryptography installed and verified"
        else
            log_warn "Trying cryptography via pip..."
            pip install cryptography 2>/dev/null || true
            if $PYTHON_CMD -c "import cryptography" 2>/dev/null; then
                log_ok "cryptography via pip verified"
            else
                log_error "cryptography import FAILED"
            fi
        fi
    fi

    # numpy
    if $PYTHON_CMD -c "import numpy" 2>/dev/null; then
        log_ok "numpy already importable"
    else
        log_info "Installing numpy via Termux pkg..."
        pkg install -y python-numpy 2>/dev/null || true
        if $PYTHON_CMD -c "import numpy" 2>/dev/null; then
            log_ok "numpy installed and verified"
        else
            log_warn "Trying numpy via pip..."
            pip install numpy 2>/dev/null || log_error "numpy install failed"
        fi
    fi
    return 0
}

# ─── Install Pip Dependencies ────────────────────────────────────────────────
install_pip_deps() {
    log_info "Installing Python dependencies via pip..."
    $PYTHON_CMD -m pip install --upgrade pip --quiet 2>/dev/null || true

    for dep in aiofiles pyyaml psutil python-telegram-bot; do
        local import_name="$dep"
        case "$dep" in
            pyyaml)              import_name="yaml" ;;
            python-telegram-bot) import_name="telegram" ;;
        esac

        if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
            log_ok "$dep already installed"
            continue
        fi

        log_info "Installing $dep..."
        pip install "$dep" 2>/dev/null || true

        if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
            log_ok "$dep installed and verified"
        else
            log_error "$dep FAILED"
        fi
    done
    return 0
}

# ─── Install LLM ─────────────────────────────────────────────────────────────
install_llm_deps() {
    log_info "Installing Local LLM Engine (llama-cpp-python)..."

    if $PYTHON_CMD -c "import llama_cpp" 2>/dev/null; then
        log_ok "llama-cpp-python already installed"
        return 0
    fi

    log_info "Attempting llama-cpp-python install (may take minutes)..."
    export CMAKE_ARGS="-DLLAMA_METAL=off -DLLAMA_CUDA=off"

    pip install llama-cpp-python --only-binary :all: 2>/dev/null || true
    if $PYTHON_CMD -c "import llama_cpp" 2>/dev/null; then
        log_ok "llama-cpp-python (pre-built) installed"
        return 0
    fi

    log_warn "No pre-built wheel. Compiling from source..."
    pip install llama-cpp-python --no-cache-dir 2>/dev/null || true
    if $PYTHON_CMD -c "import llama_cpp" 2>/dev/null; then
        log_ok "llama-cpp-python compiled and installed"
    else
        log_warn "llama-cpp-python failed (AURA will use mock LLM mode)"
        log_warn "Install manually later: pip install llama-cpp-python --no-cache-dir"
    fi
    return 0
}

# ─── Verify All Imports ──────────────────────────────────────────────────────
verify_all_imports() {
    echo ""
    echo -e "${BOLD}── Verifying All Imports ──${NC}"

    local ALL_PASS=true

    for pkg_import in "aiofiles:aiofiles" "yaml:pyyaml" "cryptography:cryptography" "psutil:psutil" "telegram:python-telegram-bot" "numpy:python-numpy" "llama_cpp:llama-cpp-python"; do
        local import_name="${pkg_import%%:*}"
        local pkg_name="${pkg_import##*:}"

        if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
            log_ok "$import_name ✓"
        else
            if [ "$import_name" = "llama_cpp" ]; then
                log_warn "$import_name ✗ (LLM mock mode)"
            else
                log_error "$import_name ✗ → pip install $pkg_name"
                ALL_PASS=false
            fi
        fi
    done

    if [ -d "$AURA_DIR/src" ]; then
        cd "$AURA_DIR"
        if $PYTHON_CMD -c "import sys; sys.path.insert(0, '.'); from src.main import AuraProduction; print('OK')" 2>/dev/null; then
            log_ok "AURA modules ✓"
        else
            log_warn "AURA module import needs debugging"
        fi
    fi

    if [ "$ALL_PASS" = true ]; then
        log_ok "All core imports verified!"
    fi
    return 0
}

# ─── Setup .env (NO heredocs — curl|bash incompatible) ───────────────────────
setup_env() {
    log_info "Setting up configuration..."
    cd "$AURA_DIR"

    if [ -f ".env" ]; then
        if grep -q "TELEGRAM_TOKEN=.\+" ".env" 2>/dev/null; then
            log_ok ".env exists with TELEGRAM_TOKEN set"
            return 0
        fi
        log_warn ".env exists but TELEGRAM_TOKEN is empty"
    fi

    echo ""
    echo -e "${BOLD}Telegram Bot Token Setup${NC}"
    echo "Get your token from @BotFather on Telegram."
    echo ""
    echo -n "Paste your bot token (or press Enter to skip): "

    read -r TOKEN </dev/tty || TOKEN=""

    if [ -n "$TOKEN" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
        fi

        if [ -f ".env" ]; then
            if grep -q "^TELEGRAM_TOKEN=" ".env"; then
                sed -i "s|^TELEGRAM_TOKEN=.*|TELEGRAM_TOKEN=$TOKEN|" .env
            else
                echo "TELEGRAM_TOKEN=$TOKEN" >> .env
            fi
            if grep -q "^TELEGRAM_BOT_TOKEN=" ".env"; then
                sed -i "s|^TELEGRAM_BOT_TOKEN=.*|TELEGRAM_BOT_TOKEN=$TOKEN|" .env
            else
                echo "TELEGRAM_BOT_TOKEN=$TOKEN" >> .env
            fi
        else
            echo "TELEGRAM_TOKEN=$TOKEN" > .env
            echo "TELEGRAM_BOT_TOKEN=$TOKEN" >> .env
            echo "AURA_ENV=production" >> .env
            echo "AURA_MOCK_LLM=false" >> .env
        fi
        log_ok "Token saved to .env"
    else
        if [ ! -f ".env" ]; then
            if [ -f ".env.example" ]; then
                cp .env.example .env
            else
                echo "TELEGRAM_TOKEN=" > .env
                echo "AURA_ENV=production" >> .env
                echo "AURA_MOCK_LLM=false" >> .env
            fi
        fi
        log_warn "No token entered — set later: nano ~/aura-v3/.env"
    fi
    return 0
}

# ─── Create Directories ─────────────────────────────────────────────────────
create_dirs() {
    log_info "Creating data directories..."
    cd "$AURA_DIR"
    mkdir -p logs data/memories data/sessions data/patterns/intents data/patterns/strategies data/memory models cache
    log_ok "Directories created"
    return 0
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo -e "${BOLD}══════════════════════════════════════════${NC}"
    echo -e "${BOLD}   AURA v3 — Termux Production Installer${NC}"
    echo -e "${BOLD}══════════════════════════════════════════${NC}"
    echo ""

    echo -e "${BLUE}[1/7]${NC} Checking platform..."
    check_platform

    echo ""
    echo -e "${BLUE}[2/7]${NC} Cloning / updating repository..."
    clone_or_update

    enable_file_logging

    echo ""
    echo -e "${BLUE}[3/7]${NC} Installing Termux-native packages..."
    install_termux_native_deps

    echo ""
    echo -e "${BLUE}[4/7]${NC} Installing Python dependencies..."
    install_pip_deps

    echo ""
    echo -e "${BLUE}[5/7]${NC} Installing Local LLM Engine..."
    install_llm_deps

    echo ""
    echo -e "${BLUE}[6/7]${NC} Creating directories..."
    create_dirs

    echo ""
    echo -e "${BLUE}[7/7]${NC} Configuring environment..."
    setup_env

    verify_all_imports

    echo ""
    echo -e "${BOLD}══════════════════════════════════════════${NC}"
    if [ "$ERRORS_FOUND" -eq 0 ]; then
        echo -e "${GREEN}${BOLD}   Installation Complete!${NC}"
    else
        echo -e "${YELLOW}${BOLD}   Installation Complete with $ERRORS_FOUND error(s)${NC}"
        echo -e "   Check: $LOG_FILE"
    fi
    echo -e "${BOLD}══════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}IMPORTANT: To keep AURA alive in background:${NC}"
    echo -e "  termux-wake-lock"
    echo ""
    echo "Next steps:"
    echo "  cd ~/aura-v3 && bash run_aura.sh telegram"
    echo "  cd ~/aura-v3 && bash run_aura.sh cli"
    echo "  cd ~/aura-v3 && python scripts/aura_doctor.py"
    echo ""
    return 0
}

main "$@"
