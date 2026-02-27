#!/data/data/com.termux/files/usr/bin/bash
# =============================================================================
# AURA v3 — Termux Production Installer (Rewrite v2)
#
# Fixes addressed:
#   Bug 1: Two-phase logging (stdout before clone, file after)
#   Bug 2: Clone with --branch production-hardening
#   Bug 3: cd "$HOME" as first action
#   Bug 4: set -o pipefail + proper exit code capture
#   Bug 5: Post-install import verification for every package
#   Bug 6: Absolute paths in "Next Steps"
#   Bug 7: No silent error suppression on critical paths
# =============================================================================
set -e
set -o pipefail

# ─── CRITICAL: Establish known CWD immediately (Bug 3) ───────────────────────
# User may run `rm -rf ~/aura-v3` while CWD IS ~/aura-v3, which
# invalidates the shell's CWD.  Fix: unconditionally cd home first.
cd "$HOME" || { echo "FATAL: Cannot cd to \$HOME"; exit 1; }

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
LOG_FILE=""  # Set AFTER clone + mkdir (Bug 1)
ERRORS_FOUND=0

# ─── Parse arguments ─────────────────────────────────────────────────────────
REPAIR_MODE=false
CHECK_MODE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --repair)  REPAIR_MODE=true; shift ;;
        --check)   CHECK_MODE=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        *)         echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# ─── Two-Phase Logging (Bug 1 + Bug 7) ──────────────────────────────────────
# Phase 1: Before clone — stdout only, no file logging
# Phase 2: After clone — stdout + file
log_info()  {
    echo -e "${BLUE}[INFO]${NC}  $1"
    [ -n "$LOG_FILE" ] && echo "[INFO]  $(date +%H:%M:%S) $1" >> "$LOG_FILE" 2>/dev/null
}
log_ok()    {
    echo -e "${GREEN}[OK]${NC}    $1"
    [ -n "$LOG_FILE" ] && echo "[OK]    $(date +%H:%M:%S) $1" >> "$LOG_FILE" 2>/dev/null
}
log_warn()  {
    echo -e "${YELLOW}[WARN]${NC}  $1"
    [ -n "$LOG_FILE" ] && echo "[WARN]  $(date +%H:%M:%S) $1" >> "$LOG_FILE" 2>/dev/null
}
log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    [ -n "$LOG_FILE" ] && echo "[ERROR] $(date +%H:%M:%S) $1" >> "$LOG_FILE" 2>/dev/null
    ERRORS_FOUND=$((ERRORS_FOUND + 1))
}

# Enable file logging (called AFTER clone + mkdir)
enable_file_logging() {
    mkdir -p "$AURA_DIR/logs"
    LOG_FILE="$AURA_DIR/logs/install.log"
    echo "=== AURA Install Log $(date) ===" > "$LOG_FILE"
    log_info "File logging enabled at $LOG_FILE"
}

# ─── Platform Check ──────────────────────────────────────────────────────────
check_platform() {
    if [ -z "$TERMUX_VERSION" ] && [ ! -d "/data/data/com.termux" ]; then
        log_error "This script is for Termux only! Use scripts/install-aura.sh for other platforms."
        exit 1
    fi

    log_ok "Termux detected (version: ${TERMUX_VERSION:-unknown})"

    # Find Python
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python not found! Run: pkg install python"
        exit 1
    fi

    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        log_error "Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi

    log_ok "Python $PYTHON_VERSION"
    log_info "Architecture: $(uname -m)"
}

# ─── Clone or Update (Bug 2: explicit branch) ───────────────────────────────
clone_or_update() {
    echo ""
    log_info "Getting AURA v3..."

    # Ensure we're in $HOME (not a deleted directory)
    cd "$HOME"

    if [ -d "$AURA_DIR/.git" ]; then
        # Existing repo — update it
        log_info "Updating existing installation..."
        cd "$AURA_DIR"

        # Make sure we're on the right branch
        local current_branch
        current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
        if [ "$current_branch" != "$AURA_BRANCH" ]; then
            log_info "Switching from '$current_branch' to '$AURA_BRANCH'..."
            git fetch origin "$AURA_BRANCH" || log_warn "Fetch failed"
            git checkout "$AURA_BRANCH" || git checkout -b "$AURA_BRANCH" "origin/$AURA_BRANCH" || log_warn "Branch checkout failed"
        fi

        git pull origin "$AURA_BRANCH" || log_warn "Could not update (using existing version)"
        log_ok "AURA updated on branch '$AURA_BRANCH'"
    elif [ -d "$AURA_DIR" ]; then
        # Directory exists but no .git — leftover from partial install
        log_warn "Found leftover $AURA_DIR without .git — cleaning up..."
        rm -rf "$AURA_DIR"
        git clone --branch "$AURA_BRANCH" --single-branch \
            https://github.com/AdityaPagare619/aura-v3.git "$AURA_DIR"
        cd "$AURA_DIR"
        log_ok "AURA cloned fresh (branch: $AURA_BRANCH)"
    else
        # Fresh install
        log_info "Cloning AURA v3 (branch: $AURA_BRANCH)..."
        git clone --branch "$AURA_BRANCH" --single-branch \
            https://github.com/AdityaPagare619/aura-v3.git "$AURA_DIR"
        cd "$AURA_DIR"
        log_ok "AURA cloned to $AURA_DIR (branch: $AURA_BRANCH)"
    fi
}

# ─── Install Termux-Native Packages ──────────────────────────────────────────
install_termux_native_deps() {
    echo ""
    log_info "Installing Termux-native packages..."

    # Update package list (non-fatal on warnings)
    pkg update -y -o Dpkg::Options::="--force-confnew" || log_warn "pkg update had warnings (continuing)"
    pkg upgrade -y -o Dpkg::Options::="--force-confnew" || log_warn "pkg upgrade had warnings (continuing)"

    # Core build tools
    local CORE_PKGS="git python clang cmake ninja"
    for pkg_name in $CORE_PKGS; do
        if command -v "$pkg_name" &>/dev/null; then
            log_ok "$pkg_name already installed"
        else
            if pkg install -y "$pkg_name"; then
                log_ok "$pkg_name installed"
            else
                log_error "Failed to install $pkg_name"
            fi
        fi
    done

    # Build deps for compilation
    pkg install -y build-essential libffi openssl 2>/dev/null || log_warn "Some build deps may have failed"

    # ── cryptography: pkg first, pip fallback (Bug 5: verify after install) ──
    if $PYTHON_CMD -c "import cryptography" 2>/dev/null; then
        log_ok "cryptography already importable"
    else
        log_info "Installing cryptography via Termux pkg (pre-built, no Rust needed)..."
        pkg install -y python-cryptography

        # VERIFY it actually works (Bug 5)
        if $PYTHON_CMD -c "import cryptography" 2>/dev/null; then
            log_ok "cryptography installed and verified"
        else
            log_warn "pkg cryptography installed but Python can't import it. Trying pip..."
            # This might need Rust, warn the user
            if pip install cryptography 2>&1; then
                if $PYTHON_CMD -c "import cryptography" 2>/dev/null; then
                    log_ok "cryptography installed via pip and verified"
                else
                    log_error "cryptography import STILL fails after pip install"
                fi
            else
                log_error "cryptography installation FAILED — secure storage will be degraded"
            fi
        fi
    fi

    # ── numpy: pkg first, pip fallback ──
    if $PYTHON_CMD -c "import numpy" 2>/dev/null; then
        log_ok "numpy already importable"
    else
        log_info "Installing numpy via Termux pkg (pre-built)..."
        pkg install -y python-numpy

        # VERIFY
        if $PYTHON_CMD -c "import numpy" 2>/dev/null; then
            log_ok "numpy installed and verified"
        else
            log_warn "pkg numpy installed but Python can't import it. Trying pip..."
            pip install numpy || log_error "Failed to install numpy"
        fi
    fi
}

# ─── Install Pip Dependencies ────────────────────────────────────────────────
install_pip_deps() {
    echo ""
    log_info "Installing Python dependencies via pip..."

    # Upgrade pip
    $PYTHON_CMD -m pip install --upgrade pip --quiet 2>/dev/null || true

    local CORE_DEPS="aiofiles pyyaml psutil python-telegram-bot"

    for dep in $CORE_DEPS; do
        # Map package name to import name
        local import_name="$dep"
        case "$dep" in
            pyyaml)              import_name="yaml" ;;
            python-telegram-bot) import_name="telegram" ;;
        esac

        # Check if already importable
        if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
            log_ok "$dep already installed"
            continue
        fi

        # Install and verify (Bug 4: don't pipe through tee for exit code)
        log_info "Installing $dep..."
        local pip_output
        if pip_output=$(pip install "$dep" 2>&1); then
            # Verify the import actually works
            if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
                log_ok "$dep installed and verified"
            else
                log_error "$dep pip says success but import failed!"
                echo "$pip_output" >> "$LOG_FILE" 2>/dev/null || true
            fi
        else
            log_error "$dep installation FAILED"
            echo "$pip_output" >> "$LOG_FILE" 2>/dev/null || true
        fi
    done
}

# ─── Install LLM Dependencies (Bug 4: proper exit code) ─────────────────────
install_llm_deps() {
    echo ""
    log_info "Installing Local LLM Engine (llama-cpp-python)..."

    if $PYTHON_CMD -c "import llama_cpp" 2>/dev/null; then
        log_ok "llama-cpp-python already installed"
        return
    fi

    log_info "Building llama-cpp-python for Termux (may take several minutes)..."

    # Set cmake flags to avoid NDK issues
    export CMAKE_ARGS="-DLLAMA_METAL=off -DLLAMA_CUDA=off"

    # Bug 4 fix: capture output WITHOUT piping through tee
    # Try pre-built first
    local pip_output
    if pip_output=$(pip install llama-cpp-python --only-binary :all: 2>&1); then
        if $PYTHON_CMD -c "import llama_cpp" 2>/dev/null; then
            log_ok "llama-cpp-python (pre-built) installed and verified"
            echo "$pip_output" >> "$LOG_FILE" 2>/dev/null || true
            return
        fi
    fi

    # Pre-built not available (expected on aarch64), try from source
    log_warn "No pre-built wheel for this platform. Compiling from source..."
    if pip_output=$(pip install llama-cpp-python --no-cache-dir 2>&1); then
        if $PYTHON_CMD -c "import llama_cpp" 2>/dev/null; then
            log_ok "llama-cpp-python compiled and installed"
        else
            log_warn "llama-cpp-python pip succeeded but import failed"
            log_warn "AURA will still run, but in MOCK LLM mode"
        fi
    else
        log_warn "llama-cpp-python compilation failed (this is OK for now)"
        log_warn "AURA will run in MOCK LLM mode. Install manually later with:"
        log_warn "  pip install llama-cpp-python --no-cache-dir"
    fi
    echo "$pip_output" >> "$LOG_FILE" 2>/dev/null || true
}

# ─── Verify All Imports ──────────────────────────────────────────────────────
verify_all_imports() {
    echo ""
    echo -e "${BOLD}── Verifying All Imports ──${NC}"

    local ALL_PASS=true

    for pkg_import in "aiofiles:aiofiles" "yaml:pyyaml" "cryptography:cryptography" \
                      "psutil:psutil" "telegram:python-telegram-bot" "numpy:python-numpy" \
                      "llama_cpp:llama-cpp-python"; do
        local import_name="${pkg_import%%:*}"
        local pkg_name="${pkg_import##*:}"

        if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
            log_ok "$import_name ✓"
        else
            if [ "$import_name" = "llama_cpp" ]; then
                log_warn "$import_name ✗ (LLM will run in mock mode)"
            else
                log_error "$import_name ✗ → pip install $pkg_name"
                ALL_PASS=false
            fi
        fi
    done

    # Test AURA module import
    if [ -d "$AURA_DIR/src" ]; then
        cd "$AURA_DIR"
        if $PYTHON_CMD -c "import sys; sys.path.insert(0, '.'); from src.main import AuraProduction; print('OK')" 2>/dev/null; then
            log_ok "AURA modules ✓"
        else
            log_warn "AURA module import needs debugging (non-fatal)"
        fi
    fi

    if [ "$ALL_PASS" = true ]; then
        log_ok "All core imports verified!"
    fi
}

# ─── Setup .env ──────────────────────────────────────────────────────────────
setup_env() {
    echo ""
    log_info "Setting up configuration..."

    cd "$AURA_DIR"

    if [ -f ".env" ]; then
        if grep -q "TELEGRAM_TOKEN=.\+" ".env" 2>/dev/null; then
            log_ok ".env exists with TELEGRAM_TOKEN set"
            return
        fi
        log_warn ".env exists but TELEGRAM_TOKEN is empty"
    fi

    # Interactive token setup
    echo ""
    echo -e "${BOLD}Telegram Bot Token Setup${NC}"
    echo "Get your token from @BotFather on Telegram."
    echo ""
    echo -n "Paste your bot token (or press Enter to skip): "

    # Read from terminal directly (script is piped via curl | bash)
    read -r TOKEN </dev/tty || true

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
            cat > .env <<EOF
TELEGRAM_TOKEN=$TOKEN
TELEGRAM_BOT_TOKEN=$TOKEN
AURA_ENV=production
AURA_MOCK_LLM=false
EOF
        fi
        log_ok "Token saved to .env"
    else
        if [ ! -f ".env" ]; then
            if [ -f ".env.example" ]; then
                cp .env.example .env
            else
                cat > .env <<EOF
TELEGRAM_TOKEN=
AURA_ENV=production
AURA_MOCK_LLM=false
EOF
            fi
        fi
        log_warn "No token entered — set later: nano ~/aura-v3/.env"
    fi
}

# ─── Create Directories ─────────────────────────────────────────────────────
create_dirs() {
    log_info "Creating data directories..."
    cd "$AURA_DIR"
    mkdir -p logs data/memories data/sessions data/patterns/intents \
             data/patterns/strategies data/memory models cache
    log_ok "Directories created"
}

# ─── Check Mode ──────────────────────────────────────────────────────────────
run_check() {
    echo ""
    log_info "Running health check..."
    if [ -f "$AURA_DIR/scripts/aura_doctor.py" ]; then
        cd "$AURA_DIR"
        $PYTHON_CMD scripts/aura_doctor.py
    else
        verify_all_imports
    fi
    exit 0
}

# ─── Repair Mode ─────────────────────────────────────────────────────────────
run_repair() {
    echo ""
    log_info "Running repair..."
    install_termux_native_deps
    install_pip_deps
    install_llm_deps
    verify_all_imports
    exit 0
}

# ─── Main Install ────────────────────────────────────────────────────────────
main() {
    echo ""
    echo -e "${BOLD}══════════════════════════════════════════${NC}"
    echo -e "${BOLD}   AURA v3 — Termux Production Installer${NC}"
    echo -e "${BOLD}══════════════════════════════════════════${NC}"
    echo ""

    # Check/Repair modes
    if [ "$CHECK_MODE" = true ]; then
        check_platform
        run_check
    fi

    if [ "$REPAIR_MODE" = true ]; then
        check_platform
        run_repair
    fi

    # ── Full Install Flow ────────────────────────────────────────────────────

    echo -e "${BLUE}[1/7]${NC} Checking platform..."
    check_platform

    echo ""
    echo -e "${BLUE}[2/7]${NC} Cloning / updating repository..."
    clone_or_update

    # ── Phase 2: Enable file logging NOW (Bug 1) ────────────────────────────
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

    # ── Final Verification ───────────────────────────────────────────────────
    verify_all_imports

    # ── Summary (Bug 6: absolute paths) ──────────────────────────────────────
    echo ""
    echo -e "${BOLD}══════════════════════════════════════════${NC}"

    if [ "$ERRORS_FOUND" -eq 0 ]; then
        echo -e "${GREEN}${BOLD}   ✅ Installation Complete!${NC}"
    else
        echo -e "${YELLOW}${BOLD}   ⚠️  Installation Complete with $ERRORS_FOUND error(s)${NC}"
        echo -e "   Check: $LOG_FILE"
    fi

    echo -e "${BOLD}══════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}IMPORTANT: Termux limits background tasks. To keep AURA alive, run:${NC}"
    echo -e "  termux-wake-lock"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Start AURA:"
    echo "     cd ~/aura-v3 && bash run_aura.sh telegram"
    echo ""
    echo "  2. Or CLI mode:"
    echo "     cd ~/aura-v3 && bash run_aura.sh cli"
    echo ""
    echo "  3. Health check:"
    echo "     cd ~/aura-v3 && python scripts/aura_doctor.py"
    echo ""
    echo "  Install log: $LOG_FILE"
    echo ""
}

main "$@"
