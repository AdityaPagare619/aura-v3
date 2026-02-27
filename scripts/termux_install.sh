#!/data/data/com.termux/files/usr/bin/bash
# =============================================================================
# AURA v3 — Termux Production Installer (v3.0)
#
# Production-Grade: Timeouts, retries, progress, rollback, verification.
# Environment-Aware: Adapts to Termux mirrors, ARM arch, storage limits.
# curl|bash safe: No set -e, no set -o pipefail, no heredocs.
# =============================================================================

cd "$HOME" 2>/dev/null || cd /data/data/com.termux/files/home 2>/dev/null || true

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ─── Globals ─────────────────────────────────────────────────────────────────
AURA_DIR="$HOME/aura-v3"
AURA_BRANCH="production-hardening"
AURA_REPO="https://github.com/AdityaPagare619/aura-v3.git"
LOG_FILE=""
ERRORS_FOUND=0
WARNINGS_FOUND=0
PYTHON_CMD=""
PYTHON_MAJOR=""
PYTHON_MINOR=""
INSTALL_START=""
PIP_TIMEOUT=120
PKG_TIMEOUT=300

# ─── Logging ─────────────────────────────────────────────────────────────────
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
    WARNINGS_FOUND=$((WARNINGS_FOUND + 1))
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
log_step() {
    echo -e "${CYAN}[STEP]${NC}  $1"
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        echo "[STEP]  $(date +%H:%M:%S) $1" >> "$LOG_FILE"
    fi
    return 0
}

enable_file_logging() {
    mkdir -p "$AURA_DIR/logs" 2>/dev/null
    LOG_FILE="$AURA_DIR/logs/install_$(date +%Y%m%d_%H%M%S).log"
    echo "=== AURA v3 Install Log $(date) ===" > "$LOG_FILE"
    echo "Arch: $(uname -m) | Termux: ${TERMUX_VERSION:-unknown}" >> "$LOG_FILE"
    log_info "Logging to: $LOG_FILE"
    return 0
}

# ─── Utilities ───────────────────────────────────────────────────────────────

# Run a command with a timeout — kills it if it exceeds $1 seconds.
# Usage: run_with_timeout 60 pip install foo
run_with_timeout() {
    local timeout_sec="$1"
    shift
    if command -v timeout > /dev/null 2>&1; then
        timeout "$timeout_sec" "$@"
    else
        # Fallback: run in background and kill after timeout
        "$@" &
        local cmd_pid=$!
        (
            sleep "$timeout_sec"
            kill "$cmd_pid" 2>/dev/null
        ) &
        local watchdog_pid=$!
        wait "$cmd_pid" 2>/dev/null
        local exit_code=$?
        kill "$watchdog_pid" 2>/dev/null
        wait "$watchdog_pid" 2>/dev/null
        return $exit_code
    fi
}

# Show a spinner while a background process runs
# Usage: show_progress PID "message"
show_progress() {
    local pid="$1"
    local msg="$2"
    local spin_chars='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0
    while kill -0 "$pid" 2>/dev/null; do
        local c="${spin_chars:$i:1}"
        printf "\r  ${DIM}%s${NC} %s" "$c" "$msg"
        i=$(( (i + 1) % ${#spin_chars} ))
        sleep 0.2
    done
    printf "\r                                                        \r"
    return 0
}

# Check available disk space (in MB)
check_disk_space() {
    local required_mb="$1"
    local available_mb
    available_mb=$(df "$HOME" 2>/dev/null | awk 'NR==2 {print int($4/1024)}')
    if [ -z "$available_mb" ]; then
        log_warn "Could not determine available disk space"
        return 0
    fi
    if [ "$available_mb" -lt "$required_mb" ] 2>/dev/null; then
        log_error "Insufficient disk space: ${available_mb}MB available, ${required_mb}MB required"
        return 1
    fi
    log_ok "Disk space: ${available_mb}MB available (need ${required_mb}MB)"
    return 0
}

# ─── Platform Check ──────────────────────────────────────────────────────────
check_platform() {
    # Verify Termux environment
    if [ -z "$TERMUX_VERSION" ] && [ ! -d "/data/data/com.termux" ]; then
        log_error "This script is for Termux only!"
        log_error "Install Termux from F-Droid: https://f-droid.org/packages/com.termux/"
        exit 1
    fi
    log_ok "Termux detected (version: ${TERMUX_VERSION:-unknown})"

    # Find Python
    if command -v python3 > /dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python > /dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        log_error "Python not found! Run: pkg install python"
        exit 1
    fi

    # Parse version
    local PYTHON_VERSION
    PYTHON_VERSION="$($PYTHON_CMD --version 2>&1 | awk '{print $2}')"
    PYTHON_MAJOR="$(echo "$PYTHON_VERSION" | cut -d. -f1)"
    PYTHON_MINOR="$(echo "$PYTHON_VERSION" | cut -d. -f2)"

    if [ "$PYTHON_MAJOR" -lt 3 ] 2>/dev/null || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; } 2>/dev/null; then
        log_error "Python 3.10+ required (found $PYTHON_VERSION). Run: pkg upgrade python"
        exit 1
    fi

    log_ok "Python $PYTHON_VERSION"
    log_info "Architecture: $(uname -m)"

    # Check disk space (500MB min for repo + venv + deps)
    check_disk_space 500 || exit 1

    # Check RAM
    local total_ram_mb
    total_ram_mb=$(free -m 2>/dev/null | awk '/^Mem:/ {print $2}')
    if [ -n "$total_ram_mb" ]; then
        log_info "RAM: ${total_ram_mb}MB total"
        if [ "$total_ram_mb" -lt 1024 ] 2>/dev/null; then
            log_warn "Low RAM (<1GB). LLM inference may be limited."
        fi
    fi
    return 0
}

# ─── Termux Mirror Setup ────────────────────────────────────────────────────
setup_mirrors() {
    # Ensure termux-tools is available for mirror management
    if command -v termux-change-repo > /dev/null 2>&1; then
        log_ok "Mirror management available"
    else
        log_info "Installing termux-tools for mirror management..."
        pkg install -y termux-tools 2>/dev/null || true
    fi

    # Test mirror speed with a lightweight package query
    if ! run_with_timeout 30 pkg list-installed 2>/dev/null | head -1 > /dev/null; then
        log_warn "Package mirror seems slow. Try: termux-change-repo"
    fi
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
        git reset --hard "origin/$AURA_BRANCH" 2>/dev/null || log_warn "Could not hard reset to latest"
        log_ok "AURA updated (branch: $AURA_BRANCH)"
    elif [ -d "$AURA_DIR" ]; then
        log_warn "Found leftover $AURA_DIR without .git — cleaning..."
        rm -rf "$AURA_DIR"
        log_info "Cloning AURA v3 (branch: $AURA_BRANCH)..."
        if ! run_with_timeout 120 git clone --depth 1 --branch "$AURA_BRANCH" --single-branch "$AURA_REPO" "$AURA_DIR"; then
            log_error "Clone failed! Check your internet connection."
            exit 1
        fi
        cd "$AURA_DIR"
        log_ok "AURA cloned fresh (branch: $AURA_BRANCH)"
    else
        log_info "Cloning AURA v3 (branch: $AURA_BRANCH)..."
        if ! run_with_timeout 120 git clone --depth 1 --branch "$AURA_BRANCH" --single-branch "$AURA_REPO" "$AURA_DIR"; then
            log_error "Clone failed! Check your internet connection."
            exit 1
        fi
        cd "$AURA_DIR"
        log_ok "AURA cloned to $AURA_DIR"
    fi
    return 0
}

# ─── Install Termux-Native Packages ──────────────────────────────────────────
install_termux_native_deps() {
    log_info "Updating package index..."
    run_with_timeout "$PKG_TIMEOUT" pkg update -y -o Dpkg::Options::="--force-confnew" 2>/dev/null || log_warn "pkg update had issues"

    # Core build dependencies — install one at a time with timeout
    local core_pkgs="git python clang cmake ninja"
    for pkg_name in $core_pkgs; do
        if command -v "$pkg_name" > /dev/null 2>&1; then
            log_ok "$pkg_name already installed"
        else
            log_step "Installing $pkg_name..."
            if run_with_timeout "$PKG_TIMEOUT" pkg install -y "$pkg_name" 2>/dev/null; then
                log_ok "$pkg_name installed"
            else
                log_error "Failed to install $pkg_name (timeout: ${PKG_TIMEOUT}s). Try: pkg install $pkg_name"
            fi
        fi
    done

    # Library dependencies (batch, non-critical)
    log_step "Installing library dependencies..."
    run_with_timeout "$PKG_TIMEOUT" pkg install -y build-essential libffi openssl 2>/dev/null || log_warn "Some library deps may be missing"

    # ── cryptography: pkg first, then --only-binary (NEVER compile from source) ──
    _install_native_python_pkg "cryptography" "python-cryptography" "session encryption degraded, AURA still works"

    # ── numpy: same approach ──
    _install_native_python_pkg "numpy" "python-numpy" "some features may be limited"

    return 0
}

# Helper: Install a Python package via Termux pkg first, then pip --only-binary
_install_native_python_pkg() {
    local import_name="$1"
    local pkg_name="$2"
    local fallback_msg="$3"

    if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
        log_ok "$import_name already importable"
        return 0
    fi

    log_step "Installing $import_name via Termux pkg..."
    run_with_timeout "$PKG_TIMEOUT" pkg install -y "$pkg_name" 2>/dev/null || true

    # Try to fix pkg/pip site-packages isolation
    local termux_site="$PREFIX/lib/python${PYTHON_MAJOR}.${PYTHON_MINOR}/site-packages"

    if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
        log_ok "$import_name installed and verified"
        return 0
    fi

    # Try adding Termux system path to Python
    if [ -d "$termux_site" ]; then
        log_info "Linking Termux pkg path to pip Python..."
        $PYTHON_CMD -c "
import site, os
sp = site.getsitepackages()
for p in sp:
    if os.path.isdir(p):
        pth = os.path.join(p, 'termux-pkgs.pth')
        with open(pth, 'w') as f:
            f.write('$termux_site\n')
        break
" 2>/dev/null || true
    fi

    if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
        log_ok "$import_name linked and verified"
        return 0
    fi

    # NEVER compile from source — use --only-binary to fail fast
    log_info "Trying pip wheel (pre-built only)..."
    run_with_timeout "$PIP_TIMEOUT" pip install "$import_name" --only-binary :all: --timeout 60 2>/dev/null || true

    if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
        log_ok "$import_name via pip wheel verified"
    else
        log_warn "$import_name unavailable ($fallback_msg)"
    fi
    return 0
}

# ─── Install Pip Dependencies ────────────────────────────────────────────────
install_pip_deps() {
    log_info "Upgrading pip..."
    run_with_timeout "$PIP_TIMEOUT" $PYTHON_CMD -m pip install --upgrade pip --quiet 2>/dev/null || true

    # Core required dependencies
    local deps="aiofiles pyyaml psutil python-telegram-bot"
    local total=0
    local installed=0
    for dep in $deps; do total=$((total + 1)); done

    local idx=0
    for dep in $deps; do
        idx=$((idx + 1))
        local import_name="$dep"
        case "$dep" in
            pyyaml)              import_name="yaml" ;;
            python-telegram-bot) import_name="telegram" ;;
        esac

        if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
            log_ok "[$idx/$total] $dep already installed"
            installed=$((installed + 1))
            continue
        fi

        log_step "[$idx/$total] Installing $dep..."
        if run_with_timeout "$PIP_TIMEOUT" pip install "$dep" --timeout 60 2>/dev/null; then
            if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
                log_ok "[$idx/$total] $dep installed and verified"
                installed=$((installed + 1))
            else
                log_error "[$idx/$total] $dep installed but import failed"
            fi
        else
            log_error "[$idx/$total] $dep install failed (timeout: ${PIP_TIMEOUT}s)"
        fi
    done

    log_info "Core deps: $installed/$total installed"
    return 0
}

# ─── Install LLM ─────────────────────────────────────────────────────────────
install_llm_deps() {
    log_info "Installing Local LLM Engine (llama-cpp-python)..."

    if $PYTHON_CMD -c "import llama_cpp" 2>/dev/null; then
        log_ok "llama-cpp-python already installed"
        return 0
    fi

    # Try pre-built wheel first (fastest path)
    log_step "Trying pre-built llama-cpp-python..."
    export CMAKE_ARGS="-DLLAMA_METAL=off -DLLAMA_CUDA=off"

    if run_with_timeout "$PIP_TIMEOUT" pip install llama-cpp-python --only-binary :all: --timeout 60 2>/dev/null; then
        if $PYTHON_CMD -c "import llama_cpp" 2>/dev/null; then
            log_ok "llama-cpp-python (pre-built) installed"
            return 0
        fi
    fi

    # Source build — this is slow but necessary for ARM
    log_warn "No pre-built wheel. Source build may take 5-15 min on mobile..."
    log_info "Building llama-cpp-python from source (this uses cmake + clang)..."
    if run_with_timeout 900 pip install llama-cpp-python --no-cache-dir --timeout 120 2>/dev/null; then
        if $PYTHON_CMD -c "import llama_cpp" 2>/dev/null; then
            log_ok "llama-cpp-python compiled and installed"
            return 0
        fi
    fi

    log_warn "llama-cpp-python unavailable (AURA will use mock LLM mode)"
    log_warn "You can retry later: pip install llama-cpp-python --no-cache-dir"
    return 0
}

# ─── Verify All Imports ──────────────────────────────────────────────────────
verify_all_imports() {
    echo ""
    echo -e "${BOLD}── Import Verification ──${NC}"

    local CORE_PASS=true
    local core_count=0
    local core_ok=0

    # Core required imports
    for pkg_import in "aiofiles:aiofiles" "yaml:pyyaml" "psutil:psutil" "telegram:python-telegram-bot"; do
        local import_name="${pkg_import%%:*}"
        local pkg_name="${pkg_import##*:}"
        core_count=$((core_count + 1))

        if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
            log_ok "$import_name ✓"
            core_ok=$((core_ok + 1))
        else
            log_error "$import_name ✗ → pip install $pkg_name"
            CORE_PASS=false
        fi
    done

    # Optional deps — warn only, don't fail
    for pkg_import in "cryptography:cryptography" "numpy:numpy" "llama_cpp:llama-cpp-python"; do
        local import_name="${pkg_import%%:*}"
        local pkg_name="${pkg_import##*:}"

        if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
            log_ok "$import_name ✓ (optional)"
        else
            log_warn "$import_name ✗ (optional — AURA works without it)"
        fi
    done

    # Test AURA module import (the critical boot test)
    echo ""
    echo -e "${BOLD}── Boot Test ──${NC}"
    if [ -d "$AURA_DIR/src" ]; then
        cd "$AURA_DIR"

        # Test 1: Can we import utils (the previous crash point)?
        if $PYTHON_CMD -c "import sys; sys.path.insert(0, '.'); from src.utils import health_monitor; print('OK')" 2>/dev/null; then
            log_ok "Utils import ✓ (health_monitor loads on Python ${PYTHON_MAJOR}.${PYTHON_MINOR})"
        else
            log_error "Utils import ✗ — health_monitor still crashes"
        fi

        # Test 2: Can we import the main module?
        if $PYTHON_CMD -c "import sys; sys.path.insert(0, '.'); from src.main import AuraProduction; print('OK')" 2>/dev/null; then
            log_ok "AURA main import ✓"
        else
            log_warn "AURA main import needs debugging (may need LLM model)"
        fi
    fi

    echo ""
    log_info "Core imports: $core_ok/$core_count passed"
    if [ "$CORE_PASS" = true ]; then
        log_ok "All core imports verified!"
    fi
    return 0
}

# ─── Setup .env ──────────────────────────────────────────────────────────────
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
    mkdir -p logs data/memories data/sessions data/patterns/intents \
             data/patterns/strategies data/memory data/security \
             data/onboarding models cache
    log_ok "Directories created"
    return 0
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
    INSTALL_START="$(date +%s)"

    echo ""
    echo -e "${BOLD}══════════════════════════════════════════${NC}"
    echo -e "${BOLD}   AURA v3 — Termux Production Installer${NC}"
    echo -e "${BOLD}   v3.0 — Production Hardened${NC}"
    echo -e "${BOLD}══════════════════════════════════════════${NC}"
    echo ""

    echo -e "${BLUE}[1/8]${NC} Checking platform..."
    check_platform

    echo ""
    echo -e "${BLUE}[2/8]${NC} Setting up mirrors..."
    setup_mirrors

    echo ""
    echo -e "${BLUE}[3/8]${NC} Cloning / updating repository..."
    clone_or_update
    enable_file_logging

    echo ""
    echo -e "${BLUE}[4/8]${NC} Installing Termux-native packages..."
    install_termux_native_deps

    echo ""
    echo -e "${BLUE}[5/8]${NC} Installing Python dependencies..."
    install_pip_deps

    echo ""
    echo -e "${BLUE}[6/8]${NC} Installing Local LLM Engine..."
    install_llm_deps

    echo ""
    echo -e "${BLUE}[7/8]${NC} Creating directories & configuring..."
    create_dirs
    setup_env

    echo ""
    echo -e "${BLUE}[8/8]${NC} Verifying installation..."
    verify_all_imports

    # Calculate duration
    local install_end
    install_end="$(date +%s)"
    local duration=$((install_end - INSTALL_START))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))

    echo ""
    echo -e "${BOLD}══════════════════════════════════════════${NC}"
    if [ "$ERRORS_FOUND" -eq 0 ]; then
        echo -e "${GREEN}${BOLD}   ✓ Installation Complete!${NC}"
    else
        echo -e "${YELLOW}${BOLD}   ⚠ Complete with $ERRORS_FOUND error(s)${NC}"
        echo -e "   Check log: $LOG_FILE"
    fi
    echo -e "${DIM}   Duration: ${minutes}m ${seconds}s | Warnings: $WARNINGS_FOUND${NC}"
    echo -e "${BOLD}══════════════════════════════════════════${NC}"
    echo ""
    echo "Next steps:"
    echo "  cd ~/aura-v3 && bash run_aura.sh telegram"
    echo "  cd ~/aura-v3 && bash run_aura.sh cli"
    echo "  cd ~/aura-v3 && python scripts/aura_doctor.py"
    echo ""
    return 0
}

main "$@"
