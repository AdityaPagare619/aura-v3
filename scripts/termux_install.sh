#!/bin/bash
# =============================================================================
# AURA v3 - Termux Production Installer
# =============================================================================
# One-liner install:
#   curl -sL https://raw.githubusercontent.com/AdityaPagare619/aura-v3/main/scripts/termux_install.sh | bash
#
# Flags:
#   --repair    Fix missing deps without full reinstall
#   --check     Check installation status only
#   --verbose   Show all command output
# =============================================================================

set -e

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# ─── Globals ─────────────────────────────────────────────────────────────────
AURA_DIR="$HOME/aura-v3"
LOG_FILE="$AURA_DIR/logs/install.log"
REPAIR_MODE=false
CHECK_MODE=false
VERBOSE=false
ERRORS_FOUND=0

# ─── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --repair)  REPAIR_MODE=true; shift ;;
        --check)   CHECK_MODE=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        *)         echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# ─── Logging ─────────────────────────────────────────────────────────────────
log_info()  { echo -e "${BLUE}[INFO]${NC}  $1"; echo "[INFO]  $(date +%H:%M:%S) $1" >> "$LOG_FILE" 2>/dev/null || true; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; echo "[OK]    $(date +%H:%M:%S) $1" >> "$LOG_FILE" 2>/dev/null || true; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; echo "[WARN]  $(date +%H:%M:%S) $1" >> "$LOG_FILE" 2>/dev/null || true; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; echo "[ERROR] $(date +%H:%M:%S) $1" >> "$LOG_FILE" 2>/dev/null || true; ERRORS_FOUND=$((ERRORS_FOUND + 1)); }

# Run command with visible output (NO silent suppression)
run_cmd() {
    local desc="$1"; shift
    log_info "$desc"
    if [ "$VERBOSE" = true ]; then
        "$@" 2>&1 | tee -a "$LOG_FILE"
    else
        "$@" >> "$LOG_FILE" 2>&1
    fi
    return ${PIPESTATUS[0]:-$?}
}

# ─── Platform Check ──────────────────────────────────────────────────────────
check_platform() {
    if [ -z "$TERMUX_VERSION" ] && [ ! -d "/data/data/com.termux" ]; then
        log_error "This script is for Termux only! Use scripts/install-aura.sh for other platforms."
        exit 1
    fi

    log_ok "Termux detected (version: ${TERMUX_VERSION:-unknown})"

    # Check Python
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

    # Check architecture
    ARCH=$(uname -m)
    log_info "Architecture: $ARCH"
}

# ─── Check Mode ──────────────────────────────────────────────────────────────
run_check() {
    echo ""
    echo -e "${BOLD}══════════════════════════════════════${NC}"
    echo -e "${BOLD}  AURA v3 — Installation Status${NC}"
    echo -e "${BOLD}══════════════════════════════════════${NC}"
    echo ""

    # Platform
    check_platform

    # Git repo
    if [ -d "$AURA_DIR/.git" ]; then
        cd "$AURA_DIR"
        log_ok "Repository: $(git rev-parse --short HEAD 2>/dev/null) on $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    else
        log_warn "Repository not cloned"
    fi

    # .env
    if [ -f "$AURA_DIR/.env" ]; then
        if grep -q "TELEGRAM_TOKEN=.\+" "$AURA_DIR/.env" 2>/dev/null; then
            log_ok ".env configured with TELEGRAM_TOKEN"
        else
            log_warn ".env exists but TELEGRAM_TOKEN not set"
        fi
    else
        log_warn ".env not found"
    fi

    # Dependencies
    echo ""
    echo "Python dependencies:"
    for pkg in aiofiles yaml cryptography psutil telegram; do
        if $PYTHON_CMD -c "import $pkg" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} $pkg"
        else
            echo -e "  ${RED}✗${NC} $pkg"
        fi
    done

    echo ""
    echo "Termux packages:"
    for pkg in git python clang cmake; do
        if command -v $pkg &>/dev/null; then
            echo -e "  ${GREEN}✓${NC} $pkg"
        else
            echo -e "  ${RED}✗${NC} $pkg (run: pkg install $pkg)"
        fi
    done

    echo ""
    exit 0
}

# ─── Repair Mode ─────────────────────────────────────────────────────────────
run_repair() {
    echo ""
    echo -e "${BOLD}══════════════════════════════════════${NC}"
    echo -e "${BOLD}  AURA v3 — Repair Mode${NC}"
    echo -e "${BOLD}══════════════════════════════════════${NC}"
    echo ""

    check_platform
    cd "$AURA_DIR" 2>/dev/null || { log_error "AURA not installed at $AURA_DIR"; exit 1; }

    install_termux_native_deps
    install_pip_deps
    verify_all_imports

    echo ""
    if [ "$ERRORS_FOUND" -eq 0 ]; then
        log_ok "Repair complete — all dependencies working!"
    else
        log_error "Repair completed with $ERRORS_FOUND error(s). See $LOG_FILE"
    fi
    exit $ERRORS_FOUND
}

# ─── Install Termux-Native Packages ──────────────────────────────────────────
install_termux_native_deps() {
    echo ""
    log_info "Installing Termux-native packages..."

    # Update package list (errors visible, but non-fatal)
    pkg update -y -o Dpkg::Options::="--force-confnew" || log_warn "pkg update had warnings (continuing)"
    pkg upgrade -y -o Dpkg::Options::="--force-confnew" || log_warn "pkg upgrade had warnings (continuing)"

    # Core packages: git, python, build tools
    local CORE_PKGS="git python clang cmake"
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

    # CRITICAL: cryptography via pkg (NOT pip — pip needs Rust toolchain which is huge)
    if $PYTHON_CMD -c "import cryptography" 2>/dev/null; then
        log_ok "cryptography already installed"
    else
        log_info "Installing cryptography via Termux pkg (pre-built, no Rust needed)..."
        if pkg install -y python-cryptography 2>/dev/null; then
            log_ok "cryptography installed via pkg"
        else
            log_warn "python-cryptography not in pkg, trying pip with Rust..."
            if pkg install -y rust 2>/dev/null; then
                if pip install cryptography --quiet; then
                    log_ok "cryptography installed via pip+Rust"
                else
                    log_error "cryptography installation FAILED — this is needed for secure storage"
                fi
            else
                log_error "Could not install rust or python-cryptography"
            fi
        fi
    fi

    # Build deps for psutil
    pkg install -y build-essential libffi openssl 2>/dev/null || true
}

# ─── Install Pip Dependencies ────────────────────────────────────────────────
install_pip_deps() {
    echo ""
    log_info "Installing Python dependencies via pip..."

    # Upgrade pip first
    $PYTHON_CMD -m pip install --upgrade pip --quiet 2>/dev/null || true

    # Install each core dep individually with verification
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

        # Install with VISIBLE errors
        log_info "Installing $dep..."
        if pip install "$dep" 2>&1 | tee -a "$LOG_FILE"; then
            # Verify the import actually works
            if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
                log_ok "$dep installed and verified"
            else
                log_error "$dep pip install succeeded but import failed!"
            fi
        else
            log_error "$dep installation FAILED — check $LOG_FILE for details"
        fi
    done
}

# ─── Verify All Imports ──────────────────────────────────────────────────────
verify_all_imports() {
    echo ""
    echo -e "${BOLD}── Verifying All Imports ──${NC}"

    local ALL_PASS=true

    for pkg_import in "aiofiles:aiofiles" "yaml:pyyaml" "cryptography:cryptography" "psutil:psutil" "telegram:python-telegram-bot"; do
        local import_name="${pkg_import%%:*}"
        local pkg_name="${pkg_import##*:}"

        if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
            log_ok "$import_name ✓"
        else
            log_error "$import_name ✗ → pip install $pkg_name"
            ALL_PASS=false
        fi
    done

    # Test AURA module import
    if [ -d "$AURA_DIR/src" ]; then
        cd "$AURA_DIR"
        if $PYTHON_CMD -c "import sys; sys.path.insert(0, '.'); from src.main import AuraProduction; print('OK')" 2>/dev/null; then
            log_ok "AURA modules ✓"
        else
            log_warn "AURA module import failed (may need debugging)"
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

    # Interactive token setup - must use /dev/tty because script is piped via curl
    echo ""
    echo -e "${BOLD}Telegram Bot Token Setup${NC}"
    echo "Get your token from @BotFather on Telegram."
    echo ""
    echo -n "Paste your bot token (or press Enter to skip): "
    
    # Read from terminal directly, handling the curl | bash pipe
    read -r TOKEN </dev/tty || true

    if [ -n "$TOKEN" ]; then
        # Create .env from example or from scratch
        if [ -f ".env.example" ]; then
            cp .env.example .env
        fi

        # Set the token (handle both var names for compatibility)
        if [ -f ".env" ]; then
            # Replace existing TELEGRAM_TOKEN line or add it
            if grep -q "^TELEGRAM_TOKEN=" ".env"; then
                sed -i "s|^TELEGRAM_TOKEN=.*|TELEGRAM_TOKEN=$TOKEN|" .env
            else
                echo "TELEGRAM_TOKEN=$TOKEN" >> .env
            fi
            # Also ensure TELEGRAM_BOT_TOKEN for backwards compat
            if grep -q "^TELEGRAM_BOT_TOKEN=" ".env"; then
                sed -i "s|^TELEGRAM_BOT_TOKEN=.*|TELEGRAM_BOT_TOKEN=$TOKEN|" .env
            else
                echo "TELEGRAM_BOT_TOKEN=$TOKEN" >> .env
            fi
        else
            echo "TELEGRAM_TOKEN=$TOKEN" > .env
            echo "TELEGRAM_BOT_TOKEN=$TOKEN" >> .env
            echo "AURA_ENV=production" >> .env
        fi

        log_ok "Token saved to .env"
    else
        # Create minimal .env if it doesn't exist
        if [ ! -f ".env" ]; then
            if [ -f ".env.example" ]; then
                cp .env.example .env
            else
                echo "TELEGRAM_TOKEN=" > .env
                echo "AURA_ENV=production" >> .env
            fi
        fi
        log_warn "No token entered — set it later: nano ~/aura-v3/.env"
    fi
}

# ─── Create Directories ─────────────────────────────────────────────────────
create_dirs() {
    log_info "Creating directories..."
    mkdir -p logs data/memories data/sessions data/patterns/intents data/patterns/strategies data/memory models cache
    log_ok "Directories created"
}

# ─── Clone or Update ─────────────────────────────────────────────────────────
clone_or_update() {
    echo ""
    log_info "Getting AURA v3..."

    cd ~

    if [ -d "$AURA_DIR/.git" ]; then
        # Existing repo — update it
        log_info "Updating existing installation..."
        cd "$AURA_DIR"
        git pull origin main || log_warn "Could not update (using existing version)"
        log_ok "AURA updated"
    elif [ -d "$AURA_DIR" ]; then
        # Directory exists but no .git — leftover from partial install
        log_warn "Found leftover $AURA_DIR without .git — cleaning up..."
        rm -rf "$AURA_DIR"
        git clone https://github.com/AdityaPagare619/aura-v3.git "$AURA_DIR"
        cd "$AURA_DIR"
        log_ok "AURA cloned fresh to $AURA_DIR"
    else
        # Fresh install
        log_info "Cloning AURA v3..."
        git clone https://github.com/AdityaPagare619/aura-v3.git "$AURA_DIR"
        cd "$AURA_DIR"
        log_ok "AURA cloned to $AURA_DIR"
    fi
}

# ─── Main Install ────────────────────────────────────────────────────────────
main() {
    echo ""
    echo -e "${BOLD}══════════════════════════════════════════${NC}"
    echo -e "${BOLD}   AURA v3 — Termux Production Installer${NC}"
    echo -e "${BOLD}══════════════════════════════════════════${NC}"
    echo ""

    # Check/Repair modes (before creating any dirs)
    if [ "$CHECK_MODE" = true ]; then
        check_platform
        run_check
    fi

    if [ "$REPAIR_MODE" = true ]; then
        check_platform
        run_repair
    fi

    # ── Full Install Flow ────────────────────────────────────────────────────
    echo -e "${BLUE}[1/6]${NC} Checking platform..."
    check_platform

    echo ""
    echo -e "${BLUE}[2/6]${NC} Cloning / updating repository..."
    clone_or_update

    # Create log dir AFTER clone (so it's inside the repo)
    mkdir -p "$AURA_DIR/logs" 2>/dev/null || true
    echo "=== AURA Install Log $(date) ===" > "$LOG_FILE" 2>/dev/null || true

    echo ""
    echo -e "${BLUE}[3/6]${NC} Installing Termux-native packages..."
    install_termux_native_deps

    echo ""
    echo -e "${BLUE}[4/6]${NC} Installing Python dependencies..."
    install_pip_deps

    echo ""
    echo -e "${BLUE}[5/6]${NC} Creating directories..."
    create_dirs

    echo ""
    echo -e "${BLUE}[6/6]${NC} Configuring environment..."
    setup_env

    # ── Final Verification ───────────────────────────────────────────────────
    verify_all_imports

    # ── LLM (optional) ───────────────────────────────────────────────────────
    echo ""
    log_info "LLM support (optional)..."
    if $PYTHON_CMD -m pip install llama-cpp-python --only-binary :all: --quiet 2>/dev/null; then
        log_ok "LLM support installed (pre-built)"
    else
        log_warn "LLM not installed — AURA will use MOCK mode (install later: pip install llama-cpp-python)"
    fi

    # ── Summary ──────────────────────────────────────────────────────────────
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
    echo "     cd ~/aura-v3"
    echo "     bash run_aura.sh telegram"
    echo ""
    echo "  2. Or CLI mode:"
    echo "     bash run_aura.sh cli"
    echo ""
    echo "  3. Health check:"
    echo "     python scripts/aura_doctor.py"
    echo ""
    echo "  Install log: $LOG_FILE"
    echo ""
}

main "$@"
