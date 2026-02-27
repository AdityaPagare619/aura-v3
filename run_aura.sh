#!/bin/bash
# =============================================================================
# AURA v3 - Start Script
# =============================================================================
# Usage: bash run_aura.sh [cli|telegram|doctor]
# Default: telegram mode
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-telegram}"

# Doctor mode shortcut
if [ "$MODE" = "doctor" ]; then
    echo "[AURA] Running health diagnostic..."
    if command -v python3 &> /dev/null; then
        python3 scripts/aura_doctor.py
    elif command -v python &> /dev/null; then
        python scripts/aura_doctor.py
    else
        echo "[AURA ERROR] Python not found!"
        exit 1
    fi
    exit $?
fi

# Load environment variables from .env file
if [ -f ".env" ]; then
    set -a  # auto-export all vars
    # Source .env, skipping comments and empty lines
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments
        case "$line" in
            ''|\#*) continue ;;
        esac
        # Only export lines that look like VAR=VALUE
        if echo "$line" | grep -q '='; then
            eval "export $line" 2>/dev/null || true
        fi
    done < .env
    set +a
    echo "[AURA] Loaded environment from .env"
else
    echo "[AURA WARN] No .env file found — create one with at least TELEGRAM_TOKEN=your-token"
fi

echo "Starting AURA v3 in $MODE mode..."

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "[AURA ERROR] Python not found!"
    exit 1
fi

# Pre-flight check: verify critical imports
$PYTHON -c "
import sys
errors = []
for mod in ['aiofiles', 'yaml', 'telegram']:
    try:
        __import__(mod)
    except ImportError:
        errors.append(mod)
if errors:
    print(f'[AURA ERROR] Missing Python packages: {errors}')
    print('Run: pip install aiofiles pyyaml python-telegram-bot')
    sys.exit(1)
" || exit 1

# Start AURA
$PYTHON main.py --mode "$MODE"
