#!/bin/bash
# =============================================================================
# AURA v3 - Start Script
# =============================================================================
# Usage: bash run_aura.sh [cli|telegram]
# Default: telegram mode
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-telegram}"

echo "Starting AURA v3 in $MODE mode..."

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "Error: Python not found!"
    exit 1
fi

# Start AURA
$PYTHON main.py --mode "$MODE"
