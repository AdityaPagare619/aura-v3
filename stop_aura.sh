#!/bin/bash
# =============================================================================
# AURA v3 - Stop Script
# =============================================================================
# Usage: bash stop_aura.sh
# Stops all running AURA processes
# =============================================================================

echo "Stopping AURA v3..."

# Find and kill AURA processes
PIDS=$(pgrep -f "python.*main.py" 2>/dev/null || true)

if [ -z "$PIDS" ]; then
    echo "No AURA processes found."
    exit 0
fi

for PID in $PIDS; do
    echo "Stopping process $PID..."
    kill -SIGTERM "$PID" 2>/dev/null || true
done

# Wait for graceful shutdown
sleep 2

# Force kill if still running
PIDS=$(pgrep -f "python.*main.py" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
    echo "Force killing remaining processes..."
    for PID in $PIDS; do
        kill -9 "$PID" 2>/dev/null || true
    done
fi

echo "AURA stopped."
