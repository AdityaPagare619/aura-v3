#!/bin/bash
# =============================================================================
# AURA v3 - Status Script
# =============================================================================
# Usage: bash status_aura.sh
# Shows AURA process status and resource usage
# =============================================================================

echo "============================================"
echo "  AURA v3 Status"
echo "============================================"
echo ""

# Check for running processes
PIDS=$(pgrep -f "python.*main.py" 2>/dev/null || true)

if [ -z "$PIDS" ]; then
    echo "Status: NOT RUNNING"
    echo ""
    echo "To start: bash run_aura.sh"
else
    echo "Status: RUNNING"
    echo ""
    echo "Processes:"
    ps -p "$PIDS" -o pid,pcpu,pmem,etime,command 2>/dev/null || echo "PID: $PIDS"
fi

echo ""

# Show memory usage (Termux-compatible)
if [ -f /proc/meminfo ]; then
    TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024)}')
    FREE_MEM=$(grep MemAvailable /proc/meminfo | awk '{print int($2/1024)}')
    USED_MEM=$((TOTAL_MEM - FREE_MEM))
    
    echo "Memory Usage:"
    echo "  Total:     ${TOTAL_MEM} MB"
    echo "  Used:      ${USED_MEM} MB"
    echo "  Available: ${FREE_MEM} MB"
    echo ""
fi

# Check data directory sizes
if [ -d "data" ]; then
    echo "Data Directory:"
    du -sh data/* 2>/dev/null || echo "  (empty)"
    echo ""
fi

# Check log file
if [ -f "logs/aura.log" ]; then
    echo "Recent Logs (last 5 lines):"
    tail -5 logs/aura.log 2>/dev/null || true
    echo ""
fi

echo "============================================"
