#!/bin/bash
# =============================================================================
# AURA v3 - Process Supervisor for Termux
# =============================================================================
# This script supervises AURA, providing:
#   - Process monitoring and auto-restart on crash
#   - Periodic state persistence
#   - Graceful shutdown handling
#   - Log rotation
#
# Usage:
#   ./aura-supervisor.sh start     - Start AURA with supervisor
#   ./aura-supervisor.sh stop      - Stop AURA gracefully
#   ./aura-supervisor.sh restart   - Restart AURA
#   ./aura-supervisor.sh status    - Check AURA status
#   ./aura-supervisor.sh logs      - Tail logs
# =============================================================================

set -o pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
AURA_DIR="$HOME/aura-v3"
SCRIPT_DIR="$AURA_DIR/scripts"
LOG_DIR="$AURA_DIR/logs"
PID_FILE="$AURA_DIR/aura-supervisor.pid"
LOG_FILE="$LOG_DIR/aura-supervisor.log"
AURA_LOG_FILE="$LOG_DIR/aura.log"

# Supervisor settings
MAX_RESTARTS=5
RESTART_WINDOW=300          # 5 minutes window for max restarts
STATE_SAVE_INTERVAL=300      # 5 minutes between state saves
LOG_ROTATE_SIZE=10485760     # 10MB, rotate logs at this size
LOG_ROTATE_COUNT=5           # Keep last 5 log files

# -----------------------------------------------------------------------------
# Logging functions
# -----------------------------------------------------------------------------
log() {
    local level="$1"
    local message="$2"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$1"; }
log_warn() { log "WARN" "$1"; }
log_error() { log "ERROR" "$1"; }

# -----------------------------------------------------------------------------
# Log rotation
# -----------------------------------------------------------------------------
rotate_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        return
    fi

    local size
    size=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)

    if [ "$size" -gt "$LOG_ROTATE_SIZE" ]; then
        log_info "Rotating logs..."

        # Rotate supervisor log
        for i in $(seq $((LOG_ROTATE_COUNT - 1)) -1 1); do
            [ -f "$LOG_FILE.$i" ] && mv "$LOG_FILE.$i" "$LOG_FILE.$((i + 1))"
        done
        [ -f "$LOG_FILE" ] && mv "$LOG_FILE" "$LOG_FILE.1"

        # Rotate AURA log
        if [ -f "$AURA_LOG_FILE" ]; then
            for i in $(seq $((LOG_ROTATE_COUNT - 1)) -1 1); do
                [ -f "$AURA_LOG_FILE.$i" ] && mv "$AURA_LOG_FILE.$i" "$AURA_LOG_FILE.$((i + 1))"
            done
            mv "$AURA_LOG_FILE" "$AURA_LOG_FILE.1"
        fi

        log_info "Log rotation complete"
    fi
}

# -----------------------------------------------------------------------------
# Check if AURA is running
# -----------------------------------------------------------------------------
is_aura_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        rm -f "$PID_FILE"
    fi
    return 1
}

# -----------------------------------------------------------------------------
# Termux wake lock - keep phone awake
# -----------------------------------------------------------------------------
acquire_wake_lock() {
    if command -v termux-wake-lock &> /dev/null; then
        termux-wake-lock
        log_info "Termux wake lock acquired"
    else
        log_warn "termux-wake-lock not found, phone may sleep"
    fi
}

release_wake_lock() {
    if command -v termux-wake-unlock &> /dev/null; then
        termux-wake-unlock
        log_info "Termux wake lock released"
    fi
}

# -----------------------------------------------------------------------------
# Save AURA state
# -----------------------------------------------------------------------------
save_aura_state() {
    log_info "Saving AURA state..."
    
    # Create a temporary script to trigger state save
    local save_script
    save_script=$(mktemp)
    
    cat > "$save_script" << 'SAVEEOF'
#!/usr/bin/env python3
import sys
import asyncio
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

async def save_state():
    try:
        from main import get_aura
        aura = await get_aura()
        if hasattr(aura, 'save_state'):
            await aura.save_state()
            print("State saved successfully")
        elif hasattr(aura, '_neural_memory') and aura._neural_memory:
            await aura._neural_memory.save("memory_state.pkl")
            print("Memory saved successfully")
        else:
            print("No save method available")
    except Exception as e:
        print(f"Save failed: {e}")
        sys.exit(1)

asyncio.run(save_state())
SAVEEOF
    
    chmod +x "$save_script"
    
    # Run save in background, don't wait
    (
        cd "$AURA_DIR" || exit 1
        python3 "$save_script" >> "$AURA_LOG_FILE" 2>&1
        rm -f "$save_script"
    ) &
    
    log_info "State save triggered"
}

# -----------------------------------------------------------------------------
# Start AURA with supervision
# -----------------------------------------------------------------------------
start_aura() {
    if is_aura_running; then
        log_error "AURA is already running (PID: $(cat $PID_FILE))"
        return 1
    fi

    # Create log directory
    mkdir -p "$LOG_DIR"

    log_info "Starting AURA supervisor..."

    # Determine mode from args or default
    local mode="${1:-telegram}"
    log_info "AURA mode: $mode"

    # Acquire wake lock
    acquire_wake_lock

    # Start supervisor loop in background
    (
        exec >> "$LOG_FILE" 2>&1
        
        # Track restarts
        local restart_count=0
        local restart_timestamps=()
        
        # Save start time for this process
        echo $$ > "$PID_FILE"
        
        log_info "Supervisor started with PID $$"
        
        # Signal handling
        trap 'log_info "SIGTERM received, initiating graceful shutdown..."; GRACEFUL_SHUTDOWN=1' SIGTERM
        trap 'log_info "SIGINT received, initiating graceful shutdown..."; GRACEFUL_SHUTDOWN=1' SIGINT
        
        GRACEFUL_SHUTDOWN=0
        LAST_STATE_SAVE=0
        
        while true; do
            # Check for graceful shutdown
            if [ "$GRACEFUL_SHUTDOWN" = "1" ]; then
                log_info "Performing graceful shutdown..."
                save_aura_state
                release_wake_lock
                rm -f "$PID_FILE"
                log_info "Supervisor stopped"
                exit 0
            fi

            # Rotate logs if needed
            rotate_logs

            # Check restart window (remove old timestamps)
            local now
            now=$(date +%s)
            restart_timestamps=($(echo "${restart_timestamps[@]}" | tr ' ' '\n' | awk -v now="$now" -v window="$RESTART_WINDOW" '$1 > now - window' | tr '\n' ' '))
            restart_count=${#restart_timestamps[@]}

            # Too many restarts?
            if [ "$restart_count" -ge "$MAX_RESTARTS" ]; then
                log_error "Too many restarts ($MAX_RESTARTS in $RESTART_WINDOW seconds), giving up!"
                log_error "Check logs at: $AURA_LOG_FILE"
                release_wake_lock
                rm -f "$PID_FILE"
                exit 1
            fi

            # Start AURA
            log_info "Starting AURA (attempt $((restart_count + 1))/$MAX_RESTARTS)..."
            
            (
                cd "$AURA_DIR" || exit 1
                python3 main.py --mode "$mode" >> "$AURA_LOG_FILE" 2>&1
            ) &
            
            local aura_pid=$!
            log_info "AURA started with PID: $aura_pid"

            # Wait for AURA to exit
            local aura_exit_code=0
            while kill -0 "$aura_pid" 2>/dev/null; do
                sleep 5
                
                # Check for graceful shutdown request
                if [ "$GRACEFUL_SHUTDOWN" = "1" ]; then
                    log_info "Sending SIGTERM to AURA (PID: $aura_pid)..."
                    kill -TERM "$aura_pid" 2>/dev/null
                    
                    # Wait for graceful exit (max 30 seconds)
                    for i in {1..30}; do
                        if ! kill -0 "$aura_pid" 2>/dev/null; then
                            break
                        fi
                        sleep 1
                    done
                    
                    # Force kill if still running
                    if kill -0 "$aura_pid" 2>/dev/null; then
                        log_warn "AURA did not exit gracefully, forcing..."
                        kill -9 "$aura_pid" 2>/dev/null
                    fi
                    
                    save_aura_state
                    release_wake_lock
                    rm -f "$PID_FILE"
                    log_info "Supervisor stopped gracefully"
                    exit 0
                fi
                
                # Periodic state save
                if [ $((now - LAST_STATE_SAVE)) -gt "$STATE_SAVE_INTERVAL" ]; then
                    save_aura_state
                    LAST_STATE_SAVE=$(date +%s)
                fi
            done
            
            wait "$aura_pid"
            aura_exit_code=$?
            
            now=$(date +%s)
            restart_timestamps+=("$now")
            
            if [ $aura_exit_code -eq 0 ]; then
                log_info "AURA exited normally (code: $aura_exit_code)"
            else
                log_warn "AURA crashed with exit code: $aura_exit_code"
            fi

            # Check if we should stop
            if [ "$GRACEFUL_SHUTDOWN" = "1" ]; then
                release_wake_lock
                rm -f "$PID_FILE"
                log_info "Supervisor stopped"
                exit 0
            fi

            # Wait before restart
            log_info "Waiting 5 seconds before restart..."
            sleep 5
        done
    ) &
    
    # Give supervisor time to start
    sleep 2
    
    if is_aura_running; then
        log_info "AURA supervisor started successfully"
        return 0
    else
        log_error "Failed to start AURA supervisor"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Stop AURA gracefully
# -----------------------------------------------------------------------------
stop_aura() {
    if ! is_aura_running; then
        log_error "AURA is not running"
        return 1
    fi

    local pid
    pid=$(cat "$PID_FILE")
    
    log_info "Stopping AURA (PID: $pid)..."
    
    # Send SIGTERM for graceful shutdown
    kill -TERM "$pid" 2>/dev/null
    
    # Wait for process to exit (max 30 seconds)
    for i in {1..30}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log_info "AURA stopped gracefully"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
    done
    
    # Force kill if still running
    log_warn "AURA did not stop gracefully, forcing..."
    kill -9 "$pid" 2>/dev/null
    rm -f "$PID_FILE"
    
    log_info "AURA force stopped"
    return 0
}

# -----------------------------------------------------------------------------
# Show status
# -----------------------------------------------------------------------------
show_status() {
    if is_aura_running; then
        local pid
        pid=$(cat "$PID_FILE")
        echo "AURA is running (PID: $pid)"
        
        if [ -f "$AURA_LOG_FILE" ]; then
            echo ""
            echo "Last 10 lines of log:"
            tail -10 "$AURA_LOG_FILE"
        fi
    else
        echo "AURA is not running"
        
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "Last 10 lines of supervisor log:"
            tail -10 "$LOG_FILE"
        fi
    fi
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
case "${1:-}" in
    start)
        start_aura "${2:-telegram}"
        ;;
    stop)
        stop_aura
        ;;
    restart)
        stop_aura
        sleep 2
        start_aura "${2:-telegram}"
        ;;
    status)
        show_status
        ;;
    logs)
        if [ -f "$AURA_LOG_FILE" ]; then
            tail -f "$AURA_LOG_FILE"
        else
            echo "No logs found"
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs} [mode]"
        echo ""
        echo "Commands:"
        echo "  start [mode]  - Start AURA with supervisor (default: telegram)"
        echo "  stop         - Stop AURA gracefully"
        echo "  restart      - Restart AURA"
        echo "  status       - Show AURA status"
        echo "  logs         - Tail AURA logs"
        echo ""
        echo "Examples:"
        echo "  $0 start telegram    # Start in Telegram mode"
        echo "  $0 start cli         # Start in CLI mode"
        echo "  $0 restart cli       # Restart in CLI mode"
        exit 1
        ;;
esac
