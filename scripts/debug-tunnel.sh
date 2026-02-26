#!/bin/bash
#
# Debug Tunnel Script
# Exposes local debug server to laptop for remote debugging
#
# Usage:
#   ./debug-tunnel.sh                    # Start with localtunnel (default)
#   ./debug-tunnel.sh --ssh              # Use SSH reverse tunnel
#   ./debug-tunnel.sh --stop             # Stop any running tunnel
#   ./debug-tunnel.sh --status           # Check tunnel status
#

DEBUG_PORT=1999
TUNNEL_PID_FILE="/tmp/aura-debug-tunnel.pid"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_debug_server() {
    # Check if debug server is running locally
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${DEBUG_PORT}/health" 2>/dev/null | grep -q "200"; then
        return 0
    else
        return 1
    fi
}

start_localtunnel() {
    log_info "Starting localtunnel for debug access..."

    # Check if localtunnel is installed
    if ! command -v npx &> /dev/null; then
        log_error "npx not found. Install Node.js to use localtunnel."
        return 1
    fi

    # Check if debug server is running
    if ! check_debug_server; then
        log_error "Debug server not running. Start AURA with DEBUG_MODE=true first:"
        echo "  DEBUG_MODE=true python -m src.main"
        return 1
    fi

    # Generate random subdomain
    SUBDOMAIN="aura-debug-$(date +%s | shasum | head -c 8)"

    # Start localtunnel in background
    npx -y localtunnel --port ${DEBUG_PORT} --subdomain ${SUBDOMAIN} > /tmp/localtunnel.log 2>&1 &
    LT_PID=$!

    echo ${LT_PID} > ${TUNNEL_PID_FILE}

    sleep 3

    if ps -p ${LT_PID} > /dev/null 2>&1; then
        log_info "Localtunnel started!"
        log_info "Your debug server is accessible at:"
        echo ""
        echo "  Click/tap this link on your LAPTOP:"
        echo "  https://${SUBDOMAIN}.loca.lt"
        echo ""
        echo "  Or directly access logs/status:"
        echo "  https://${SUBDOMAIN}.loca.lt/logs"
        echo "  https://${SUBDOMAIN}.loca.lt/status"
        echo ""
        log_warn "Note: Each tunnel link is one-time use. Generate new one if needed."
    else
        log_error "Localtunnel failed to start. Check /tmp/localtunnel.log"
        cat /tmp/localtunnel.log
        return 1
    fi
}

start_ssh_tunnel() {
    log_info "Starting SSH reverse tunnel..."

    # Check if debug server is running
    if ! check_debug_server; then
        log_error "Debug server not running. Start AURA with DEBUG_MODE=true first:"
        echo "  DEBUG_MODE=true python -m src.main"
        return 1
    fi

    # Check for SSH
    if ! command -v ssh &> /dev/null; then
        log_error "ssh not found. Install OpenSSH."
        return 1
    fi

    # Check for SSH key-based auth to remote server
    SSH_HOST="${SSH_TUNNEL_HOST:-localhost}"
    SSH_PORT="${SSH_TUNNEL_PORT:-22}"

    log_info "Creating SSH reverse tunnel to ${SSH_HOST}:${SSH_PORT}"
    log_info "Remote port will be printed when connected"

    # Create reverse tunnel (run in background)
    ssh -N -R ${DEBUG_PORT}:127.0.0.1:${DEBUG_PORT} -p ${SSH_PORT} ${SSH_HOST} &
    SSH_PID=$!

    echo ${SSH_PID} > ${TUNNEL_PID_FILE}

    sleep 2

    if ps -p ${SSH_PID} > /dev/null 2>&1; then
        log_info "SSH tunnel started (PID: ${SSH_PID})"
        log_info "Debug server should be accessible on remote at port ${DEBUG_PORT}"
    else
        log_error "SSH tunnel failed. Check SSH configuration."
        return 1
    fi
}

start_socat() {
    log_info "Starting socat for local network access..."

    if ! command -v socat &> /dev/null; then
        log_error "socat not found. Install it: pkg install socat"
        return 1
    fi

    # Get device IP
    DEVICE_IP=$(ip route get 1 | grep -oP 'src \K\S+' 2>/dev/null || hostname -I | awk '{print $1}')

    if [ -z "${DEVICE_IP}" ]; then
        log_error "Could not determine device IP"
        return 1
    fi

    # Start socat to expose localhost port to network
    socat TCP-LISTEN:${DEBUG_PORT},reuseaddr,fork TCP:127.0.0.1:${DEBUG_PORT} &
    SOCAT_PID=$!

    echo ${SOCAT_PID} > ${TUNNEL_PID_FILE}

    sleep 1

    if ps -p ${SOCAT_PID} > /dev/null 2>&1; then
        log_info " socat started!"
        log_info "Access debug server from laptop at:"
        echo "  http://${DEVICE_IP}:${DEBUG_PORT}/logs"
        echo "  http://${DEVICE_IP}:${DEBUG_PORT}/status"
        echo ""
        log_warn "Warning: This exposes debug port to LOCAL NETWORK, not just localhost"
    else
        log_error "socat failed to start"
        return 1
    fi
}

stop_tunnel() {
    if [ -f "${TUNNEL_PID_FILE}" ]; then
        PID=$(cat ${TUNNEL_PID_FILE})
        if ps -p ${PID} > /dev/null 2>&1; then
            kill ${PID} 2>/dev/null
            log_info "Tunnel stopped (was PID: ${PID})"
        else
            log_warn "Tunnel process not running"
        fi
        rm -f ${TUNNEL_PID_FILE}
    else
        log_info "No tunnel PID file found"
    fi

    # Also kill any orphan localtunnel processes
    pkill -f "localtunnel.*${DEBUG_PORT}" 2>/dev/null
    pkill -f "socat.*${DEBUG_PORT}" 2>/dev/null
}

show_status() {
    log_info "Debug Tunnel Status"
    echo "========================"

    # Check debug server
    if check_debug_server; then
        echo -e "Debug Server: ${GREEN}Running${NC} on port ${DEBUG_PORT}"
    else
        echo -e "Debug Server: ${RED}Not Running${NC}"
        echo "  Start with: DEBUG_MODE=true python -m src.main"
    fi

    echo ""

    # Check tunnel
    if [ -f "${TUNNEL_PID_FILE}" ]; then
        PID=$(cat ${TUNNEL_PID_FILE})
        if ps -p ${PID} > /dev/null 2>&1; then
            echo -e "Tunnel: ${GREEN}Active${NC} (PID: ${PID})"
        else
            echo -e "Tunnel: ${RED}Stale PID file${NC} (process dead)"
            rm -f ${TUNNEL_PID_FILE}
        fi
    else
        echo -e "Tunnel: ${YELLOW}Not Active${NC}"
    fi

    echo ""
    echo "Quick Commands:"
    echo "  ./debug-tunnel.sh --local     # Expose to local network (IP)"
    echo "  ./debug-tunnel.sh             # Expose via localtunnel (internet)"
    echo "  ./debug-tunnel.sh --ssh       # Expose via SSH reverse tunnel"
    echo "  ./debug-tunnel.sh --stop      # Stop tunnel"
}

show_help() {
    echo "AURA Debug Tunnel - Expose debug server to laptop"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --local     Expose debug port to local network (via socat)"
    echo "              Use device IP to access from laptop"
    echo ""
    echo "  (default)   Expose debug port via localtunnel (internet)"
    echo "              Provides a shareable link"
    echo ""
    echo "  --ssh       Expose debug port via SSH reverse tunnel"
    echo "              Requires SSH_TUNNEL_HOST env var or config"
    echo ""
    echo "  --stop      Stop any running tunnel"
    echo ""
    echo "  --status    Show tunnel status and debug server status"
    echo ""
    echo "  --help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DEBUG_PORT      Debug server port (default: 1999)"
    echo "  SSH_TUNNEL_HOST SSH server for reverse tunnel"
    echo "  SSH_TUNNEL_PORT SSH port (default: 22)"
    echo ""
    echo "Examples:"
    echo "  # Start AURA with debug mode"
    echo "  DEBUG_MODE=true python -m src.main"
    echo ""
    echo "  # In another terminal, expose debug to laptop"
    echo "  ./debug-tunnel.sh"
    echo ""
    echo "  # On laptop, access logs:"
    echo "  curl http://<phone-ip>:1999/logs"
    echo "  curl http://<phone-ip>:1999/status"
}

case "${1:-}" in
    --local)
        start_socat
        ;;
    --ssh)
        start_ssh_tunnel
        ;;
    --stop)
        stop_tunnel
        ;;
    --status)
        show_status
        ;;
    --help|-h)
        show_help
        ;;
    "")
        start_localtunnel
        ;;
    *)
        log_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
