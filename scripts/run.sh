#!/bin/bash
# Quick start script

# Load .env if exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default to CLI mode
MODE=${1:-cli}
PORT=${2:-8080}

python main.py --mode $MODE --port $PORT
