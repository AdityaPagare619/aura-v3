#!/usr/bin/env python3
"""
DEPRECATED: Use scripts/install.sh instead
===============================================
This script is deprecated as of Wave 5.2.
Please use the unified installer:

    bash scripts/install.sh

The unified installer handles:
  - All environments (Termux, Desktop, Docker)
  - Python version checking
  - Proper dependency management
  - Configuration setup

This file is kept for backwards compatibility only.
===============================================

AURA v3 - Auto Setup Script (LEGACY)
Run this on Termux to install everything automatically

Usage:
    python scripts/auto_setup.py
    OR
    bash <(curl -s https://raw.githubusercontent.com/AdityaPagare619/aura-v3/main/scripts/setup.sh)
"""

import os
import sys
import subprocess
import shutil


def run_cmd(cmd, desc="", capture=True, check=False):
    """Run command and return success"""
    print(f"\n{'=' * 50}")
    print(f"  {desc or cmd}")
    print("=" * 50)
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture, text=True, timeout=300
        )
        if result.returncode != 0:
            print(
                f"WARNING: {result.stderr[:200] if result.stderr else 'Command failed'}"
            )
            if check:
                return False
        print(f"OK: {result.stdout[:200] if result.stdout else 'Done'}")
        return True
    except subprocess.TimeoutExpired:
        print("ERROR: Command timed out")
        return False
    except Exception as e:
        print(f"ERROR: {str(e)[:100]}")
        return check


def check_package_installed(package_name):
    """Check if a package is installed"""
    result = subprocess.run(
        f"which {package_name} || pip show {package_name} 2>/dev/null",
        shell=True,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║          AURA v3 - Auto Setup for Termux             ║
║                                                           ║
║  This will install everything automatically            ║
╚═══════════════════════════════════════════════════════════╝
    """)

    # Telegram token - must be set by user via environment variable
    TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
    if not TELEGRAM_TOKEN:
        print("\n⚠️  TELEGRAM_TOKEN not set. Telegram integration will be disabled.")
        print("    Set it later via: export TELEGRAM_TOKEN='your-bot-token'")

    # Step 1: Fix existing aura-v3 directory
    print("\n[1/9] Checking existing installation...")

    # Check if we're already inside aura-v3 directory
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Parent of scripts/

    # Check if we're inside aura-v3 directory structure
    aura_dir = os.path.expanduser("~/aura-v3")

    # Determine the correct project directory
    # Priority: 1. Current dir if it looks like project root, 2. ~/aura-v3
    if os.path.exists(os.path.join(current_dir, "main.py")):
        project_dir = current_dir
        print(f"Using current directory as project: {project_dir}")
    elif os.path.exists(os.path.join(project_root, "main.py")):
        project_dir = project_root
        print(f"Using script parent as project: {project_dir}")
    elif os.path.exists(aura_dir):
        project_dir = aura_dir
        print(f"Using home aura-v3: {project_dir}")
    else:
        # Need to clone
        project_dir = None

    # If we need to clone (no existing installation)
    if project_dir is None:
        os.chdir(os.path.expanduser("~"))
        run_cmd(
            "git clone https://github.com/AdityaPagare619/aura-v3.git",
            "Cloning AURA v3",
        )
        project_dir = aura_dir

    os.chdir(project_dir)
    print(f"Working in: {os.getcwd()}")

    # Pull latest if git exists
    if os.path.exists(os.path.join(project_dir, ".git")):
        run_cmd("git pull origin main", "Pulling latest updates")

    # Step 2: Create required directories
    print("\n[2/9] Creating directories...")
    for d in [
        "logs",
        "data/memories",
        "data/sessions",
        "data/patterns",
        "models",
        "data/patterns/intents",
        "data/patterns/strategies",
    ]:
        os.makedirs(d, exist_ok=True)
        print(f"  ✓ Created: {d}/")

    # Step 3: Update pip
    print("\n[3/9] Updating pip...")
    run_cmd("pip install --upgrade pip", "Upgrading pip")

    # Step 4: Install Python dependencies
    print("\n[4/9] Installing Python dependencies...")
    deps = [
        "aiofiles>=23.0.0",
        "pyyaml>=6.0",
        "cryptography>=41.0.0",
        "python-telegram-bot>=20.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
    ]

    for dep in deps:
        run_cmd(f"pip install {dep}", f"Installing {dep}")

    # Step 5: Try to install LLM (optional)
    print("\n[5/9] Setting up LLM (optional)...")
    # Try pre-built wheel first
    llm_success = run_cmd(
        "pip install llama-cpp-python --only-binary :all:",
        "Trying pre-built LLM wheel",
        check=False,
    )

    if not llm_success:
        print("\n  ⚠️ Note: LLM will use MOCK mode (no actual AI)")
        print("  You can install manually later with:")
        print("  pip install llama-cpp-python")

    # Step 6: Install espeak (TTS)
    print("\n[6/9] Installing TTS...")
    # Try different package names
    tts_installed = False
    for pkg in ["espeak-ng", "espeak"]:
        if run_cmd(f"apt install {pkg} -y", f"Installing {pkg}", check=False):
            tts_installed = True
            break

    if not tts_installed:
        print("  ⚠️ Note: TTS optional, will use mock voice")

    # Also try pyttsx3
    run_cmd("pip install pyttsx3", "Installing pyttsx3", check=False)

    # Step 7: Configure Telegram token
    print("\n[7/9] Configuring Telegram...")

    # Create .env file
    with open(".env", "w") as f:
        f.write(f"TELEGRAM_TOKEN={TELEGRAM_TOKEN}\n")
    print(f"  ✓ Token configured: {TELEGRAM_TOKEN[:15]}...")

    # Update config.yaml with token
    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        try:
            import yaml

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            if config is None:
                config = {}

            config["telegram"] = {"token": TELEGRAM_TOKEN}

            with open(config_path, "w") as f:
                yaml.dump(config, f)
            print("  ✓ Config updated")
        except Exception as e:
            print(f"  ⚠️ Config update warning: {e}")

    # Step 8: Test imports
    print("\n[8/9] Testing imports...")
    sys.path.insert(0, ".")

    test_results = []

    try:
        from src.llm import LLMRunner, MockLLM

        print("  ✓ LLM module")
        test_results.append(True)
    except Exception as e:
        print(f"  ⚠️ LLM module: {e}")
        test_results.append(False)

    try:
        from src.agent.loop import ReActAgent

        print("  ✓ Agent module")
        test_results.append(True)
    except Exception as e:
        print(f"  ✗ Agent module: {e}")
        test_results.append(False)

    try:
        from src.channels.telegram_bot import AuraTelegramBot

        print("  ✓ Telegram module")
        test_results.append(True)
    except Exception as e:
        print(f"  ✗ Telegram module: {e}")
        test_results.append(False)

    try:
        from src.memory import HierarchicalMemory

        print("  ✓ Memory module")
        test_results.append(True)
    except Exception as e:
        print(f"  ✗ Memory module: {e}")
        test_results.append(False)

    # Step 9: Create run and stop scripts
    print("\n[9/9] Creating scripts...")

    # Create run script
    run_script = """#!/bin/bash
# AURA v3 - Run in background
cd ~/aura-v3
export TELEGRAM_TOKEN="${TELEGRAM_TOKEN:-}"

# Check if already running
if pgrep -f "python.*main.py" > /dev/null; then
    echo "AURA is already running!"
    exit 1
fi

nohup python main.py --mode telegram > aura.log 2>&1 &
AURA_PID=$!
echo $AURA_PID > aura.pid
echo "AURA started with PID: $AURA_PID"
echo "Check Telegram and aura.log"
"""
    with open("run_aura.sh", "w") as f:
        f.write(run_script)
    os.chmod("run_aura.sh", 0o755)
    print("  ✓ Created: run_aura.sh")

    # Create stop script
    stop_script = """#!/bin/bash
# AURA v3 - Stop running instance
cd ~/aura-v3

# Kill by PID if available
if [ -f aura.pid ]; then
    PID=$(cat aura.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        rm aura.pid
        echo "AURA stopped (PID: $PID)"
    else
        rm aura.pid
        echo "Process not running"
    fi
fi

# Also try to kill any python main.py processes
pkill -f "python.*main.py" 2>/dev/null
echo "AURA stopped"
"""
    with open("stop_aura.sh", "w") as f:
        f.write(stop_script)
    os.chmod("stop_aura.sh", 0o755)
    print("  ✓ Created: stop_aura.sh")

    # Create status script
    status_script = """#!/bin/bash
# AURA v3 - Check status
cd ~/aura-v3

echo "=== AURA v3 Status ==="

# Check if running
if [ -f aura.pid ]; then
    PID=$(cat aura.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "Status: RUNNING (PID: $PID)"
    else
        echo "Status: NOT RUNNING (stale PID file)"
    fi
else
    # Check if process exists
    if pgrep -f "python.*main.py" > /dev/null; then
        echo "Status: RUNNING (found process)"
    else
        echo "Status: NOT RUNNING"
    fi
fi

# Show last few log lines
if [ -f aura.log ]; then
    echo ""
    echo "=== Last 10 lines of log ==="
    tail -n 10 aura.log
fi
"""
    with open("status_aura.sh", "w") as f:
        f.write(status_script)
    os.chmod("status_aura.sh", 0o755)
    print("  ✓ Created: status_aura.sh")

    print("""
╔═══════════════════════════════════════════════════════════╗
║                   SETUP COMPLETE!                      ║
╚═══════════════════════════════════════════════════════════╝

📱 TO RUN:
    cd ~/aura-v3
    bash run_aura.sh
    OR
    python main.py --mode telegram

🤖 TO CONTROL:
    bash run_aura.sh    # Start AURA
    bash stop_aura.sh  # Stop AURA  
    bash status_aura.sh # Check status

📋 COMMANDS IN TELEGRAM:
    /start  - Start AURA
    /stop   - Stop AURA
    /status - Check system status
    /help   - Show all commands
    /test   - Test functionality
    /setup  - Run setup wizard

🔧 IF ISSUES:
    # Check logs
    tail -f aura.log
    
    # Reinstall dependencies
    pip install -r requirements.txt
    
    # Test imports
    python -c "from src.agent.loop import ReActAgent; print('OK')"
""")

    # Offer to start
    print("\n❓ Start AURA now? (y/n)")
    try:
        response = input("> ").strip().lower()
        if response == "y":
            print("\nStarting AURA...")
            os.environ["TELEGRAM_TOKEN"] = TELEGRAM_TOKEN
            subprocess.Popen(
                ["python", "main.py", "--mode", "telegram"],
                stdout=open("aura.log", "w"),
                stderr=subprocess.STDOUT,
            )
            print("✓ AURA started! Check aura.log")
            print("📱 Open Telegram and send /start to your bot")
    except EOFError:
        print("\nSkipping auto-start (not interactive)")


if __name__ == "__main__":
    main()
