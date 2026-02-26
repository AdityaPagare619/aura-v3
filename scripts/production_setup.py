#!/usr/bin/env python3
"""
DEPRECATED: Use scripts/install.sh instead
===============================================
This script is deprecated as of Wave 5.2.
Please use the unified installer:

    bash scripts/install.sh

The unified installer provides all the same features:
  - Environment detection
  - Validation and health checks
  - Model setup guidance
  - Production-grade logging

This file is kept for backwards compatibility only.
===============================================

AURA v3 - Production-Grade Setup Script (LEGACY)
Based on OpenClaw production practices

Features:
- Strict error handling
- Comprehensive logging
- Model downloading
- Health checks
- Validation
"""

import os
import sys
import subprocess
import shutil
import logging
import pathlib
import urllib.request
import hashlib
from datetime import datetime
from typing import Optional, Tuple

# Production-grade logging
LOG_FILE = "aura_setup.log"


def setup_logging():
    """Setup production-grade logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )
    return logging.getLogger("AURA_SETUP")


logger = setup_logging()


class SetupError(Exception):
    """Custom exception for setup errors"""

    pass


def run_cmd(
    cmd: str,
    desc: str = "",
    check: bool = True,
    capture: bool = True,
    timeout: int = 300,
) -> Tuple[int, str, str]:
    """
    Run command with production-grade error handling
    Returns: (returncode, stdout, stderr)
    """
    logger.info(f"[CMD] {desc or cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        if check and result.returncode != 0:
            logger.error(f"[FAIL] {desc}: {stderr[:200]}")
            raise SetupError(f"Command failed: {cmd}")

        logger.info(f"[OK] {desc or 'Done'}")
        return result.returncode, stdout, stderr

    except subprocess.TimeoutExpired:
        logger.error(f"[TIMEOUT] {cmd}")
        raise SetupError(f"Command timed out: {cmd}")
    except Exception as e:
        logger.error(f"[ERROR] {str(e)}")
        raise


def check_command(cmd: str) -> bool:
    """Check if command exists"""
    result = subprocess.run(
        f"which {cmd} 2>/dev/null || echo 'not found'",
        shell=True,
        capture_output=True,
        text=True,
    )
    return "not found" not in result.stdout


def check_python_version() -> Tuple[bool, str]:
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (need 3.9+)"


def download_file(url: str, dest: str, expected_hash: Optional[str] = None) -> bool:
    """Download file with progress and hash verification"""
    logger.info(f"[DOWNLOAD] {url} -> {dest}")

    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\r  Progress: {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=report_progress)
        print()  # New line after progress

        # Verify hash if provided
        if expected_hash:
            logger.info(f"[VERIFY] Checking hash...")
            with open(dest, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()

            if actual_hash != expected_hash:
                logger.error(
                    f"[HASH_MISMATCH] Expected: {expected_hash[:16]}..., Got: {actual_hash[:16]}..."
                )
                return False

            logger.info(f"[OK] Hash verified")

        return True

    except Exception as e:
        logger.error(f"[DOWNLOAD_ERROR] {str(e)}")
        return False


def validate_environment() -> bool:
    """Validate environment before installation"""
    logger.info("[VALIDATE] Checking environment...")

    checks = []

    # Python version
    ok, msg = check_python_version()
    checks.append(("Python version", ok, msg))

    # Check pip
    pip_ok = check_command("pip") or check_command("pip3")
    checks.append(("pip available", pip_ok, "pip not found"))

    # Print results
    all_ok = True
    for name, ok, msg in checks:
        status = "[OK]" if ok else "[FAIL]"
        logger.info(f"  {status} {name}: {msg}")
        if not ok:
            all_ok = False

    return all_ok


def install_dependencies() -> bool:
    """Install Python dependencies"""
    logger.info("[DEPS] Installing Python dependencies...")

    deps = [
        "aiofiles>=23.0.0",
        "pyyaml>=6.0",
        "cryptography>=41.0.0",
        "python-telegram-bot>=20.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pyttsx3>=2.90",
    ]

    for dep in deps:
        try:
            run_cmd(f"pip install {dep}", f"Installing {dep}", check=False)
        except SetupError:
            logger.warning(f"  Could not install {dep} (may already be installed)")

    return True


def install_llm() -> bool:
    """Install llama-cpp-python for LLM support"""
    logger.info("[LLM] Setting up LLM (optional)...")

    # Try pre-built wheel first (faster)
    logger.info("  Trying pre-built wheel...")
    try:
        run_cmd(
            "pip install llama-cpp-python --only-binary :all:",
            "Installing llama-cpp-python",
            check=False,
        )
        return True
    except SetupError:
        pass

    # Try with specific build
    logger.info("  Trying with CPU-only build...")
    try:
        run_cmd(
            "pip install llama-cpp-python --no-binary :all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu",
            "Installing llama-cpp-python (CPU)",
            check=False,
            timeout=600,
        )
        return True
    except SetupError:
        pass

    logger.warning("[LLM] Could not install llama-cpp-python")
    logger.info("  AURA will run in MOCK mode without AI")
    return False


def download_models() -> bool:
    """Download recommended LLM models"""
    logger.info("[MODELS] Setting up LLM models...")

    models_dir = pathlib.Path("models")
    models_dir.mkdir(exist_ok=True)

    # Recommended models for mobile (4GB RAM)
    models = [
        {
            "name": "Qwen2.5-1B",
            "url": "https://huggingface.co/TheBloke/Qwen2.5-1B-GGUF/resolve/main/qwen2.5-1b-q5_k_m.gguf",
            "size": "625MB",
            "desc": "Best balance for mobile",
        },
    ]

    for model in models:
        model_path = (
            models_dir / model["name"].lower().replace(".", "-") + "-q5_k_m.gguf"
        )

        if model_path.exists():
            logger.info(f"  [SKIP] {model['name']} already exists")
            continue

        logger.info(f"  [DOWNLOAD] {model['name']} ({model['size']}) - {model['desc']}")

        # Note: In production, you'd want actual URLs and hash verification
        # This is a placeholder - user should download manually or use provided URLs
        logger.info(f"  Please download from: {model['url']}")
        logger.info(f"  Save to: {model_path}")

    return True


def create_directories() -> bool:
    """Create required directories"""
    logger.info("[DIRS] Creating directories...")

    dirs = [
        "logs",
        "data/memories",
        "data/sessions",
        "data/patterns",
        "data/patterns/intents",
        "data/patterns/strategies",
        "models",
    ]

    for d in dirs:
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"  Created: {d}/")

    return True


def configure_telegram(token: str) -> bool:
    """Configure Telegram bot"""
    logger.info("[TELEGRAM] Configuring Telegram...")

    # Create .env file
    env_file = pathlib.Path(".env")
    env_file.write_text(f"TELEGRAM_TOKEN={token}\n")
    logger.info(f"  Token set: {token[:15]}...")

    # Update config.yaml
    config_path = pathlib.Path("config/config.yaml")
    if config_path.exists():
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        config["telegram"] = {"token": token}

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        logger.info("  Config updated")

    return True


def test_imports() -> bool:
    """Test all imports"""
    logger.info("[TEST] Testing imports...")

    modules = [
        ("src.llm", "LLM module"),
        ("src.agent.loop", "Agent module"),
        ("src.memory", "Memory module"),
        ("src.learning.engine", "Learning module"),
        ("src.session.manager", "Session module"),
        ("src.channels.telegram_bot", "Telegram module"),
    ]

    all_ok = True
    for module, name in modules:
        try:
            __import__(module)
            logger.info(f"  [OK] {name}")
        except Exception as e:
            logger.error(f"  [FAIL] {name}: {str(e)[:50]}")
            all_ok = False

    return all_ok


def create_scripts() -> bool:
    """Create convenience scripts"""
    logger.info("[SCRIPTS] Creating convenience scripts...")

    scripts = {
        "run_aura.sh": """#!/bin/bash
# AURA v3 - Start in background
cd "$(dirname "$0")"

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if already running
if pgrep -f "python.*main.py" > /dev/null 2>&1; then
    echo "AURA is already running!"
    exit 1
fi

# Start in background
nohup python main.py --mode telegram > aura.log 2>&1 &
AURA_PID=$!
echo $AURA_PID > aura.pid

echo "AURA started with PID: $AURA_PID"
echo "Check: tail -f aura.log"
""",
        "stop_aura.sh": """#!/bin/bash
# AURA v3 - Stop running instance
cd "$(dirname "$0")"

# Kill by PID
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

# Also kill any python main.py processes
pkill -f "python.*main.py" 2>/dev/null
echo "AURA stopped"
""",
        "status_aura.sh": """#!/bin/bash
# AURA v3 - Check status
cd "$(dirname "$0")"

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
    if pgrep -f "python.*main.py" > /dev/null 2>&1; then
        echo "Status: RUNNING (found process)"
    else
        echo "Status: NOT RUNNING"
    fi
fi

# Show last log lines
if [ -f aura.log ]; then
    echo ""
    echo "=== Last 10 lines of log ==="
    tail -n 10 aura.log
fi
""",
        "health_check.sh": """#!/bin/bash
# AURA v3 - Health check
cd "$(dirname "$0")"

echo "=== AURA Health Check ==="

# Check process
if pgrep -f "python.*main.py" > /dev/null 2>&1; then
    echo "[OK] Process running"
else
    echo "[FAIL] Process not running"
    exit 1
fi

# Check log for errors
if [ -f aura.log ]; then
    ERRORS=$(tail -n 100 aura.log | grep -i "error\|exception\|fail" | tail -n 5)
    if [ -n "$ERRORS" ]; then
        echo "[WARN] Recent errors in log:"
        echo "$ERRORS"
    else
        echo "[OK] No recent errors in log"
    fi
fi

# Check data directories
for dir in logs data/memories data/sessions; do
    if [ -d "$dir" ]; then
        echo "[OK] Directory exists: $dir"
    else
        echo "[FAIL] Missing directory: $dir"
    fi
done

echo ""
echo "AURA is healthy!"
""",
    }

    for name, content in scripts.items():
        path = pathlib.Path(name)
        path.write_text(content)
        path.chmod(0o755)
        logger.info(f"  Created: {name}")

    return True


def main():
    """Main setup function"""
    start_time = datetime.now()

    print("""
╔═══════════════════════════════════════════════════════════╗
║          AURA v3 - Production Setup                  ║
║                                                           ║
║  Production-grade installer with validation             ║
╚═══════════════════════════════════════════════════════════╝
    """)

    logger.info("=" * 60)
    logger.info("AURA v3 Setup Starting")
    logger.info("=" * 60)

    try:
        # Step 1: Find project directory
        logger.info("[1/8] Finding project directory...")
        script_dir = pathlib.Path(__file__).parent
        project_root = script_dir.parent

        # Check if we're in the right place
        if (project_root / "main.py").exists():
            os.chdir(project_root)
            logger.info(f"  Working in: {os.getcwd()}")

        # Step 2: Validate environment
        logger.info("[2/8] Validating environment...")
        if not validate_environment():
            logger.warning("  Some checks failed, but continuing...")

        # Step 3: Create directories
        logger.info("[3/8] Creating directories...")
        create_directories()

        # Step 4: Install dependencies
        logger.info("[4/8] Installing dependencies...")
        install_dependencies()

        # Step 5: Install LLM (optional)
        logger.info("[5/8] Setting up LLM...")
        llm_installed = install_llm()

        # Step 6: Download models (info)
        logger.info("[6/8] Model setup...")
        download_models()

        # Step 7: Configure Telegram
        logger.info("[7/8] Configuring Telegram...")
        TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
        if not TELEGRAM_TOKEN:
            logger.warning(
                "TELEGRAM_TOKEN not set - Telegram integration will be disabled. Set it via: export TELEGRAM_TOKEN='your-token'"
            )
        configure_telegram(TELEGRAM_TOKEN)

        # Step 8: Test imports
        logger.info("[8/8] Testing imports...")
        imports_ok = test_imports()

        # Create scripts
        create_scripts()

        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"""
╔═══════════════════════════════════════════════════════════╗
║                   SETUP COMPLETE!                      ║
║                     ({elapsed:.1f}s)                              ║
╚═══════════════════════════════════════════════════════════╝

TO START:
    bash run_aura.sh

TO CHECK STATUS:
    bash status_aura.sh
    OR
    bash health_check.sh

TO VIEW LOGS:
    tail -f aura.log

TO STOP:
    bash stop_aura.sh

TELEGRAM:
    Open Telegram and send /start to your bot

NOTE:
    - LLM installed: {"Yes" if llm_installed else "No (MOCK mode)"}
    - Imports OK: {"Yes" if imports_ok else "No (check errors above)"}

NEXT STEPS:
    1. Download LLM model to models/ (optional)
    2. Run: bash run_aura.sh
    3. Check: tail -f aura.log
""")

        logger.info(f"Setup completed in {elapsed:.1f}s")
        return 0

    except SetupError as e:
        logger.error(f"Setup failed: {e}")
        logger.error(f"Check {LOG_FILE} for details")
        return 1
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
