#!/usr/bin/env python3
"""
DEPRECATED: Use scripts/install.sh for installation
====================================================
This interactive setup wizard is deprecated for installation.
Please use the unified installer:

    bash scripts/install.sh

This script can still be used for:
  - Interactive configuration: python scripts/setup.py
  - Status check: python scripts/setup.py --status
  - Testing: python scripts/setup.py --test

For installation, use install.sh instead.
====================================================

AURA v3 Interactive Setup Script

Guides users through:
1. Checking existing installations
2. Configuring Telegram bot
3. Optional LLM model setup
4. Testing functionality
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path

AURA_DIR = Path(__file__).parent.parent
CONFIG_DIR = AURA_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
MODELS_DIR = AURA_DIR / "models"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text):
    print(f"\n{BLUE}{BOLD}{'=' * 50}{RESET}")
    print(f"{BLUE}{BOLD}{text}{RESET}")
    print(f"{BLUE}{BOLD}{'=' * 50}{RESET}\n")


def print_success(text):
    print(f"{GREEN}✓{RESET} {text}")


def print_warning(text):
    print(f"{YELLOW}⚠{RESET} {text}")


def print_error(text):
    print(f"{RED}✗{RESET} {text}")


def print_info(text):
    print(f"{BLUE}ℹ{RESET} {text}")


def run_command(cmd, capture=True):
    """Run shell command and return output"""
    try:
        if capture:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        else:
            return subprocess.run(cmd, shell=True, timeout=30).returncode == 0, "", ""
    except Exception as e:
        return False, "", str(e)


def check_python_version():
    """Check Python version"""
    print_info("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} - Need 3.10+")
        return False


def check_installed_packages():
    """Check which packages are already installed"""
    print_info("Checking installed packages...")

    packages = {
        "aiofiles": "aiofiles",
        "pyyaml": "yaml",
        "cryptography": "cryptography",
        "telegram": "telegram",
    }

    installed = {}
    for name, import_name in packages.items():
        try:
            __import__(import_name)
            installed[name] = True
            print_success(f"{name} installed")
        except ImportError:
            installed[name] = False
            print_warning(f"{name} NOT installed")

    return installed


def install_core_packages():
    """Install core required packages"""
    print_header("Installing Core Packages")

    packages = ["aiofiles>=23.0.0", "pyyaml>=6.0", "cryptography>=41.0.0"]

    print_info(f"Installing: {', '.join(packages)}")

    cmd = f"pip install {' '.join(packages)}"
    success, stdout, stderr = run_command(cmd)

    if success:
        print_success("Core packages installed!")
    else:
        print_error(f"Install failed: {stderr}")
        return False

    return True


def install_telegram():
    """Install Telegram bot package"""
    print_info("Installing Telegram bot package...")

    cmd = "pip install python-telegram-bot>=20.0"
    success, stdout, stderr = run_command(cmd)

    if success:
        print_success("Telegram bot installed!")
        return True
    else:
        print_error(f"Install failed: {stderr}")
        return False


def get_telegram_token():
    """Ask user for Telegram token"""
    print_header("Telegram Bot Setup")

    print_info("To control AURA from Telegram, you need a bot token.")
    print("\nSteps to get a token:")
    print("  1. Open Telegram and search for @BotFather")
    print("  2. Send /newbot command")
    print("  3. Follow prompts to name your bot")
    print("  4. Copy the token BotFather gives you")
    print()

    token = input(
        f"{BLUE}Enter your Telegram bot token (or press Enter to skip):{RESET} "
    ).strip()

    if not token:
        print_warning("Telegram setup skipped")
        return None

    if token:
        print_info("Testing bot connection...")
        cmd = f'curl -s "https://api.telegram.org/bot{token}/getMe"'
        success, stdout, stderr = run_command(cmd)

        if success and "ok" in stdout and "true" in stdout:
            print_success("Bot connection successful!")

            env_file = AURA_DIR / ".env"
            with open(env_file, "a") as f:
                f.write(f"\nTELEGRAM_TOKEN={token}\n")

            print_success(f"Token saved to .env file")
            return token
        else:
            print_error("Invalid token. Please check and try again.")
            return get_telegram_token()

    return None


def setup_llm():
    """Offer to help download LLM model"""
    print_header("LLM Model Setup (Optional)")

    print("""
AURA can work without an LLM for basic tasks.
For advanced AI features, you need to download a model.

Available models:
  • Qwen2.5-1B - Best for mobile (625MB, 4GB RAM)
  • Phi-3-Mini - Fast inference (2.5GB)
  • Mistral-7B - Powerful (4GB+)
""")

    choice = input(f"{BLUE}Download an LLM model? (y/n):{RESET} ").strip().lower()

    if choice == "y":
        print("""
To download a model:
1. Visit https://huggingface.co/models?sort=downloads
2. Search for GGUF format models
3. Download and place in: {MODELS_DIR}

Or use wget/curl from CLI:
  wget -O models/model.gguf <model-url>

After downloading, configure with:
  /setmodel <name> in Telegram
  Or edit config/config.yaml
""")
    else:
        print_info("LLM setup skipped. You can add it later.")

    return False


def create_default_config(telegram_token=None):
    """Create default configuration file"""
    print_info("Creating configuration file...")

    config = {
        "llm": {
            "model_type": "llama",
            "model_path": "models/",
            "max_context": 4096,
            "n_gpu_layers": 0,
        },
        "voice": {
            "engine": "espeak",
            "language": "en",
        },
        "security": {
            "default_level": "L2",
            "banking_protection": True,
            "allowed_contacts": [],
        },
        "memory": {
            "working_size": 10,
            "short_term_size": 100,
            "long_term_enabled": False,
        },
        "telegram": {
            "enabled": telegram_token is not None,
            "token": telegram_token if telegram_token else "",
        },
    }

    CONFIG_DIR.mkdir(exist_ok=True)

    with open(CONFIG_FILE, "w") as f:
        import yaml

        yaml.dump(config, f, default_flow_style=False)

    print_success(f"Config created: {CONFIG_FILE}")
    return config


def test_basic_functionality():
    """Test basic AURA functionality"""
    print_header("Testing Basic Functionality")

    tests_passed = 0
    tests_total = 0

    tests_total += 1
    print_info("Testing imports...")
    try:
        import aiofiles
        import yaml
        import cryptography

        print_success("Core imports OK")
        tests_passed += 1
    except ImportError as e:
        print_error(f"Import failed: {e}")

    tests_total += 1
    print_info("Testing config loading...")
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                config = yaml.safe_load(f)
            print_success("Config loads OK")
            tests_passed += 1
        else:
            print_warning("Config file not found")
    except Exception as e:
        print_error(f"Config load failed: {e}")

    tests_total += 1
    print_info("Testing Telegram bot...")
    try:
        from telegram import Update, Bot

        print_success("Telegram imports OK")
        tests_passed += 1
    except ImportError:
        print_warning("Telegram not installed (optional)")
        tests_passed += 1

    return tests_passed, tests_total


def show_status():
    """Show final setup status"""
    print_header("Setup Status")

    print(f"{BOLD}Installation:{RESET}")

    installed = check_installed_packages()

    for pkg, status in installed.items():
        if status:
            print_success(f"{pkg}")
        else:
            print_warning(f"{pkg} (not installed)")

    print(f"\n{BOLD}Configuration:{RESET}")

    if CONFIG_FILE.exists():
        print_success(f"Config: {CONFIG_FILE}")
    else:
        print_warning("Config not created")

    env_file = AURA_DIR / ".env"
    if env_file.exists():
        print_success(f"Environment: {env_file}")

    print(f"\n{BOLD}Directories:{RESET}")
    print(f"  Models: {MODELS_DIR}")

    print(f"\n{BOLD}Next Steps:{RESET}")
    print("""
1. Start AURA:
   python main.py

2. Or start Telegram bot:
   python -m src.channels.telegram_bot

3. Message your bot on Telegram!

For help: python scripts/setup.py --help
""")


def main():
    """Main setup flow"""
    print(f"""
{BLUE}{BOLD}
╔═══════════════════════════════════════════╗
║         AURA v3 Setup Wizard             ║
║    Personal Mobile AGI - Easy Setup       ║
╚═══════════════════════════════════════════╝
{RESET}
    """)

    print_header("Step 1: System Check")

    if not check_python_version():
        print_error("Python 3.10+ required")
        sys.exit(1)

    installed = check_installed_packages()

    needs_install = not all(installed.values())

    if needs_install:
        choice = (
            input(f"\n{BLUE}Install missing packages? (y/n):{RESET} ").strip().lower()
        )

        if choice == "y":
            if not install_core_packages():
                print_error("Core installation failed")
                sys.exit(1)

            if not installed.get("telegram", False):
                install_telegram()

    print_header("Step 2: Configuration")

    telegram_token = get_telegram_token()

    create_default_config(telegram_token)

    print_header("Step 3: LLM Model (Optional)")

    setup_llm()

    print_header("Step 4: Testing")

    passed, total = test_basic_functionality()

    print(f"\nTests: {passed}/{total} passed")

    show_status()

    print_success("Setup complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("""
AURA v3 Setup Script

Usage: python setup.py

Options:
  --help     Show this help
  --status   Show current status
  --test     Run functionality tests
        """)
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        show_status()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_basic_functionality()
        sys.exit(0)

    main()
