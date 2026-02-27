#!/usr/bin/env python3
"""
AURA v3 — Installation Verifier
================================
Checks every import AURA needs at startup.
Reports: PASS/FAIL for each module with version and error.
Exit code 0 = all good, 1 = failures found.

Usage:
    python scripts/verify_install.py
"""

import sys
import os

# Ensure project root is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

failures = []
warnings = []


def check_import(module_name: str, pip_name: str = None, required: bool = True):
    """Check if a module can be imported."""
    pip_name = pip_name or module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", "?"))
        print(f"  {GREEN}\u2713{RESET} {module_name} ({version})")
        return True
    except ImportError as e:
        if required:
            print(f"  {RED}\u2717{RESET} {module_name} \u2014 MISSING \u2192 pip install {pip_name}")
            failures.append((module_name, pip_name, str(e)))
        else:
            print(f"  {YELLOW}\u25CB{RESET} {module_name} \u2014 optional, not installed")
            warnings.append((module_name, pip_name))
        return False


def check_env():
    """Check .env configuration."""
    env_path = os.path.join(PROJECT_ROOT, ".env")

    if not os.path.exists(env_path):
        print(f"  {RED}\u2717{RESET} .env file \u2014 NOT FOUND")
        print(f"      Create: cp .env.example .env && nano .env")
        failures.append((".env", "file", "not found"))
        return

    # Read .env manually (no dotenv dependency)
    env_vars = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip()

    # Check TELEGRAM_TOKEN
    token = env_vars.get("TELEGRAM_TOKEN") or env_vars.get("TELEGRAM_BOT_TOKEN", "")
    if token and len(token) > 10:
        print(f"  {GREEN}\u2713{RESET} TELEGRAM_TOKEN set ({token[:8]}...)")
    elif token:
        print(f"  {YELLOW}!{RESET} TELEGRAM_TOKEN looks incomplete ({token[:5]}...)")
        warnings.append(("TELEGRAM_TOKEN", "value", "suspiciously short"))
    else:
        print(f"  {RED}\u2717{RESET} TELEGRAM_TOKEN not set in .env")
        print(f"      Edit: nano ~/aura-v3/.env")
        failures.append(("TELEGRAM_TOKEN", "env", "not set"))


def check_aura_modules():
    """Check AURA's own modules load correctly."""
    modules_to_check = [
        ("src.main", "AuraProduction"),
        ("src.agent.loop", "ReActAgent"),
        ("src.memory", "HierarchicalMemory"),
        ("src.llm", "LLMRunner"),
        ("src.channels.telegram_bot", "AuraTelegramBot"),
        ("src.utils", "HealthMonitor"),
    ]

    for mod_path, class_name in modules_to_check:
        try:
            mod = __import__(mod_path, fromlist=[class_name])
            cls = getattr(mod, class_name, None)
            if cls:
                print(f"  {GREEN}\u2713{RESET} {mod_path}.{class_name}")
            else:
                print(f"  {YELLOW}!{RESET} {mod_path} loaded but {class_name} not found")
                warnings.append((mod_path, class_name, "class missing"))
        except Exception as e:
            print(f"  {RED}\u2717{RESET} {mod_path} \u2014 {str(e)[:80]}")
            failures.append((mod_path, class_name, str(e)))


def check_directories():
    """Check required directories exist."""
    required_dirs = ["logs", "data", "data/memories", "data/sessions", "models"]
    for d in required_dirs:
        full_path = os.path.join(PROJECT_ROOT, d)
        if os.path.isdir(full_path):
            print(f"  {GREEN}\u2713{RESET} {d}/")
        else:
            print(f"  {YELLOW}\u25CB{RESET} {d}/ \u2014 will be created on first run")


def main():
    os.chdir(PROJECT_ROOT)

    print()
    print(f"{BOLD}{'='*38}{RESET}")
    print(f"{BOLD}  AURA v3 \u2014 Installation Verifier{RESET}")
    print(f"{BOLD}{'='*38}{RESET}")
    print()

    # 1. Core Python dependencies
    print(f"{BOLD}Core Dependencies:{RESET}")
    check_import("aiofiles")
    check_import("yaml", "pyyaml")
    check_import("cryptography")
    check_import("psutil")
    check_import("telegram", "python-telegram-bot")
    print()

    # 2. Optional dependencies
    print(f"{BOLD}Optional Dependencies:{RESET}")
    check_import("llama_cpp", "llama-cpp-python", required=False)
    check_import("pyttsx3", required=False)
    check_import("pytest", required=False)
    print()

    # 3. Environment
    print(f"{BOLD}Configuration:{RESET}")
    check_env()
    print()

    # 4. Directories
    print(f"{BOLD}Directories:{RESET}")
    check_directories()
    print()

    # 5. AURA modules
    print(f"{BOLD}AURA Modules:{RESET}")
    check_aura_modules()
    print()

    # Summary
    print(f"{BOLD}{'='*38}{RESET}")
    if not failures:
        print(f"{GREEN}{BOLD}  \u2705 All checks passed!{RESET}")
        if warnings:
            print(f"  ({len(warnings)} optional items noted)")
        print(f"{BOLD}{'='*38}{RESET}")
        return 0
    else:
        print(f"{RED}{BOLD}  \u274C {len(failures)} issue(s) found{RESET}")
        print()
        print("  Fix with:")
        pip_deps = set()
        for name, fix, _ in failures:
            if fix not in ("file", "env", "value"):
                pip_deps.add(fix)
        if pip_deps:
            print(f"    pip install {' '.join(pip_deps)}")
        for name, fix, _ in failures:
            if fix == "file":
                print(f"    cp .env.example .env && nano .env")
            elif fix == "env":
                print(f"    nano ~/aura-v3/.env  # Add: TELEGRAM_TOKEN=your-token")
        print()
        print("  Or run the repair script:")
        print("    bash scripts/termux_install.sh --repair")
        print(f"{BOLD}{'='*38}{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
