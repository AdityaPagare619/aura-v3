#!/usr/bin/env python3
"""
AURA v3 Doctor — Production Health Diagnostic
==============================================
Comprehensive health check for AURA on any device.

Usage:
    python scripts/aura_doctor.py
    python scripts/aura_doctor.py --fix      # Auto-fix what it can
    python scripts/aura_doctor.py --json     # Machine-readable output
"""

import asyncio
import importlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

# Ensure project root is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# ── Colors ──────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

results = []


def ok(section, msg):
    results.append({"status": "ok", "section": section, "message": msg})
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(section, msg):
    results.append({"status": "warn", "section": section, "message": msg})
    print(f"  {YELLOW}!{RESET} {msg}")


def fail(section, msg, fix=None):
    results.append({"status": "fail", "section": section, "message": msg, "fix": fix})
    fix_str = f" → Fix: {fix}" if fix else ""
    print(f"  {RED}✗{RESET} {msg}{fix_str}")


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 1: Platform & Environment
# ═══════════════════════════════════════════════════════════════════════════
def check_platform():
    section = "Platform"
    print(f"\n{BOLD}[1/6] Platform & Environment{RESET}")

    # OS
    is_termux = os.environ.get("TERMUX_VERSION") or os.path.isdir("/data/data/com.termux")
    os_name = "Termux" if is_termux else platform.system()
    ok(section, f"OS: {os_name} ({platform.machine()})")

    # Python version
    py_ver = sys.version_info
    if py_ver >= (3, 8):
        ok(section, f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    else:
        fail(section, f"Python {py_ver.major}.{py_ver.minor} — need 3.8+", "pkg install python")

    # RAM check
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb >= 4:
            ok(section, f"RAM: {ram_gb:.1f} GB")
        elif ram_gb >= 2:
            warn(section, f"RAM: {ram_gb:.1f} GB — AURA will run but may be slow")
        else:
            fail(section, f"RAM: {ram_gb:.1f} GB — minimum 2GB recommended")
    except ImportError:
        warn(section, "psutil not available — cannot check RAM")

    # Disk space
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
        free_gb = usage.free / (1024**3)
        if free_gb >= 1:
            ok(section, f"Disk free: {free_gb:.1f} GB")
        elif free_gb >= 0.3:
            warn(section, f"Disk free: {free_gb:.1f} GB — getting low")
        else:
            fail(section, f"Disk free: {free_gb:.1f} GB — critically low!")
    except Exception:
        warn(section, "Could not check disk space")


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 2: Dependencies
# ═══════════════════════════════════════════════════════════════════════════
def check_dependencies():
    section = "Dependencies"
    print(f"\n{BOLD}[2/6] Dependencies{RESET}")

    required = [
        ("aiofiles", "aiofiles"),
        ("yaml", "pyyaml"),
        ("cryptography", "cryptography"),
        ("psutil", "psutil"),
        ("telegram", "python-telegram-bot"),
    ]

    optional = [
        ("numpy", "numpy"),
        ("llama_cpp", "llama-cpp-python"),
    ]

    missing_required = []
    for import_name, pip_name in required:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            ok(section, f"{import_name} ({ver})")
        except ImportError:
            fail(section, f"{import_name} — MISSING", f"pip install {pip_name}")
            missing_required.append(pip_name)

    for import_name, pip_name in optional:
        try:
            importlib.import_module(import_name)
            ok(section, f"{import_name} (optional)")
        except ImportError:
            warn(section, f"{import_name} (optional, not installed)")

    return missing_required


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 3: Core Module Imports
# ═══════════════════════════════════════════════════════════════════════════
def check_modules():
    section = "Modules"
    print(f"\n{BOLD}[3/6] AURA Modules{RESET}")

    critical_modules = [
        "src.main",
        "src.agent.loop",
        "src.memory",
        "src.llm.manager",
        "src.tools.registry",
        "src.security.security",
        "src.session.manager",
        "src.core.neuromorphic_engine",
        "src.core.adaptive_personality",
        "src.channels.communication",
    ]

    all_ok = True
    for mod_name in critical_modules:
        try:
            importlib.import_module(mod_name)
            ok(section, mod_name)
        except Exception as e:
            fail(section, f"{mod_name}: {str(e)[:80]}")
            all_ok = False

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 4: Boot Test
# ═══════════════════════════════════════════════════════════════════════════
def check_boot():
    section = "Boot"
    print(f"\n{BOLD}[4/6] Boot Test{RESET}")

    import logging
    logging.disable(logging.WARNING)

    try:
        from src.main import AuraProduction
        start = time.time()
        a = AuraProduction()
        elapsed_init = time.time() - start
        ok(section, f"AuraProduction() created ({elapsed_init:.2f}s)")

        # Test initialize (tier 0)
        start = time.time()
        asyncio.get_event_loop().run_until_complete(a.initialize())
        elapsed_boot = time.time() - start

        if elapsed_boot < 5:
            ok(section, f"Tier 0 initialized ({elapsed_boot:.2f}s)")
        elif elapsed_boot < 15:
            warn(section, f"Tier 0 initialized but slow ({elapsed_boot:.2f}s)")
        else:
            warn(section, f"Tier 0 very slow ({elapsed_boot:.2f}s) — check LLM config")

        # Count initialized components
        attrs = [x for x in dir(a) if x.startswith('_') and not x.startswith('__')]
        none_count = len([x for x in attrs if getattr(a, x, 'Z') is None])
        init_count = len([x for x in attrs if getattr(a, x, None) is not None and not callable(getattr(a, x))])
        ok(section, f"Tier 0: {init_count} components initialized, {none_count} deferred")

        return a

    except Exception as e:
        fail(section, f"BOOT FAILED: {e}")
        return None

    finally:
        logging.disable(logging.NOTSET)


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 5: Configuration
# ═══════════════════════════════════════════════════════════════════════════
def check_config():
    section = "Config"
    print(f"\n{BOLD}[5/6] Configuration{RESET}")

    # .env
    env_path = Path(PROJECT_ROOT) / ".env"
    if env_path.exists():
        env_vars = {}
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip()

        token = env_vars.get("TELEGRAM_TOKEN") or env_vars.get("TELEGRAM_BOT_TOKEN", "")
        if token and len(token) > 10:
            ok(section, f"TELEGRAM_TOKEN: ...{token[-6:]}")
        elif token:
            warn(section, f"TELEGRAM_TOKEN looks incomplete ({len(token)} chars)")
        else:
            fail(section, "TELEGRAM_TOKEN not set", "nano .env")
    else:
        fail(section, ".env file missing", "cp .env.example .env && nano .env")

    # Data directories
    for d in ["data", "data/sessions", "data/memories", "logs"]:
        p = Path(PROJECT_ROOT) / d
        if p.is_dir():
            ok(section, f"{d}/ exists")
        else:
            warn(section, f"{d}/ missing — will be created on first run")


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 6: Data Integrity
# ═══════════════════════════════════════════════════════════════════════════
def check_data():
    section = "Data"
    print(f"\n{BOLD}[6/6] Data Integrity{RESET}")

    data_dir = Path(PROJECT_ROOT) / "data"
    if not data_dir.exists():
        warn(section, "No data directory yet (fresh install)")
        return

    # Count session files
    sessions_dir = data_dir / "sessions"
    if sessions_dir.exists():
        enc_files = list(sessions_dir.glob("*.enc"))
        ok(section, f"{len(enc_files)} encrypted session file(s)")
    else:
        warn(section, "No sessions directory")

    # Check DB files
    for db_name in ["memory/vectors.db", "personality/personality_state.db"]:
        db_path = data_dir / db_name
        if db_path.exists():
            size_kb = db_path.stat().st_size / 1024
            ok(section, f"{db_name} ({size_kb:.0f} KB)")
        else:
            warn(section, f"{db_name} — not created yet")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    json_mode = "--json" in sys.argv
    fix_mode = "--fix" in sys.argv

    if not json_mode:
        print()
        print(f"{BOLD}{'='*50}{RESET}")
        print(f"{BOLD}  AURA v3 Doctor — Health Diagnostic{RESET}")
        print(f"{BOLD}{'='*50}{RESET}")

    check_platform()
    missing_deps = check_dependencies()
    modules_ok = check_modules()
    aura_instance = check_boot()
    check_config()
    check_data()

    # ── Summary ──
    ok_count = len([r for r in results if r["status"] == "ok"])
    warn_count = len([r for r in results if r["status"] == "warn"])
    fail_count = len([r for r in results if r["status"] == "fail"])

    if json_mode:
        print(json.dumps({
            "ok": ok_count, "warnings": warn_count, "failures": fail_count,
            "results": results
        }, indent=2))
        return 1 if fail_count > 0 else 0

    print()
    print(f"{BOLD}{'='*50}{RESET}")
    if fail_count == 0:
        print(f"{GREEN}{BOLD}  HEALTHY: {ok_count} checks passed{RESET}")
        if warn_count:
            print(f"  {warn_count} warning(s) noted")
    else:
        print(f"{RED}{BOLD}  {fail_count} ISSUE(S) FOUND{RESET}")
        print(f"  {ok_count} passed, {warn_count} warnings")
        print()
        print(f"  Fixes:")
        for r in results:
            if r["status"] == "fail" and r.get("fix"):
                print(f"    {r['fix']}")
    print(f"{BOLD}{'='*50}{RESET}")
    print()

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
