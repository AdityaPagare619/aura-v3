"""
AURA v3 Doctor - Comprehensive System Audit & Health Verification
Inspired by OpenClaw's `openclaw security audit --deep` pattern

Provides:
- Pre-flight installation verification
- Security audit with actionable recommendations
- Database integrity checks
- API key validation
- Termux/Android-specific checks
- Production readiness assessment

Usage:
    from src.utils.doctor import AuraDoctor

    doctor = AuraDoctor()
    report = await doctor.run_full_audit()
    print(report.summary())

    # Or specific checks:
    report = await doctor.check_security_only()
"""

import asyncio
import hashlib
import logging
import os
import platform
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CheckStatus(Enum):
    """Status of a single check"""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class CheckCategory(Enum):
    """Categories of checks"""

    ENVIRONMENT = "environment"
    FILES = "files"
    SECURITY = "security"
    DATABASE = "database"
    API_KEYS = "api_keys"
    COMPONENTS = "components"
    RESOURCES = "resources"
    TERMUX = "termux"


@dataclass
class CheckResult:
    """Result of a single check"""

    name: str
    category: CheckCategory
    status: CheckStatus
    message: str
    details: str = ""
    fix_command: str = ""
    fix_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "fix_command": self.fix_command,
            "fix_description": self.fix_description,
        }


@dataclass
class AuditReport:
    """Complete audit report"""

    timestamp: datetime = field(default_factory=datetime.now)
    checks: List[CheckResult] = field(default_factory=list)
    duration_seconds: float = 0.0
    platform: str = ""
    python_version: str = ""
    aura_version: str = ""
    is_termux: bool = False

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.PASS)

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.WARN)

    @property
    def failures(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

    @property
    def skipped(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.SKIP)

    @property
    def is_healthy(self) -> bool:
        """No failures (warnings are OK)"""
        return self.failures == 0

    @property
    def is_production_ready(self) -> bool:
        """No failures and no warnings"""
        return self.failures == 0 and self.warnings == 0

    def get_by_category(self, category: CheckCategory) -> List[CheckResult]:
        return [c for c in self.checks if c.category == category]

    def get_failures(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    def get_warnings(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.WARN]

    def get_fixable(self) -> List[CheckResult]:
        """Get issues with available fixes"""
        return [
            c
            for c in self.checks
            if c.fix_command and c.status in (CheckStatus.FAIL, CheckStatus.WARN)
        ]

    def summary(self, verbose: bool = False) -> str:
        """Generate human-readable summary"""
        lines = [
            "=" * 60,
            "AURA DOCTOR REPORT",
            "=" * 60,
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Platform: {self.platform} | Python: {self.python_version}",
            f"Termux: {'Yes' if self.is_termux else 'No'}",
            f"Duration: {self.duration_seconds:.2f}s",
            "",
            f"Results: {self.passed} passed, {self.warnings} warnings, {self.failures} failures, {self.skipped} skipped",
            "",
        ]

        # Status emoji
        if self.is_production_ready:
            lines.append("✅ PRODUCTION READY - All checks passed!")
        elif self.is_healthy:
            lines.append("⚠️  HEALTHY WITH WARNINGS - Review recommended")
        else:
            lines.append("❌ NOT HEALTHY - Critical issues found")

        lines.append("")

        # Show failures first
        if self.failures > 0:
            lines.append("FAILURES:")
            lines.append("-" * 40)
            for check in self.get_failures():
                lines.append(f"  ❌ {check.name}: {check.message}")
                if check.details:
                    lines.append(f"     Details: {check.details}")
                if check.fix_command:
                    lines.append(f"     Fix: {check.fix_command}")
            lines.append("")

        # Show warnings
        if self.warnings > 0:
            lines.append("WARNINGS:")
            lines.append("-" * 40)
            for check in self.get_warnings():
                lines.append(f"  ⚠️  {check.name}: {check.message}")
                if verbose and check.details:
                    lines.append(f"     Details: {check.details}")
            lines.append("")

        # Show verbose details
        if verbose:
            lines.append("ALL CHECKS:")
            lines.append("-" * 40)
            for category in CheckCategory:
                category_checks = self.get_by_category(category)
                if category_checks:
                    lines.append(f"\n{category.value.upper()}:")
                    for check in category_checks:
                        icon = {"pass": "✓", "warn": "⚠", "fail": "✗", "skip": "○"}[
                            check.status.value
                        ]
                        lines.append(f"  {icon} {check.name}: {check.message}")

        # Fixable issues
        fixable = self.get_fixable()
        if fixable:
            lines.append("")
            lines.append("QUICK FIXES:")
            lines.append("-" * 40)
            for check in fixable:
                lines.append(f"  • {check.fix_description or check.name}:")
                lines.append(f"    $ {check.fix_command}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


class AuraDoctor:
    """
    AURA's Doctor - Comprehensive system audit and health verification

    Inspired by:
    - OpenClaw's `openclaw security audit --deep`
    - Agent Zero's initialization verification

    Provides actionable diagnostics for production deployment.
    """

    def __init__(self, project_root: Optional[Path] = None):
        # Auto-detect project root
        if project_root:
            self.root = Path(project_root)
        else:
            # Try to find aura-v3 root
            current = Path(__file__).resolve()
            for parent in [current] + list(current.parents):
                if (parent / "main.py").exists() and (parent / "src").exists():
                    self.root = parent
                    break
            else:
                self.root = Path.cwd()

        self.is_termux = self._detect_termux()
        self.report = AuditReport()

    def _detect_termux(self) -> bool:
        """Detect if running in Termux environment"""
        return (
            os.environ.get("TERMUX_VERSION") is not None
            or os.environ.get("PREFIX", "").startswith("/data/data/com.termux")
            or Path("/data/data/com.termux").exists()
        )

    async def run_full_audit(self, fix: bool = False) -> AuditReport:
        """
        Run all audit checks

        Args:
            fix: If True, attempt to fix issues automatically

        Returns:
            AuditReport with all check results
        """
        start_time = datetime.now()

        self.report = AuditReport(
            platform=platform.system(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            aura_version=self._get_aura_version(),
            is_termux=self.is_termux,
        )

        # Run all check categories
        await self._check_environment()
        await self._check_files()
        await self._check_security()
        await self._check_databases()
        await self._check_api_keys()
        await self._check_components()
        await self._check_resources()

        if self.is_termux:
            await self._check_termux()

        self.report.duration_seconds = (datetime.now() - start_time).total_seconds()

        return self.report

    async def run_quick_check(self) -> AuditReport:
        """Run minimal checks for quick status"""
        start_time = datetime.now()

        self.report = AuditReport(
            platform=platform.system(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
            is_termux=self.is_termux,
        )

        # Quick checks only
        await self._check_environment()
        await self._check_resources()

        self.report.duration_seconds = (datetime.now() - start_time).total_seconds()
        return self.report

    async def check_security_only(self) -> AuditReport:
        """Run security-focused audit"""
        start_time = datetime.now()

        self.report = AuditReport(is_termux=self.is_termux)

        await self._check_security()
        await self._check_api_keys()

        self.report.duration_seconds = (datetime.now() - start_time).total_seconds()
        return self.report

    def _get_aura_version(self) -> str:
        """Get AURA version from source"""
        try:
            version_file = self.root / "src" / "__version__.py"
            if version_file.exists():
                content = version_file.read_text()
                for line in content.split("\n"):
                    if line.startswith("__version__"):
                        return line.split("=")[1].strip().strip("'\"")
            return "unknown"
        except Exception:
            return "unknown"

    def _add_check(self, result: CheckResult):
        """Add a check result to the report"""
        self.report.checks.append(result)

        # Log based on status
        if result.status == CheckStatus.FAIL:
            logger.error(f"[DOCTOR] FAIL: {result.name} - {result.message}")
        elif result.status == CheckStatus.WARN:
            logger.warning(f"[DOCTOR] WARN: {result.name} - {result.message}")
        else:
            logger.debug(f"[DOCTOR] {result.status.value}: {result.name}")

    # =========================================================================
    # Environment Checks
    # =========================================================================

    async def _check_environment(self):
        """Check Python environment and dependencies"""

        # Python version
        py_version = sys.version_info
        if py_version >= (3, 8):
            self._add_check(
                CheckResult(
                    name="Python Version",
                    category=CheckCategory.ENVIRONMENT,
                    status=CheckStatus.PASS,
                    message=f"Python {py_version.major}.{py_version.minor}.{py_version.micro}",
                )
            )
        else:
            self._add_check(
                CheckResult(
                    name="Python Version",
                    category=CheckCategory.ENVIRONMENT,
                    status=CheckStatus.FAIL,
                    message=f"Python {py_version.major}.{py_version.minor} - requires 3.8+",
                    fix_description="Upgrade Python",
                    fix_command="pkg install python"
                    if self.is_termux
                    else "python -m pip install --upgrade python",
                )
            )

        # Required packages
        required_packages = [
            ("aiofiles", "aiofiles"),
            ("yaml", "pyyaml"),
            ("cryptography", "cryptography"),
            ("telegram", "python-telegram-bot"),
        ]

        for module_name, pip_name in required_packages:
            try:
                __import__(module_name)
                self._add_check(
                    CheckResult(
                        name=f"Package: {pip_name}",
                        category=CheckCategory.ENVIRONMENT,
                        status=CheckStatus.PASS,
                        message="Installed",
                    )
                )
            except ImportError:
                self._add_check(
                    CheckResult(
                        name=f"Package: {pip_name}",
                        category=CheckCategory.ENVIRONMENT,
                        status=CheckStatus.FAIL,
                        message="Not installed",
                        fix_description=f"Install {pip_name}",
                        fix_command=f"pip install {pip_name}",
                    )
                )

        # Optional packages
        optional_packages = [
            ("llama_cpp", "llama-cpp-python", "Local LLM support"),
            ("openai", "openai", "OpenAI API support"),
            ("anthropic", "anthropic", "Anthropic API support"),
            ("psutil", "psutil", "System monitoring"),
        ]

        for module_name, pip_name, description in optional_packages:
            try:
                __import__(module_name)
                self._add_check(
                    CheckResult(
                        name=f"Optional: {pip_name}",
                        category=CheckCategory.ENVIRONMENT,
                        status=CheckStatus.PASS,
                        message=f"Installed - {description}",
                    )
                )
            except ImportError:
                self._add_check(
                    CheckResult(
                        name=f"Optional: {pip_name}",
                        category=CheckCategory.ENVIRONMENT,
                        status=CheckStatus.SKIP,
                        message=f"Not installed - {description}",
                        fix_command=f"pip install {pip_name}",
                    )
                )

    # =========================================================================
    # File System Checks
    # =========================================================================

    async def _check_files(self):
        """Check file system structure and permissions"""

        # Required directories
        required_dirs = [
            ("data", "Data storage"),
            ("logs", "Log files"),
            ("src", "Source code"),
        ]

        for dir_name, description in required_dirs:
            dir_path = self.root / dir_name
            if dir_path.exists():
                # Check if writable
                if os.access(dir_path, os.W_OK):
                    self._add_check(
                        CheckResult(
                            name=f"Directory: {dir_name}/",
                            category=CheckCategory.FILES,
                            status=CheckStatus.PASS,
                            message=f"Exists and writable - {description}",
                        )
                    )
                else:
                    self._add_check(
                        CheckResult(
                            name=f"Directory: {dir_name}/",
                            category=CheckCategory.FILES,
                            status=CheckStatus.FAIL,
                            message=f"Not writable - {description}",
                            fix_command=f"chmod 755 {dir_path}",
                        )
                    )
            else:
                self._add_check(
                    CheckResult(
                        name=f"Directory: {dir_name}/",
                        category=CheckCategory.FILES,
                        status=CheckStatus.WARN,
                        message=f"Missing - {description}",
                        fix_command=f"mkdir -p {dir_path}",
                    )
                )

        # Config files
        config_files = [
            ("config.yaml", False, "Main configuration"),
            (".env", False, "Environment variables"),
            ("main.py", True, "Entry point"),
        ]

        for filename, required, description in config_files:
            file_path = self.root / filename
            if file_path.exists():
                self._add_check(
                    CheckResult(
                        name=f"File: {filename}",
                        category=CheckCategory.FILES,
                        status=CheckStatus.PASS,
                        message=f"Exists - {description}",
                    )
                )

                # YAML syntax check
                if filename.endswith((".yaml", ".yml")):
                    try:
                        import yaml

                        with open(file_path) as f:
                            yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        self._add_check(
                            CheckResult(
                                name=f"Syntax: {filename}",
                                category=CheckCategory.FILES,
                                status=CheckStatus.FAIL,
                                message="Invalid YAML syntax",
                                details=str(e)[:200],
                            )
                        )
            else:
                status = CheckStatus.FAIL if required else CheckStatus.WARN
                self._add_check(
                    CheckResult(
                        name=f"File: {filename}",
                        category=CheckCategory.FILES,
                        status=status,
                        message=f"Missing - {description}",
                        fix_description=f"Create {filename}",
                        fix_command=f"cp {filename}.example {filename}"
                        if not required
                        else "",
                    )
                )

    # =========================================================================
    # Security Checks
    # =========================================================================

    async def _check_security(self):
        """Check security configuration"""

        # Check for exposed secrets in common files
        sensitive_patterns = [
            "password",
            "token",
            "secret",
            "api_key",
            "apikey",
            "private_key",
        ]
        files_to_scan = ["config.yaml", ".env"]

        for filename in files_to_scan:
            file_path = self.root / filename
            if file_path.exists():
                try:
                    content = file_path.read_text().lower()
                    # Check for hardcoded values (not just variable names)
                    exposed = []
                    for line in content.split("\n"):
                        if "=" in line and not line.strip().startswith("#"):
                            key, _, value = line.partition("=")
                            value = value.strip().strip("'\"")
                            # Check if value looks like a real secret (not a placeholder)
                            if any(p in key.lower() for p in sensitive_patterns):
                                if (
                                    value
                                    and len(value) > 10
                                    and not value.startswith("${")
                                    and value
                                    not in ["your_token_here", "xxx", "changeme"]
                                ):
                                    exposed.append(key.strip())

                    if exposed:
                        self._add_check(
                            CheckResult(
                                name=f"Secrets in {filename}",
                                category=CheckCategory.SECURITY,
                                status=CheckStatus.WARN,
                                message=f"Potential exposed secrets: {', '.join(exposed[:3])}",
                                details="Consider using environment variables instead",
                            )
                        )
                    else:
                        self._add_check(
                            CheckResult(
                                name=f"Secrets in {filename}",
                                category=CheckCategory.SECURITY,
                                status=CheckStatus.PASS,
                                message="No exposed secrets detected",
                            )
                        )
                except Exception as e:
                    logger.debug(f"Could not scan {filename}: {e}")

        # Check encryption setup
        try:
            from src.core.security_layers import get_security_layers

            security = get_security_layers()

            # Check if encryption is available
            if hasattr(security, "encryption") and security.encryption:
                self._add_check(
                    CheckResult(
                        name="Encryption",
                        category=CheckCategory.SECURITY,
                        status=CheckStatus.PASS,
                        message="Encryption manager available",
                    )
                )
            else:
                self._add_check(
                    CheckResult(
                        name="Encryption",
                        category=CheckCategory.SECURITY,
                        status=CheckStatus.WARN,
                        message="Encryption not initialized",
                    )
                )
        except ImportError:
            self._add_check(
                CheckResult(
                    name="Security Module",
                    category=CheckCategory.SECURITY,
                    status=CheckStatus.SKIP,
                    message="Security module not available",
                )
            )
        except Exception as e:
            self._add_check(
                CheckResult(
                    name="Security Module",
                    category=CheckCategory.SECURITY,
                    status=CheckStatus.WARN,
                    message=f"Could not verify: {str(e)[:100]}",
                )
            )

        # Check file permissions on sensitive files
        sensitive_files = [".env", "config.yaml", "data/keys"]
        for filename in sensitive_files:
            file_path = self.root / filename
            if file_path.exists():
                try:
                    # Check if file is world-readable (on Unix)
                    if platform.system() != "Windows":
                        import stat

                        mode = file_path.stat().st_mode
                        if mode & stat.S_IROTH:
                            self._add_check(
                                CheckResult(
                                    name=f"Permissions: {filename}",
                                    category=CheckCategory.SECURITY,
                                    status=CheckStatus.WARN,
                                    message="World-readable - should be restricted",
                                    fix_command=f"chmod 600 {file_path}",
                                )
                            )
                except Exception:
                    pass

    # =========================================================================
    # Database Checks
    # =========================================================================

    async def _check_databases(self):
        """Check database integrity"""

        # Find SQLite databases
        db_patterns = ["*.db", "*.sqlite", "*.sqlite3"]
        db_paths = []

        data_dir = self.root / "data"
        if data_dir.exists():
            for pattern in db_patterns:
                db_paths.extend(data_dir.glob(f"**/{pattern}"))

        if not db_paths:
            self._add_check(
                CheckResult(
                    name="Databases",
                    category=CheckCategory.DATABASE,
                    status=CheckStatus.SKIP,
                    message="No SQLite databases found",
                )
            )
            return

        for db_path in db_paths[:5]:  # Limit to 5 databases
            rel_path = (
                db_path.relative_to(self.root)
                if db_path.is_relative_to(self.root)
                else db_path.name
            )

            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()

                # Integrity check
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()

                if result and result[0] == "ok":
                    # Get table count
                    cursor.execute(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                    )
                    table_count = cursor.fetchone()[0]

                    self._add_check(
                        CheckResult(
                            name=f"DB: {rel_path}",
                            category=CheckCategory.DATABASE,
                            status=CheckStatus.PASS,
                            message=f"Integrity OK - {table_count} tables",
                        )
                    )
                else:
                    self._add_check(
                        CheckResult(
                            name=f"DB: {rel_path}",
                            category=CheckCategory.DATABASE,
                            status=CheckStatus.FAIL,
                            message=f"Integrity check failed: {result}",
                            details="Database may be corrupted",
                        )
                    )

                conn.close()

            except sqlite3.DatabaseError as e:
                self._add_check(
                    CheckResult(
                        name=f"DB: {rel_path}",
                        category=CheckCategory.DATABASE,
                        status=CheckStatus.FAIL,
                        message=f"Database error: {str(e)[:100]}",
                    )
                )
            except Exception as e:
                self._add_check(
                    CheckResult(
                        name=f"DB: {rel_path}",
                        category=CheckCategory.DATABASE,
                        status=CheckStatus.WARN,
                        message=f"Could not check: {str(e)[:100]}",
                    )
                )

    # =========================================================================
    # API Key Checks
    # =========================================================================

    async def _check_api_keys(self):
        """Check API key configuration"""

        api_keys = [
            ("TELEGRAM_TOKEN", "TELEGRAM_BOT_TOKEN", True, "Telegram bot"),
            ("OPENAI_API_KEY", None, False, "OpenAI"),
            ("ANTHROPIC_API_KEY", None, False, "Anthropic"),
            ("GOOGLE_API_KEY", None, False, "Google AI"),
        ]

        for primary_key, alt_key, required, description in api_keys:
            value = os.environ.get(primary_key) or (
                os.environ.get(alt_key) if alt_key else None
            )

            if value:
                # Basic validation (not empty, reasonable length)
                if len(value) > 10:
                    self._add_check(
                        CheckResult(
                            name=f"API Key: {description}",
                            category=CheckCategory.API_KEYS,
                            status=CheckStatus.PASS,
                            message=f"Configured ({len(value)} chars)",
                        )
                    )
                else:
                    self._add_check(
                        CheckResult(
                            name=f"API Key: {description}",
                            category=CheckCategory.API_KEYS,
                            status=CheckStatus.WARN,
                            message="Value seems too short",
                        )
                    )
            else:
                status = CheckStatus.FAIL if required else CheckStatus.SKIP
                self._add_check(
                    CheckResult(
                        name=f"API Key: {description}",
                        category=CheckCategory.API_KEYS,
                        status=status,
                        message="Not set",
                        fix_description=f"Set {primary_key} environment variable",
                        fix_command=f"export {primary_key}=your_key_here",
                    )
                )

    # =========================================================================
    # Component Checks
    # =========================================================================

    async def _check_components(self):
        """Check AURA components can load"""

        components = [
            ("src.agent.loop", "ReActAgent", "ReAct Agent"),
            ("src.memory.hierarchical", "HierarchicalMemory", "Memory System"),
            ("src.security.security", "SecurityManager", "Security"),
            ("src.tools.registry", "ToolRegistry", "Tool System"),
        ]

        for module_path, class_name, description in components:
            try:
                module = __import__(module_path, fromlist=[class_name])
                if hasattr(module, class_name):
                    self._add_check(
                        CheckResult(
                            name=f"Component: {description}",
                            category=CheckCategory.COMPONENTS,
                            status=CheckStatus.PASS,
                            message="Loadable",
                        )
                    )
                else:
                    self._add_check(
                        CheckResult(
                            name=f"Component: {description}",
                            category=CheckCategory.COMPONENTS,
                            status=CheckStatus.WARN,
                            message=f"Module exists but {class_name} not found",
                        )
                    )
            except ImportError as e:
                self._add_check(
                    CheckResult(
                        name=f"Component: {description}",
                        category=CheckCategory.COMPONENTS,
                        status=CheckStatus.FAIL,
                        message=f"Import failed: {str(e)[:100]}",
                    )
                )
            except Exception as e:
                self._add_check(
                    CheckResult(
                        name=f"Component: {description}",
                        category=CheckCategory.COMPONENTS,
                        status=CheckStatus.WARN,
                        message=f"Error loading: {str(e)[:100]}",
                    )
                )

    # =========================================================================
    # Resource Checks
    # =========================================================================

    async def _check_resources(self):
        """Check system resources"""

        try:
            import psutil

            # Memory
            memory = psutil.virtual_memory()
            mem_available_mb = memory.available / (1024 * 1024)
            mem_percent = memory.percent

            if mem_available_mb > 512:
                self._add_check(
                    CheckResult(
                        name="Memory Available",
                        category=CheckCategory.RESOURCES,
                        status=CheckStatus.PASS,
                        message=f"{mem_available_mb:.0f}MB free ({100 - mem_percent:.0f}%)",
                    )
                )
            elif mem_available_mb > 256:
                self._add_check(
                    CheckResult(
                        name="Memory Available",
                        category=CheckCategory.RESOURCES,
                        status=CheckStatus.WARN,
                        message=f"{mem_available_mb:.0f}MB free - may be constrained",
                    )
                )
            else:
                self._add_check(
                    CheckResult(
                        name="Memory Available",
                        category=CheckCategory.RESOURCES,
                        status=CheckStatus.FAIL,
                        message=f"{mem_available_mb:.0f}MB free - insufficient",
                        details="AURA needs at least 256MB free RAM",
                    )
                )

            # Disk space
            disk = psutil.disk_usage(str(self.root))
            disk_free_mb = disk.free / (1024 * 1024)

            if disk_free_mb > 500:
                self._add_check(
                    CheckResult(
                        name="Disk Space",
                        category=CheckCategory.RESOURCES,
                        status=CheckStatus.PASS,
                        message=f"{disk_free_mb:.0f}MB free",
                    )
                )
            elif disk_free_mb > 100:
                self._add_check(
                    CheckResult(
                        name="Disk Space",
                        category=CheckCategory.RESOURCES,
                        status=CheckStatus.WARN,
                        message=f"{disk_free_mb:.0f}MB free - running low",
                    )
                )
            else:
                self._add_check(
                    CheckResult(
                        name="Disk Space",
                        category=CheckCategory.RESOURCES,
                        status=CheckStatus.FAIL,
                        message=f"{disk_free_mb:.0f}MB free - critically low",
                    )
                )

            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self._add_check(
                CheckResult(
                    name="CPU Usage",
                    category=CheckCategory.RESOURCES,
                    status=CheckStatus.PASS if cpu_percent < 90 else CheckStatus.WARN,
                    message=f"{cpu_percent}%",
                )
            )

        except ImportError:
            self._add_check(
                CheckResult(
                    name="Resource Monitoring",
                    category=CheckCategory.RESOURCES,
                    status=CheckStatus.SKIP,
                    message="psutil not installed",
                    fix_command="pip install psutil",
                )
            )

    # =========================================================================
    # Termux-Specific Checks
    # =========================================================================

    async def _check_termux(self):
        """Termux/Android-specific checks"""

        # Check Termux-API
        termux_api = shutil.which("termux-notification")
        if termux_api:
            self._add_check(
                CheckResult(
                    name="Termux-API",
                    category=CheckCategory.TERMUX,
                    status=CheckStatus.PASS,
                    message="Installed",
                )
            )
        else:
            self._add_check(
                CheckResult(
                    name="Termux-API",
                    category=CheckCategory.TERMUX,
                    status=CheckStatus.WARN,
                    message="Not installed - some features unavailable",
                    fix_command="pkg install termux-api",
                )
            )

        # Check storage permission
        storage_path = Path(os.environ.get("HOME", "")) / "storage"
        if storage_path.exists():
            self._add_check(
                CheckResult(
                    name="Storage Access",
                    category=CheckCategory.TERMUX,
                    status=CheckStatus.PASS,
                    message="Storage symlinks available",
                )
            )
        else:
            self._add_check(
                CheckResult(
                    name="Storage Access",
                    category=CheckCategory.TERMUX,
                    status=CheckStatus.WARN,
                    message="Storage not linked",
                    fix_command="termux-setup-storage",
                )
            )

        # Check wake-lock capability
        wake_lock = shutil.which("termux-wake-lock")
        if wake_lock:
            self._add_check(
                CheckResult(
                    name="Wake Lock",
                    category=CheckCategory.TERMUX,
                    status=CheckStatus.PASS,
                    message="Available for background execution",
                )
            )
        else:
            self._add_check(
                CheckResult(
                    name="Wake Lock",
                    category=CheckCategory.TERMUX,
                    status=CheckStatus.WARN,
                    message="Not available - background may be killed",
                    fix_command="pkg install termux-api",
                )
            )

        # Check battery optimization
        self._add_check(
            CheckResult(
                name="Battery Optimization",
                category=CheckCategory.TERMUX,
                status=CheckStatus.WARN,
                message="Ensure Termux is excluded from battery optimization",
                details="Settings > Apps > Termux > Battery > Unrestricted",
            )
        )


# =============================================================================
# CLI Entry Point
# =============================================================================


async def main():
    """CLI entry point for `python -m src.utils.doctor`"""
    import argparse

    parser = argparse.ArgumentParser(
        description="AURA Doctor - System audit and health verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.utils.doctor              # Full audit
  python -m src.utils.doctor --quick      # Quick check
  python -m src.utils.doctor --security   # Security audit only
  python -m src.utils.doctor --verbose    # Detailed output
        """,
    )
    parser.add_argument("--quick", action="store_true", help="Quick status check")
    parser.add_argument("--security", action="store_true", help="Security audit only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    doctor = AuraDoctor()

    if args.quick:
        report = await doctor.run_quick_check()
    elif args.security:
        report = await doctor.check_security_only()
    else:
        report = await doctor.run_full_audit()

    if args.json:
        import json

        print(
            json.dumps(
                {
                    "timestamp": report.timestamp.isoformat(),
                    "summary": {
                        "total": report.total_checks,
                        "passed": report.passed,
                        "warnings": report.warnings,
                        "failures": report.failures,
                    },
                    "is_healthy": report.is_healthy,
                    "is_production_ready": report.is_production_ready,
                    "checks": [c.to_dict() for c in report.checks],
                },
                indent=2,
            )
        )
    else:
        print(report.summary(verbose=args.verbose))

    # Exit with appropriate code
    sys.exit(0 if report.is_healthy else 1)


if __name__ == "__main__":
    asyncio.run(main())
