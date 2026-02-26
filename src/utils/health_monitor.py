"""
AURA v3 Health Monitor
Monitors system health, component status, and resource usage
Designed for mobile (Termux) constraints - lightweight and efficient
"""

import asyncio
import logging
import psutil
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    MEMORY = "memory"
    LLM = "llm"
    VOICE = "voice"
    ANDROID = "android"
    AGENT = "agent"
    SECURITY = "security"
    NETWORK = "network"
    STORAGE = "storage"


@dataclass
class HealthCheck:
    """Result of a single health check"""

    component: ComponentType
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recovery_actions: List[str] = field(default_factory=list)


@dataclass
class SystemHealth:
    """Overall system health snapshot"""

    overall_status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    battery_percent: Optional[int] = None
    checks: List[HealthCheck] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class HealthMonitor:
    """
    AURA's Health Monitor - continuously checks system and component health

    Designed for mobile constraints:
    - Runs lightweight checks
    - Caches results to avoid repeated checks
    - Provides recovery recommendations
    - Can trigger automatic recovery actions
    """

    def __init__(
        self,
        check_interval: int = 60,
        critical_threshold: float = 0.90,
        degraded_threshold: float = 0.75,
    ):
        self.check_interval = check_interval
        self.critical_threshold = critical_threshold
        self.degraded_threshold = degraded_threshold

        self._start_time = time.time()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_health: Optional[SystemHealth] = None

        self._check_cache: Dict[str, tuple] = {}
        self._cache_ttl = 10

        self._custom_checks: Dict[ComponentType, Callable] = {}
        self._health_history = deque(maxlen=100)

        self._recovery_actions: Dict[ComponentType, Callable] = {}

    def register_custom_check(
        self, component: ComponentType, check_func: Callable[[], HealthCheck]
    ):
        """Register a custom health check for a component"""
        self._custom_checks[component] = check_func
        logger.info(f"Registered custom health check for {component.value}")

    def register_recovery_action(
        self, component: ComponentType, action_func: Callable[[], asyncio.coroutine]
    ):
        """Register an automatic recovery action for a component"""
        self._recovery_actions[component] = action_func
        logger.info(f"Registered recovery action for {component.value}")

    async def start(self):
        """Start the health monitoring loop"""
        if self._running:
            logger.warning("Health monitor already running")
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")

    async def stop(self):
        """Stop the health monitoring loop"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                health = await self.check_health()
                self._last_health = health
                self._health_history.append(health)

                await self._process_health_status(health)

            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")

            await asyncio.sleep(self.check_interval)

    async def _process_health_status(self, health: SystemHealth):
        """Process health status and trigger recovery if needed"""
        if health.overall_status == HealthStatus.CRITICAL:
            logger.warning("System health is CRITICAL")
            for check in health.checks:
                if check.status == HealthStatus.CRITICAL:
                    component = check.component
                    if component in self._recovery_actions:
                        try:
                            logger.info(f"Attempting recovery for {component.value}")
                            await self._recovery_actions[component]()
                        except Exception as e:
                            logger.error(f"Recovery failed for {component.value}: {e}")

    async def check_health(self) -> SystemHealth:
        """Perform full system health check"""
        checks = []

        checks.append(await self._check_memory())
        checks.append(await self._check_cpu())
        checks.append(await self._check_storage())

        for component, check_func in self._custom_checks.items():
            try:
                check = check_func()
                if asyncio.iscoroutine(check):
                    check = await check
                checks.append(check)
            except Exception as e:
                logger.error(f"Custom check failed for {component.value}: {e}")

        overall = self._calculate_overall_status(checks)
        recommendations = self._generate_recommendations(checks)

        return SystemHealth(
            overall_status=overall,
            timestamp=datetime.now(),
            uptime_seconds=time.time() - self._start_time,
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=psutil.virtual_memory().percent,
            memory_available_mb=psutil.virtual_memory().available / (1024 * 1024),
            battery_percent=self._get_battery_level(),
            checks=checks,
            recommendations=recommendations,
        )

    async def _check_memory(self) -> HealthCheck:
        """Check memory health"""
        try:
            vm = psutil.virtual_memory()

            if vm.percent >= self.critical_threshold * 100:
                status = HealthStatus.CRITICAL
                message = f"Memory critical: {vm.percent:.1f}% used"
                recovery = [
                    "Clear working memory cache",
                    "Reduce concurrent agent tasks",
                    "Consider restarting non-essential services",
                ]
            elif vm.percent >= self.degraded_threshold * 100:
                status = HealthStatus.DEGRADED
                message = f"Memory degraded: {vm.percent:.1f}% used"
                recovery = ["Monitor memory usage", "Clear unnecessary caches"]
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory OK: {vm.percent:.1f}% used"
                recovery = []

            return HealthCheck(
                component=ComponentType.MEMORY,
                status=status,
                message=message,
                metrics={
                    "total_mb": vm.total / (1024 * 1024),
                    "available_mb": vm.available / (1024 * 1024),
                    "used_mb": vm.used / (1024 * 1024),
                    "percent": vm.percent,
                },
                recovery_actions=recovery,
            )

        except Exception as e:
            return HealthCheck(
                component=ComponentType.MEMORY,
                status=HealthStatus.UNKNOWN,
                message=f"Memory check failed: {e}",
                recovery_actions=["Check system resources"],
            )

    async def _check_cpu(self) -> HealthCheck:
        """Check CPU health"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            cpu_count = psutil.cpu_count()

            if cpu_percent >= self.critical_threshold * 100:
                status = HealthStatus.CRITICAL
                message = f"CPU critical: {cpu_percent:.1f}% usage"
                recovery = ["Reduce concurrent tasks", "Check for runaway processes"]
            elif cpu_percent >= self.degraded_threshold * 100:
                status = HealthStatus.DEGRADED
                message = f"CPU degraded: {cpu_percent:.1f}% usage"
                recovery = ["Monitor CPU usage"]
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU OK: {cpu_percent:.1f}% usage"
                recovery = []

            return HealthCheck(
                component=ComponentType.LLM,
                status=status,
                message=message,
                metrics={"percent": cpu_percent, "count": cpu_count},
                recovery_actions=recovery,
            )

        except Exception as e:
            return HealthCheck(
                component=ComponentType.LLM,
                status=HealthStatus.UNKNOWN,
                message=f"CPU check failed: {e}",
                recovery_actions=["Check system processes"],
            )

    async def _check_storage(self) -> HealthCheck:
        """Check storage health"""
        try:
            disk = psutil.disk_usage("/")

            if disk.percent >= 95:
                status = HealthStatus.CRITICAL
                message = f"Storage critical: {disk.percent:.1f}% used"
                recovery = ["Clear logs", "Remove unused files", "Clean memory cache"]
            elif disk.percent >= 85:
                status = HealthStatus.DEGRADED
                message = f"Storage degraded: {disk.percent:.1f}% used"
                recovery = ["Monitor storage usage"]
            else:
                status = HealthStatus.HEALTHY
                message = f"Storage OK: {disk.percent:.1f}% used"
                recovery = []

            return HealthCheck(
                component=ComponentType.STORAGE,
                status=status,
                message=message,
                metrics={
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent": disk.percent,
                },
                recovery_actions=recovery,
            )

        except Exception as e:
            return HealthCheck(
                component=ComponentType.STORAGE,
                status=HealthStatus.UNKNOWN,
                message=f"Storage check failed: {e}",
                recovery_actions=["Check disk space"],
            )

    def _get_battery_level(self) -> Optional[int]:
        """Get battery level (mobile-specific)"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return battery.percent
        except Exception:
            pass
        return None

    def _calculate_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Calculate overall system status from individual checks"""
        if not checks:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in checks]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses and all(
            s == HealthStatus.UNKNOWN for s in statuses
        ):
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY

    def _generate_recommendations(self, checks: List[HealthCheck]) -> List[str]:
        """Generate recommendations based on health checks"""
        recommendations = []

        for check in checks:
            if check.status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                recommendations.extend(check.recovery_actions)

        return list(set(recommendations))

    def get_last_health(self) -> Optional[SystemHealth]:
        """Get the last health check result"""
        return self._last_health

    def get_health_history(self) -> List[SystemHealth]:
        """Get health check history"""
        return list(self._health_history)

    async def quick_check(self) -> Dict[str, Any]:
        """Perform a quick health check (lightweight)"""
        return {
            "status": self._last_health.overall_status.value
            if self._last_health
            else "unknown",
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "uptime_seconds": time.time() - self._start_time,
        }

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics including resource monitor data if available"""
        try:
            from src.utils.resource_monitor import get_resource_monitor

            resource_monitor = get_resource_monitor()
            return resource_monitor.get_summary()
        except ImportError:
            # Fallback if resource monitor not available
            return {
                "current": {
                    "ram": {
                        "percent": psutil.virtual_memory().percent,
                        "used_mb": psutil.virtual_memory().used / (1024 * 1024),
                    },
                    "cpu": {
                        "percent": psutil.cpu_percent(interval=0.1),
                    },
                },
                "peaks": {},
                "llm": {},
            }


# ==============================================================================
# INTEGRATION WITH RESOURCE MONITOR
# ==============================================================================


def get_health_and_resources() -> Dict[str, Any]:
    """Get combined health and resource information"""
    try:
        from src.utils.resource_monitor import get_resource_monitor

        # Get resource monitor data
        resource_monitor = get_resource_monitor()
        resource_summary = resource_monitor.get_summary()

        # Get current metrics
        current = resource_summary.get("current", {})
        ram = current.get("ram", {})
        cpu = current.get("cpu", {})

        # Determine health status
        ram_percent = ram.get("percent", 0)
        cpu_percent = cpu.get("percent", 0)

        if ram_percent >= 90 or cpu_percent >= 85:
            status = "critical"
        elif ram_percent >= 75 or cpu_percent >= 70:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "resources": resource_summary,
            "alerts": [],
        }

    except ImportError:
        return {
            "status": "unknown",
            "resources": {},
            "alerts": ["Resource monitor not available"],
        }
