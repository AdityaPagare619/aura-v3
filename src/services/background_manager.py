"""
AURA v3 Background Resource Manager
Manages AURA running in background on mobile without draining resources
Handles screen state, battery awareness, and smart task scheduling
"""

import asyncio
import logging
import psutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ResourcePriority(Enum):
    """Priority levels for background tasks"""

    CRITICAL = 3
    HIGH = 2
    MEDIUM = 1
    LOW = 0


class ScreenState(Enum):
    """Mobile screen states"""

    ON = "on"
    OFF = "off"
    LOCKED = "locked"
    UNKNOWN = "unknown"


class TaskIntensity(Enum):
    """How resource-intensive a task is"""

    HEAVY = 3  # LLM inference, video processing
    MEDIUM = 2  # File operations, web scraping
    LIGHT = 1  # Memory updates, notifications
    MINIMAL = 0  # Just monitoring


@dataclass
class ResourceBudget:
    """Budget for resource usage"""

    cpu_percent_limit: float = 30.0
    memory_mb_limit: int = 512
    battery_percent_min: int = 20
    max_concurrent_heavy: int = 1


@dataclass
class BackgroundTask:
    """A task that can run in background"""

    id: str
    name: str
    intensity: TaskIntensity
    priority: ResourcePriority

    # Execution control
    can_pause: bool = True
    can_resume: bool = True
    pause_on_low_battery: bool = True
    pause_on_screen_off: bool = False
    pause_on_heavy_use: bool = True

    # State
    is_running: bool = False
    is_paused: bool = False
    progress: float = 0.0


class BackgroundResourceManager:
    """
    Background Resource Manager

    Manages AURA operations on mobile when:
    - Screen is off
    - Phone is locked
    - Battery is low
    - User is in a meeting
    - User is using other apps

    Key features:
    - Battery-aware task scheduling
    - Screen state detection
    - Resource budgeting
    - Pause/resume logic
    - Priority-based execution
    """

    def __init__(self):
        self._running = False
        self._manager_task: Optional[asyncio.Task] = None

        # Background tasks
        self._background_tasks: Dict[str, BackgroundTask] = {}

        # Resource budgets
        self._budget = ResourceBudget()

        # Current state
        self._screen_state = ScreenState.UNKNOWN
        self._battery_level: Optional[int] = None
        self._cpu_usage: float = 0.0
        self._memory_usage: float = 0.0

        # Settings
        self.check_interval = 30  # seconds
        self.enable_battery_saving = True
        self.enable_screen_aware = True

        # Callbacks
        self._state_callbacks: List[Callable] = []

    async def start(self):
        """Start resource manager"""
        self._running = True
        self._manager_task = asyncio.create_task(self._manager_loop())

        # Initial state check
        await self._update_resource_state()

        logger.info("Background Resource Manager started")

    async def stop(self):
        """Stop resource manager"""
        self._running = False

        # Pause all background tasks
        for task in self._background_tasks.values():
            if task.is_running:
                task.is_running = False

        if self._manager_task:
            self._manager_task.cancel()

        logger.info("Background Resource Manager stopped")

    # =========================================================================
    # TASK REGISTRATION
    # =========================================================================

    async def register_task(
        self,
        task_id: str,
        name: str,
        intensity: TaskIntensity,
        priority: ResourcePriority = ResourcePriority.MEDIUM,
        can_pause: bool = True,
        **options,
    ) -> BackgroundTask:
        """Register a task for background management"""

        task = BackgroundTask(
            id=task_id,
            name=name,
            intensity=intensity,
            priority=priority,
            can_pause=can_pause,
            pause_on_low_battery=options.get("pause_on_low_battery", True),
            pause_on_screen_off=options.get("pause_on_screen_off", False),
            pause_on_heavy_use=options.get("pause_on_heavy_use", True),
        )

        self._background_tasks[task_id] = task

        logger.info(f"Registered background task: {task_id} ({intensity.name})")

        return task

    async def unregister_task(self, task_id: str):
        """Unregister a background task"""
        if task_id in self._background_tasks:
            task = self._background_tasks[task_id]
            if task.is_running:
                await self._stop_task(task)
            del self._background_tasks[task_id]

    async def start_task(self, task_id: str):
        """Start a background task if resources allow"""
        if task_id not in self._background_tasks:
            return False

        task = self._background_tasks[task_id]

        # Check if we can start
        if not await self._can_start_task(task):
            logger.info(f"Cannot start task {task_id} - resources limited")
            return False

        task.is_running = True
        task.is_paused = False

        logger.info(f"Started background task: {task_id}")
        return True

    async def stop_task(self, task_id: str):
        """Stop a background task"""
        if task_id in self._background_tasks:
            await self._stop_task(self._background_tasks[task_id])

    async def pause_task(self, task_id: str):
        """Pause a background task"""
        if task_id in self._background_tasks:
            task = self._background_tasks[task_id]
            if task.can_pause and task.is_running:
                task.is_paused = True
                logger.info(f"Paused background task: {task_id}")

    async def resume_task(self, task_id: str):
        """Resume a paused background task"""
        if task_id in self._background_tasks:
            task = self._background_tasks[task_id]
            if task.can_resume and task.is_paused:
                if await self._can_start_task(task):
                    task.is_paused = False
                    logger.info(f"Resumed background task: {task_id}")

    # =========================================================================
    # RESOURCE MANAGEMENT
    # =========================================================================

    async def _update_resource_state(self):
        """Update current resource usage (non-blocking)"""
        try:
            loop = asyncio.get_event_loop()

            # Batch all psutil calls into a single executor call to avoid
            # blocking the asyncio event loop (psutil I/O can take 100ms+)
            def _read_system_stats():
                cpu = psutil.cpu_percent(interval=0.5)
                vm = psutil.virtual_memory()
                try:
                    battery = psutil.sensors_battery()
                except Exception:
                    battery = None
                return cpu, vm, battery

            cpu, vm, battery = await loop.run_in_executor(None, _read_system_stats)

            self._cpu_usage = cpu
            self._memory_usage = vm.percent
            self._battery_level = battery.percent if battery else None

        except Exception as e:
            logger.error(f"Error updating resource state: {e}")

    async def _can_start_task(self, task: BackgroundTask) -> bool:
        """Check if a task can be started given current resources"""

        # Check heavy task limits
        if task.intensity == TaskIntensity.HEAVY:
            heavy_running = len(
                [
                    t
                    for t in self._background_tasks.values()
                    if t.is_running and t.intensity == TaskIntensity.HEAVY
                ]
            )
            if heavy_running >= self._budget.max_concurrent_heavy:
                return False

        # Check CPU limits
        if self._cpu_usage > self._budget.cpu_percent_limit:
            if task.pause_on_heavy_use:
                return False

        # Check memory limits
        if self._memory_usage > 80:  # High memory usage
            if task.intensity in [TaskIntensity.HEAVY, TaskIntensity.MEDIUM]:
                return False

        # Check battery
        if task.pause_on_low_battery and self._battery_level:
            if self._battery_level < self._budget.battery_percent_min:
                return False

        # Check screen state
        if task.pause_on_screen_off:
            if self._screen_state in [ScreenState.OFF, ScreenState.LOCKED]:
                return False

        return True

    async def _apply_resource_limits(self):
        """Apply resource limits to running tasks"""

        await self._update_resource_state()

        for task in self._background_tasks.values():
            if not task.is_running:
                continue

            should_pause = False
            reason = ""

            # Battery check
            if task.pause_on_low_battery and self._battery_level:
                if self._battery_level < self._budget.battery_percent_min:
                    should_pause = True
                    reason = f"low_battery_{self._battery_level}%"

            # CPU check
            if task.pause_on_heavy_use and self._cpu_usage > 70:
                should_pause = True
                reason = f"high_cpu_{self._cpu_usage}%"

            # Screen check
            if task.pause_on_screen_off:
                if self._screen_state in [ScreenState.OFF, ScreenState.LOCKED]:
                    should_pause = True
                    reason = f"screen_{self._screen_state.value}"

            if should_pause and not task.is_paused and task.can_pause:
                task.is_paused = True
                logger.info(f"Paused task {task.id} due to: {reason}")

    async def _manager_loop(self):
        """Main management loop"""
        while self._running:
            try:
                await self._apply_resource_limits()

                # Resume paused tasks if resources available
                for task in self._background_tasks.values():
                    if task.is_paused and task.can_resume:
                        if await self._can_start_task(task):
                            task.is_paused = False
                            logger.info(f"Resumed task {task.id}")

            except Exception as e:
                logger.error(f"Resource manager error: {e}")

            await asyncio.sleep(self.check_interval)

    # =========================================================================
    # SCREEN STATE
    # =========================================================================

    def set_screen_state(self, state: ScreenState):
        """Set current screen state (called by system integration)"""
        old_state = self._screen_state
        self._screen_state = state

        if old_state != state:
            logger.info(f"Screen state changed: {old_state.value} -> {state.value}")

            # Notify callbacks
            for callback in self._state_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")

    def register_state_callback(self, callback: Callable):
        """Register callback for state changes"""
        self._state_callbacks.append(callback)

    # =========================================================================
    # STATUS & CONTROLS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current resource manager status"""
        return {
            "running": self._running,
            "screen_state": self._screen_state.value,
            "battery_level": self._battery_level,
            "cpu_usage": self._cpu_usage,
            "memory_usage": self._memory_usage,
            "tasks": {
                "total": len(self._background_tasks),
                "running": len(
                    [t for t in self._background_tasks.values() if t.is_running]
                ),
                "paused": len(
                    [t for t in self._background_tasks.values() if t.is_paused]
                ),
            },
            "budget": {
                "cpu_limit": self._budget.cpu_percent_limit,
                "battery_min": self._budget.battery_percent_min,
            },
        }

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific task"""
        task = self._background_tasks.get(task_id)
        if not task:
            return None

        return {
            "id": task.id,
            "name": task.name,
            "intensity": task.intensity.name,
            "priority": task.priority.name,
            "is_running": task.is_running,
            "is_paused": task.is_paused,
            "progress": task.progress,
            "can_pause": task.can_pause,
        }

    def set_budget(self, cpu_limit: float = None, battery_min: int = None):
        """Adjust resource budgets"""
        if cpu_limit is not None:
            self._budget.cpu_percent_limit = cpu_limit
        if battery_min is not None:
            self._budget.battery_percent_min = battery_min

        logger.info(
            f"Updated budget: CPU={self._budget.cpu_percent_limit}%, Battery min={self._budget.battery_percent_min}%"
        )

    async def force_pause_all(self):
        """Force pause all running tasks"""
        for task in self._background_tasks.values():
            if task.is_running:
                task.is_paused = True

        logger.info("Force paused all background tasks")

    async def force_resume_all(self):
        """Force resume all paused tasks"""
        for task in self._background_tasks.values():
            if task.is_paused:
                task.is_paused = False

        logger.info("Force resumed all background tasks")
