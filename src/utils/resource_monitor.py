"""
AURA v3 Resource Monitor
Comprehensive resource monitoring with detailed metrics tracking
Designed for mobile (Termux) constraints with minimal overhead
"""

import asyncio
import logging
import os
import psutil
import time
import sqlite3
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked"""

    RAM = "ram"
    CPU = "cpu"
    DISK = "disk"
    LLM = "llm"
    TASKS = "tasks"
    DATABASE = "database"


@dataclass
class RAMMetrics:
    """RAM usage metrics"""

    timestamp: datetime
    total_mb: float
    available_mb: float
    used_mb: float
    percent: float
    peak_mb: float = 0.0
    process_mb: float = 0.0  # AURA process memory
    component_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class CPUMetrics:
    """CPU usage metrics"""

    timestamp: datetime
    percent: float
    per_cpu: List[float] = field(default_factory=list)
    process_percent: float = 0.0
    load_avg: tuple = (0.0, 0.0, 0.0)


@dataclass
class LLMMetrics:
    """LLM inference metrics"""

    timestamp: datetime
    model_name: str
    inference_time_ms: float
    tokens_generated: int
    tokens_per_second: float
    prompt_tokens: int = 0
    memory_used_mb: float = 0.0


@dataclass
class DatabaseMetrics:
    """SQLite database metrics"""

    timestamp: datetime
    name: str
    path: str
    size_mb: float
    tables_count: int = 0
    row_count: int = 0


@dataclass
class TaskMetrics:
    """Background task metrics"""

    timestamp: datetime
    active_count: int
    paused_count: int
    completed_count: int
    task_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ResourceSnapshot:
    """Complete resource snapshot"""

    timestamp: datetime
    ram: RAMMetrics
    cpu: CPUMetrics
    databases: List[DatabaseMetrics]
    llm_inference: Optional[LLMMetrics] = None
    tasks: Optional[TaskMetrics] = None


class ResourceMonitor:
    """
    Comprehensive Resource Monitor for AURA

    Tracks:
    - RAM usage (current, peak, per-component)
    - CPU usage (overall, per-core, process-specific)
    - SQLite database sizes
    - LLM inference times
    - Active background tasks

    Features:
    - Lightweight sampling
    - Historical data with circular buffers
    - Component-level tracking
    - Export capabilities
    """

    def __init__(
        self,
        sample_interval: int = 10,
        history_size: int = 360,  # 1 hour at 10s interval
        data_dir: str = "data",
    ):
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.data_dir = Path(data_dir)

        # Process reference
        self._process = psutil.Process(os.getpid())

        # Running state
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Historical data
        self._ram_history: deque = deque(maxlen=history_size)
        self._cpu_history: deque = deque(maxlen=history_size)
        self._llm_history: deque = deque(maxlen=100)
        self._snapshot_history: deque = deque(maxlen=history_size)

        # Peak tracking
        self._peak_ram_mb: float = 0.0
        self._peak_cpu_percent: float = 0.0
        self._peak_process_ram_mb: float = 0.0

        # Component memory tracking
        self._component_memory: Dict[str, float] = {}

        # LLM inference tracking
        self._total_inferences: int = 0
        self._total_inference_time_ms: float = 0.0
        self._avg_inference_time_ms: float = 0.0

        # Task tracking reference
        self._background_manager = None

        # Callbacks
        self._alert_callbacks: List[Callable] = []

        # Alert thresholds
        self.ram_warning_threshold = 80.0  # percent
        self.ram_critical_threshold = 90.0
        self.cpu_warning_threshold = 70.0
        self.cpu_critical_threshold = 85.0

    async def start(self):
        """Start resource monitoring"""
        if self._running:
            logger.warning("Resource monitor already running")
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Resource monitor started (interval={self.sample_interval}s)")

    async def stop(self):
        """Stop resource monitoring"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitor stopped")

    def set_background_manager(self, manager):
        """Set reference to background manager for task tracking"""
        self._background_manager = manager

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                snapshot = await self._collect_snapshot()
                self._snapshot_history.append(snapshot)

                # Check alerts
                await self._check_alerts(snapshot)

            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")

            await asyncio.sleep(self.sample_interval)

    async def _collect_snapshot(self) -> ResourceSnapshot:
        """Collect a complete resource snapshot"""
        now = datetime.now()

        # Collect all metrics
        ram = await self._collect_ram_metrics()
        cpu = await self._collect_cpu_metrics()
        databases = await self._collect_database_metrics()
        tasks = await self._collect_task_metrics()

        # Store in history
        self._ram_history.append(ram)
        self._cpu_history.append(cpu)

        # Get last LLM inference if any
        llm = self._llm_history[-1] if self._llm_history else None

        return ResourceSnapshot(
            timestamp=now,
            ram=ram,
            cpu=cpu,
            databases=databases,
            llm_inference=llm,
            tasks=tasks,
        )

    async def _collect_ram_metrics(self) -> RAMMetrics:
        """Collect RAM metrics"""
        now = datetime.now()

        # System memory
        vm = psutil.virtual_memory()

        # Process memory
        try:
            process_mem = self._process.memory_info()
            process_mb = process_mem.rss / (1024 * 1024)
        except Exception:
            process_mb = 0.0

        # Update peaks
        if vm.used / (1024 * 1024) > self._peak_ram_mb:
            self._peak_ram_mb = vm.used / (1024 * 1024)
        if process_mb > self._peak_process_ram_mb:
            self._peak_process_ram_mb = process_mb

        return RAMMetrics(
            timestamp=now,
            total_mb=vm.total / (1024 * 1024),
            available_mb=vm.available / (1024 * 1024),
            used_mb=vm.used / (1024 * 1024),
            percent=vm.percent,
            peak_mb=self._peak_ram_mb,
            process_mb=process_mb,
            component_breakdown=self._component_memory.copy(),
        )

    async def _collect_cpu_metrics(self) -> CPUMetrics:
        """Collect CPU metrics"""
        now = datetime.now()

        # Overall CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        per_cpu = psutil.cpu_percent(percpu=True)

        # Process CPU
        try:
            process_cpu = self._process.cpu_percent(interval=0.1)
        except Exception:
            process_cpu = 0.0

        # Load average (Unix only)
        try:
            load_avg = os.getloadavg()
        except (AttributeError, OSError):
            load_avg = (0.0, 0.0, 0.0)

        # Update peak
        if cpu_percent > self._peak_cpu_percent:
            self._peak_cpu_percent = cpu_percent

        return CPUMetrics(
            timestamp=now,
            percent=cpu_percent,
            per_cpu=per_cpu,
            process_percent=process_cpu,
            load_avg=load_avg,
        )

    async def _collect_database_metrics(self) -> List[DatabaseMetrics]:
        """Collect SQLite database metrics"""
        now = datetime.now()
        databases = []

        # Find all .db files
        db_patterns = [
            self.data_dir / "**/*.db",
        ]

        for pattern in db_patterns:
            for db_path in self.data_dir.glob("**/*.db"):
                try:
                    size_mb = db_path.stat().st_size / (1024 * 1024)

                    # Try to get table count
                    tables_count = 0
                    row_count = 0
                    try:
                        conn = sqlite3.connect(str(db_path), timeout=1)
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )
                        tables = cursor.fetchall()
                        tables_count = len(tables)

                        # Get approximate row count (sample first table)
                        if tables:
                            cursor.execute(f"SELECT COUNT(*) FROM {tables[0][0]}")
                            row_count = cursor.fetchone()[0]
                        conn.close()
                    except Exception:
                        pass

                    databases.append(
                        DatabaseMetrics(
                            timestamp=now,
                            name=db_path.stem,
                            path=str(db_path),
                            size_mb=size_mb,
                            tables_count=tables_count,
                            row_count=row_count,
                        )
                    )
                except Exception as e:
                    logger.debug(f"Error reading database {db_path}: {e}")

        return databases

    async def _collect_task_metrics(self) -> Optional[TaskMetrics]:
        """Collect background task metrics"""
        if not self._background_manager:
            return None

        now = datetime.now()

        try:
            status = self._background_manager.get_status()
            tasks_info = status.get("tasks", {})

            # Get individual task details
            task_details = []
            if hasattr(self._background_manager, "_background_tasks"):
                for task_id, task in self._background_manager._background_tasks.items():
                    task_details.append(
                        {
                            "id": task.id,
                            "name": task.name,
                            "running": task.is_running,
                            "paused": task.is_paused,
                            "progress": task.progress,
                        }
                    )

            return TaskMetrics(
                timestamp=now,
                active_count=tasks_info.get("running", 0),
                paused_count=tasks_info.get("paused", 0),
                completed_count=tasks_info.get("total", 0)
                - tasks_info.get("running", 0)
                - tasks_info.get("paused", 0),
                task_details=task_details,
            )
        except Exception as e:
            logger.debug(f"Error collecting task metrics: {e}")
            return None

    # =========================================================================
    # LLM INFERENCE TRACKING
    # =========================================================================

    def record_llm_inference(
        self,
        model_name: str,
        inference_time_ms: float,
        tokens_generated: int,
        prompt_tokens: int = 0,
        memory_used_mb: float = 0.0,
    ):
        """Record an LLM inference for tracking"""
        now = datetime.now()

        tokens_per_second = (
            (tokens_generated / inference_time_ms * 1000)
            if inference_time_ms > 0
            else 0
        )

        metrics = LLMMetrics(
            timestamp=now,
            model_name=model_name,
            inference_time_ms=inference_time_ms,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            prompt_tokens=prompt_tokens,
            memory_used_mb=memory_used_mb,
        )

        self._llm_history.append(metrics)

        # Update aggregates
        self._total_inferences += 1
        self._total_inference_time_ms += inference_time_ms
        self._avg_inference_time_ms = (
            self._total_inference_time_ms / self._total_inferences
        )

        logger.debug(
            f"Recorded LLM inference: {inference_time_ms:.1f}ms, {tokens_generated} tokens"
        )

    def get_llm_stats(self) -> Dict[str, Any]:
        """Get LLM inference statistics"""
        if not self._llm_history:
            return {
                "total_inferences": 0,
                "avg_inference_time_ms": 0,
                "total_tokens_generated": 0,
            }

        recent = list(self._llm_history)[-10:]  # Last 10 inferences

        return {
            "total_inferences": self._total_inferences,
            "avg_inference_time_ms": self._avg_inference_time_ms,
            "recent_avg_time_ms": sum(m.inference_time_ms for m in recent)
            / len(recent),
            "total_tokens_generated": sum(
                m.tokens_generated for m in self._llm_history
            ),
            "avg_tokens_per_second": sum(m.tokens_per_second for m in recent)
            / len(recent)
            if recent
            else 0,
            "last_inference": recent[-1].timestamp.isoformat() if recent else None,
        }

    # =========================================================================
    # COMPONENT MEMORY TRACKING
    # =========================================================================

    def register_component_memory(self, component_name: str, memory_mb: float):
        """Register memory usage for a component"""
        self._component_memory[component_name] = memory_mb

    def unregister_component(self, component_name: str):
        """Unregister a component"""
        self._component_memory.pop(component_name, None)

    def get_component_breakdown(self) -> Dict[str, float]:
        """Get memory breakdown by component"""
        return self._component_memory.copy()

    # =========================================================================
    # ALERTS
    # =========================================================================

    def register_alert_callback(self, callback: Callable):
        """Register callback for resource alerts"""
        self._alert_callbacks.append(callback)

    async def _check_alerts(self, snapshot: ResourceSnapshot):
        """Check for resource alerts"""
        alerts = []

        # RAM alerts
        if snapshot.ram.percent >= self.ram_critical_threshold:
            alerts.append(
                {
                    "type": "ram_critical",
                    "message": f"RAM usage critical: {snapshot.ram.percent:.1f}%",
                    "value": snapshot.ram.percent,
                }
            )
        elif snapshot.ram.percent >= self.ram_warning_threshold:
            alerts.append(
                {
                    "type": "ram_warning",
                    "message": f"RAM usage high: {snapshot.ram.percent:.1f}%",
                    "value": snapshot.ram.percent,
                }
            )

        # CPU alerts
        if snapshot.cpu.percent >= self.cpu_critical_threshold:
            alerts.append(
                {
                    "type": "cpu_critical",
                    "message": f"CPU usage critical: {snapshot.cpu.percent:.1f}%",
                    "value": snapshot.cpu.percent,
                }
            )
        elif snapshot.cpu.percent >= self.cpu_warning_threshold:
            alerts.append(
                {
                    "type": "cpu_warning",
                    "message": f"CPU usage high: {snapshot.cpu.percent:.1f}%",
                    "value": snapshot.cpu.percent,
                }
            )

        # Notify callbacks
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

    # =========================================================================
    # DATA ACCESS
    # =========================================================================

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics"""
        if not self._snapshot_history:
            return {"status": "no_data"}

        snapshot = self._snapshot_history[-1]

        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "ram": {
                "used_mb": snapshot.ram.used_mb,
                "available_mb": snapshot.ram.available_mb,
                "percent": snapshot.ram.percent,
                "peak_mb": snapshot.ram.peak_mb,
                "process_mb": snapshot.ram.process_mb,
            },
            "cpu": {
                "percent": snapshot.cpu.percent,
                "process_percent": snapshot.cpu.process_percent,
                "peak_percent": self._peak_cpu_percent,
            },
            "databases": [
                {"name": db.name, "size_mb": db.size_mb} for db in snapshot.databases
            ],
            "tasks": {
                "active": snapshot.tasks.active_count if snapshot.tasks else 0,
                "paused": snapshot.tasks.paused_count if snapshot.tasks else 0,
            },
        }

    def get_history(
        self, metric_type: MetricType, limit: int = 60
    ) -> List[Dict[str, Any]]:
        """Get historical metrics"""
        if metric_type == MetricType.RAM:
            return [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "percent": m.percent,
                    "used_mb": m.used_mb,
                    "process_mb": m.process_mb,
                }
                for m in list(self._ram_history)[-limit:]
            ]
        elif metric_type == MetricType.CPU:
            return [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "percent": m.percent,
                    "process_percent": m.process_percent,
                }
                for m in list(self._cpu_history)[-limit:]
            ]
        elif metric_type == MetricType.LLM:
            return [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "model": m.model_name,
                    "inference_time_ms": m.inference_time_ms,
                    "tokens_per_second": m.tokens_per_second,
                }
                for m in list(self._llm_history)[-limit:]
            ]
        return []

    def get_peaks(self) -> Dict[str, float]:
        """Get peak resource usage"""
        return {
            "ram_mb": self._peak_ram_mb,
            "process_ram_mb": self._peak_process_ram_mb,
            "cpu_percent": self._peak_cpu_percent,
        }

    def reset_peaks(self):
        """Reset peak tracking"""
        self._peak_ram_mb = 0.0
        self._peak_cpu_percent = 0.0
        self._peak_process_ram_mb = 0.0
        logger.info("Peak metrics reset")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all resource metrics"""
        current = self.get_current_metrics()
        llm_stats = self.get_llm_stats()
        peaks = self.get_peaks()

        # Calculate database totals
        total_db_size = 0.0
        if self._snapshot_history:
            total_db_size = sum(
                db.size_mb for db in self._snapshot_history[-1].databases
            )

        return {
            "current": current,
            "peaks": peaks,
            "llm": llm_stats,
            "databases": {
                "total_size_mb": total_db_size,
                "count": len(self._snapshot_history[-1].databases)
                if self._snapshot_history
                else 0,
            },
            "components": self._component_memory,
            "history_size": len(self._snapshot_history),
        }


# ==============================================================================
# SINGLETON INSTANCE
# ==============================================================================

_resource_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor() -> ResourceMonitor:
    """Get or create the resource monitor instance"""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor
