"""
AURA v3 Graceful Shutdown
Ensures clean shutdown of all components with proper resource cleanup
Mobile-optimized: handles interrupted operations, saves state
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ShutdownPhase(Enum):
    INITIATED = "initiated"
    CANCELLING_TASKS = "cancelling_tasks"
    SAVING_STATE = "saving_state"
    CLEANING_RESOURCES = "cleaning_resources"
    FINALIZING = "finalizing"
    COMPLETE = "complete"


@dataclass
class ShutdownState:
    """Current state of shutdown process"""

    phase: ShutdownPhase = ShutdownPhase.INITIATED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    components_shutdown: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    success: bool = True


class GracefulShutdown:
    """
    Manages graceful shutdown of AURA system

    Ensures:
    - All running tasks are cancelled or completed
    - State is persisted
    - Resources are properly cleaned up
    - External connections are closed

    Mobile-optimized:
    - Handles interrupted operations
    - Saves critical state before exit
    - Minimal memory footprint during shutdown
    """

    def __init__(
        self,
        timeout: float = 30.0,
        save_state_callback: Optional[Callable] = None,
    ):
        self.timeout = timeout
        self.save_state_callback = save_state_callback

        self._running = False
        self._shutdown_state: Optional[ShutdownState] = None
        self._components: Dict[str, Callable] = {}
        self._priority_order: List[str] = []
        self._lock = asyncio.Lock()

        self._original_sigint_handler = None
        self._original_sigterm_handler = None

    def register_component(self, name: str, shutdown_func: Callable, priority: int = 0):
        """Register a component for graceful shutdown"""
        self._components[name] = shutdown_func
        self._priority_order.append((priority, name))
        self._priority_order.sort(key=lambda x: x[0])
        logger.info(f"Registered component for shutdown: {name}")

    def remove_component(self, name: str):
        """Remove a component from shutdown list"""
        if name in self._components:
            del self._components[name]
            self._priority_order = [
                (p, n) for p, n in self._priority_order if n != name
            ]
            logger.info(f"Removed component from shutdown: {name}")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        if sys.platform != "win32":
            self._original_sigint_handler = signal.getsignal(signal.SIGINT)
            self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)

            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.info("Signal handlers registered")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.shutdown())
        except RuntimeError:
            asyncio.run(self.shutdown())

    async def shutdown(self, reason: str = "unknown"):
        """Initiate graceful shutdown"""
        async with self._lock:
            if self._running:
                logger.warning("Shutdown already in progress")
                return

            self._running = True
            self._shutdown_state = ShutdownState(
                phase=ShutdownPhase.INITIATED, start_time=datetime.now()
            )

        logger.info(f"Starting graceful shutdown: {reason}")

        try:
            await self._cancel_running_tasks()
            await self._save_state()
            await self._cleanup_components()
            await self._finalize()

            self._shutdown_state.phase = ShutdownPhase.COMPLETE
            self._shutdown_state.end_time = datetime.now()
            self._shutdown_state.success = True

            duration = (
                self._shutdown_state.end_time - self._shutdown_state.start_time
            ).total_seconds()
            logger.info(f"Graceful shutdown complete in {duration:.2f}s")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self._shutdown_state.errors.append(str(e))
            self._shutdown_state.success = False

        finally:
            self._restore_signal_handlers()

    async def _cancel_running_tasks(self):
        """Cancel all running tasks"""
        self._shutdown_state.phase = ShutdownPhase.CANCELLING_TASKS

        try:
            tasks = [t for t in asyncio.all_tasks() if not t.done()]

            if tasks:
                logger.info(f"Cancelling {len(tasks)} running tasks")

                for task in tasks:
                    task.cancel()

                await asyncio.wait(
                    tasks, timeout=self.timeout / 2, return_when=asyncio.ALL_COMPLETED
                )

                cancelled = [t for t in tasks if t.cancelled()]
                logger.info(f"Cancelled {len(cancelled)} tasks")

        except Exception as e:
            logger.error(f"Error cancelling tasks: {e}")
            self._shutdown_state.errors.append(f"Task cancellation: {e}")

    async def _save_state(self):
        """Save current state before shutdown"""
        self._shutdown_state.phase = ShutdownPhase.SAVING_STATE

        try:
            if self.save_state_callback:
                logger.info("Saving system state")
                await self.save_state_callback()
                logger.info("State saved successfully")
            else:
                logger.info("No state save callback registered")

        except Exception as e:
            logger.error(f"Error saving state: {e}")
            self._shutdown_state.errors.append(f"State save: {e}")

    async def _cleanup_components(self):
        """Clean up all registered components"""
        self._shutdown_state.phase = ShutdownPhase.CLEANING_RESOURCES

        for priority, name in self._priority_order:
            if name not in self._components:
                continue

            try:
                logger.info(f"Shutting down component: {name}")
                shutdown_func = self._components[name]

                if asyncio.iscoroutinefunction(shutdown_func):
                    await asyncio.wait_for(
                        shutdown_func(), timeout=self.timeout / len(self._components)
                    )
                else:
                    shutdown_func()

                self._shutdown_state.components_shutdown.append(name)
                logger.info(f"Component shutdown complete: {name}")

            except asyncio.TimeoutError:
                logger.warning(f"Component shutdown timeout: {name}")
                self._shutdown_state.errors.append(f"Timeout: {name}")

            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")
                self._shutdown_state.errors.append(f"{name}: {e}")

    async def _finalize(self):
        """Final cleanup and logging"""
        self._shutdown_state.phase = ShutdownPhase.FINALIZING

        logger.info("Performing final cleanup")

        if self._shutdown_state.errors:
            logger.warning(
                f"Shutdown completed with {len(self._shutdown_state.errors)} errors"
            )
        else:
            logger.info("Shutdown completed successfully")

    def _restore_signal_handlers(self):
        """Restore original signal handlers"""
        if sys.platform != "win32":
            if self._original_sigint_handler:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            if self._original_sigterm_handler:
                signal.signal(signal.SIGTERM, self._original_sigterm_handler)

    def get_shutdown_state(self) -> Optional[ShutdownState]:
        """Get current shutdown state"""
        return self._shutdown_state

    @asynccontextmanager
    async def running_context(self):
        """Context manager for running operations with shutdown support"""
        self._running = True
        try:
            yield self
        finally:
            if self._running:
                await self.shutdown("context_exit")

    async def emergency_shutdown(self):
        """Immediate shutdown without cleanup (for critical errors)"""
        logger.warning("EMERGENCY SHUTDOWN - no cleanup")
        self._running = False

        tasks = [t for t in asyncio.all_tasks() if not t.done()]
        for task in tasks:
            task.cancel()

        if sys.platform != "win32":
            if self._original_sigint_handler:
                signal.signal(signal.SIGINT, signal.SIG_DFL)
            if self._original_sigterm_handler:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)

        sys.exit(1)


class ComponentShutdownHelper:
    """Helper class for components to implement shutdown"""

    def __init__(self, name: str, shutdown_manager: GracefulShutdown):
        self.name = name
        self._manager = shutdown_manager
        self._is_shutting_down = False

    async def shutdown(self):
        """Component-specific shutdown logic - override in subclass"""
        self._is_shutting_down = True
        logger.info(f"Component {self.name} shutting down")

    def register(self, priority: int = 0):
        """Register this component with shutdown manager"""
        self._manager.register_component(self.name, self.shutdown, priority)

    @property
    def is_shutting_down(self) -> bool:
        """Check if system is shutting down"""
        return self._is_shutting_down
