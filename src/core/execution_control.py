"""
AURA v3 Execution Control System
Production-grade Force Stop/Kill system for reliable AURA control

Features:
- Graceful stop: finish current atomic action, then stop
- Force stop: finish current tool, stop agent loop immediately
- Emergency kill: stop everything, save state, cleanup resources
- Atomic action tracking - know what's safe to stop
- Operation timeout tracking - detect stuck operations
- State persistence - resume after kill if needed
- Resource cleanup - release all resources on kill
"""

import asyncio
import logging
import json
import os
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
from contextlib import asynccontextmanager
import uuid
import traceback

logger = logging.getLogger(__name__)


class ExecutionState(Enum):
    """Execution states for AURA"""

    IDLE = "idle"
    THINKING = "thinking"  # LLM generating response
    ACTING = "acting"  # Tool executing
    WAITING = "waiting"  # Waiting for user input
    STOPPING = "stopping"  # Graceful stop in progress
    KILLING = "killing"  # Emergency kill in progress
    STOPPED = "stopped"  # Fully stopped
    ERROR = "error"


class StopLevel(Enum):
    """Stop levels in order of severity"""

    NONE = "none"
    STOP = "stop"  # Graceful - finish current atomic action
    FORCE_STOP = "force_stop"  # Immediate - finish current tool only
    KILL = "kill"  # Emergency - stop everything now


class AtomicActionType(Enum):
    """Types of atomic actions that can be stopped"""

    LLM_THINKING = "llm_thinking"
    TOOL_EXECUTION = "tool_execution"
    STATE_SAVE = "state_save"
    MEMORY_OPERATION = "memory_operation"
    NETWORK_REQUEST = "network_request"
    USER_WAIT = "user_wait"


@dataclass
class AtomicAction:
    """Represents an atomic action that can be in progress"""

    id: str
    action_type: AtomicActionType
    description: str
    started_at: datetime
    tool_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    can_interrupt: bool = True
    is_safe_to_stop: bool = True


@dataclass
class OperationTimeout:
    """Tracks operation timeouts"""

    operation_id: str
    operation_type: str
    started_at: datetime
    timeout_seconds: float
    is_hung: bool = False


@dataclass
class StopEvent:
    """Records a stop/kill event"""

    event_id: str
    level: StopLevel
    triggered_at: datetime
    completed_at: Optional[datetime] = None
    reason: str = ""
    state_before: ExecutionState = ExecutionState.IDLE
    atomic_actions_interrupted: List[str] = field(default_factory=list)
    cleanup_performed: List[str] = field(default_factory=list)
    error: Optional[str] = None
    success: bool = True


@dataclass
class ExecutionSnapshot:
    """Snapshot of execution state for persistence"""

    snapshot_id: str
    timestamp: datetime
    execution_state: str
    current_action: Optional[Dict]
    conversation_id: str
    thought_history_count: int
    pending_actions: List[Dict]
    loop_count: int


class ExecutionController:
    """
    Production-grade execution controller for AURA.

    Provides reliable stop/kill functionality with proper state management,
    timeout tracking, and resource cleanup.
    """

    def __init__(self, state_file: str = "data/execution_state.json"):
        self.state_file = state_file

        # Core state
        self._state = ExecutionState.IDLE
        self._stop_level = StopLevel.NONE
        self._target_state = ExecutionState.IDLE

        # Tracking
        self._current_action: Optional[AtomicAction] = None
        self._action_stack: deque = deque(maxlen=50)
        self._operation_timeouts: Dict[str, OperationTimeout] = {}
        self._loop_count = 0
        self._conversation_id = str(uuid.uuid4())[:8]

        # Callbacks
        self._on_stop_callbacks: List[Callable] = []
        self._on_kill_callbacks: List[Callable] = []
        self._cleanup_handlers: Dict[str, Callable] = {}

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Timeouts configuration
        self.default_timeout = 60.0
        self.llm_timeout = 120.0
        self.tool_timeout = 30.0
        self.state_save_timeout = 10.0

        # Infinite loop detection
        self._loop_detection_window = 10
        self._recent_actions: deque = deque(maxlen=self._loop_detection_window)
        self._max_repeat_actions = 5

        # Event history
        self._stop_events: deque = deque(maxlen=100)

        # Persistence
        self._last_snapshot: Optional[ExecutionSnapshot] = None

        logger.info("ExecutionController initialized")

    def set_loop_detector(self, loop_detector):
        """Connect loop detector for automatic infinite loop prevention"""
        self._loop_detector = loop_detector
        logger.info("Loop detector connected to execution controller")

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    @property
    def state(self) -> ExecutionState:
        """Get current execution state"""
        return self._state

    @property
    def stop_level(self) -> StopLevel:
        """Get current stop level"""
        return self._stop_level

    @property
    def conversation_id(self) -> str:
        """Get current conversation ID"""
        return self._conversation_id

    @property
    def is_stopping(self) -> bool:
        """Check if stop is in progress"""
        return self._state in (ExecutionState.STOPPING, ExecutionState.KILLING)

    @property
    def can_accept_input(self) -> bool:
        """Check if can accept new input"""
        return self._state not in (
            ExecutionState.STOPPING,
            ExecutionState.KILLING,
            ExecutionState.STOPPED,
        )

    def set_state(self, new_state: ExecutionState):
        """Set execution state (internal use)"""
        old_state = self._state
        self._state = new_state
        logger.info(f"Execution state: {old_state.value} -> {new_state.value}")

    def new_conversation(self):
        """Start a new conversation"""
        self._conversation_id = str(uuid.uuid4())[:8]
        self._loop_count = 0
        self._recent_actions.clear()
        logger.info(f"New conversation: {self._conversation_id}")

    # =========================================================================
    # ATOMIC ACTION TRACKING
    # =========================================================================

    async def start_action(
        self,
        action_type: AtomicActionType,
        description: str,
        tool_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        can_interrupt: bool = True,
        is_safe_to_stop: bool = True,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Start tracking an atomic action.
        Returns action ID.
        """
        async with self._lock:
            action_id = str(uuid.uuid4())[:8]

            action = AtomicAction(
                id=action_id,
                action_type=action_type,
                description=description,
                started_at=datetime.now(),
                tool_name=tool_name,
                params=params or {},
                can_interrupt=can_interrupt,
                is_safe_to_stop=is_safe_to_stop,
            )

            self._current_action = action
            self._action_stack.append(action)

            # Start timeout tracking
            timeout_duration = timeout or self.default_timeout
            if action_type == AtomicActionType.LLM_THINKING:
                timeout_duration = self.llm_timeout
            elif action_type == AtomicActionType.TOOL_EXECUTION:
                timeout_duration = self.tool_timeout

            self._operation_timeouts[action_id] = OperationTimeout(
                operation_id=action_id,
                operation_type=action_type.value,
                started_at=datetime.now(),
                timeout_seconds=timeout_duration,
            )

            # Track for loop detection
            self._recent_actions.append(
                {"type": action_type.value, "tool": tool_name, "time": datetime.now()}
            )

            self._loop_count += 1

            logger.info(
                f"Started action: {action_id} ({action_type.value}) - {description}"
            )
            return action_id

    async def end_action(self, action_id: str, success: bool = True):
        """End tracking an atomic action"""
        async with self._lock:
            if self._current_action and self._current_action.id == action_id:
                self._current_action = None

            if action_id in self._operation_timeouts:
                del self._operation_timeouts[action_id]

            status = "completed" if success else "failed"
            logger.info(f"Action {action_id} {status}")

    def get_current_action(self) -> Optional[AtomicAction]:
        """Get the current atomic action"""
        return self._current_action

    def is_action_safe_to_stop(self) -> bool:
        """Check if current action can be safely stopped"""
        if not self._current_action:
            return True
        return self._current_action.is_safe_to_stop

    # =========================================================================
    # STOP/KILL COMMANDS
    # =========================================================================

    async def stop(self, reason: str = "user_request") -> StopEvent:
        """
        Graceful stop - finish current atomic action, then stop.
        /stop command handler
        """
        async with self._lock:
            if self.is_stopping:
                logger.warning("Stop already in progress")
                return self._get_latest_stop_event()

            event = StopEvent(
                event_id=str(uuid.uuid4())[:8],
                level=StopLevel.STOP,
                triggered_at=datetime.now(),
                reason=reason,
                state_before=self._state,
            )

            self._stop_level = StopLevel.STOP
            self._state = ExecutionState.STOPPING
            self._target_state = ExecutionState.IDLE

            self._stop_events.append(event)

            logger.info(f"STOP initiated: {reason}")

            # If in waiting state, stop immediately
            if self._state == ExecutionState.WAITING:
                logger.info("In WAITING state, stopping immediately")
                await self._complete_stop(event)

            return event

    async def force_stop(self, reason: str = "user_request") -> StopEvent:
        """
        Force stop - finish current tool execution, then stop agent loop.
        /force-stop command handler
        """
        async with self._lock:
            if self._state == ExecutionState.KILLING:
                logger.warning("Kill already in progress")
                return self._get_latest_stop_event()

            event = StopEvent(
                event_id=str(uuid.uuid4())[:8],
                level=StopLevel.FORCE_STOP,
                triggered_at=datetime.now(),
                reason=reason,
                state_before=self._state,
            )

            self._stop_level = StopLevel.FORCE_STOP
            self._state = ExecutionState.STOPPING
            self._target_state = ExecutionState.IDLE

            self._stop_events.append(event)

            logger.info(f"FORCE STOP initiated: {reason}")

            # If in WAITING or if current action is interruptible, stop now
            if self._state == ExecutionState.WAITING or self.is_action_safe_to_stop():
                logger.info("Force stop conditions met, stopping now")
                await self._complete_stop(event)

            return event

    async def kill(self, reason: str = "user_request") -> StopEvent:
        """
        Emergency kill - stop everything immediately, save state, cleanup.
        /kill command handler
        """
        async with self._lock:
            event = StopEvent(
                event_id=str(uuid.uuid4())[:8],
                level=StopLevel.KILL,
                triggered_at=datetime.now(),
                reason=reason,
                state_before=self._state,
            )

            self._stop_level = StopLevel.KILL
            self._state = ExecutionState.KILLING
            self._target_state = ExecutionState.STOPPED

            self._stop_events.append(event)

            logger.warning(f"KILL initiated: {reason}")

            # List actions that will be interrupted
            if self._current_action:
                event.atomic_actions_interrupted.append(self._current_action.id)

            # Perform immediate cleanup
            await self._perform_cleanup(event)

            # Complete the kill
            await self._complete_kill(event)

            return event

    async def _complete_stop(self, event: StopEvent):
        """Complete a graceful or force stop"""
        logger.info(f"Completing STOP (level: {event.level.value})")

        # Wait for current action if needed (for graceful stop)
        if event.level == StopLevel.STOP and self._current_action:
            if not self._current_action.can_interrupt:
                logger.info(
                    f"Waiting for non-interruptible action: {self._current_action.id}"
                )
                # Let it complete naturally
                await asyncio.sleep(0.5)

        # For force stop, we complete current tool but don't start new ones
        # The agent loop should check stop_level and stop accordingly

        # Save state
        await self._save_state(event)

        # Update event
        event.completed_at = datetime.now()
        event.success = True

        # Reset state
        self._state = self._target_state
        self._stop_level = StopLevel.NONE

        logger.info(f"STOP completed successfully")

        # Notify callbacks
        for callback in self._on_stop_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Stop callback error: {e}")

    async def _complete_kill(self, event: StopEvent):
        """Complete an emergency kill"""
        logger.warning("Completing KILL")

        # Force reset all tracking
        self._current_action = None
        self._action_stack.clear()
        self._operation_timeouts.clear()

        # Save final state
        await self._save_state(event)

        # Update event
        event.completed_at = datetime.now()
        event.success = True

        # Set final state
        self._state = ExecutionState.STOPPED
        self._stop_level = StopLevel.NONE

        logger.warning("KILL completed")

        # Notify kill callbacks
        for callback in self._on_kill_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Kill callback error: {e}")

    # =========================================================================
    # RESOURCE CLEANUP
    # =========================================================================

    def register_cleanup_handler(self, name: str, handler: Callable):
        """Register a cleanup handler to be called on kill"""
        self._cleanup_handlers[name] = handler
        logger.info(f"Registered cleanup handler: {name}")

    async def _perform_cleanup(self, event: StopEvent):
        """Perform resource cleanup"""
        logger.info("Performing resource cleanup...")

        cleanup_results = []

        for name, handler in self._cleanup_handlers.items():
            try:
                if asyncio.iscoroutinefunction(handler):
                    await asyncio.wait_for(handler(), timeout=5.0)
                else:
                    handler()
                cleanup_results.append(name)
                logger.info(f"Cleaned up: {name}")
            except asyncio.TimeoutError:
                logger.warning(f"Cleanup timeout: {name}")
                event.cleanup_performed.append(f"{name}_timeout")
            except Exception as e:
                logger.error(f"Cleanup error for {name}: {e}")
                event.error = str(e)

        event.cleanup_performed = cleanup_results

    async def _save_state(self, event: StopEvent):
        """Save current execution state"""
        try:
            snapshot = ExecutionSnapshot(
                snapshot_id=str(uuid.uuid4())[:8],
                timestamp=datetime.now(),
                execution_state=self._state.value,
                current_action={
                    "id": self._current_action.id,
                    "type": self._current_action.action_type.value,
                    "description": self._current_action.description,
                }
                if self._current_action
                else None,
                conversation_id=self._conversation_id,
                thought_history_count=0,  # Will be set by agent
                pending_actions=[
                    {"id": a.id, "type": a.action_type.value}
                    for a in self._action_stack
                ],
                loop_count=self._loop_count,
            )

            self._last_snapshot = snapshot

            # Persist to disk
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(
                    {
                        "snapshot_id": snapshot.snapshot_id,
                        "timestamp": snapshot.timestamp.isoformat(),
                        "execution_state": snapshot.execution_state,
                        "conversation_id": snapshot.conversation_id,
                        "loop_count": snapshot.loop_count,
                        "pending_actions": snapshot.pending_actions,
                    },
                    f,
                    indent=2,
                    default=str,
                )

            logger.info(f"State saved: {snapshot.snapshot_id}")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            event.error = f"State save failed: {e}"

    # =========================================================================
    # TIMEOUT & HANG DETECTION
    # =========================================================================

    async def check_timeouts(self) -> List[str]:
        """Check for operations that have timed out"""
        hung_operations = []
        now = datetime.now()

        for op_id, op in list(self._operation_timeouts.items()):
            elapsed = (now - op.started_at).total_seconds()
            if elapsed > op.timeout_seconds:
                if not op.is_hung:
                    op.is_hung = True
                    hung_operations.append(op_id)
                    logger.warning(
                        f"Operation hung: {op_id} ({op.operation_type}) - {elapsed:.1f}s elapsed"
                    )

        return hung_operations

    async def handle_hung_operations(self) -> bool:
        """
        Handle hung operations.
        Returns True if a hang was detected and handled.
        """
        hung = await self.check_timeouts()

        if hung:
            logger.warning(f"Detected {len(hung)} hung operations")

            # If too many loops with same action, it's likely an infinite loop
            if self._detect_infinite_loop():
                logger.error("INFINITE LOOP DETECTED - initiating force kill")
                await self.kill(reason="infinite_loop_detected")
                return True

            # Otherwise, force stop current operation
            logger.info("Force stopping hung operations")
            await self.force_stop(reason="operation_timeout")
            return True

        return False

    def _detect_infinite_loop(self) -> bool:
        """Detect if we're in an infinite loop"""
        if len(self._recent_actions) < self._loop_detection_window:
            return False

        # Count recent actions
        action_counts: Dict[str, int] = {}
        for action in self._recent_actions:
            key = f"{action['type']}_{action['tool']}"
            action_counts[key] = action_counts.get(key, 0) + 1

        # Check if any action repeated too many times
        for action_key, count in action_counts.items():
            if count >= self._max_repeat_actions:
                logger.error(
                    f"Infinite loop detected: {action_key} repeated {count} times"
                )
                return True

        return False

    # =========================================================================
    # CALLBACK REGISTRATION
    # =========================================================================

    def on_stop(self, callback: Callable):
        """Register a callback to be called on stop"""
        self._on_stop_callbacks.append(callback)

    def on_kill(self, callback: Callable):
        """Register a callback to be called on kill"""
        self._on_kill_callbacks.append(callback)

    # =========================================================================
    # STATUS & INFO
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get execution controller status"""
        current = None
        if self._current_action:
            current = {
                "id": self._current_action.id,
                "type": self._current_action.action_type.value,
                "description": self._current_action.description,
                "can_interrupt": self._current_action.can_interrupt,
            }

        return {
            "state": self._state.value,
            "stop_level": self._stop_level.value,
            "conversation_id": self._conversation_id,
            "loop_count": self._loop_count,
            "current_action": current,
            "pending_actions": len(self._action_stack),
            "active_timeouts": len(self._operation_timeouts),
            "can_accept_input": self.can_accept_input,
            "is_stopping": self.is_stopping,
        }

    def _get_latest_stop_event(self) -> Optional[StopEvent]:
        """Get the most recent stop event"""
        if self._stop_events:
            return list(self._stop_events)[-1]
        return None

    def get_recent_events(self, limit: int = 10) -> List[Dict]:
        """Get recent stop/kill events"""
        events = list(self._stop_events)[-limit:]
        return [
            {
                "event_id": e.event_id,
                "level": e.level.value,
                "triggered_at": e.triggered_at.isoformat(),
                "completed_at": e.completed_at.isoformat() if e.completed_at else None,
                "reason": e.reason,
                "success": e.success,
            }
            for e in events
        ]

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    @asynccontextmanager
    async def action_context(
        self, action_type: AtomicActionType, description: str, **kwargs
    ):
        """Context manager for atomic actions"""
        action_id = await self.start_action(action_type, description, **kwargs)
        try:
            yield action_id
            await self.end_action(action_id, success=True)
        except asyncio.CancelledError:
            await self.end_action(action_id, success=False)
            raise
        except Exception as e:
            await self.end_action(action_id, success=False)
            raise

    @asynccontextmanager
    async def state_context(self, state: ExecutionState):
        """Context manager for execution state"""
        old_state = self._state
        self._state = state
        try:
            yield
        finally:
            self._state = old_state


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================


class AgentLoopIntegration:
    """Integration helper for AuraAgentLoop"""

    def __init__(self, controller: ExecutionController, agent_loop):
        self.controller = controller
        self.agent_loop = agent_loop

    async def wrap_thinking(self, coro):
        """Wrap LLM thinking with execution control"""
        async with self.controller.action_context(
            AtomicActionType.LLM_THINKING,
            "LLM generating response",
            timeout=self.controller.llm_timeout,
        ):
            # Check stop level before starting
            if self.controller.stop_level == StopLevel.KILL:
                raise asyncio.CancelledError("Kill requested")

            result = await coro

            # Check again after completion
            if self.controller.stop_level >= StopLevel.FORCE_STOP:
                raise asyncio.CancelledError("Stop requested")

            return result

    async def wrap_tool_execution(self, tool_name: str, coro):
        """Wrap tool execution with execution control"""
        async with self.controller.action_context(
            AtomicActionType.TOOL_EXECUTION,
            f"Executing tool: {tool_name}",
            tool_name=tool_name,
            timeout=self.controller.tool_timeout,
        ):
            if self.controller.stop_level == StopLevel.KILL:
                raise asyncio.CancelledError("Kill requested")

            result = await coro

            if self.controller.stop_level == StopLevel.KILL:
                raise asyncio.CancelledError("Kill requested")

            return result

    def should_continue(self) -> bool:
        """Check if agent loop should continue"""
        if self.controller.stop_level == StopLevel.NONE:
            return True

        if self.controller.stop_level == StopLevel.KILL:
            return False

        # For STOP and FORCE_STOP, continue if current action can complete
        return True


# =============================================================================
# TELEGRAM COMMAND HANDLERS
# =============================================================================


async def handle_stop_command(update, context, controller: ExecutionController):
    """Handle /stop command"""
    await update.message.reply_text(
        "🛑 *Stop Requested*\n\n"
        "Finishing current atomic action, then stopping...\n"
        "Use /force-stop for immediate stop.\n"
        "Use /kill for emergency kill.",
        parse_mode="Markdown",
    )

    event = await controller.stop(reason="telegram_stop_command")

    await update.message.reply_text(
        f"✅ Stop initiated (Event: {event.event_id})", parse_mode="Markdown"
    )


async def handle_force_stop_command(update, context, controller: ExecutionController):
    """Handle /force-stop command"""
    await update.message.reply_text(
        "⚠️ *Force Stop Requested*\n\n"
        "Finishing current tool, then stopping agent loop...\n"
        "Use /kill for emergency kill.",
        parse_mode="Markdown",
    )

    event = await controller.force_stop(reason="telegram_force_stop_command")

    await update.message.reply_text(
        f"✅ Force stop initiated (Event: {event.event_id})", parse_mode="Markdown"
    )


async def handle_kill_command(update, context, controller: ExecutionController):
    """Handle /kill command"""
    await update.message.reply_text(
        "🚨 *EMERGENCY KILL*\n\n"
        "Stopping everything immediately!\n"
        "Saving state and cleaning up...",
        parse_mode="Markdown",
    )

    event = await controller.kill(reason="telegram_kill_command")

    status = controller.get_status()
    await update.message.reply_text(
        f"🛑 Kill completed!\n\n"
        f"State: {status['state']}\n"
        f"Conversation: {status['conversation_id']}\n"
        f"Loop count: {status['loop_count']}",
        parse_mode="Markdown",
    )


async def handle_execution_status_command(
    update, context, controller: ExecutionController
):
    """Handle /execution-status command"""
    status = controller.get_status()

    current = status["current_action"]
    current_str = f"{current['type']}: {current['description']}" if current else "None"

    message = f"""
*⚙️ Execution Status*

*State:* {status["state"]}
*Stop Level:* {status["stop_level"]}
*Conversation:* {status["conversation_id"]}
*Loop Count:* {status["loop_count"]}

*Current Action:*
{current_str}

*Pending Actions:* {status["pending_actions"]}
*Active Timeouts:* {status["active_timeouts"]}
*Can Accept Input:* {status["can_accept_input"]}
"""

    await update.message.reply_text(message, parse_mode="Markdown")


# =============================================================================
# FACTORY
# =============================================================================

_controller_instance: Optional[ExecutionController] = None


def get_execution_controller() -> ExecutionController:
    """Get or create the global execution controller"""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = ExecutionController()
    return _controller_instance


__all__ = [
    "ExecutionController",
    "ExecutionState",
    "StopLevel",
    "AtomicActionType",
    "AtomicAction",
    "OperationTimeout",
    "StopEvent",
    "ExecutionSnapshot",
    "AgentLoopIntegration",
    "handle_stop_command",
    "handle_force_stop_command",
    "handle_kill_command",
    "handle_execution_status_command",
    "get_execution_controller",
]
