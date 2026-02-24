"""
AURA v3 Task Context Preservation System
Handles interruptions gracefully - preserves task state when user is busy
CRITICAL: This is what prevents AURA from "losing" work when interrupted
"""

import asyncio
import logging
import json
import pickle
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import deque
import uuid
import copy

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """State of a tracked task"""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_USER = "waiting_user"
    WAITING_RESOURCE = "waiting_resource"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


class InterruptionType(Enum):
    """Types of interruptions"""

    USER_PRIORITY = "user_priority"  # User gave higher priority task
    MEETING = "meeting"  # User in meeting
    PHONE_IN_USE = "phone_in_use"  # User using phone
    BATTERY_LOW = "battery_low"  # Low battery
    CALL_INCOMING = "call_incoming"  # Incoming call
    APP_SWITCH = "app_switch"  # User switched apps
    MANUAL_PAUSE = "manual_pause"  # User paused
    BACKGROUND = "background"  # AURA moved to background


@dataclass
class TaskCheckpoint:
    """A checkpoint in task execution"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    step: str = ""
    step_number: int = 0
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    data_preserved: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackedTask:
    """A task with full context preservation"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    task_type: str = ""
    priority: int = 5

    # State management
    state: TaskState = TaskState.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    resumed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Checkpoints
    checkpoints: List[TaskCheckpoint] = field(default_factory=list)
    current_checkpoint_id: Optional[str] = None

    # Execution context
    context: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Interruption tracking
    interruptions: List[Dict[str, Any]] = field(default_factory=list)
    interruption_count: int = 0

    # Resume info
    resume_point: Optional[str] = None
    resume_data: Dict[str, Any] = field(default_factory=dict)

    # Progress
    progress: float = 0.0
    estimated_completion: Optional[datetime] = None

    # Error handling
    error: Optional[str] = None
    retry_count: int = 0


class TaskContextPreservation:
    """
    Task Context Preservation System

    CRITICAL FUNCTIONALITY:
    - When AURA is working on Task A and Task B interrupts:
      1. Save complete state of Task A
      2. Create checkpoint
      3. Store all intermediate results
      4. Store decision points
      5. Save what was about to happen next
      6. Resume Task A exactly where left off

    - When user returns:
      1. Detect user is available
      2. Offer to resume or complete pending tasks
      3. Show what was in progress

    Key features:
    - Full state serialization
    - Checkpoint management
    - Interruption tracking
    - Resume point tracking
    - Context data preservation
    """

    def __init__(self, storage_path: str = "data/task_context"):
        self.storage_path = storage_path

        # Active tasks
        self._tasks: Dict[str, TrackedTask] = {}

        # Task queue (priority order)
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

        # Currently running task
        self._current_task_id: Optional[str] = None

        # Interruption handlers
        self._interruption_handlers: Dict[InterruptionType, Callable] = {}

        # Task history
        self._task_history: deque = deque(maxlen=200)

        # Settings
        self.auto_checkpoint_interval = 60  # seconds
        self.max_checkpoints_per_task = 10
        self.context_preservation_timeout = 3600  # 1 hour

    # =========================================================================
    # TASK CREATION & TRACKING
    # =========================================================================

    async def create_task(
        self,
        name: str,
        task_type: str,
        description: str = "",
        priority: int = 5,
        context: Dict[str, Any] = None,
        depends_on: List[str] = None,
    ) -> TrackedTask:
        """Create a new tracked task with full context"""

        task = TrackedTask(
            name=name,
            task_type=task_type,
            description=description,
            priority=priority,
            context=context or {},
            depends_on=depends_on or [],
        )

        self._tasks[task.id] = task

        # Create initial checkpoint
        await self._create_checkpoint(task, "task_created")

        logger.info(f"Created tracked task: {task.id} - {name}")

        return task

    async def start_task(self, task_id: str):
        """Start executing a task"""
        if task_id not in self._tasks:
            logger.error(f"Task not found: {task_id}")
            return

        task = self._tasks[task_id]
        task.state = TaskState.RUNNING
        task.started_at = datetime.now()
        self._current_task_id = task_id

        await self._create_checkpoint(task, "task_started")

        logger.info(f"Started task: {task_id}")

    async def pause_task(
        self, task_id: str, reason: str = "manual", save_state: Dict[str, Any] = None
    ):
        """Pause a task and save all context"""
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task.state = TaskState.PAUSED
        task.paused_at = datetime.now()

        # Save current state
        if save_state:
            task.context.update(save_state)

        # Create checkpoint
        await self._create_checkpoint(task, f"paused_{reason}")

        # If this was current task, clear it
        if self._current_task_id == task_id:
            self._current_task_id = None

        logger.info(f"Paused task: {task_id} - {reason}")

    async def resume_task(self, task_id: str) -> bool:
        """Resume a paused task from exact checkpoint"""
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]

        if task.state not in [TaskState.PAUSED, TaskState.INTERRUPTED]:
            return False

        # Restore context from checkpoint
        checkpoint = self._get_latest_checkpoint(task)
        if checkpoint:
            task.context = checkpoint.state_snapshot.get("context", {})
            task.intermediate_results = checkpoint.state_snapshot.get(
                "intermediate_results", {}
            )

        task.state = TaskState.RUNNING
        task.resumed_at = datetime.now()
        self._current_task_id = task_id

        await self._create_checkpoint(task, "task_resumed")

        logger.info(f"Resumed task: {task_id}")
        return True

    async def complete_task(
        self, task_id: str, result: Any = None, error: Optional[str] = None
    ):
        """Mark task as completed"""
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]

        if error:
            task.state = TaskState.FAILED
            task.error = error
        else:
            task.state = TaskState.COMPLETED
            if result:
                task.intermediate_results["final_result"] = result

        task.completed_at = datetime.now()
        task.progress = 100.0

        if self._current_task_id == task_id:
            self._current_task_id = None

        # Keep in history
        self._task_history.append(task)

        logger.info(f"Completed task: {task_id}")

    # =========================================================================
    # INTERRUPTION HANDLING
    # =========================================================================

    async def handle_interruption(
        self, interruption_type: InterruptionType, metadata: Dict[str, Any] = None
    ):
        """
        Handle an interruption - save state before handling
        This is CRITICAL for context preservation
        """

        # If there's a current task, pause it
        if self._current_task_id:
            current_task = self._tasks.get(self._current_task_id)
            if current_task and current_task.state == TaskState.RUNNING:
                # Create interruption checkpoint
                checkpoint = await self._create_checkpoint(
                    current_task, f"interrupted_{interruption_type.value}"
                )

                # Store interruption info
                current_task.interruptions.append(
                    {
                        "type": interruption_type.value,
                        "at_checkpoint": checkpoint.id,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata or {},
                    }
                )
                current_task.interruption_count += 1
                current_task.state = TaskState.INTERRUPTED

                logger.info(
                    f"Task {current_task.id} interrupted by {interruption_type.value}"
                )

        # Call specific handler if exists
        if interruption_type in self._interruption_handlers:
            handler = self._interruption_handlers[interruption_type]
            if asyncio.iscoroutinefunction(handler):
                await handler(metadata or {})
            else:
                handler(metadata or {})

    def register_interruption_handler(
        self, interruption_type: InterruptionType, handler: Callable
    ):
        """Register handler for specific interruption type"""
        self._interruption_handlers[interruption_type] = handler

    # =========================================================================
    # CHECKPOINT MANAGEMENT
    # =========================================================================

    async def _create_checkpoint(
        self, task: TrackedTask, step: str, data: Dict[str, Any] = None
    ) -> TaskCheckpoint:
        """Create a checkpoint of current task state"""

        checkpoint = TaskCheckpoint(
            step=step,
            step_number=len(task.checkpoints),
            state_snapshot={
                "context": copy.deepcopy(task.context),
                "intermediate_results": copy.deepcopy(task.intermediate_results),
                "progress": task.progress,
            },
            data_preserved=data or {},
        )

        task.checkpoints.append(checkpoint)
        task.current_checkpoint_id = checkpoint.id

        # Limit checkpoints
        if len(task.checkpoints) > self.max_checkpoints_per_task:
            task.checkpoints.pop(0)

        # Persist checkpoint
        await self._persist_checkpoint(task.id, checkpoint)

        return checkpoint

    def _get_latest_checkpoint(self, task: TrackedTask) -> Optional[TaskCheckpoint]:
        """Get the most recent checkpoint"""
        if task.checkpoints:
            return task.checkpoints[-1]
        return None

    # =========================================================================
    # PROGRESS TRACKING
    # =========================================================================

    async def update_progress(
        self, task_id: str, progress: float, intermediate_result: Dict[str, Any] = None
    ):
        """Update task progress and save checkpoint"""
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task.progress = progress

        if intermediate_result:
            task.intermediate_results.update(intermediate_result)

        # Auto checkpoint at intervals
        if (
            len(task.checkpoints) == 0
            or (datetime.now() - task.checkpoints[-1].created_at).total_seconds()
            > self.auto_checkpoint_interval
        ):
            await self._create_checkpoint(task, f"progress_{int(progress)}%")

    # =========================================================================
    # TASK QUEUE MANAGEMENT
    # =========================================================================

    async def queue_task(self, task_id: str, priority: int = None):
        """Add task to execution queue"""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            if priority is not None:
                task.priority = priority
            await self._task_queue.put((task.priority, task_id))

    async def get_next_task(self) -> Optional[TrackedTask]:
        """Get next task from queue"""
        try:
            priority, task_id = await asyncio.wait_for(
                self._task_queue.get(), timeout=1.0
            )
            return self._tasks.get(task_id)
        except asyncio.TimeoutError:
            return None

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    async def get_task_status(self, task_id: str) -> Optional[TrackedTask]:
        """Get full task status"""
        return self._tasks.get(task_id)

    async def get_active_tasks(
        self, state: Optional[TaskState] = None
    ) -> List[TrackedTask]:
        """Get all active (non-completed) tasks"""
        tasks = [
            t
            for t in self._tasks.values()
            if t.state not in [TaskState.COMPLETED, TaskState.FAILED]
        ]

        if state:
            tasks = [t for t in tasks if t.state == state]

        return sorted(tasks, key=lambda t: (t.priority, t.created_at), reverse=True)

    async def get_interrupted_tasks(self) -> List[TrackedTask]:
        """Get tasks that were interrupted"""
        return [t for t in self._tasks.values() if t.state == TaskState.INTERRUPTED]

    async def can_resume_task(self, task_id: str) -> bool:
        """Check if a task can be resumed"""
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        return task.state in [
            TaskState.PAUSED,
            TaskState.INTERRUPTED,
            TaskState.WAITING_RESOURCE,
        ]

    async def get_task_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked tasks"""
        return {
            "total": len(self._tasks),
            "running": len(
                [t for t in self._tasks.values() if t.state == TaskState.RUNNING]
            ),
            "paused": len(
                [t for t in self._tasks.values() if t.state == TaskState.PAUSED]
            ),
            "interrupted": len(
                [t for t in self._tasks.values() if t.state == TaskState.INTERRUPTED]
            ),
            "completed": len(
                [t for t in self._tasks.values() if t.state == TaskState.COMPLETED]
            ),
            "current_task": self._current_task_id,
            "pending_resume": len(await self.get_interrupted_tasks()),
        }

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    async def _persist_checkpoint(self, task_id: str, checkpoint: TaskCheckpoint):
        """Persist checkpoint to disk"""
        try:
            import os

            os.makedirs(self.storage_path, exist_ok=True)

            filename = f"{self.storage_path}/{task_id}_{checkpoint.id}.cpkt"
            with open(filename, "wb") as f:
                pickle.dump(checkpoint, f)
        except Exception as e:
            logger.error(f"Failed to persist checkpoint: {e}")

    async def load_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load full task state from disk"""
        try:
            task = self._tasks.get(task_id)
            if not task:
                return None

            # Load from latest checkpoint
            checkpoint = self._get_latest_checkpoint(task)
            if checkpoint:
                return checkpoint.state_snapshot
        except Exception as e:
            logger.error(f"Failed to load task state: {e}")

        return None
