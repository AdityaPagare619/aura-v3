"""
AURA v3 Parallel Task Execution Engine

Handles task decomposition, dependency analysis, parallel execution,
and result merging for mobile automation tasks.

Mobile Constraints:
- Limited background processing
- Battery considerations
- Sequential UI interactions

Example:
- "Read emails and WhatsApp" → Task1 (emails) + Task2 (WhatsApp) → PARALLEL
- "Read email THEN reply" → Task1 → Task2 → SEQUENTIAL
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks based on their characteristics"""

    READ = "read"  # Reading data (non-mutating)
    WRITE = "write"  # Writing/sending data (mutating)
    QUERY = "query"  # Querying information
    ACTION = "action"  # Performing an action
    NAVIGATION = "navigation"  # Navigating between screens


class TaskPriority(Enum):
    """Task priority levels"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING = "waiting"  # Waiting for dependencies


@dataclass
class Task:
    """A single executable task"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    task_type: TaskType = TaskType.ACTION
    priority: TaskPriority = TaskPriority.NORMAL

    # Execution details
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Callable] = None

    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Task IDs
    produces: List[str] = field(default_factory=list)  # Data keys this task produces

    # State
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    progress: float = 0.0

    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Metadata
    estimated_duration: float = 5.0  # seconds
    battery_impact: str = "medium"  # low, medium, high


@dataclass
class TaskGroup:
    """A group of tasks that can execute together"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    tasks: List[Task] = field(default_factory=list)
    execution_mode: str = "parallel"  # parallel or sequential
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None


@dataclass
class ExecutionPlan:
    """Complete execution plan with ordered task groups"""

    task_groups: List[TaskGroup] = field(default_factory=list)
    total_tasks: int = 0
    parallel_capable: int = 0
    sequential_required: int = 0
    estimated_duration: float = 0.0


@dataclass
class ProgressUpdate:
    """Progress update for a task or group"""

    task_id: str = ""
    group_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    timestamp: float = field(default_factory=time.time)


class TaskParser:
    """
    Task 1: Parse user request into individual tasks

    Analyzes natural language to decompose into executable tasks.
    """

    TASK_PATTERNS = {
        "read": [
            "read",
            "check",
            "get",
            "fetch",
            "retrieve",
            "list",
            "view",
            "show",
            "search",
            "find",
            "open",
        ],
        "write": [
            "send",
            "write",
            "reply",
            "create",
            "update",
            "delete",
            "post",
            "share",
            "submit",
            "compose",
        ],
        "action": [
            "call",
            "open",
            "launch",
            "start",
            "run",
            "execute",
            "do",
            "make",
            "perform",
            "enable",
            "disable",
        ],
    }

    APP_KEYWORDS = {
        "email": {"tool": "get_emails", "apps": ["gmail", "outlook", "email"]},
        "whatsapp": {"tool": "read_whatsapp", "apps": ["whatsapp"]},
        "sms": {"tool": "get_sms", "apps": ["messages", "sms"]},
        "contacts": {"tool": "get_contacts", "apps": ["contacts"]},
        "camera": {"tool": "open_camera", "apps": ["camera"]},
        "photos": {"tool": "get_photos", "apps": ["gallery", "photos"]},
        "calendar": {"tool": "get_calendar", "apps": ["calendar"]},
        "maps": {"tool": "open_maps", "apps": ["maps", "google maps"]},
        "browser": {"tool": "open_browser", "apps": ["chrome", "browser"]},
        "weather": {"tool": "get_weather", "apps": ["weather"]},
    }

    def __init__(self):
        self.llm_parser: Optional[Callable] = None

    def set_llm_parser(self, parser: Callable):
        """Set LLM-based parser for complex requests"""
        self.llm_parser = parser

    def parse(self, user_request: str) -> List[Task]:
        """
        Parse user request into tasks.

        Args:
            user_request: Natural language request

        Returns:
            List of Task objects
        """
        user_request = user_request.lower().strip()

        # Try LLM parser first for complex requests
        if self.llm_parser and self._is_complex_request(user_request):
            return self._parse_with_llm(user_request)

        # Rule-based parsing for simple cases
        return self._parse_rule_based(user_request)

    def _is_complex_request(self, request: str) -> bool:
        """Determine if request needs LLM parsing"""
        indicators = [
            " and then ",
            " after that ",
            " followed by ",
            " based on ",
            " depending on ",
            " if ",
            " but ",
            " however ",
            " or ",
        ]
        return any(ind in request for ind in indicators)

    def _parse_with_llm(self, request: str) -> List[Task]:
        """Use LLM to parse complex requests"""
        try:
            prompt = f"""
Parse this user request into individual tasks. Each task should have:
- name: Short task name
- type: read, write, action, or query
- tool: The tool needed (e.g., get_emails, send_whatsapp, open_app)
- parameters: Any parameters needed
- produces: What data this task produces (for dependencies)

User request: {request}

Respond with JSON array of tasks.
"""
            result = asyncio.run(self.llm_parser(prompt))
            data = json.loads(result)
            return [self._dict_to_task(t) for t in data]
        except Exception as e:
            logger.warning(f"LLM parsing failed: {e}, falling back to rule-based")
            return self._parse_rule_based(request)

    def _dict_to_task(self, d: Dict) -> Task:
        """Convert dict to Task object"""
        return Task(
            name=d.get("name", ""),
            description=d.get("description", ""),
            task_type=TaskType(d.get("type", "action")),
            tool_name=d.get("tool", ""),
            parameters=d.get("parameters", {}),
            produces=d.get("produces", []),
        )

    def _parse_rule_based(self, request: str) -> List[Task]:
        """Rule-based parsing for simple requests"""
        tasks = []

        # Split by "and" for parallel tasks
        parts = [p.strip() for p in request.replace(" and ", "|").split("|")]

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            task = self._create_task_from_phrase(part)
            task.id = f"task_{i}_{task.id}"
            tasks.append(task)

        return tasks

    def _create_task_from_phrase(self, phrase: str) -> Task:
        """Create a task from a phrase"""
        phrase = phrase.lower().strip()

        # Determine task type
        task_type = TaskType.ACTION
        for ttype, keywords in self.TASK_PATTERNS.items():
            if any(kw in phrase for kw in keywords):
                task_type = TaskType(ttype)
                break

        # Determine target app/tool
        tool_name = "open_app"
        parameters = {}

        for app, config in self.APP_KEYWORDS.items():
            if app in phrase:
                tool_name = config["tool"]
                if "apps" in config:
                    parameters["app_name"] = config["apps"][0]
                break

        # Handle specific actions
        if "send" in phrase and "message" in phrase:
            tool_name = "send_sms"
            if "whatsapp" in phrase:
                tool_name = "send_whatsapp"

        if "call" in phrase:
            tool_name = "make_call"

        return Task(
            name=phrase[:50],
            description=phrase,
            task_type=task_type,
            tool_name=tool_name,
            parameters=parameters,
        )


class DependencyAnalyzer:
    """
    Task 2: Analyze dependencies between tasks

    Determines which tasks can run in parallel and which must be sequential.
    """

    CONFLICTING_TOOLS = {
        "open_app": ["open_app", "start_app"],
        "send_sms": ["send_sms", "send_whatsapp"],
        "make_call": ["make_call"],
        "send_whatsapp": ["send_whatsapp", "send_sms"],
    }

    # Same app operations should be sequential (UI state)
    SAME_APP_SEQUENTIAL = True

    def analyze(self, tasks: List[Task]) -> Dict[str, List[str]]:
        """
        Analyze dependencies between tasks.

        Args:
            tasks: List of tasks to analyze

        Returns:
            Dict mapping task_id to list of task_ids it depends on
        """
        dependencies = {task.id: set() for task in tasks}

        # Add explicit dependencies from task.depends_on
        for task in tasks:
            for dep_id in task.depends_on:
                dependencies[task.id].add(dep_id)

        # Analyze implicit dependencies
        for i, task in enumerate(tasks):
            for j, other in enumerate(tasks[:i]):
                # Check if task depends on other's output
                if self._has_data_dependency(task, other):
                    dependencies[task.id].add(other.id)

                # Check for conflicts
                if self._has_conflict(task, other):
                    dependencies[task.id].add(other.id)

                # Same app = sequential (mobile UI constraint)
                if self.SAME_APP_SEQUENTIAL and self._same_app(task, other):
                    dependencies[task.id].add(other.id)

        return {k: list(v) for k, v in dependencies.items()}

    def _has_data_dependency(self, task: Task, producer: Task) -> bool:
        """Check if task needs data from producer"""
        # Task needs data that producer creates
        for prod in task.produces:
            if prod in [p for p in producer.produces]:
                return True

        # Check for read-after-write patterns
        if task.task_type == TaskType.READ and producer.task_type == TaskType.WRITE:
            # E.g., "send message then check if sent"
            return True

        return False

    def _has_conflict(self, task: Task, other: Task) -> bool:
        """Check if tasks conflict (can't run simultaneously)"""
        if task.tool_name == other.tool_name:
            return True

        for tool, conflicts in self.CONFLICTING_TOOLS.items():
            if task.tool_name in conflicts and other.tool_name in conflicts:
                return True

        return False

    def _same_app(self, task: Task, other: Task) -> bool:
        """Check if tasks target the same app"""
        app1 = task.parameters.get("app_name", "")
        app2 = other.parameters.get("app_name", "")

        if app1 and app2 and app1 == app2:
            return True

        # Infer from tool names
        tool1_apps = ["whatsapp", "gmail", "email", "messages", "chrome"]
        tool2_apps = ["whatsapp", "gmail", "email", "messages", "chrome"]

        for app in tool1_apps:
            if app in task.tool_name.lower():
                for app2 in tool2_apps:
                    if app2 in other.tool_name.lower():
                        return app == app2

        return False

    def build_execution_plan(self, tasks: List[Task]) -> ExecutionPlan:
        """
        Build an execution plan with task groups.

        Groups independent tasks for parallel execution.
        """
        dependencies = self.analyze(tasks)

        # Topological sort into levels
        levels = self._topological_sort_levels(tasks, dependencies)

        # Create task groups
        task_groups = []
        parallel_count = 0
        sequential_count = 0

        for level_tasks in levels:
            if len(level_tasks) == 1:
                # Single task - execute sequentially
                group = TaskGroup(
                    name=f"Sequential: {level_tasks[0].name}",
                    tasks=level_tasks,
                    execution_mode="sequential",
                )
                sequential_count += 1
            else:
                # Multiple tasks - check if they can run parallel
                can_parallel = all(not dependencies[t.id] for t in level_tasks) and all(
                    not self._has_conflict(t1, t2)
                    for i, t1 in enumerate(level_tasks)
                    for t2 in level_tasks[i + 1 :]
                )

                if can_parallel:
                    group = TaskGroup(
                        name=f"Parallel: {', '.join(t.name for t in level_tasks)}",
                        tasks=level_tasks,
                        execution_mode="parallel",
                    )
                    parallel_count += len(level_tasks)
                else:
                    # Conflict - run sequentially
                    group = TaskGroup(
                        name=f"Conflict: {level_tasks[0].name}",
                        tasks=level_tasks,
                        execution_mode="sequential",
                    )
                    sequential_count += len(level_tasks)

            task_groups.append(group)

        # Calculate estimated duration
        est_duration = sum(t.estimated_duration for t in tasks) / max(
            1, parallel_count
        ) + sum(t.estimated_duration for t in tasks if t.status == TaskStatus.PENDING)

        return ExecutionPlan(
            task_groups=task_groups,
            total_tasks=len(tasks),
            parallel_capable=parallel_count,
            sequential_required=sequential_count,
            estimated_duration=est_duration,
        )

    def _topological_sort_levels(
        self, tasks: List[Task], dependencies: Dict[str, List[str]]
    ) -> List[List[Task]]:
        """Sort tasks into levels for parallel execution"""
        task_map = {t.id: t for t in tasks}
        deps = {k: set(v) for k, v in dependencies.items()}

        levels = []
        remaining = set(task_map.keys())

        while remaining:
            # Find tasks with no remaining dependencies
            ready = [tid for tid in remaining if not deps[tid]]

            if not ready:
                # Circular dependency - break by taking any remaining
                ready = [remaining.pop()]
            else:
                for r in ready:
                    remaining.discard(r)

            # Remove these from other dependencies
            for tid in ready:
                for d in deps:
                    deps[d].discard(tid)

            levels.append([task_map[tid] for tid in ready])

        return levels


class ParallelExecutor:
    """
    Task 3: Execute tasks in parallel where possible

    Handles both parallel and sequential execution with proper
    resource management for mobile constraints.
    """

    MAX_PARALLEL_TASKS = 3  # Mobile battery constraint
    TASK_TIMEOUT = 30.0

    def __init__(self):
        self.progress_callback: Optional[Callable[[ProgressUpdate], None]] = None

    def set_progress_callback(self, callback: Callable[[ProgressUpdate], None]):
        """Set callback for progress updates"""
        self.progress_callback = callback

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        tool_executor: Callable[[str, Dict], Dict],
    ) -> Dict[str, Any]:
        """
        Execute the complete task plan.

        Args:
            plan: ExecutionPlan with task groups
            tool_executor: Async function to execute a tool

        Returns:
            Dict with results and status
        """
        results = {}
        all_tasks = []

        for group in plan.task_groups:
            # Update group status
            group.status = TaskStatus.RUNNING

            if group.execution_mode == "parallel":
                group_result = await self._execute_parallel(group.tasks, tool_executor)
            else:
                group_result = await self._execute_sequential(
                    group.tasks, tool_executor
                )

            group.status = TaskStatus.COMPLETED
            group.result = group_result

            # Collect results
            for task in group.tasks:
                results[task.id] = {
                    "name": task.name,
                    "status": task.status.value,
                    "result": task.result,
                    "error": task.error,
                    "duration": (
                        task.completed_at - task.started_at
                        if task.completed_at and task.started_at
                        else 0
                    ),
                }
                all_tasks.append(task)

        return {
            "success": all(t.status == TaskStatus.COMPLETED for t in all_tasks),
            "results": results,
            "total_tasks": len(all_tasks),
            "completed": sum(1 for t in all_tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in all_tasks if t.status == TaskStatus.FAILED),
        }

    async def _execute_parallel(
        self,
        tasks: List[Task],
        tool_executor: Callable[[str, Dict], Dict],
    ) -> List[Dict]:
        """Execute tasks in parallel (limited by MAX_PARALLEL_TASKS)"""
        # Limit concurrent tasks for mobile
        limited_tasks = tasks[: self.MAX_PARALLEL_TASKS]

        self._emit_progress(
            group_id=tasks[0].id if tasks else "",
            status=TaskStatus.RUNNING,
            progress=0.0,
            message=f"Running {len(limited_tasks)} tasks in parallel",
        )

        # Execute in parallel
        coroutines = [self._execute_task(task, tool_executor) for task in limited_tasks]

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Handle any remaining tasks sequentially
        remaining = tasks[self.MAX_PARALLEL_TASKS :]
        if remaining:
            for task in remaining:
                result = await self._execute_task(task, tool_executor)
                results.append(result)

        self._emit_progress(
            group_id=tasks[0].id if tasks else "",
            status=TaskStatus.COMPLETED,
            progress=1.0,
            message="Parallel execution complete",
        )

        return results

    async def _execute_sequential(
        self,
        tasks: List[Task],
        tool_executor: Callable[[str, Dict], Dict],
    ) -> List[Dict]:
        """Execute tasks sequentially"""
        results = []

        for i, task in enumerate(tasks):
            self._emit_progress(
                group_id=task.id,
                status=TaskStatus.RUNNING,
                progress=i / len(tasks) if tasks else 0,
                message=f"Executing: {task.name}",
            )

            result = await self._execute_task(task, tool_executor)
            results.append(result)

        self._emit_progress(
            group_id=tasks[0].id if tasks else "",
            status=TaskStatus.COMPLETED,
            progress=1.0,
            message="Sequential execution complete",
        )

        return results

    async def _execute_task(
        self,
        task: Task,
        tool_executor: Callable[[str, Dict], Dict],
    ) -> Dict:
        """Execute a single task"""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        self._emit_progress(
            task_id=task.id,
            status=TaskStatus.RUNNING,
            progress=0.1,
            message=f"Starting: {task.name}",
        )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                tool_executor(task.tool_name, task.parameters),
                timeout=self.TASK_TIMEOUT,
            )

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0

            self._emit_progress(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                progress=1.0,
                message=f"Completed: {task.name}",
            )

        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = "Task timed out"
            task.progress = 0.0

            self._emit_progress(
                task_id=task.id,
                status=TaskStatus.FAILED,
                progress=0.0,
                message=f"Timeout: {task.name}",
            )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.progress = 0.0

            self._emit_progress(
                task_id=task.id,
                status=TaskStatus.FAILED,
                progress=0.0,
                message=f"Failed: {task.name} - {str(e)}",
            )

        task.completed_at = time.time()

        return {
            "task_id": task.id,
            "status": task.status.value,
            "result": task.result,
            "error": task.error,
        }

    def _emit_progress(
        self,
        task_id: str = "",
        group_id: str = "",
        status: TaskStatus = TaskStatus.PENDING,
        progress: float = 0.0,
        message: str = "",
    ):
        """Emit progress update if callback is set"""
        if self.progress_callback:
            update = ProgressUpdate(
                task_id=task_id,
                group_id=group_id,
                status=status,
                progress=progress,
                message=message,
            )
            self.progress_callback(update)


class ResultMerger:
    """
    Task 4: Merge results from parallel tasks

    Combines results from multiple tasks into a cohesive response.
    """

    def merge(self, execution_result: Dict, original_request: str) -> Dict:
        """
        Merge results from task execution.

        Args:
            execution_result: Result from ParallelExecutor
            original_request: Original user request

        Returns:
            Merged result with message and details
        """
        results = execution_result.get("results", {})
        total = execution_result.get("total_tasks", 0)
        completed = execution_result.get("completed", 0)
        failed = execution_result.get("failed", 0)

        # Generate summary message
        if failed == 0:
            message = self._generate_success_message(results, original_request)
        elif completed == 0:
            message = f"Unable to complete any of the {total} tasks."
        else:
            message = self._generate_partial_message(completed, failed, results)

        return {
            "success": execution_result.get("success", False),
            "message": message,
            "results": results,
            "summary": {
                "total": total,
                "completed": completed,
                "failed": failed,
            },
            "details": self._extract_details(results),
        }

    def _generate_success_message(self, results: Dict, request: str) -> str:
        """Generate message for successful execution"""
        task_names = [r.get("name", "task") for r in results.values()]

        if len(task_names) == 1:
            return f"Completed: {task_names[0]}"

        return f"Completed {len(task_names)} tasks: {', '.join(task_names[:3])}"

    def _generate_partial_message(
        self, completed: int, failed: int, results: Dict
    ) -> str:
        """Generate message for partial success"""
        failed_tasks = [
            name
            for tid, r in results.items()
            if r.get("status") == "failed"
            for name in [r.get("name", "task")]
        ]

        msg = f"Completed {completed} tasks, {failed} failed."
        if failed_tasks:
            msg += f" Failed: {', '.join(failed_tasks[:2])}"

        return msg

    def _extract_details(self, results: Dict) -> List[Dict]:
        """Extract relevant details from results"""
        details = []

        for tid, result in results.items():
            if result.get("status") == "completed":
                detail = {
                    "task": result.get("name", tid),
                    "data": self._extract_relevant_data(result.get("result", {})),
                }
                details.append(detail)

        return details

    def _extract_relevant_data(self, result: Any) -> Any:
        """Extract relevant data from tool result"""
        if isinstance(result, dict):
            # Filter to important fields
            important_keys = ["messages", "emails", "contacts", "data", "content"]
            filtered = {
                k: v
                for k, v in result.items()
                if any(ik in k.lower() for ik in important_keys)
            }
            return filtered if filtered else result

        return result


class ProgressTracker:
    """
    Task 5: Track progress for long-running tasks

    Provides real-time progress updates for task execution.
    """

    def __init__(self):
        self.updates: List[ProgressUpdate] = []
        self.current_task: Optional[str] = None
        self.listeners: List[Callable[[ProgressUpdate], None]] = []

    def add_listener(self, listener: Callable[[ProgressUpdate], None]):
        """Add a progress listener"""
        self.listeners.append(listener)

    def remove_listener(self, listener: Callable[[ProgressUpdate], None]):
        """Remove a progress listener"""
        self.listeners.remove(listener)

    def track(self, update: ProgressUpdate):
        """Track a progress update"""
        self.updates.append(update)
        self.current_task = update.task_id or update.group_id

        # Notify listeners
        for listener in self.listeners:
            try:
                listener(update)
            except Exception as e:
                logger.warning(f"Progress listener error: {e}")

    def get_status(self) -> Dict:
        """Get current execution status"""
        if not self.updates:
            return {"status": "idle", "progress": 0.0}

        latest = self.updates[-1]

        # Calculate overall progress
        total_updates = len(self.updates)
        completed_updates = sum(
            1
            for u in self.updates
            if u.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
        )
        overall_progress = completed_updates / max(1, total_updates)

        return {
            "status": latest.status.value,
            "current_task": self.current_task,
            "progress": overall_progress,
            "message": latest.message,
            "timestamp": latest.timestamp,
        }

    def reset(self):
        """Reset tracker state"""
        self.updates.clear()
        self.current_task = None


class TaskEngine:
    """
    Main Task Engine - orchestrates all components

    Coordinates task parsing, dependency analysis, execution,
    and result merging.
    """

    def __init__(self):
        self.parser = TaskParser()
        self.analyzer = DependencyAnalyzer()
        self.executor = ParallelExecutor()
        self.merger = ResultMerger()
        self.tracker = ProgressTracker()

        # Connect progress tracking
        self.executor.set_progress_callback(self.tracker.track)

    def set_llm_parser(self, parser: Callable):
        """Set LLM-based parser for complex requests"""
        self.parser.set_llm_parser(parser)

    async def execute(
        self,
        user_request: str,
        tool_executor: Callable[[str, Dict], Dict],
    ) -> Dict:
        """
        Execute user request as parallel tasks.

        Args:
            user_request: Natural language request
            tool_executor: Async function to execute tools

        Returns:
            Execution result with merged message
        """
        logger.info(f"TaskEngine: Processing '{user_request}'")

        # Step 1: Parse into tasks
        self.tracker.reset()
        tasks = self.parser.parse(user_request)

        if not tasks:
            return {
                "success": False,
                "message": "Could not parse request into tasks",
                "results": {},
            }

        logger.info(f"TaskEngine: Parsed {len(tasks)} tasks")

        # Step 2: Analyze dependencies
        plan = self.analyzer.build_execution_plan(tasks)

        logger.info(
            f"TaskEngine: Plan - {plan.parallel_capable} parallel, "
            f"{plan.sequential_required} sequential"
        )

        # Step 3: Execute plan
        execution_result = await self.executor.execute_plan(plan, tool_executor)

        # Step 4: Merge results
        merged = self.merger.merge(execution_result, user_request)

        return merged

    def get_progress(self) -> Dict:
        """Get current execution progress"""
        return self.tracker.get_status()

    def get_task_info(self, request: str) -> Dict:
        """
        Preview what tasks would be executed (without executing).

        Useful for confirmation dialogs.
        """
        tasks = self.parser.parse(request)
        plan = self.analyzer.build_execution_plan(tasks)

        return {
            "task_count": len(tasks),
            "can_parallel": plan.parallel_capable > 1,
            "estimated_duration": plan.estimated_duration,
            "tasks": [
                {
                    "name": t.name,
                    "type": t.task_type.value,
                    "tool": t.tool_name,
                }
                for t in tasks
            ],
            "groups": [
                {
                    "mode": g.execution_mode,
                    "tasks": [t.name for t in g.tasks],
                }
                for g in plan.task_groups
            ],
        }


class TaskEngineIntegration:
    """
    Integration layer for TaskEngine with Agent Loop

    Provides seamless integration with the existing ReAct agent.
    """

    def __init__(self, task_engine: TaskEngine):
        self.task_engine = task_engine

    async def handle_request(
        self,
        user_message: str,
        agent,
    ) -> AgentResponse:
        """
        Handle request that may benefit from parallel execution.

        Returns AgentResponse with results.
        """
        # Check if request is multi-task
        task_info = self.task_engine.get_task_info(user_message)

        if task_info["task_count"] <= 1:
            # Single task - let agent handle normally
            return None  # Signal to use normal agent flow

        # Execute as parallel tasks
        async def tool_executor(tool_name: str, params: Dict) -> Dict:
            tool = agent.tools.get_tool(tool_name)
            if not tool:
                return {"success": False, "error": f"Tool {tool_name} not found"}

            try:
                return await tool(**params)
            except Exception as e:
                return {"success": False, "error": str(e)}

        result = await self.task_engine.execute(user_message, tool_executor)

        # Convert to AgentResponse
        return AgentResponse(
            message=result.get("message", ""),
            state=AgentState.COMPLETED if result.get("success") else AgentState.ERROR,
            context={
                "task_results": result.get("results", {}),
                "task_summary": result.get("summary", {}),
            },
        )


# For backward compatibility - AgentResponse import
class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_USER = "waiting_user"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentResponse:
    message: str
    state: AgentState = AgentState.IDLE
    context: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "TaskEngine",
    "TaskParser",
    "DependencyAnalyzer",
    "ParallelExecutor",
    "ResultMerger",
    "ProgressTracker",
    "TaskEngineIntegration",
    "Task",
    "TaskGroup",
    "ExecutionPlan",
    "TaskType",
    "TaskPriority",
    "TaskStatus",
    "ProgressUpdate",
    "AgentResponse",
    "AgentState",
]
