"""
AURA v3 Internal Agent Coordinator
Manages multiple specialized agents working in parallel
Inspired by OpenClaw but adapted for mobile/offline operation
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of specialized agents in AURA"""

    ORCHESTRATOR = "orchestrator"
    ANALYZER = "analyzer"
    EXECUTOR = "executor"
    MEMORY = "memory"
    VOICE = "voice"
    ANDROID = "android"
    SECURITY = "security"
    MONITOR = "monitor"
    LEARNING = "learning"


class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"


@dataclass
class AgentTask:
    """A task assigned to an agent"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: str = ""
    description: str = ""
    priority: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None


@dataclass
class AgentInfo:
    """Information about a registered agent"""

    id: str
    name: str
    agent_type: AgentType
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[AgentTask] = None
    completed_tasks: int = 0
    failed_tasks: int = 0
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent:
    """
    Base class for specialized agents in AURA

    Each agent has:
    - Specific capabilities
    - Task queue
    - Status tracking
    - Communication with coordinator
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        agent_type: AgentType,
        coordinator: "AgentCoordinator",
    ):
        self.id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.coordinator = coordinator

        self._status = AgentStatus.IDLE
        self._current_task: Optional[AgentTask] = None
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    @property
    def status(self) -> AgentStatus:
        return self._status

    async def start(self):
        """Start the agent's worker loop"""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info(f"Agent {self.name} started")

    async def stop(self):
        """Stop the agent"""
        self._running = False

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Agent {self.name} stopped")

    async def submit_task(self, task: AgentTask):
        """Submit a task to this agent"""
        await self._task_queue.put(task)

    async def _worker_loop(self):
        """Main worker loop for processing tasks"""
        while self._running:
            try:
                task = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)

                self._status = AgentStatus.BUSY
                self._current_task = task
                task.started_at = datetime.now()

                logger.info(f"Agent {self.name} processing task: {task.id}")

                try:
                    result = await self.process_task(task)
                    task.result = result
                    task.completed_at = datetime.now()
                    await self.coordinator.report_task_complete(task)

                except Exception as e:
                    task.error = str(e)
                    task.completed_at = datetime.now()
                    logger.error(f"Agent {self.name} task failed: {e}")
                    await self.coordinator.report_task_failed(task)

                self._current_task = None
                self._status = AgentStatus.IDLE

            except asyncio.TimeoutError:
                self._status = AgentStatus.IDLE

            except asyncio.CancelledError:
                break

    async def process_task(self, task: AgentTask) -> Any:
        """Process a task - override in subclasses"""
        raise NotImplementedError

    def get_info(self) -> AgentInfo:
        """Get agent information"""
        return AgentInfo(
            id=self.id,
            name=self.name,
            agent_type=self.agent_type,
            status=self._status,
            current_task=self._current_task,
            capabilities=self.get_capabilities(),
        )

    def get_capabilities(self) -> List[str]:
        """Get agent capabilities - override in subclasses"""
        return []


class AgentCoordinator:
    """
    Coordinates multiple agents working in parallel

    Features:
    - Task distribution based on agent capabilities
    - Task prioritization
    - Status monitoring
    - Inter-agent communication
    - Mobile-optimized (lightweight, efficient)
    """

    def __init__(self, max_concurrent_tasks: int = 5):
        self.max_concurrent_tasks = max_concurrent_tasks

        self._agents: Dict[str, Agent] = {}
        self._agent_types: Dict[AgentType, List[str]] = {}
        self._tasks: Dict[str, AgentTask] = {}
        self._task_results: deque = deque(maxlen=100)

        self._running = False
        self._lock = asyncio.Lock()

        self._event_handlers: Dict[str, List[Callable]] = {
            "task_complete": [],
            "task_failed": [],
            "agent_status_change": [],
        }

    def register_agent(self, agent: Agent):
        """Register an agent with the coordinator"""
        self._agents[agent.id] = agent

        if agent.agent_type not in self._agent_types:
            self._agent_types[agent.agent_type] = []
        self._agent_types[agent.agent_type].append(agent.id)

        logger.info(f"Registered agent: {agent.name} ({agent.agent_type.value})")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self._agents:
            agent = self._agents[agent_id]

            if agent.agent_type in self._agent_types:
                self._agent_types[agent.agent_type].remove(agent_id)

            del self._agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")

    async def start(self):
        """Start all registered agents"""
        self._running = True

        for agent in self._agents.values():
            await agent.start()

        logger.info(f"Coordinator started with {len(self._agents)} agents")

    async def stop(self):
        """Stop all agents"""
        self._running = False

        for agent in self._agents.values():
            await agent.stop()

        logger.info("Coordinator stopped")

    async def submit_task(
        self,
        task: AgentTask,
        target_agent_type: Optional[AgentType] = None,
    ) -> str:
        """Submit a task to be processed by an appropriate agent"""
        async with self._lock:
            self._tasks[task.id] = task

        if target_agent_type:
            agents = self._agent_types.get(target_agent_type, [])
        else:
            agents = await self._find_agents_for_task(task)

        if not agents:
            task.error = "No suitable agent found"
            task.completed_at = datetime.now()
            return task.id

        agent_id = await self._select_best_agent(agents)
        agent = self._agents.get(agent_id)

        if agent:
            await agent.submit_task(task)
            logger.info(f"Task {task.id} assigned to agent {agent.name}")
        else:
            task.error = "Agent not found"
            task.completed_at = datetime.now()

        return task.id

    async def _find_agents_for_task(self, task: AgentTask) -> List[str]:
        """Find agents capable of handling a task"""
        capable = []

        for agent in self._agents.values():
            if agent.get_capabilities():
                capable.append(agent.id)

        return capable

    async def _select_best_agent(self, agent_ids: List[str]) -> Optional[str]:
        """Select the best agent for a task based on current load"""
        best_id = None
        best_load = float("inf")

        for agent_id in agent_ids:
            agent = self._agents.get(agent_id)
            if agent and agent.status != AgentStatus.ERROR:
                queue_size = agent._task_queue.qsize()
                if queue_size < best_load:
                    best_load = queue_size
                    best_id = agent_id

        return best_id

    async def report_task_complete(self, task: AgentTask):
        """Report task completion"""
        async with self._lock:
            self._task_results.append(task)

        for handler in self._event_handlers["task_complete"]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(task)
                else:
                    handler(task)
            except Exception as e:
                logger.error(f"Task complete handler error: {e}")

    async def report_task_failed(self, task: AgentTask):
        """Report task failure"""
        async with self._lock:
            self._task_results.append(task)

        for handler in self._event_handlers["task_failed"]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(task)
                else:
                    handler(task)
            except Exception as e:
                logger.error(f"Task failed handler error: {e}")

    def on(self, event: str, handler: Callable):
        """Register an event handler"""
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)

    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents"""
        status = {}

        for agent in self._agents.values():
            info = agent.get_info()
            status[agent.id] = {
                "name": info.name,
                "type": info.agent_type.value,
                "status": info.status.value,
                "current_task": info.current_task.id if info.current_task else None,
                "completed_tasks": info.completed_tasks,
                "failed_tasks": info.failed_tasks,
            }

        return status

    def get_task_status(self, task_id: str) -> Optional[AgentTask]:
        """Get status of a specific task"""
        return self._tasks.get(task_id)

    def get_coordinator_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        return {
            "total_agents": len(self._agents),
            "total_tasks": len(self._tasks),
            "completed_tasks": len([t for t in self._task_results if not t.error]),
            "failed_tasks": len([t for t in self._task_results if t.error]),
            "agents_by_type": {
                agent_type.value: len(agent_ids)
                for agent_type, agent_ids in self._agent_types.items()
            },
        }


class OrchestratorAgent(Agent):
    """Main orchestrator agent - coordinates other agents"""

    def __init__(self, coordinator: AgentCoordinator):
        super().__init__(
            agent_id="orchestrator",
            name="Orchestrator",
            agent_type=AgentType.ORCHESTRATOR,
            coordinator=coordinator,
        )

    def get_capabilities(self) -> List[str]:
        return ["coordination", "task_planning", "resource_allocation"]

    async def process_task(self, task: AgentTask) -> Any:
        """Process orchestration tasks"""
        await asyncio.sleep(0.1)
        return {"status": "orchestrated", "task_id": task.id}


class AnalyzerAgent(Agent):
    """Analyzes data and provides insights"""

    def __init__(self, coordinator: AgentCoordinator):
        super().__init__(
            agent_id="analyzer",
            name="Analyzer",
            agent_type=AgentType.ANALYZER,
            coordinator=coordinator,
        )

    def get_capabilities(self) -> List[str]:
        return ["analysis", "pattern_recognition", "insights"]

    async def process_task(self, task: AgentTask) -> Any:
        """Process analysis tasks"""
        await asyncio.sleep(0.1)
        return {"status": "analyzed", "task_id": task.id}


class ExecutorAgent(Agent):
    """Executes actions and tasks"""

    def __init__(self, coordinator: AgentCoordinator):
        super().__init__(
            agent_id="executor",
            name="Executor",
            agent_type=AgentType.EXECUTOR,
            coordinator=coordinator,
        )

    def get_capabilities(self) -> List[str]:
        return ["execution", "action", "automation"]

    async def process_task(self, task: AgentTask) -> Any:
        """Process execution tasks"""
        await asyncio.sleep(0.1)
        return {"status": "executed", "task_id": task.id}
