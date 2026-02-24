"""
AURA v3 Dashboard Service
User-facing dashboard to see what's happening with AURA
Like a cockpit to monitor all agents and activities
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class DashboardView(Enum):
    """Types of dashboard views"""

    OVERVIEW = "overview"
    AGENTS = "agents"
    TASKS = "tasks"
    MEMORY = "memory"
    DISCOVERY = "discovery"
    SETTINGS = "settings"


class ActivityType(Enum):
    """Types of dashboard activities"""

    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    TASK_CREATED = "task_created"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    INSIGHT_DISCOVERED = "insight_discovered"
    EVENT_DETECTED = "event_detected"
    ACTION_PROACTIVE = "action_proactive"
    USER_INTERACTION = "user_interaction"
    SYSTEM_ALERT = "system_alert"


@dataclass
class DashboardActivity:
    """Single activity in the dashboard"""

    id: str
    activity_type: ActivityType
    title: str
    description: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    viewed: bool = False


@dataclass
class AgentDashboardInfo:
    """Dashboard info for an agent"""

    agent_id: str
    agent_name: str
    status: str
    current_task: Optional[str]
    task_progress: float
    last_active: datetime
    actions_count: int


@dataclass
class DashboardState:
    """Complete dashboard state"""

    last_updated: datetime
    active_agents: int
    active_tasks: int
    pending_actions: int
    recent_activities: List[DashboardActivity]
    system_health: Dict[str, Any]
    user_profile_summary: Dict[str, Any]


class DashboardService:
    """
    Dashboard Service - The Cockpit for AURA

    Users can see:
    - What agents are working
    - What tasks are in progress
    - What's been discovered
    - What's been prepared
    - Activity feed
    - Quick actions

    Features:
    - Real-time updates
    - Activity history
    - Agent status
    - Task tracking
    - Discovery feed
    """

    def __init__(self):
        self._running = False

        # Activity feed
        self._activities: deque = deque(maxlen=500)

        # Agent status
        self._agent_status: Dict[str, AgentDashboardInfo] = {}

        # Task status
        self._task_status: Dict[str, Dict[str, Any]] = {}

        # Subscribers for real-time updates
        self._subscribers: List[Callable] = []

        # Settings
        self.max_visible_activities = 50

    async def start(self):
        """Start dashboard service"""
        self._running = True
        logger.info("Dashboard service started")

    async def stop(self):
        """Stop dashboard service"""
        self._running = False
        logger.info("Dashboard service stopped")

    # =========================================================================
    # ACTIVITY TRACKING
    # =========================================================================

    async def log_activity(
        self,
        activity_type: ActivityType,
        title: str,
        description: str,
        source: str,
        metadata: Dict[str, Any] = None,
    ):
        """Log an activity to the dashboard"""
        activity = DashboardActivity(
            id=f"{activity_type.value}_{datetime.now().timestamp()}",
            activity_type=activity_type,
            title=title,
            description=description,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {},
        )

        self._activities.append(activity)

        # Notify subscribers
        await self._notify_subscribers(activity)

    async def get_activities(
        self,
        activity_type: Optional[ActivityType] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
        include_viewed: bool = False,
    ) -> List[DashboardActivity]:
        """Get dashboard activities"""
        activities = list(self._activities)

        # Filter by type
        if activity_type:
            activities = [a for a in activities if a.activity_type == activity_type]

        # Filter by time
        if since:
            activities = [a for a in activities if a.timestamp >= since]

        # Filter viewed
        if not include_viewed:
            activities = [a for a in activities if not a.viewed]

        # Sort by time (newest first)
        activities = sorted(activities, key=lambda a: a.timestamp, reverse=True)

        return activities[:limit]

    async def mark_activity_viewed(self, activity_id: str):
        """Mark activity as viewed"""
        for activity in self._activities:
            if activity.id == activity_id:
                activity.viewed = True
                break

    async def mark_all_viewed(self):
        """Mark all activities as viewed"""
        for activity in self._activities:
            activity.viewed = True

    # =========================================================================
    # AGENT STATUS
    # =========================================================================

    async def update_agent_status(
        self,
        agent_id: str,
        agent_name: str,
        status: str,
        current_task: Optional[str] = None,
        task_progress: float = 0.0,
    ):
        """Update agent status"""
        self._agent_status[agent_id] = AgentDashboardInfo(
            agent_id=agent_id,
            agent_name=agent_name,
            status=status,
            current_task=current_task,
            task_progress=task_progress,
            last_active=datetime.now(),
            actions_count=self._agent_status.get(
                agent_id,
                AgentDashboardInfo(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    status=status,
                    current_task=current_task,
                    task_progress=task_progress,
                    last_active=datetime.now(),
                    actions_count=0,
                ),
            ).actions_count
            + 1,
        )

    async def get_agent_status(self) -> List[AgentDashboardInfo]:
        """Get status of all agents"""
        return list(self._agent_status.values())

    async def get_agent_by_id(self, agent_id: str) -> Optional[AgentDashboardInfo]:
        """Get specific agent status"""
        return self._agent_status.get(agent_id)

    # =========================================================================
    # TASK TRACKING
    # =========================================================================

    async def create_task(
        self,
        task_id: str,
        title: str,
        description: str,
        task_type: str,
        priority: str = "medium",
    ) -> Dict[str, Any]:
        """Create a tracked task"""
        task_info = {
            "id": task_id,
            "title": title,
            "description": description,
            "type": task_type,
            "priority": priority,
            "status": "created",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "logs": [],
        }

        self._task_status[task_id] = task_info

        await self.log_activity(
            ActivityType.TASK_CREATED,
            f"Task created: {title}",
            description,
            "system",
            {"task_id": task_id, "type": task_type},
        )

        return task_info

    async def update_task_progress(
        self,
        task_id: str,
        progress: float,
        status: Optional[str] = None,
        log: Optional[str] = None,
    ):
        """Update task progress"""
        if task_id in self._task_status:
            task = self._task_status[task_id]
            task["progress"] = progress
            task["updated_at"] = datetime.now().isoformat()

            if status:
                task["status"] = status

            if log:
                task["logs"].append(
                    {"time": datetime.now().isoformat(), "message": log}
                )

            if status == "completed":
                await self.log_activity(
                    ActivityType.TASK_COMPLETE,
                    f"Task completed: {task['title']}",
                    f"Progress: {progress}%",
                    "system",
                    {"task_id": task_id},
                )

    async def get_tasks(
        self, status: Optional[str] = None, task_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get tasks with optional filters"""
        tasks = list(self._task_status.values())

        if status:
            tasks = [t for t in tasks if t["status"] == status]

        if task_type:
            tasks = [t for t in tasks if t["type"] == task_type]

        return sorted(tasks, key=lambda t: t["created_at"], reverse=True)

    # =========================================================================
    # DISCOVERY FEED
    # =========================================================================

    async def add_discovery(
        self,
        title: str,
        description: str,
        discovery_type: str,
        relevance: float = 0.5,
        action_items: List[str] = None,
    ):
        """Add a discovery to the feed"""
        await self.log_activity(
            ActivityType.INSIGHT_DISCOVERED,
            title,
            description,
            "discovery",
            {
                "type": discovery_type,
                "relevance": relevance,
                "action_items": action_items or [],
            },
        )

    # =========================================================================
    # DASHBOARD STATE
    # =========================================================================

    async def get_dashboard_state(self) -> DashboardState:
        """Get complete dashboard state"""
        now = datetime.now()

        # Count active items
        active_agents = len(
            [a for a in self._agent_status.values() if a.status in ["running", "busy"]]
        )
        active_tasks = len(
            [
                t
                for t in self._task_status.values()
                if t["status"] not in ["completed", "cancelled"]
            ]
        )

        # Get recent activities
        recent = await self.get_activities(limit=10)

        return DashboardState(
            last_updated=now,
            active_agents=active_agents,
            active_tasks=active_tasks,
            pending_actions=0,  # Would come from proactive engine
            recent_activities=recent,
            system_health={"memory": "good", "cpu": "normal", "battery": "unknown"},
            user_profile_summary={
                "events_tracked": 0,
                "insights_generated": 0,
                "patterns_learned": 0,
            },
        )

    # =========================================================================
    # REAL-TIME SUBSCRIPTIONS
    # =========================================================================

    async def subscribe(self, callback: Callable):
        """Subscribe to dashboard updates"""
        self._subscribers.append(callback)

    async def unsubscribe(self, callback: Callable):
        """Unsubscribe from dashboard updates"""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    async def _notify_subscribers(self, activity: DashboardActivity):
        """Notify all subscribers of new activity"""
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(activity)
                else:
                    callback(activity)
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")

    # =========================================================================
    # FORMATTED OUTPUT
    # =========================================================================

    async def generate_dashboard_text(self) -> str:
        """Generate text representation of dashboard"""
        state = await self.get_dashboard_state()

        lines = [
            "╔═══════════════════════════════════════════════════════════════╗",
            "║                    AURA DASHBOARD                              ║",
            "╠═══════════════════════════════════════════════════════════════╣",
            f"║ Last Updated: {state.last_updated.strftime('%H:%M:%S')}                                      ║",
            "╠═══════════════════════════════════════════════════════════════╣",
            f"║ Active Agents: {state.active_agents}                                          ║",
            f"║ Active Tasks: {state.active_tasks}                                           ║",
            "╠═══════════════════════════════════════════════════════════════╣",
            "║ RECENT ACTIVITY:                                            ║",
        ]

        for activity in state.recent_activities[:5]:
            icon = {
                ActivityType.AGENT_START: "🤖",
                ActivityType.TASK_COMPLETE: "✅",
                ActivityType.INSIGHT_DISCOVERED: "💡",
                ActivityType.EVENT_DETECTED: "📅",
                ActivityType.ACTION_PROACTIVE: "⚡",
                ActivityType.USER_INTERACTION: "👤",
            }.get(activity.activity_type, "•")

            lines.append(f"║ {icon} {activity.title[:50]:<50} ║")

        lines.append(
            "╚═══════════════════════════════════════════════════════════════╝"
        )

        return "\n".join(lines)

    async def get_quick_stats(self) -> Dict[str, Any]:
        """Get quick statistics for overview"""
        today = datetime.now().date()

        today_activities = [a for a in self._activities if a.timestamp.date() == today]

        return {
            "activities_today": len(today_activities),
            "unread_count": len([a for a in self._activities if not a.viewed]),
            "active_agents": len(
                [a for a in self._agent_status.values() if a.status == "running"]
            ),
            "tasks_in_progress": len(
                [t for t in self._task_status.values() if t["status"] == "in_progress"]
            ),
            "completed_today": len(
                [
                    t
                    for t in self._task_status.values()
                    if t["status"] == "completed"
                    and t.get("updated_at", "").startswith(str(today))
                ]
            ),
        }
