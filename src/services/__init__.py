"""
AURA v3 Services Package
Core services that make AURA a true personal assistant
"""

from src.services.life_tracker import (
    LifeTracker,
    LifeEvent,
    EventCategory,
    EventPriority,
    EventStatus,
    UserPattern,
    SocialInsight,
)

from src.services.proactive_engine import (
    ProactiveEngine,
    ProactiveAction,
    ActionType,
    ActionPriority,
    ActionStatus,
    TriggerRule,
)

from src.services.dashboard import (
    DashboardService,
    DashboardActivity,
    AgentDashboardInfo,
    DashboardState,
    ActivityType,
    DashboardView,
)

from src.services.task_context import (
    TaskContextPreservation,
    TrackedTask,
    TaskState,
    TaskCheckpoint,
    InterruptionType,
)

from src.services.background_manager import (
    BackgroundResourceManager,
    BackgroundTask,
    ResourcePriority,
    ScreenState,
    TaskIntensity,
    ResourceBudget,
)

__all__ = [
    # Life Tracker
    "LifeTracker",
    "LifeEvent",
    "EventCategory",
    "EventPriority",
    "EventStatus",
    "UserPattern",
    "SocialInsight",
    # Proactive Engine
    "ProactiveEngine",
    "ProactiveAction",
    "ActionType",
    "ActionPriority",
    "ActionStatus",
    "TriggerRule",
    # Dashboard
    "DashboardService",
    "DashboardActivity",
    "AgentDashboardInfo",
    "DashboardState",
    "ActivityType",
    "DashboardView",
    # Task Context
    "TaskContextPreservation",
    "TrackedTask",
    "TaskState",
    "TaskCheckpoint",
    "InterruptionType",
    # Background Manager
    "BackgroundResourceManager",
    "BackgroundTask",
    "ResourcePriority",
    "ScreenState",
    "TaskIntensity",
    "ResourceBudget",
]
