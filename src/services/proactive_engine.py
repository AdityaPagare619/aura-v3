"""
AURA v3 Proactive Intelligence Engine
Decides when and how to act PROACTIVELY without user prompting
The brain behind AURA's "anticipation" capabilities
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import uuid
import random

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of proactive actions"""

    RESEARCH = "research"
    PREPARE = "prepare"
    NOTIFY = "notify"
    SUGGEST = "suggest"
    PLAN = "plan"
    EXECUTE = "execute"
    MONITOR = "monitor"


class ActionPriority(Enum):
    """Priority of proactive action"""

    URGENT = 3
    HIGH = 2
    MEDIUM = 1
    LOW = 0


class ActionStatus(Enum):
    """Status of proactive action"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProactiveAction:
    """A proactive action AURA decides to take"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    action_type: ActionType = ActionType.RESEARCH
    title: str = ""
    description: str = ""
    priority: ActionPriority = ActionPriority.MEDIUM
    status: ActionStatus = ActionStatus.PENDING

    # Trigger information
    trigger_reason: str = ""
    trigger_source: str = ""  # event, pattern, insight, schedule

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Execution
    target: Optional[str] = None  # What this action targets
    tools_needed: List[str] = field(default_factory=list)
    result: Any = None
    error: Optional[str] = None

    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    resources_used: List[str] = field(default_factory=list)

    # User notification
    notify_user: bool = False
    notification_message: Optional[str] = None


@dataclass
class TriggerRule:
    """Rule that triggers proactive actions"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""

    # Trigger conditions
    trigger_type: str = ""  # event, pattern, time, insight
    conditions: Dict[str, Any] = field(default_factory=dict)

    # Action to take
    action_type: ActionType = ActionType.RESEARCH
    action_template: str = ""

    # Settings
    enabled: bool = True
    cooldown_minutes: int = 60
    last_triggered: Optional[datetime] = None

    priority: ActionPriority = ActionPriority.MEDIUM


class ProactiveEngine:
    """
    Proactive Intelligence Engine

    This is what makes AURA truly "smart" - it doesn't wait for commands.
    It observes, analyzes, and ACTS proactively.

    Key responsibilities:
    - Monitor triggers (events, patterns, time, insights)
    - Decide WHEN to act
    - Determine WHAT action to take
    - Execute actions in background
    - Notify user when appropriate
    """

    def __init__(self, life_tracker=None, memory_system=None):
        self.life_tracker = life_tracker
        self.memory = memory_system

        self._running = False
        self._engine_task: Optional[asyncio.Task] = None

        # Active actions
        self._active_actions: Dict[str, ProactiveAction] = {}
        self._action_history: deque = deque(maxlen=100)

        # Trigger rules
        self._trigger_rules: Dict[str, TriggerRule] = {}

        # Callbacks
        self._action_callbacks: List[Callable] = []

        # Settings
        self.max_concurrent_actions = 3
        self.check_interval = 300  # 5 minutes
        self.enabled = True

        # Connected services (set via setters)
        self.event_tracker = None
        self.call_manager = None
        self.life_explorer = None

        # Built-in trigger rules
        self._setup_default_rules()

    def set_services(self, event_tracker=None, call_manager=None, life_explorer=None):
        """Wire proactive services to the engine"""
        self.event_tracker = event_tracker
        self.call_manager = call_manager
        self.life_explorer = life_explorer
        logger.info("Proactive services wired to engine")

    def _setup_default_rules(self):
        """Setup default proactive trigger rules"""

        # Event approaching - prepare
        self.add_trigger_rule(
            TriggerRule(
                name="event_approaching",
                description="When an important event is coming up",
                trigger_type="event",
                conditions={"days_until": 3, "min_priority": "HIGH"},
                action_type=ActionType.PREPARE,
                action_template="Prepare for {event_title}",
                priority=ActionPriority.HIGH,
            )
        )

        # Shopping interest detected
        self.add_trigger_rule(
            TriggerRule(
                name="shopping_interest",
                description="User saved or liked shopping items",
                trigger_type="insight",
                conditions={"insight_type": "shopping_interest"},
                action_type=ActionType.RESEARCH,
                action_template="Find options for {interest}",
                priority=ActionPriority.MEDIUM,
            )
        )

        # Pattern detected - learn more
        self.add_trigger_rule(
            TriggerRule(
                name="new_pattern",
                description="New user pattern detected",
                trigger_type="pattern",
                conditions={"confidence": 0.7},
                action_type=ActionType.MONITOR,
                action_template="Monitor pattern: {pattern_type}",
                priority=ActionPriority.LOW,
            )
        )

        # Daily check-in
        self.add_trigger_rule(
            TriggerRule(
                name="daily_checkin",
                description="Daily routine check",
                trigger_type="time",
                conditions={"hour": 9},
                action_type=ActionType.NOTIFY,
                action_template="Daily summary for user",
                priority=ActionPriority.MEDIUM,
            )
        )

    def add_trigger_rule(self, rule: TriggerRule):
        """Add a trigger rule"""
        self._trigger_rules[rule.id] = rule
        logger.info(f"Added trigger rule: {rule.name}")

    def remove_trigger_rule(self, rule_id: str):
        """Remove a trigger rule"""
        if rule_id in self._trigger_rules:
            del self._trigger_rules[rule_id]

    async def start(self):
        """Start the proactive engine"""
        self._running = True
        self._engine_task = asyncio.create_task(self._engine_loop())
        logger.info("Proactive Engine started")

    async def stop(self):
        """Stop the proactive engine"""
        self._running = False
        if self._engine_task:
            self._engine_task.cancel()
        logger.info("Proactive Engine stopped")

    async def trigger_action(
        self,
        action_type: ActionType,
        title: str,
        description: str,
        trigger_reason: str,
        priority: ActionPriority = ActionPriority.MEDIUM,
        context: Dict[str, Any] = None,
        notify_user: bool = False,
        schedule_delay: Optional[timedelta] = None,
    ) -> ProactiveAction:
        """Manually trigger a proactive action"""

        action = ProactiveAction(
            action_type=action_type,
            title=title,
            description=description,
            trigger_reason=trigger_reason,
            priority=priority,
            context=context or {},
            notify_user=notify_user,
            scheduled_at=datetime.now() + schedule_delay if schedule_delay else None,
        )

        self._active_actions[action.id] = action

        # Execute if not delayed
        if not schedule_delay:
            asyncio.create_task(self._execute_action(action))

        return action

    async def decide_proactive_actions(self) -> List[ProactiveAction]:
        """Decide what proactive actions to take based on triggers"""
        new_actions = []

        # Check event-based triggers
        if self.life_tracker:
            upcoming = await self.life_tracker.get_upcoming_events(7)

            for rule in self._trigger_rules.values():
                if not rule.enabled or rule.trigger_type != "event":
                    continue

                # Check cooldown
                if rule.last_triggered:
                    if (
                        datetime.now() - rule.last_triggered
                    ).total_seconds() < rule.cooldown_minutes * 60:
                        continue

                for event in upcoming:
                    if event.event_date:
                        days_until = (event.event_date - datetime.now()).days

                        # Check conditions
                        if "days_until" in rule.conditions:
                            if days_until == rule.conditions["days_until"]:
                                # Create action
                                action = ProactiveAction(
                                    action_type=rule.action_type,
                                    title=rule.action_template.format(
                                        event_title=event.title
                                    ),
                                    description=rule.description,
                                    trigger_reason=f"Event: {event.title}",
                                    trigger_source="event",
                                    priority=rule.priority,
                                    context={"event_id": event.id},
                                )
                                new_actions.append(action)
                                rule.last_triggered = datetime.now()
                                break

        # Check insight-based triggers
        if self.life_tracker:
            insights = await self.life_tracker.get_social_insights(actionable_only=True)

            for rule in self._trigger_rules.values():
                if not rule.enabled or rule.trigger_type != "insight":
                    continue

                for insight in insights:
                    if insight.insight_type == rule.conditions.get("insight_type"):
                        action = ProactiveAction(
                            action_type=rule.action_type,
                            title=rule.action_template.format(interest=insight.content),
                            description=f"Based on {insight.platform} activity",
                            trigger_source="insight",
                            priority=rule.priority,
                            context={"insight_id": insight.id},
                        )
                        new_actions.append(action)
                        break

        return new_actions

    async def _engine_loop(self):
        """Main proactive engine loop"""
        while self._running:
            try:
                if self.enabled:
                    # Decide what actions to take
                    actions = await self.decide_proactive_actions()

                    for action in actions:
                        if (
                            len(
                                [
                                    a
                                    for a in self._active_actions.values()
                                    if a.status == ActionStatus.RUNNING
                                ]
                            )
                            < self.max_concurrent_actions
                        ):
                            self._active_actions[action.id] = action
                            asyncio.create_task(self._execute_action(action))

                # Clean up old actions
                await self._cleanup_completed_actions()

            except Exception as e:
                logger.error(f"Proactive engine error: {e}")

            await asyncio.sleep(self.check_interval)

    async def _execute_action(self, action: ProactiveAction):
        """Execute a proactive action"""
        logger.info(f"Executing proactive action: {action.title}")

        action.status = ActionStatus.RUNNING
        action.started_at = datetime.now()

        try:
            # Execute based on action type
            if action.action_type == ActionType.RESEARCH:
                result = await self._do_research(action)
            elif action.action_type == ActionType.PREPARE:
                result = await self._do_prepare(action)
            elif action.action_type == ActionType.PLAN:
                result = await self._do_plan(action)
            elif action.action_type == ActionType.MONITOR:
                result = await self._do_monitor(action)
            else:
                result = {"status": "completed"}

            action.result = result
            action.status = ActionStatus.COMPLETED
            action.completed_at = datetime.now()

            # Notify user if needed
            if action.notify_user:
                await self._notify_user(action)

        except Exception as e:
            action.error = str(e)
            action.status = ActionStatus.FAILED
            logger.error(f"Action failed: {e}")

        self._action_history.append(action)

        # Trigger callbacks
        for callback in self._action_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(action)
                else:
                    callback(action)
            except Exception as e:
                logger.error(f"Action callback error: {e}")

    async def _do_research(self, action: ProactiveAction) -> Dict[str, Any]:
        """Perform research action"""
        await asyncio.sleep(1)  # Placeholder - would use browser/tools

        return {
            "status": "research_complete",
            "findings": [],
            "summary": f"Research completed for: {action.title}",
        }

    async def _do_prepare(self, action: ProactiveAction) -> Dict[str, Any]:
        """Perform preparation action"""
        # If this is for an event, use life tracker
        if self.life_tracker and "event_id" in action.context:
            prep_result = await self.life_tracker.prepare_for_event(
                action.context["event_id"]
            )

            await self.life_tracker.add_action_to_event(
                action.context["event_id"], f"Proactive preparation: {action.title}"
            )

            return prep_result

        return {"status": "preparation_complete"}

    async def _do_plan(self, action: ProactiveAction) -> Dict[str, Any]:
        """Perform planning action"""
        return {"status": "plan_created", "plan": []}

    async def _do_monitor(self, action: ProactiveAction) -> Dict[str, Any]:
        """Perform monitoring action"""
        return {"status": "monitoring_active"}

    async def _notify_user(self, action: ProactiveAction):
        """Send notification to user"""
        if action.notification_message:
            logger.info(f"Would notify user: {action.notification_message}")
            # Would integrate with notification system

    async def _cleanup_completed_actions(self):
        """Clean up completed actions"""
        to_remove = [
            aid
            for aid, action in self._active_actions.items()
            if action.status
            in [ActionStatus.COMPLETED, ActionStatus.FAILED, ActionStatus.CANCELLED]
        ]

        for aid in to_remove:
            del self._active_actions[aid]

    def get_active_actions(self) -> List[ProactiveAction]:
        """Get all active actions"""
        return list(self._active_actions.values())

    def get_action_history(self) -> List[ProactiveAction]:
        """Get action history"""
        return list(self._action_history)

    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            "running": self._running,
            "enabled": self.enabled,
            "active_actions": len(self._active_actions),
            "trigger_rules": len(self._trigger_rules),
            "completed_today": len(
                [
                    a
                    for a in self._action_history
                    if a.completed_at and a.completed_at.date() == datetime.now().date()
                ]
            ),
        }
