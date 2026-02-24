"""
AURA v3 Life Tracker Service
Tracks user's life events, patterns, schedules, and social activity
Core to AURA being a TRUE personal assistant that never forgets
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import uuid

logger = logging.getLogger(__name__)


class EventCategory(Enum):
    """Categories of life events"""

    FAMILY = "family"
    WORK = "work"
    SOCIAL = "social"
    HEALTH = "health"
    PERSONAL = "personal"
    SHOPPING = "shopping"
    EDUCATION = "education"
    FINANCIAL = "financial"
    REMINDER = "reminder"
    TRAVEL = "travel"


class EventPriority(Enum):
    """Priority levels for events"""

    CRITICAL = 3
    HIGH = 2
    MEDIUM = 1
    LOW = 0


class EventStatus(Enum):
    """Status of tracked events"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class LifeEvent:
    """A life event that AURA tracks"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    category: EventCategory = EventCategory.PERSONAL
    priority: EventPriority = EventPriority.MEDIUM
    status: EventStatus = EventStatus.PENDING

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    event_date: Optional[datetime] = None
    deadline: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Source information
    source: str = "manual"  # whatsapp, message, calendar, manual
    source_app: Optional[str] = None

    # Context
    related_people: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    location: Optional[str] = None

    # Actions taken
    actions_taken: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # AI insights
    insights: Dict[str, Any] = field(default_factory=dict)
    prep_status: str = "not_started"  # not_started, in_progress, prepared

    # Metadata
    is_recurring: bool = False
    recurring_pattern: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class UserPattern:
    """Pattern detected in user behavior"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    pattern_type: str = ""
    description: str = ""
    confidence: float = 0.0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    frequency: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SocialInsight:
    """Insight from social media analysis"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    platform: str = ""
    insight_type: str = ""  # interest, preference, behavior
    content: str = ""
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    actionable: bool = False
    suggested_action: Optional[str] = None


class LifeTracker:
    """
    Life Tracker - AURA's memory of user's life

    This is what makes AURA truly personal:
    - Remembers events user might forget
    - Tracks patterns in user behavior
    - Analyzes social media (with permission)
    - Proactively prepares for upcoming events
    - Never forgets important dates/deadlines
    """

    def __init__(self, storage_path: str = "data/life_tracker.json"):
        self.storage_path = storage_path
        self._events: Dict[str, LifeEvent] = {}
        self._patterns: Dict[str, UserPattern] = {}
        self._social_insights: Dict[str, SocialInsight] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._tracker_task: Optional[asyncio.Task] = None

        # Callbacks for proactive actions
        self._event_callbacks: List[Callable] = []

    async def start(self):
        """Start the life tracker"""
        self._running = True
        self._tracker_task = asyncio.create_task(self._tracker_loop())
        await self._load_data()
        logger.info("Life Tracker started")

    async def stop(self):
        """Stop the life tracker"""
        self._running = False
        await self._save_data()
        if self._tracker_task:
            self._tracker_task.cancel()
        logger.info("Life Tracker stopped")

    async def add_event(
        self,
        title: str,
        category: EventCategory = EventCategory.PERSONAL,
        event_date: Optional[datetime] = None,
        deadline: Optional[datetime] = None,
        description: str = "",
        source: str = "manual",
        related_people: List[str] = None,
        **kwargs,
    ) -> LifeEvent:
        """Add a new life event to track"""
        event = LifeEvent(
            title=title,
            description=description,
            category=category,
            event_date=event_date,
            deadline=deadline,
            source=source,
            related_people=related_people or [],
            **kwargs,
        )

        self._events[event.id] = event

        # Analyze event for proactive preparation
        await self._analyze_event_for_preparation(event)

        # Trigger callbacks
        for callback in self._event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

        await self._save_data()
        logger.info(f"Added life event: {event.title}")
        return event

    async def add_event_from_message(
        self, message_text: str, sender: str, app: str, timestamp: datetime = None
    ) -> Optional[LifeEvent]:
        """Add event from WhatsApp/message analysis"""
        # This would integrate with message analysis
        # For now, create placeholder logic
        timestamp = timestamp or datetime.now()

        # Extract potential event keywords
        event_keywords = {
            "wedding": EventCategory.FAMILY,
            "birthday": EventCategory.FAMILY,
            "meeting": EventCategory.WORK,
            "deadline": EventCategory.WORK,
            "doctor": EventCategory.HEALTH,
            "appointment": EventCategory.PERSONAL,
            "trip": EventCategory.SOCIAL,
            "flight": EventCategory.TRAVEL,
        }

        message_lower = message_text.lower()
        for keyword, category in event_keywords.items():
            if keyword in message_lower:
                return await self.add_event(
                    title=f"Detected from {sender}: {keyword.title()}",
                    category=category,
                    source=app,
                    source_app=app,
                    description=message_text,
                    notes=[f"From message by {sender}"],
                )

        return None

    async def get_upcoming_events(
        self, days: int = 7, category: Optional[EventCategory] = None
    ) -> List[LifeEvent]:
        """Get events coming up in the next N days"""
        now = datetime.now()
        cutoff = now + timedelta(days=days)

        upcoming = []
        for event in self._events.values():
            if event.status == EventStatus.COMPLETED:
                continue

            if event.event_date:
                if now <= event.event_date <= cutoff:
                    if category is None or event.category == category:
                        upcoming.append(event)

        return sorted(upcoming, key=lambda e: e.event_date or datetime.max)

    async def get_events_by_category(
        self, category: EventCategory, status: Optional[EventStatus] = None
    ) -> List[LifeEvent]:
        """Get all events of a specific category"""
        events = [
            e
            for e in self._events.values()
            if e.category == category and (status is None or e.status == status)
        ]
        return sorted(events, key=lambda e: e.event_date or datetime.max)

    async def update_event_status(
        self, event_id: str, status: EventStatus, note: Optional[str] = None
    ):
        """Update event status"""
        if event_id in self._events:
            event = self._events[event_id]
            event.status = status

            if status == EventStatus.COMPLETED:
                event.completed_at = datetime.now()

            if note:
                event.notes.append(note)

            await self._save_data()

    async def add_action_to_event(self, event_id: str, action: str):
        """Record an action taken for an event"""
        if event_id in self._events:
            event = self._events[event_id]
            event.actions_taken.append(action)

            # Check if event is now prepared
            if event.actions_taken:
                event.prep_status = "prepared"

            await self._save_data()

    async def detect_patterns(self) -> List[UserPattern]:
        """Detect patterns in user behavior"""
        patterns = []

        # Analyze event timing patterns
        event_times = {}
        for event in self._events.values():
            if event.event_date:
                hour = event.event_date.hour
                day = event.event_date.weekday()
                key = f"{event.category.value}_{day}_{hour}"
                event_times[key] = event_times.get(key, 0) + 1

        # Create patterns for frequent combinations
        for key, count in event_times.items():
            if count >= 3:
                category, day, hour = key.rsplit("_", 2)
                pattern = UserPattern(
                    pattern_type="timing",
                    description=f"Tends to have {category} events on day {day} around {hour}:00",
                    confidence=min(count / 10, 1.0),
                    frequency=count,
                    context={"category": category, "day": day, "hour": hour},
                )
                patterns.append(pattern)
                self._patterns[pattern.id] = pattern

        return patterns

    async def add_social_insight(
        self,
        platform: str,
        insight_type: str,
        content: str,
        confidence: float = 0.5,
        actionable: bool = False,
        suggested_action: Optional[str] = None,
    ) -> SocialInsight:
        """Add insight from social media analysis"""
        insight = SocialInsight(
            platform=platform,
            insight_type=insight_type,
            content=content,
            confidence=confidence,
            actionable=actionable,
            suggested_action=suggested_action,
        )

        self._social_insights[insight.id] = insight
        await self._save_data()

        return insight

    async def get_social_insights(
        self, platform: Optional[str] = None, actionable_only: bool = False
    ) -> List[SocialInsight]:
        """Get social insights"""
        insights = list(self._social_insights.values())

        if platform:
            insights = [i for i in insights if i.platform == platform]

        if actionable_only:
            insights = [i for i in insights if i.actionable]

        return sorted(insights, key=lambda i: i.created_at, reverse=True)

    async def get_user_interests(self) -> Dict[str, float]:
        """Get user's interests based on social insights"""
        interests = {}

        for insight in self._social_insights.values():
            if insight.insight_type == "interest":
                interests[insight.content] = max(
                    interests.get(insight.content, 0), insight.confidence
                )

        return dict(sorted(interests.items(), key=lambda x: x[1], reverse=True))

    async def prepare_for_event(self, event_id: str) -> Dict[str, Any]:
        """Prepare resources/actions for an event"""
        if event_id not in self._events:
            return {"error": "Event not found"}

        event = self._events[event_id]
        preparations = []

        days_until = (event.event_date - datetime.now()).days if event.event_date else 0

        # Different preparation based on category and timing
        if event.category == EventCategory.FAMILY:
            if days_until <= 7:
                preparations.append(
                    {
                        "type": "shopping",
                        "action": "Research outfit options for event",
                        "priority": "high",
                    }
                )
                preparations.append(
                    {
                        "type": "schedule",
                        "action": "Check for scheduling conflicts",
                        "priority": "high",
                    }
                )

        elif event.category == EventCategory.WORK:
            if days_until <= 1:
                preparations.append(
                    {
                        "type": "research",
                        "action": "Prepare meeting materials",
                        "priority": "high",
                    }
                )

        elif event.category == EventCategory.SHOPPING:
            preparations.append(
                {
                    "type": "price_check",
                    "action": "Monitor for price changes",
                    "priority": "medium",
                }
            )

        # Update event prep status
        event.prep_status = "in_progress"
        await self._save_data()

        return {
            "event": event.title,
            "days_until": days_until,
            "preparations": preparations,
        }

    async def get_life_summary(self) -> Dict[str, Any]:
        """Get a summary of tracked life data"""
        now = datetime.now()

        return {
            "total_events": len(self._events),
            "upcoming_7_days": len(await self.get_upcoming_events(7)),
            "completed": len(
                [e for e in self._events.values() if e.status == EventStatus.COMPLETED]
            ),
            "pending": len(
                [e for e in self._events.values() if e.status == EventStatus.PENDING]
            ),
            "by_category": {
                cat.value: len([e for e in self._events.values() if e.category == cat])
                for cat in EventCategory
            },
            "patterns_detected": len(self._patterns),
            "social_insights": len(self._social_insights),
            "top_interests": dict(list((await self.get_user_interests()).items())[:5]),
        }

    def register_event_callback(self, callback: Callable):
        """Register callback for new events"""
        self._event_callbacks.append(callback)

    async def _tracker_loop(self):
        """Background loop for time-based checks"""
        while self._running:
            try:
                # Check for upcoming events
                upcoming = await self.get_upcoming_events(1)

                for event in upcoming:
                    if event.event_date:
                        time_diff = event.event_date - datetime.now()

                        # Notify for events in next hour
                        if 0 < time_diff.total_seconds() < 3600:
                            logger.info(f"Event approaching: {event.title}")

                # Run pattern detection daily
                await self.detect_patterns()

            except Exception as e:
                logger.error(f"Tracker loop error: {e}")

            await asyncio.sleep(3600)  # Check every hour

    async def _analyze_event_for_preparation(self, event: LifeEvent):
        """Analyze event and schedule proactive preparation"""
        if not event.event_date:
            return

        days_until = (event.event_date - datetime.now()).days

        # Schedule preparation based on importance and timing
        prep_schedule = {
            EventPriority.CRITICAL: 7,  # Start 7 days before
            EventPriority.HIGH: 3,
            EventPriority.MEDIUM: 1,
            EventPriority.LOW: 0,
        }

        prep_days = prep_schedule.get(event.priority, 1)

        if days_until <= prep_days:
            await self.prepare_for_event(event.id)

    async def _load_data(self):
        """Load data from storage"""
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            # Load events
            for e_data in data.get("events", []):
                event = LifeEvent(**e_data)
                self._events[event.id] = event

            # Load patterns
            for p_data in data.get("patterns", []):
                pattern = UserPattern(**p_data)
                self._patterns[pattern.id] = pattern

            # Load insights
            for i_data in data.get("insights", []):
                insight = SocialInsight(**i_data)
                self._social_insights[insight.id] = insight

            logger.info(f"Loaded {len(self._events)} events")

        except FileNotFoundError:
            logger.info("No existing data found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading data: {e}")

    async def _save_data(self):
        """Save data to storage"""
        try:
            data = {
                "events": [vars(e) for e in self._events.values()],
                "patterns": [vars(p) for p in self._patterns.values()],
                "insights": [vars(i) for i in self._social_insights.values()],
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, default=str)

        except Exception as e:
            logger.error(f"Error saving data: {e}")
