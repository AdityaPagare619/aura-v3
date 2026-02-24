"""
AURA v3 Event Manager
Tracks social events, reminders, and follow-ups
100% offline - all data stored locally
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of social events"""

    BIRTHDAY = "birthday"
    ANNIVERSARY = "anniversary"
    MEETUP = "meetup"
    CALL = "call"
    DINNER = "dinner"
    CELEBRATION = "celebration"
    FOLLOWUP = "followup"
    REMINDER = "reminder"
    CUSTOM = "custom"


class ReminderType(Enum):
    """Types of reminders"""

    ONE_DAY = "one_day"
    THREE_DAYS = "three_days"
    ONE_WEEK = "one_week"
    CUSTOM = "custom"


@dataclass
class SocialEvent:
    """A social event"""

    id: str
    title: str
    event_type: EventType

    scheduled_at: datetime
    duration_minutes: int = 60

    location: str = ""
    description: str = ""

    related_contacts: List[str] = field(default_factory=list)

    recurring: bool = False
    recurrence_pattern: str = ""

    reminder_set: bool = True
    reminder_times: List[ReminderType] = field(default_factory=list)

    status: str = "scheduled"
    notes: str = ""

    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class Reminder:
    """A reminder for an event or follow-up"""

    id: str
    title: str
    description: str

    due_at: datetime
    reminder_type: ReminderType

    related_event_id: Optional[str] = None
    related_contact: Optional[str] = None

    dismissed: bool = False
    completed: bool = False

    snooze_count: int = 0
    snooze_until: Optional[datetime] = None

    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventManager:
    """
    Manages social events and reminders
    Tracks upcoming events, follow-ups, and important dates
    """

    def __init__(self, data_dir: str = "data/social_life/events"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._events: Dict[str, SocialEvent] = {}
        self._reminders: Dict[str, Reminder] = {}

        self._reminder_callbacks: List[Callable] = []

        self._default_reminder_times = [
            ReminderType.ONE_DAY,
            ReminderType.ONE_WEEK,
        ]

    async def initialize(self):
        """Initialize event manager"""
        logger.info("Initializing Event Manager...")
        await self._load_events()
        await self._load_reminders()
        await self._check_due_reminders()
        logger.info(f"Event Manager initialized with {len(self._events)} events")

    async def _load_events(self):
        """Load events from disk"""
        events_file = self.data_dir / "events.json"
        if events_file.exists():
            try:
                with open(events_file, "r") as f:
                    data = json.load(f)
                    for e_data in data.get("events", []):
                        e_data["scheduled_at"] = datetime.fromisoformat(
                            e_data["scheduled_at"]
                        )
                        e_data["created_at"] = datetime.fromisoformat(
                            e_data["created_at"]
                        )
                        if e_data.get("completed_at"):
                            e_data["completed_at"] = datetime.fromisoformat(
                                e_data["completed_at"]
                            )
                        self._events[e_data["id"]] = SocialEvent(**e_data)
                logger.info(f"Loaded {len(self._events)} events")
            except Exception as e:
                logger.error(f"Error loading events: {e}")

    async def _save_events(self):
        """Save events to disk"""
        events_file = self.data_dir / "events.json"
        try:
            data = {
                "events": [
                    {
                        **vars(e),
                        "scheduled_at": e.scheduled_at.isoformat(),
                        "created_at": e.created_at.isoformat(),
                        "completed_at": e.completed_at.isoformat()
                        if e.completed_at
                        else None,
                    }
                    for e in self._events.values()
                ]
            }
            with open(events_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving events: {e}")

    async def _load_reminders(self):
        """Load reminders from disk"""
        reminders_file = self.data_dir / "reminders.json"
        if reminders_file.exists():
            try:
                with open(reminders_file, "r") as f:
                    data = json.load(f)
                    for r_data in data.get("reminders", []):
                        r_data["due_at"] = datetime.fromisoformat(r_data["due_at"])
                        r_data["created_at"] = datetime.fromisoformat(
                            r_data["created_at"]
                        )
                        if r_data.get("snooze_until"):
                            r_data["snooze_until"] = datetime.fromisoformat(
                                r_data["snooze_until"]
                            )
                        self._reminders[r_data["id"]] = Reminder(**r_data)
                logger.info(f"Loaded {len(self._reminders)} reminders")
            except Exception as e:
                logger.error(f"Error loading reminders: {e}")

    async def _save_reminders(self):
        """Save reminders to disk"""
        reminders_file = self.data_dir / "reminders.json"
        try:
            data = {
                "reminders": [
                    {
                        **vars(r),
                        "due_at": r.due_at.isoformat(),
                        "created_at": r.created_at.isoformat(),
                        "snooze_until": r.snooze_until.isoformat()
                        if r.snooze_until
                        else None,
                    }
                    for r in self._reminders.values()
                ]
            }
            with open(reminders_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving reminders: {e}")

    async def create_event(
        self,
        title: str,
        event_type: EventType,
        scheduled_at: datetime,
        duration_minutes: int = 60,
        related_contacts: List[str] = None,
        recurring: bool = False,
        recurrence_pattern: str = "",
        **kwargs,
    ) -> SocialEvent:
        """Create a new social event"""
        event_id = f"event_{len(self._events)}_{int(datetime.now().timestamp())}"

        event = SocialEvent(
            id=event_id,
            title=title,
            event_type=event_type,
            scheduled_at=scheduled_at,
            duration_minutes=duration_minutes,
            related_contacts=related_contacts or [],
            recurring=recurring,
            recurrence_pattern=recurrence_pattern,
            **kwargs,
        )

        self._events[event_id] = event

        if event.reminder_set:
            await self._create_event_reminders(event)

        await self._save_events()
        logger.info(f"Created event: {title}")

        return event

    async def _create_event_reminders(self, event: SocialEvent):
        """Create reminders for an event"""
        reminder_offsets = {
            ReminderType.ONE_DAY: timedelta(days=1),
            ReminderType.THREE_DAYS: timedelta(days=3),
            ReminderType.ONE_WEEK: timedelta(weeks=1),
        }

        for reminder_type in event.reminder_times or self._default_reminder_times:
            if reminder_type in reminder_offsets:
                due = event.scheduled_at - reminder_offsets[reminder_type]
                if due > datetime.now():
                    await self.create_reminder(
                        title=f"Upcoming: {event.title}",
                        description=event.description,
                        due_at=due,
                        reminder_type=reminder_type,
                        related_event_id=event.id,
                        related_contact=event.related_contacts[0]
                        if event.related_contacts
                        else None,
                    )

    async def create_reminder(
        self,
        title: str,
        description: str,
        due_at: datetime,
        reminder_type: ReminderType,
        related_event_id: str = None,
        related_contact: str = None,
        **metadata,
    ) -> Reminder:
        """Create a reminder"""
        reminder_id = (
            f"reminder_{len(self._reminders)}_{int(datetime.now().timestamp())}"
        )

        reminder = Reminder(
            id=reminder_id,
            title=title,
            description=description,
            due_at=due_at,
            reminder_type=reminder_type,
            related_event_id=related_event_id,
            related_contact=related_contact,
            metadata=metadata,
        )

        self._reminders[reminder_id] = reminder

        await self._save_reminders()
        logger.info(f"Created reminder: {title}")

        return reminder

    async def create_birthday_event(
        self, contact_name: str, birth_date: datetime, year: int = None
    ) -> SocialEvent:
        """Create a birthday event"""
        this_year = datetime.now().year
        event_year = year or this_year

        scheduled = datetime(event_year, birth_date.month, birth_date.day, 9, 0)

        if scheduled < datetime.now():
            scheduled = datetime(event_year + 1, birth_date.month, birth_date.day, 9, 0)

        return await self.create_event(
            title=f"{contact_name}'s Birthday",
            event_type=EventType.BIRTHDAY,
            scheduled_at=scheduled,
            duration_minutes=0,
            related_contacts=[contact_name],
            recurring=True,
            recurrence_pattern="yearly",
            description=f"Birthday for {contact_name}",
            reminder_times=[ReminderType.ONE_WEEK, ReminderType.ONE_DAY],
        )

    async def create_followup_reminder(
        self,
        contact_name: str,
        followup_type: str,
        due_in_days: int = 7,
    ) -> Reminder:
        """Create a follow-up reminder"""
        due_at = datetime.now() + timedelta(days=due_in_days)

        return await self.create_reminder(
            title=f"Follow up with {contact_name}",
            description=f"Follow up regarding: {followup_type}",
            due_at=due_at,
            reminder_type=ReminderType.CUSTOM,
            related_contact=contact_name,
        )

    async def get_upcoming_events(self, days: int = 30, limit: int = 20) -> List[Dict]:
        """Get upcoming events"""
        now = datetime.now()
        cutoff = now + timedelta(days=days)

        upcoming = [
            e
            for e in self._events.values()
            if now <= e.scheduled_at <= cutoff and e.status == "scheduled"
        ]

        upcoming.sort(key=lambda x: x.scheduled_at)

        return [
            {
                "id": e.id,
                "title": e.title,
                "type": e.event_type.value,
                "scheduled_at": e.scheduled_at.isoformat(),
                "location": e.location,
                "related_contacts": e.related_contacts,
                "recurring": e.recurring,
            }
            for e in upcoming[:limit]
        ]

    async def get_due_reminders(self) -> List[Dict]:
        """Get reminders that are due"""
        now = datetime.now()

        due = [
            r
            for r in self._reminders.values()
            if not r.dismissed
            and not r.completed
            and (not r.snooze_until or r.snooze_until <= now)
            and r.due_at <= now
        ]

        due.sort(key=lambda x: x.due_at)

        return [
            {
                "id": r.id,
                "title": r.title,
                "description": r.description,
                "due_at": r.due_at.isoformat(),
                "type": r.reminder_type.value,
                "related_contact": r.related_contact,
                "snooze_count": r.snooze_count,
            }
            for r in due
        ]

    async def check_reminders(self, contact_name: str) -> List[Dict]:
        """Check for reminders related to a contact"""
        contact_reminders = [
            r
            for r in self._reminders.values()
            if r.related_contact
            and r.related_contact.lower() == contact_name.lower()
            and not r.dismissed
            and not r.completed
        ]

        return [
            {
                "id": r.id,
                "title": r.title,
                "description": r.description,
                "due_at": r.due_at.isoformat(),
            }
            for r in contact_reminders
        ]

    async def complete_event(self, event_id: str):
        """Mark an event as completed"""
        if event_id in self._events:
            event = self._events[event_id]
            event.status = "completed"
            event.completed_at = datetime.now()

            if event.recurring:
                await self._create_next_recurring_event(event)

            await self._save_events()
            logger.info(f"Completed event: {event.title}")

    async def _create_next_recurring_event(self, event: SocialEvent):
        """Create next occurrence of recurring event"""
        if event.event_type == EventType.BIRTHDAY:
            new_date = event.scheduled_at.replace(year=event.scheduled_at.year + 1)
            # Await directly instead of create_task
            await self.create_event(
                title=event.title,
                event_type=event.event_type,
                scheduled_at=new_date,
                duration_minutes=event.duration_minutes,
                related_contacts=event.related_contacts,
                recurring=True,
                recurrence_pattern=event.recurrence_pattern,
                description=event.description,
            )

    async def dismiss_reminder(self, reminder_id: str):
        """Dismiss a reminder"""
        if reminder_id in self._reminders:
            self._reminders[reminder_id].dismissed = True
            await self._save_reminders()

    async def snooze_reminder(self, reminder_id: str, snooze_hours: int = 1):
        """Snooze a reminder"""
        if reminder_id in self._reminders:
            reminder = self._reminders[reminder_id]
            reminder.snooze_count += 1
            reminder.snooze_until = datetime.now() + timedelta(hours=snooze_hours)
            await self._save_reminders()

    async def complete_reminder(self, reminder_id: str):
        """Mark a reminder as completed"""
        if reminder_id in self._reminders:
            self._reminders[reminder_id].completed = True
            await self._save_reminders()

    async def get_events_by_contact(self, contact_name: str) -> List[SocialEvent]:
        """Get events for a specific contact"""
        return [
            e
            for e in self._events.values()
            if contact_name.lower() in [c.lower() for c in e.related_contacts]
        ]

    async def _check_due_reminders(self):
        """Check for due reminders and trigger callbacks"""
        due = await self.get_due_reminders()
        if due:
            for callback in self._reminder_callbacks:
                try:
                    await callback(due)
                except Exception as e:
                    logger.error(f"Error in reminder callback: {e}")

    def register_reminder_callback(self, callback: Callable):
        """Register a callback for due reminders"""
        self._reminder_callbacks.append(callback)

    async def get_event_summary(self) -> Dict[str, Any]:
        """Get event summary"""
        now = datetime.now()

        upcoming = [
            e
            for e in self._events.values()
            if e.scheduled_at > now and e.status == "scheduled"
        ]

        due_reminders = await self.get_due_reminders()

        event_types = {}
        for event in self._events.values():
            event_types[event.event_type.value] = (
                event_types.get(event.event_type.value, 0) + 1
            )

        return {
            "total_events": len(self._events),
            "upcoming_events": len(upcoming),
            "due_reminders": len(due_reminders),
            "event_types": event_types,
        }


_event_manager: Optional[EventManager] = None


def get_event_manager() -> EventManager:
    """Get or create event manager"""
    global _event_manager
    if _event_manager is None:
        _event_manager = EventManager()
    return _event_manager
