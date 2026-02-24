"""
AURA v3 Proactive Event & Date Tracker
Automatically tracks dates, events, schedules without being asked

POLYMATH INSPIRATION:
- Biology: Memory consolidation (important events get stronger memories)
- Psychology: Prospective memory (remembering to do things in future)
- Economics: Opportunity cost (tracking what's being missed)
- Military: Reconnaissance (continuous scanning for date/event patterns)

This makes AURA PROACTIVE - it learns from context and acts without prompting!
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Priority of tracked events"""

    CRITICAL = 1  # Deadlines, important meetings
    HIGH = 2  # Work events, appointments
    MEDIUM = 3  # Social events, reminders
    LOW = 4  # Optional events
    RECURRING = 5  # Regular events


class EventSource(Enum):
    """Where event was detected"""

    CHAT = "chat"  # From conversation
    CALENDAR = "calendar"  # From calendar app
    EMAIL = "email"  # From email analysis
    SOCIAL = "social"  # From social apps
    CONTEXT = "context"  # From context (location, time)
    NEURAL_PATTERN = "neural"  # From neural pattern recognition


@dataclass
class TrackedEvent:
    """An event AURA is tracking"""

    id: str
    title: str
    description: str

    # Timing
    event_time: datetime
    reminder_time: Optional[datetime]
    duration_minutes: int = 60

    # Priority & Source
    priority: EventPriority = EventPriority.MEDIUM
    source: EventSource = EventSource.CHAT

    # Status
    is_completed: bool = False
    is_cancelled: bool = False
    completion_summary: str = ""

    # Context
    participants: List[str] = field(default_factory=list)
    location: str = ""
    related_entity: str = ""  # Who this is about

    # Neural memory integration
    neural_importance: float = 0.5
    emotional_valence: float = 0.0

    # Tracking metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    reminder_sent: bool = False


class EventRelationship:
    """Relationships between events (what relates to what)"""

    def __init__(self):
        self.related_events: Dict[str, List[str]] = defaultdict(list)
        self.event_chains: Dict[str, List[str]] = defaultdict(list)

    def add_relationship(self, event_id1: str, event_id2: str, relationship_type: str):
        """Add relationship between events"""
        self.related_events[event_id1].append(event_id2)
        self.related_events[event_id2].append(event_id1)

    def get_related(self, event_id: str) -> List[str]:
        """Get related event IDs"""
        return self.related_events.get(event_id, [])


class ProactiveEventTracker:
    """
    PROACTIVE Event Tracker - learns and tracks without being asked

    POLYMATH APPROACH:
    - Biology: Important events get neural priority boost
    - Psychology: Prospective memory for future intentions
    - Economics: Opportunity cost of missed events
    - Military: Early warning system for approaching deadlines

    KEY FEATURES:
    - Auto-detection from chat context
    - Neural importance scoring
    - Smart reminder timing
    - Relationship tracking
    - Completion follow-up
    """

    def __init__(self, neural_memory=None, user_profile=None, knowledge_graph=None):
        self.neural_memory = neural_memory
        self.knowledge_graph = knowledge_graph
        self.user_profile = user_profile

        # Event storage
        self.events: Dict[str, TrackedEvent] = {}
        self.relationships = EventRelationship()

        # Patterns learned
        self.recurring_patterns: Dict[str, Dict] = {}  # event_template -> pattern
        self.participant_patterns: Dict[str, List[str]] = defaultdict(
            list
        )  # person -> event types

        # Settings
        self.default_reminder_minutes = 30
        self.auto_detect_confidence_threshold = 0.6
        self.max_events_tracked = 500

        # Stats
        self.events_completed = 0
        self.events_missed = 0
        self.reminders_sent = 0

    async def initialize(self):
        """Initialize the event tracker - set up background monitoring"""
        logger.info("Initializing ProactiveEventTracker...")

        try:
            if self.neural_memory:
                await self._load_events_from_memory()
        except Exception as e:
            logger.warning(f"Could not load events from memory: {e}")

        logger.info("ProactiveEventTracker initialized")

    async def start(self):
        """Start the proactive event tracker - begin background monitoring"""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("ProactiveEventTracker started")

    async def stop(self):
        """Stop the proactive event tracker"""
        self._running = False
        if hasattr(self, "_monitor_task") and self._monitor_task:
            self._monitor_task.cancel()
        logger.info("ProactiveEventTracker stopped")

    async def _monitor_loop(self):
        """Background loop for monitoring events"""
        while self._running:
            try:
                # Check for upcoming events
                await self._check_upcoming_events()
                # Check for overdue follow-ups
                await self._check_overdue_followups()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in monitor loop: {e}")
            await asyncio.sleep(60)  # Check every minute

    async def _check_upcoming_events(self):
        """Check for events that need reminders"""
        # Implementation would check upcoming events and send reminders
        pass

    async def _check_overdue_followups(self):
        """Check for events that need follow-up"""
        # Implementation would check for missed follow-ups
        pass

    _running = False
    _monitor_task = None

    async def _load_events_from_memory(self):
        """Load tracked events from neural memory"""
        pass

    # =========================================================================
    # AUTO-DETECTION (The "Proactive" Part)
    # =========================================================================

    async def analyze_for_events(self, text: str, context: Dict) -> List[TrackedEvent]:
        """
        Analyze text/context for event-like patterns WITHOUT being asked.
        This is the PROACTIVE part - AURA notices things!
        """
        detected_events = []

        # Extract potential dates/times
        date_patterns = self._extract_date_patterns(text)

        # Check for event keywords
        event_keywords = [
            "meeting",
            "call",
            "appointment",
            "deadline",
            "due",
            "schedule",
            "planned",
            "booked",
            "reserved",
            "remind me",
            "don't forget",
            "make sure",
            "tomorrow",
            "next week",
            "monday",
            "tuesday",
            "at 3",
            "at 5",
            "at noon",
            "evening",
        ]

        has_event_keyword = any(kw in text.lower() for kw in event_keywords)

        if date_patterns or has_event_keyword:
            # Try to extract event
            event = await self._extract_event_from_text(text, date_patterns, context)
            if event:
                detected_events.append(event)

        # Check context for implicit events
        context_events = await self._check_context_for_events(context)
        detected_events.extend(context_events)

        return detected_events

    def _extract_date_patterns(self, text: str) -> List[Dict]:
        """Extract date/time patterns from text"""
        import re

        patterns = []
        text_lower = text.lower()

        # Relative dates
        relative_patterns = [
            (r"tomorrow", 1),
            (r"today", 0),
            (r"next week", 7),
            (r"monday", None),  # Will calculate
            (r"tuesday", None),
            (r"wednesday", None),
            (r"thursday", None),
            (r"friday", None),
            (r"saturday", None),
            (r"sunday", None),
        ]

        now = datetime.now()

        for keyword, days_ahead in relative_patterns:
            if keyword in text_lower:
                if days_ahead is not None:
                    event_date = now + timedelta(days=days_ahead)
                else:
                    # Calculate next weekday
                    weekday_map = {
                        "monday": 0,
                        "tuesday": 1,
                        "wednesday": 2,
                        "thursday": 3,
                        "friday": 4,
                        "saturday": 5,
                        "sunday": 6,
                    }
                    target_weekday = weekday_map.get(keyword)
                    days_until = (target_weekday - now.weekday() + 7) % 7
                    if days_until == 0:
                        days_until = 7  # Next week
                    event_date = now + timedelta(days=days_until)

                patterns.append({"date": event_date, "keyword": keyword})

        # Time patterns
        time_pattern = re.search(r"(\d{1,2}):(\d{2})", text)
        if time_pattern:
            hour = int(time_pattern.group(1))
            minute = int(time_pattern.group(2))
            for pattern in patterns:
                pattern["time"] = f"{hour:02d}:{minute:02d}"

        return patterns

    async def _extract_event_from_text(
        self, text: str, date_patterns: List[Dict], context: Dict
    ) -> Optional[TrackedEvent]:
        """Extract event from text and patterns"""

        if not date_patterns:
            return None

        # Get primary date
        primary = date_patterns[0]
        event_time = primary["date"]

        # Add time if found
        if "time" in primary:
            time_parts = primary["time"].split(":")
            event_time = event_time.replace(
                hour=int(time_parts[0]), minute=int(time_parts[1])
            )
        else:
            # Default to 9 AM if not specified
            event_time = event_time.replace(hour=9, minute=0)

        # Extract title
        title = self._extract_event_title(text)

        # Determine priority from context
        priority = self._determine_priority(text, context)

        # Determine source
        source = EventSource.CHAT

        # Calculate neural importance
        neural_importance = await self._calculate_neural_importance(title, context)

        # Create event
        event = TrackedEvent(
            id=f"evt_{datetime.now().timestamp()}",
            title=title,
            description=text,
            event_time=event_time,
            reminder_time=event_time - timedelta(minutes=self.default_reminder_minutes),
            priority=priority,
            source=source,
            neural_importance=neural_importance,
            participants=self._extract_participants(text),
            location=self._extract_location(text),
        )

        return event

    def _extract_event_title(self, text: str) -> str:
        """Extract event title from text"""

        # Remove common prefixes
        clean_text = text.lower()
        for prefix in [
            "remind me to",
            "remind me that",
            "don't forget to",
            "make sure to",
        ]:
            clean_text = clean_text.replace(prefix, "")

        # Take first meaningful part
        title = clean_text.strip()[:50]

        # Capitalize
        return title.capitalize()

    def _determine_priority(self, text: str, context: Dict) -> EventPriority:
        """Determine event priority"""

        text_lower = text.lower()

        # Critical keywords
        if any(
            kw in text_lower for kw in ["deadline", "due", "must", "critical", "urgent"]
        ):
            return EventPriority.CRITICAL

        # High priority
        if any(
            kw in text_lower for kw in ["meeting", "call", "presentation", "interview"]
        ):
            return EventPriority.HIGH

        # Low priority
        if any(kw in text_lower for kw in ["optional", "maybe", "if possible"]):
            return EventPriority.LOW

        return EventPriority.MEDIUM

    async def _calculate_neural_importance(self, title: str, context: Dict) -> float:
        """Calculate neural importance score (0-1)"""

        importance = 0.5  # Base

        if not self.neural_memory:
            return importance

        try:
            # Check if this type of event is important historically
            neurons = await self.neural_memory.recall(
                query=title, memory_types=["episodic"], limit=3
            )

            if neurons:
                # Average importance of similar events
                avg_importance = sum(n.importance for n in neurons) / len(neurons)
                importance = max(importance, avg_importance)

            # Boost based on emotional valence
            if neurons and hasattr(neurons[0], "emotional_valence"):
                emotion = neurons[0].emotional_valence
                if emotion > 0.3:
                    importance += 0.2
                elif emotion < -0.3:
                    importance -= 0.1

        except Exception as e:
            logger.warning(f"Neural importance calculation error: {e}")

        return min(max(importance, 0.1), 1.0)

    def _extract_participants(self, text: str) -> List[str]:
        """Extract participants from text"""

        import re

        # Simple name extraction (would use NER in production)
        # Look for capitalized words that might be names
        words = text.split()
        potential_names = []

        for i, word in enumerate(words):
            if word and word[0].isupper() and len(word) > 2:
                # Check if it's likely a name (not at sentence start, not common words)
                if i > 0 and word.lower() not in [
                    "the",
                    "a",
                    "an",
                    "this",
                    "that",
                    "meeting",
                    "call",
                ]:
                    potential_names.append(word.strip(".,!?"))

        return potential_names[:5]  # Limit

    def _extract_location(self, text: str) -> str:
        """Extract location from text"""

        location_keywords = ["at", "in", "from", "to"]
        text_lower = text.lower()

        for keyword in location_keywords:
            idx = text_lower.find(keyword)
            if idx >= 0:
                # Extract next few words after keyword
                start = idx + len(keyword) + 1
                end = min(start + 30, len(text))
                location = text[start:end].strip()
                # Clean up
                location = location.rstrip(".,!?")
                if location:
                    return location

        return ""

    async def _check_context_for_events(self, context: Dict) -> List[TrackedEvent]:
        """Check context for implicit events"""

        events = []

        # Check calendar if available
        if context.get("calendar_events"):
            for cal_event in context["calendar_events"]:
                # Already have calendar events - track them
                event = TrackedEvent(
                    id=f"cal_{cal_event.get('id', datetime.now().timestamp())}",
                    title=cal_event.get("title", "Calendar Event"),
                    description=cal_event.get("description", ""),
                    event_time=datetime.fromisoformat(cal_event.get("start", "")),
                    priority=EventPriority.HIGH,
                    source=EventSource.CALENDAR,
                    participants=cal_event.get("attendees", []),
                    location=cal_event.get("location", ""),
                )
                events.append(event)

        return events

    # =========================================================================
    # EVENT MANAGEMENT
    # =========================================================================

    async def track_event(self, event: TrackedEvent) -> str:
        """Add event to tracking"""

        # Check capacity
        if len(self.events) >= self.max_events_tracked:
            # Remove old completed events
            await self._cleanup_old_events()

        self.events[event.id] = event

        # Learn from this event
        await self._learn_event_pattern(event)

        logger.info(f"Tracking event: {event.title} at {event.event_time}")
        return event.id

    async def _learn_event_pattern(self, event: TrackedEvent):
        """Learn patterns from tracked event"""

        # Track recurring patterns
        title_lower = event.title.lower()

        # Check for recurring keywords
        recurring_keywords = ["weekly", "daily", "every", "recurring", "routine"]
        if any(kw in title_lower for kw in recurring_keywords):
            pattern_key = f"recurring_{event.title.lower()}"
            self.recurring_patterns[pattern_key] = {
                "title": event.title,
                "time": event.event_time.strftime("%H:%M"),
                "participants": event.participants,
                "last_seen": datetime.now(),
            }

        # Track participant patterns
        for participant in event.participants:
            self.participant_patterns[participant].append(event.title)

    async def _cleanup_old_events(self):
        """Remove old completed events to make space"""

        # Remove completed events older than 7 days
        cutoff = datetime.now() - timedelta(days=7)

        to_remove = [
            eid
            for eid, event in self.events.items()
            if event.is_completed and event.last_updated < cutoff
        ]

        for eid in to_remove:
            del self.events[eid]

        logger.info(f"Cleaned up {len(to_remove)} old events")

    # =========================================================================
    # REMINDERS (The Proactive Part)
    # =========================================================================

    async def check_and_trigger_reminders(self) -> List[Dict]:
        """Check for events needing reminders"""

        now = datetime.now()
        reminders = []

        for event in self.events.values():
            if event.reminder_sent or event.is_completed or event.is_cancelled:
                continue

            # Check if reminder time has passed
            if event.reminder_time and now >= event.reminder_time:
                # Calculate how late
                minutes_late = (now - event.reminder_time).total_seconds() / 60

                reminder = {
                    "event_id": event.id,
                    "title": event.title,
                    "event_time": event.event_time,
                    "minutes_until": max(
                        0, (event.event_time - now).total_seconds() / 60
                    ),
                    "minutes_late": minutes_late,
                    "priority": event.priority.value,
                    "location": event.location,
                    "participants": event.participants,
                }

                reminders.append(reminder)
                event.reminder_sent = True
                self.reminders_sent += 1

        return reminders

    async def get_upcoming_events(self, hours: int = 24) -> List[TrackedEvent]:
        """Get events in next N hours"""

        now = datetime.now()
        cutoff = now + timedelta(hours=hours)

        upcoming = [
            event
            for event in self.events.values()
            if now <= event.event_time <= cutoff
            and not event.is_completed
            and not event.is_cancelled
        ]

        # Sort by time
        upcoming.sort(key=lambda e: e.event_time)

        return upcoming

    async def get_daily_summary(self) -> Dict:
        """Get summary of today's events"""

        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)

        today_events = [
            e for e in self.events.values() if today_start <= e.event_time < today_end
        ]

        completed = [e for e in today_events if e.is_completed]
        upcoming = [e for e in today_events if not e.is_completed]

        # Calculate opportunity cost
        missed = [
            e
            for e in today_events
            if e.is_cancelled or (e.event_time < now and not e.is_completed)
        ]

        return {
            "date": now.strftime("%Y-%m-%d"),
            "total_events": len(today_events),
            "completed": len(completed),
            "upcoming": len(upcoming),
            "missed": len(missed),
            "completion_rate": len(completed) / max(len(today_events), 1),
            "events": [
                {
                    "title": e.title,
                    "time": e.event_time.strftime("%H:%M"),
                    "priority": e.priority.name,
                    "completed": e.is_completed,
                }
                for e in today_events
            ],
        }

    # =========================================================================
    # EVENT COMPLETION
    # =========================================================================

    async def complete_event(self, event_id: str, summary: str = "") -> bool:
        """Mark event as completed"""

        if event_id not in self.events:
            return False

        event = self.events[event_id]
        event.is_completed = True
        event.completion_summary = summary
        event.last_updated = datetime.now()

        # Strengthen neural memory
        if self.neural_memory:
            await self._strengthen_event_memory(event)

        self.events_completed += 1

        # Check for related events
        related = self.relationships.get_related(event_id)
        if related:
            # Could trigger follow-up events
            pass

        return True

    async def cancel_event(self, event_id: str, reason: str = "") -> bool:
        """Cancel an event"""

        if event_id not in self.events:
            return False

        event = self.events[event_id]
        event.is_cancelled = True
        event.completion_summary = reason
        event.last_updated = datetime.now()

        # Weaken neural memory
        if self.neural_memory:
            await self._weaken_event_memory(event)

        self.events_missed += 1

        return True

    async def _strengthen_event_memory(self, event: TrackedEvent):
        """Strengthen memory after successful completion"""

        try:
            await self.neural_memory.learn(
                content=f"Completed: {event.title}",
                memory_type="episodic",
                importance=event.neural_importance,
                emotional_valence=0.3,  # Positive completion
            )
        except Exception as e:
            logger.warning(f"Memory strengthening error: {e}")

    async def _weaken_event_memory(self, event: TrackedEvent):
        """Weaken memory after cancellation"""

        try:
            await self.neural_memory.learn(
                content=f"Cancelled: {event.title}",
                memory_type="episodic",
                importance=max(event.neural_importance - 0.2, 0.1),
                emotional_valence=-0.2,  # Slight negative
            )
        except Exception as e:
            logger.warning(f"Memory weakening error: {e}")

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get tracking statistics"""

        total = len(self.events)
        completed = sum(1 for e in self.events.values() if e.is_completed)
        cancelled = sum(1 for e in self.events.values() if e.is_cancelled)

        return {
            "total_tracked": total,
            "completed": completed,
            "cancelled": cancelled,
            "active": total - completed - cancelled,
            "completion_rate": completed / max(total - cancelled, 1),
            "miss_rate": self.events_missed
            / max(self.events_completed + self.events_missed, 1),
            "reminders_sent": self.reminders_sent,
            "recurring_patterns": len(self.recurring_patterns),
        }


# Factory function
def create_event_tracker(
    neural_memory=None, knowledge_graph=None
) -> ProactiveEventTracker:
    """Create proactive event tracker"""
    return ProactiveEventTracker(neural_memory, knowledge_graph)
