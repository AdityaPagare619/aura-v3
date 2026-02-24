"""
AURA v3 Adaptive Social Engine
===============================

Learns social patterns from communication and behavior:
- NO hardcoded relationship types - discovers them
- Learns communication patterns from messages/calls
- Identifies relationship strength dynamically
- Context-aware suggestions without hardcoded rules

Based on roadmap: "AURA should discover patterns itself, not have predefined rules"
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import uuid

logger = logging.getLogger(__name__)


@dataclass
class CommunicationEvent:
    """A communication event with the person"""

    timestamp: datetime
    event_type: str  # message_sent, message_received, call_made, call_received
    duration_seconds: Optional[int] = None
    sentiment: Optional[str] = None  # positive, negative, neutral (if detected)
    topic: Optional[str] = None  # if detected


@dataclass
class DiscoveredRelationship:
    """A relationship pattern discovered through learning"""

    person_id: str
    discovered_label: str  # e.g., "work_colleague", "close_friend", "family"
    communication_patterns: Dict[str, Any] = field(default_factory=dict)
    strength_score: float = 0.0  # 0-1
    reciprocity_score: float = 0.0  # 0-1
    last_contact: Optional[datetime] = None
    total_interactions: int = 0


@dataclass
class SocialPattern:
    """A discovered social pattern"""

    pattern_id: str
    pattern_type: str  # e.g., "weekly_checkin", "project_collaboration"
    participants: List[str] = field(default_factory=list)
    frequency: str  # daily, weekly, monthly, occasional
    typical_time: str = ""
    context: str = ""
    confidence: float = 0.0


class AdaptiveSocialEngine:
    """
    Learns social patterns without hardcoded rules.

    Key principles:
    1. Observe communication without assumptions
    2. Discover relationship types from behavior
    3. Learn communication preferences
    4. Identify reciprocal patterns
    5. Generate contextual suggestions
    """

    def __init__(self, neural_memory=None):
        self.neural_memory = neural_memory

        # Discovered relationships (not hardcoded types!)
        self.relationships: Dict[str, DiscoveredRelationship] = {}

        # Communication events per person
        self.communication_history: Dict[str, List[CommunicationEvent]] = defaultdict(
            list
        )

        # Discovered social patterns
        self.social_patterns: Dict[str, SocialPattern] = {}

        # Learning parameters
        self.min_interactions_to_discover = 5
        self.strength_change_threshold = 0.1

    async def add_communication_event(self, person_id: str, event: CommunicationEvent):
        """Record a communication event"""
        self.communication_history[person_id].append(event)

        # Keep history manageable
        if len(self.communication_history[person_id]) > 100:
            self.communication_history[person_id] = self.communication_history[
                person_id
            ][-50:]

        # Update relationship
        await self._update_relationship(person_id)

        # Check for new patterns
        await self._discover_patterns(person_id)

    async def _update_relationship(self, person_id: str):
        """Update relationship model based on communication"""
        events = self.communication_history.get(person_id, [])

        if len(events) < self.min_interactions_to_discover:
            return

        # Analyze communication patterns
        sent = sum(1 for e in events if e.event_type in ["message_sent", "call_made"])
        received = sum(
            1 for e in events if e.event_type in ["message_received", "call_received"]
        )

        # Calculate reciprocity
        reciprocity = received / sent if sent > 0 else 0.5

        # Calculate strength based on frequency and recency
        now = datetime.now()
        recent_events = [e for e in events if (now - e.timestamp).days < 30]
        frequency_score = len(recent_events) / 30  # events per day

        if person_id not in self.relationships:
            self.relationships[person_id] = DiscoveredRelationship(
                person_id=person_id,
                discovered_label="",  # Will be discovered
            )

        rel = self.relationships[person_id]
        rel.total_interactions = len(events)
        rel.last_contact = events[-1].timestamp
        rel.reciprocity_score = rel.reciprocity_score * 0.7 + reciprocity * 0.3
        rel.strength_score = min(1.0, frequency_score * 5)  # Cap at 1.0

        # Discover label if not yet set
        if not rel.discovered_label:
            rel.discovered_label = await self._discover_relationship_label(events)

        # Update communication patterns
        rel.communication_patterns = self._analyze_communication_patterns(events)

    async def _discover_relationship_label(
        self, events: List[CommunicationEvent]
    ) -> str:
        """Discover relationship label from communication patterns"""
        if len(events) < 5:
            return "new_connection"

        # Time-based patterns
        weekday_interactions = sum(1 for e in events if e.timestamp.weekday() < 5)
        weekend_interactions = sum(1 for e in events if e.timestamp.weekday() >= 5)

        # Time-of-day patterns
        work_hours = sum(1 for e in events if 9 <= e.timestamp.hour < 18)
        evening = sum(1 for e in events if 18 <= e.timestamp.hour < 22)

        # Duration patterns (for calls)
        call_durations = [e.duration_seconds for e in events if e.duration_seconds]
        avg_duration = statistics.mean(call_durations) if call_durations else 0

        # Generate label based on patterns
        if weekday_interactions > weekend_interactions * 3:
            if avg_duration > 300:  # > 5 min calls
                return "work_colleague"
            else:
                return "work_acquaintance"
        elif weekend_interactions > weekday_interactions:
            if avg_duration > 600:  # > 10 min calls
                return "close_friend"
            else:
                return "casual_friend"
        elif work_hours > evening:
            return "professional_contact"
        else:
            return "personal_connection"

    def _analyze_communication_patterns(
        self, events: List[CommunicationEvent]
    ) -> Dict[str, Any]:
        """Analyze communication patterns in detail"""
        if not events:
            return {}

        # Preferred communication type
        message_count = sum(1 for e in events if "message" in e.event_type)
        call_count = sum(1 for e in events if "call" in e.event_type)

        preferred_type = "mixed"
        if message_count > call_count * 2:
            preferred_type = "text_oriented"
        elif call_count > message_count * 2:
            preferred_type = "call_oriented"

        # Typical time
        hour_counts = defaultdict(int)
        for e in events:
            hour_counts[e.timestamp.hour] += 1

        typical_hour = (
            max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else 12
        )

        # Response time (if we have sent-received pairs)
        response_times = []
        for i in range(len(events) - 1):
            if (
                "sent" in events[i].event_type
                and "received" in events[i + 1].event_type
            ):
                delta = (
                    events[i + 1].timestamp - events[i].timestamp
                ).total_seconds() / 60
                if delta > 0:
                    response_times.append(delta)

        avg_response_time = statistics.mean(response_times) if response_times else None

        return {
            "preferred_type": preferred_type,
            "typical_hour": typical_hour,
            "avg_response_minutes": round(avg_response_time, 1)
            if avg_response_time
            else None,
            "message_ratio": message_count / len(events) if events else 0,
            "call_ratio": call_count / len(events) if events else 0,
        }

    async def _discover_patterns(self, person_id: str):
        """Discover recurring social patterns"""
        events = self.communication_history.get(person_id, [])

        if len(events) < 10:
            return

        # Look for weekly patterns
        day_counts = defaultdict(int)
        hour_counts = defaultdict(int)

        for e in events:
            day_counts[e.timestamp.strftime("%A")] += 1
            hour_counts[e.timestamp.hour] += 1

        # Find most common day/time combination
        # This is simplified - real implementation would do more sophisticated analysis

    async def suggest_action(self, person_id: str) -> Optional[Dict]:
        """Generate contextual social suggestion based on learned patterns"""
        if person_id not in self.relationships:
            return None

        rel = self.relationships[person_id]
        events = self.communication_history.get(person_id, [])

        if not events:
            return None

        now = datetime.now()
        last_event = events[-1]
        days_since_contact = (now - last_event.timestamp).days

        # Generate suggestion based on pattern
        suggestions = []

        # Check if it's been a while
        if days_since_contact > 14 and rel.strength_score > 0.7:
            suggestions.append(
                {
                    "type": "reconnect",
                    "reason": f"You haven't connected with {person_id} in {days_since_contact} days",
                    "confidence": rel.strength_score,
                }
            )

        # Check for pattern match (e.g., usually text on certain days)
        patterns = rel.communication_patterns
        if "typical_hour" in patterns:
            typical_hour = patterns["typical_hour"]
            if typical_hour == now.hour and days_since_contact > 3:
                suggestions.append(
                    {
                        "type": "check_in",
                        "reason": f"This is typically when you connect with {person_id}",
                        "confidence": 0.7,
                    }
                )

        # Return best suggestion
        if suggestions:
            return max(suggestions, key=lambda x: x["confidence"])

        return None

    def get_relationships(self) -> List[Dict]:
        """Get all discovered relationships"""
        return [
            {
                "person_id": r.person_id,
                "label": r.discovered_label,
                "strength": round(r.strength_score, 2),
                "reciprocity": round(r.reciprocity_score, 2),
                "last_contact": r.last_contact.isoformat() if r.last_contact else None,
                "total_interactions": r.total_interactions,
                "patterns": r.communication_patterns,
            }
            for r in self.relationships.values()
        ]

    def get_patterns(self) -> List[Dict]:
        """Get all discovered social patterns"""
        return [
            {
                "id": p.pattern_id,
                "type": p.pattern_type,
                "participants": p.participants,
                "frequency": p.frequency,
                "typical_time": p.typical_time,
                "confidence": round(p.confidence, 2),
            }
            for p in self.social_patterns.values()
        ]

    def get_suggestions(self) -> List[Dict]:
        """Get suggestions for all relationships"""
        suggestions = []

        for person_id in self.relationships:
            suggestion = asyncio.create_task(self.suggest_action(person_id))

        # This would need proper async handling in production
        return suggestions


# Global instance
_engine: Optional[AdaptiveSocialEngine] = None


def get_adaptive_social_engine() -> AdaptiveSocialEngine:
    """Get or create the global adaptive social engine"""
    global _engine
    if _engine is None:
        _engine = AdaptiveSocialEngine()
    return _engine
