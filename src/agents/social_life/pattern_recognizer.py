"""
AURA v3 Pattern Recognizer
Identifies patterns in social behavior
Uses learning engine for adaptive pattern detection
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of social patterns"""

    TEMPORAL = "temporal"
    COMMUNICATION = "communication"
    EMOTIONAL = "emotional"
    RELATIONSHIP = "relationship"
    BEHAVIORAL = "behavioral"
    TRIGGER = "trigger"


@dataclass
class SocialPattern:
    """A detected social pattern"""

    id: str
    pattern_type: PatternType
    name: str
    description: str

    trigger_conditions: Dict[str, Any] = field(default_factory=dict)
    frequency: float = 0.0
    confidence: float = 0.0

    first_detected: datetime = field(default_factory=datetime.now)
    last_detected: datetime = field(default_factory=datetime.now)
    occurrence_count: int = 0

    related_contacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternInstance:
    """An instance of a pattern occurring"""

    pattern_id: str
    detected_at: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


class PatternRecognizer:
    """
    Recognizes patterns in social behavior
    Uses temporal, communication, emotional, and behavioral analysis
    """

    def __init__(self, data_dir: str = "data/social_life/patterns"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._patterns: Dict[str, SocialPattern] = {}
        self._pattern_instances: List[PatternInstance] = []

        self._temporal_buffers: Dict[str, List[datetime]] = defaultdict(list)
        self._communication_buffers: Dict[str, List[Dict]] = defaultdict(list)

        self._min_occurrences = 3
        self._pattern_window_days = 30

    async def initialize(self):
        """Initialize pattern recognizer"""
        logger.info("Initializing Pattern Recognizer...")
        await self._load_patterns()
        await self._setup_default_patterns()
        logger.info("Pattern Recognizer initialized")

    async def _load_patterns(self):
        """Load saved patterns"""
        patterns_file = self.data_dir / "patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, "r") as f:
                    data = json.load(f)
                    for p_data in data.get("patterns", []):
                        p_data["first_detected"] = datetime.fromisoformat(
                            p_data["first_detected"]
                        )
                        p_data["last_detected"] = datetime.fromisoformat(
                            p_data["last_detected"]
                        )
                        self._patterns[p_data["id"]] = SocialPattern(**p_data)
                logger.info(f"Loaded {len(self._patterns)} patterns")
            except Exception as e:
                logger.error(f"Error loading patterns: {e}")

    async def _save_patterns(self):
        """Save patterns to disk"""
        patterns_file = self.data_dir / "patterns.json"
        try:
            data = {
                "patterns": [
                    {
                        **vars(p),
                        "first_detected": p.first_detected.isoformat(),
                        "last_detected": p.last_detected.isoformat(),
                    }
                    for p in self._patterns.values()
                ]
            }
            with open(patterns_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")

    async def _setup_default_patterns(self):
        """Set up default pattern templates"""
        default_patterns = [
            SocialPattern(
                id="weekend_connector",
                pattern_type=PatternType.TEMPORAL,
                name="Weekend Connector",
                description="Tends to connect with specific contacts on weekends",
                trigger_conditions={"day_of_week": [5, 6]},
                confidence=0.5,
            ),
            SocialPattern(
                id="morning_person",
                pattern_type=PatternType.TEMPORAL,
                name="Morning Person",
                description="Most active in morning hours",
                trigger_conditions={"hour_range": (6, 12)},
                confidence=0.5,
            ),
            SocialPattern(
                id="night_owl",
                pattern_type=PatternType.TEMPORAL,
                name="Night Owl",
                description="Most active in late night hours",
                trigger_conditions={"hour_range": (22, 6)},
                confidence=0.5,
            ),
            SocialPattern(
                id="quick_responder",
                pattern_type=PatternType.COMMUNICATION,
                name="Quick Responder",
                description="Responds quickly to messages",
                trigger_conditions={"max_response_minutes": 30},
                confidence=0.5,
            ),
            SocialPattern(
                id="delayed_responder",
                pattern_type=PatternType.COMMUNICATION,
                name="Delayed Responder",
                description="Takes longer to respond",
                trigger_conditions={"min_response_hours": 24},
                confidence=0.5,
            ),
            SocialPattern(
                id="family_checkin",
                pattern_type=PatternType.RELATIONSHIP,
                name="Family Check-in",
                description="Regular family contact pattern",
                trigger_conditions={"contact_category": "family"},
                confidence=0.5,
            ),
            SocialPattern(
                id="work_colleague",
                pattern_type=PatternType.RELATIONSHIP,
                name="Work Colleague",
                description="Work-related communication pattern",
                trigger_conditions={"contact_category": "work"},
                confidence=0.5,
            ),
        ]

        for pattern in default_patterns:
            if pattern.id not in self._patterns:
                self._patterns[pattern.id] = pattern

    async def record_temporal_event(
        self, contact: str, timestamp: datetime, event_type: str
    ):
        """Record a temporal event for pattern detection"""
        key = f"{contact}:{event_type}"
        self._temporal_buffers[key].append(timestamp)

        if len(self._temporal_buffers[key]) >= self._min_occurrences:
            await self._check_temporal_pattern(contact, event_type)

    async def record_communication(
        self, contact: str, message: str, timestamp: datetime, is_outgoing: bool
    ):
        """Record communication for pattern detection"""
        self._communication_buffers[contact].append(
            {
                "message": message,
                "timestamp": timestamp,
                "is_outgoing": is_outgoing,
            }
        )

        if len(self._communication_buffers[contact]) >= self._min_occurrences:
            await self._check_communication_pattern(contact)

    async def _check_temporal_pattern(self, contact: str, event_type: str):
        """Check for temporal patterns"""
        key = f"{contact}:{event_type}"
        timestamps = self._temporal_buffers[key]

        if len(timestamps) < self._min_occurrences:
            return

        weekday_counts = defaultdict(int)
        hour_counts = defaultdict(int)

        for ts in timestamps:
            weekday_counts[ts.weekday()] += 1
            hour_counts[ts.hour] += 1

        total = len(timestamps)

        for weekday, count in weekday_counts.items():
            if count / total >= 0.6:
                pattern_id = f"weekend_contact_{contact}"
                await self._update_or_create_pattern(
                    pattern_id=pattern_id,
                    name=f"Weekend Contact - {contact}",
                    pattern_type=PatternType.TEMPORAL,
                    contact=contact,
                    conditions={"day_of_week": weekday, "frequency": count / total},
                )

        for hour, count in hour_counts.items():
            if count / total >= 0.5:
                if 6 <= hour < 12:
                    pattern_id = f"morning_contact_{contact}"
                    await self._update_or_create_pattern(
                        pattern_id=pattern_id,
                        name=f"Morning Contact - {contact}",
                        pattern_type=PatternType.TEMPORAL,
                        contact=contact,
                        conditions={"hour_range": "morning"},
                    )
                elif 22 <= hour or hour < 6:
                    pattern_id = f"night_contact_{contact}"
                    await self._update_or_create_pattern(
                        pattern_id=pattern_id,
                        name=f"Night Contact - {contact}",
                        pattern_type=PatternType.TEMPORAL,
                        contact=contact,
                        conditions={"hour_range": "night"},
                    )

    async def _check_communication_pattern(self, contact: str):
        """Check for communication patterns"""
        messages = self._communication_buffers[contact]

        if len(messages) < 2:
            return

        outgoing_times = []
        incoming_times = []

        for i, msg in enumerate(messages):
            if msg["is_outgoing"]:
                outgoing_times.append(msg["timestamp"])
            else:
                incoming_times.append(msg["timestamp"])

        if outgoing_times and incoming_times:
            response_times = []
            for in_time in incoming_times:
                next_out = None
                for out_time in outgoing_times:
                    if out_time > in_time:
                        next_out = out_time
                        break
                if next_out:
                    diff = (next_out - in_time).total_seconds() / 60
                    if 0 < diff < 1440:
                        response_times.append(diff)

            if response_times:
                avg_response = statistics.mean(response_times)

                if avg_response < 30:
                    pattern_id = f"quick_response_{contact}"
                    await self._update_or_create_pattern(
                        pattern_id=pattern_id,
                        name=f"Quick Responder - {contact}",
                        pattern_type=PatternType.COMMUNICATION,
                        contact=contact,
                        conditions={"avg_response_minutes": avg_response},
                    )
                elif avg_response > 1440:
                    pattern_id = f"delayed_response_{contact}"
                    await self._update_or_create_pattern(
                        pattern_id=pattern_id,
                        name=f"Delayed Responder - {contact}",
                        pattern_type=PatternType.COMMUNICATION,
                        contact=contact,
                        conditions={"avg_response_hours": avg_response / 60},
                    )

    async def _update_or_create_pattern(
        self,
        pattern_id: str,
        name: str,
        pattern_type: PatternType,
        contact: str,
        conditions: Dict[str, Any],
    ):
        """Update or create a pattern"""
        now = datetime.now()

        if pattern_id in self._patterns:
            pattern = self._patterns[pattern_id]
            pattern.occurrence_count += 1
            pattern.last_detected = now
            pattern.frequency = pattern.occurrence_count / max(
                1, (now - pattern.first_detected).days
            )
            pattern.confidence = min(pattern.confidence + 0.1, 1.0)
        else:
            pattern = SocialPattern(
                id=pattern_id,
                pattern_type=pattern_type,
                name=name,
                description=f"Auto-detected pattern for {contact}",
                trigger_conditions=conditions,
                first_detected=now,
                last_detected=now,
                occurrence_count=1,
                related_contacts=[contact],
            )
            self._patterns[pattern_id] = pattern

        await self._save_patterns()

    async def detect_patterns(self) -> Dict[str, List[SocialPattern]]:
        """Detect all patterns"""
        detected = {
            "temporal": [],
            "communication": [],
            "emotional": [],
            "relationship": [],
            "behavioral": [],
            "trigger": [],
        }

        for pattern in self._patterns.values():
            if pattern.occurrence_count >= self._min_occurrences:
                detected[pattern.pattern_type.value].append(pattern)

        return detected

    async def check_pattern_triggers(
        self, contact: str, message: str
    ) -> List[Dict[str, Any]]:
        """Check if any patterns are triggered by a message"""
        triggers = []
        message_lower = message.lower()

        for pattern in self._patterns.values():
            if contact not in pattern.related_contacts:
                continue

            triggered = False

            if pattern.pattern_type == PatternType.TRIGGER:
                trigger_words = pattern.trigger_conditions.get("keywords", [])
                if any(word in message_lower for word in trigger_words):
                    triggered = True

            if triggered:
                triggers.append(
                    {
                        "pattern_id": pattern.id,
                        "pattern_name": pattern.name,
                        "confidence": pattern.confidence,
                        "context": pattern.trigger_conditions,
                    }
                )

        return triggers

    async def get_patterns_for_contact(self, contact: str) -> List[SocialPattern]:
        """Get all patterns related to a contact"""
        return [p for p in self._patterns.values() if contact in p.related_contacts]

    async def get_strongest_patterns(self, limit: int = 5) -> List[SocialPattern]:
        """Get the strongest patterns"""
        sorted_patterns = sorted(
            self._patterns.values(),
            key=lambda p: (p.confidence, p.occurrence_count),
            reverse=True,
        )
        return sorted_patterns[:limit]

    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get pattern statistics"""
        pattern_types = defaultdict(int)
        total_occurrences = 0
        high_confidence = 0

        for pattern in self._patterns.values():
            pattern_types[pattern.pattern_type.value] += 1
            total_occurrences += pattern.occurrence_count
            if pattern.confidence >= 0.7:
                high_confidence += 1

        return {
            "total_patterns": len(self._patterns),
            "pattern_types": dict(pattern_types),
            "total_occurrences": total_occurrences,
            "high_confidence_patterns": high_confidence,
            "average_confidence": statistics.mean(
                [p.confidence for p in self._patterns.values()]
            )
            if self._patterns
            else 0,
        }


_pattern_recognizer: Optional[PatternRecognizer] = None


def get_pattern_recognizer() -> PatternRecognizer:
    """Get or create pattern recognizer"""
    global _pattern_recognizer
    if _pattern_recognizer is None:
        _pattern_recognizer = PatternRecognizer()
    return _pattern_recognizer
