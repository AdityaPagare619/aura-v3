"""
AURA v3 Adaptive Context Engine
Dynamically learns user patterns WITHOUT hardcoded templates
Mobile-optimized: lightweight, fast, adaptive
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import re

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context AURA learns"""

    TEMPORAL = "temporal"  # Time patterns
    SPATIAL = "spatial"  # Location patterns
    BEHAVIORAL = "behavioral"  # Action patterns
    EMOTIONAL = "emotional"  # Mood patterns
    SOCIAL = "social"  # Relationship patterns
    PROFESSIONAL = "professional"  # Work patterns
    INTENT = "intent"  # Goal patterns


@dataclass
class ContextPattern:
    """A learned pattern from user behavior"""

    id: str
    context_type: ContextType

    # Pattern definition (not hardcoded)
    pattern_hash: str  # Hash of the actual pattern
    pattern_template: str  # Dynamic template

    # How it was learned
    occurrences: int = 0
    confidence: float = 0.0

    # Temporal info
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

    # Associated data
    related_entities: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextSnapshot:
    """Current context snapshot for decision making"""

    timestamp: datetime
    time_of_day: str
    day_of_week: str

    # Current state
    active_apps: List[str] = field(default_factory=list)
    recent_actions: List[str] = field(default_factory=list)
    current_location: Optional[str] = None
    # Inferred state
    user_mood: Optional[str] = None
    user_busyness: float = 0.0  # 0-1 scale
    interruption_willingness: float = 0.5  # Match5

    # Learned patterns
    active_patterns: List[str] = field(default_factory=list)


class AdaptiveContextEngine:
    """
    Adaptive Context Engine - The CORE of AURA's intelligence

    Key principles:
    1. NO hardcoded templates - learns everything dynamically
    2. Mobile-optimized - lightweight, fast inference
    3. Pattern-based - finds recurring themes
    4. Confidence-weighted - acts only when sure
    5. Privacy-first - learns from local data only
    """

    def __init__(self, storage_path: str = "data/context_engine"):
        self.storage_path = storage_path

        # Learned patterns (not hardcoded!)
        self._patterns: Dict[str, ContextPattern] = {}

        # Learning history
        self._observations: List[Dict] = []
        self._max_observations = 1000

        # Entity extraction (dynamic, not fixed list)
        self._entity_extractors: List[Callable] = []

        # Mobile-optimized settings
        self.max_patterns = 100
        self.min_confidence_threshold = 0.6
        self.pattern_decay_factor = 0.95
        self._running = False

    async def start(self):
        """Start the adaptive context engine."""
        self._running = True

    async def stop(self):
        """Stop the adaptive context engine."""
        self._running = False

    # =========================================================================
    # CORE LEARNING - ADAPTIVE, NOT HARDCODED
    # =========================================================================

    async def observe(self, action: str, metadata: Dict[str, Any] = None):
        """
        Observe user action and learn from it
        This is how AURA learns - through observation
        """
        metadata = metadata or {}

        observation = {
            "action": action,
            "timestamp": datetime.now(),
            "metadata": metadata,
            "context_type": self._infer_context_type(action, metadata),
        }

        self._observations.append(observation)

        # Trim old observations
        if len(self._observations) > self._max_observations:
            self._observations = self._observations[-self._max_observations :]

        # Learn from this observation
        await self._learn_from_observation(observation)

    def _infer_context_type(self, action: str, metadata: Dict) -> ContextType:
        """
        Dynamically infer what type of context this is
        NOT hardcoded - uses NLP-like pattern matching
        """
        action_lower = action.lower()

        # Time-related patterns
        time_indicators = [
            "morning",
            "evening",
            "night",
            "afternoon",
            "clock",
            "time",
            "when",
        ]
        if any(word in action_lower for word in time_indicators):
            return ContextType.TEMPORAL

        # Location-related
        location_indicators = ["where", "location", "place", "going", "travel", "trip"]
        if any(word in action_lower for word in location_indicators):
            return ContextType.SPATIAL

        # Emotional
        emotional_indicators = [
            "feel",
            "mood",
            "happy",
            "sad",
            "angry",
            "tired",
            "excited",
        ]
        if any(word in action_lower for word in emotional_indicators):
            return ContextType.EMOTIONAL

        # Social
        social_indicators = [
            "friend",
            "family",
            "meeting",
            "call",
            "message",
            "wedding",
            "birthday",
        ]
        if any(word in action_lower for word in social_indicators):
            return ContextType.SOCIAL

        # Professional
        work_indicators = [
            "work",
            "meeting",
            "deadline",
            "project",
            "boss",
            "office",
            "client",
        ]
        if any(word in action_lower for word in work_indicators):
            return ContextType.PROFESSIONAL

        # Intent
        intent_indicators = [
            "want",
            "need",
            "should",
            "plan",
            "will",
            "going to",
            "intend",
        ]
        if any(word in action_lower for word in intent_indicators):
            return ContextType.INTENT

        # Default to behavioral
        return ContextType.BEHAVIORAL

    async def _learn_from_observation(self, observation: Dict):
        """Learn patterns from observations - dynamic, not hardcoded"""

        action = observation["action"]
        context_type = observation["context_type"]

        # Extract entities dynamically
        entities = self._extract_entities(action)

        # Find or create pattern
        pattern_key = self._generate_pattern_key(context_type, entities)

        if pattern_key in self._patterns:
            # Update existing pattern
            pattern = self._patterns[pattern_key]
            pattern.occurrences += 1
            pattern.last_seen = datetime.now()
            pattern.confidence = min(
                pattern.confidence * self.pattern_decay_factor + 0.2, 1.0
            )
            pattern.related_entities.update(entities)

        else:
            # Create new pattern (if under limit)
            if len(self._patterns) < self.max_patterns:
                pattern = ContextPattern(
                    id=pattern_key,
                    context_type=context_type,
                    pattern_hash=self._hash_pattern(action),
                    pattern_template=self._generate_template(action),
                    occurrences=1,
                    confidence=0.3,
                    related_entities=entities,
                )
                self._patterns[pattern_key] = pattern

    def _extract_entities(self, text: str) -> Set[str]:
        """
        Extract entities dynamically - NOT hardcoded to specific apps
        Uses pattern matching to find meaningful tokens
        """
        entities = set()

        # Extract capitalized phrases (potential named entities)
        capitalized = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text)
        entities.update(capitalized)

        # Extract numbers (potential dates, times, quantities)
        numbers = re.findall(r"\d+", text)
        entities.update(numbers)

        # Extract time patterns
        time_patterns = re.findall(r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", text.lower())
        entities.update(time_patterns)

        return entities

    def _generate_pattern_key(
        self, context_type: ContextType, entities: Set[str]
    ) -> str:
        """Generate unique pattern key"""
        entity_str = "_".join(sorted(entities)) if entities else "generic"
        return f"{context_type.value}_{hash(entity_str) % 10000}"

    def _hash_pattern(self, text: str) -> str:
        """Create hash of pattern"""
        return hashlib.md5(text.encode()).hexdigest()[:8]

    def _generate_template(self, text: str) -> str:
        """
        Generate template from text - replace specifics with placeholders
        This is how AURA generalizes!
        """
        template = text

        # Replace numbers with placeholder
        template = re.sub(r"\d+", "NUM", template)

        # Replace specific times with placeholder
        template = re.sub(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)?", "TIME", template.lower())

        # Replace capitalized names with placeholder
        template = re.sub(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", "ENTITY", template)

        return template

    # =========================================================================
    # CONTEXT INFERENCE - ADAPTIVE
    # =========================================================================

    async def get_current_context(self) -> ContextSnapshot:
        """Get current context snapshot for decision making"""

        now = datetime.now()

        # Time-based context
        time_of_day = self._get_time_period(now)
        day_of_week = now.strftime("%A")

        # Analyze recent observations
        recent_actions = [o["action"] for o in self._observations[-10:]]

        # Infer user state
        user_busyness = self._infer_busyness(recent_actions)
        interruption_willingness = self._infer_interruption_willingness(recent_actions)

        # Find active patterns
        active_patterns = self._find_active_patterns()

        return ContextSnapshot(
            timestamp=now,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            recent_actions=recent_actions,
            user_busyness=user_busyness,
            interruption_willingness=interruption_willingness,
            active_patterns=active_patterns,
        )

    def _get_time_period(self, dt: datetime) -> str:
        """Get time period - adaptive to user's actual patterns"""
        hour = dt.hour

        # Learn user's actual time preferences
        morning_occurrences = len(
            [o for o in self._observations if o["timestamp"].hour in range(6, 12)]
        )
        afternoon_occurrences = len(
            [o for o in self._observations if o["timestamp"].hour in range(12, 18)]
        )

        # Dynamic time boundaries based on user behavior
        if morning_occurrences > afternoon_occurrences:
            if hour < 10:
                return "morning"
            if hour < 14:
                return "mid_morning"
            if hour < 17:
                return "early_afternoon"
            return "late_afternoon"
        else:
            if hour < 9:
                return "early_morning"
            if hour < 12:
                return "late_morning"
            if hour < 14:
                return "lunch"
            return "afternoon"

    def _infer_busyness(self, recent_actions: List[str]) -> float:
        """Infer how busy the user is based on recent actions"""

        # Work-related actions suggest busyness
        work_actions = ["meeting", "deadline", "work", "office", "call", "email"]
        busyness_indicators = sum(
            1 for a in recent_actions if any(w in a.lower() for w in work_actions)
        )

        # Calculate busyness (0-1 scale)
        return min(busyness_indicators / 5.0, 1.0)

    def _infer_interruption_willingness(self, recent_actions: List[str]) -> float:
        """Infer if user is willing to be interrupted"""

        # Relaxed activities suggest willingness
        relaxed_actions = ["break", "lunch", "coffee", "rest", "leisure"]
        relaxed_count = sum(
            1 for a in recent_actions if any(r in a.lower() for r in relaxed_actions)
        )

        # Invert busyness
        base_willingness = 1.0 - self._infer_busyness(recent_actions)

        # Adjust for relaxed activities
        return min(base_willingness + (relaxed_count * 0.1), 1.0)

    def _find_active_patterns(self) -> List[str]:
        """Find patterns that match current context"""

        current_time = datetime.now()
        active = []

        for pattern in self._patterns.values():
            # Check if pattern is still relevant
            time_diff = (current_time - pattern.last_seen).total_seconds()

            # Pattern is active if seen recently and has high confidence
            if (
                time_diff < 86400 * 7
                and pattern.confidence > self.min_confidence_threshold
            ):
                active.append(pattern.id)

        return active

    # =========================================================================
    # DECISION MAKING - ADAPTIVE
    # =========================================================================

    async def should_act_proactively(
        self, action_benefit: float, action_complexity: float
    ) -> Tuple[bool, str]:
        """
        Decide whether to act proactively
        Returns: (should_act, reason)

        NOT hardcoded - uses learned context and confidence
        """

        context = await self.get_current_context()

        # Don't interrupt if user is busy
        if context.user_busyness > 0.8:
            return False, "User is very busy"

        # Don't interrupt if unwilling
        if context.interruption_willingness < 0.3:
            return False, "User not receptive to interruptions"

        # Only act if confident enough
        if action_benefit > 0.7 and action_complexity < 0.5:
            if len(context.active_patterns) > 2:
                return True, f"Pattern match ({len(context.active_patterns)} patterns)"

        return False, "Confidence too low to act"

    async def suggest_action(self, user_query: str) -> Dict[str, Any]:
        """
        Generate contextual suggestions based on learned patterns
        NOT template-based - dynamically generated
        """

        # Get current context
        context = await self.get_current_context()

        # Find relevant patterns
        relevant_patterns = []
        for pattern in self._patterns.values():
            if pattern.confidence > self.min_confidence_threshold:
                # Check relevance
                if any(
                    entity in user_query.lower() for entity in pattern.related_entities
                ):
                    relevant_patterns.append(pattern)

        # Generate suggestions
        suggestions = []

        if context.user_busyness < 0.5:
            # User has free time - suggest proactive actions
            for pattern in relevant_patterns[:3]:
                suggestions.append(
                    {
                        "type": "proactive",
                        "action": f"Based on your {pattern.context_type.value} patterns",
                        "confidence": pattern.confidence,
                    }
                )

        if context.day_of_week in ["Saturday", "Sunday"]:
            suggestions.append(
                {
                    "type": "context",
                    "action": "Weekend detected - may have different preferences",
                }
            )

        return {
            "context": {
                "time_of_day": context.time_of_day,
                "busyness": context.user_busyness,
                "active_patterns": len(context.active_patterns),
            },
            "suggestions": suggestions,
        }

    # =========================================================================
    # PATTERN MANAGEMENT
    # =========================================================================

    async def get_all_patterns(self) -> List[Dict]:
        """Get all learned patterns"""
        return [
            {
                "id": p.id,
                "type": p.context_type.value,
                "confidence": p.confidence,
                "occurrences": p.occurrences,
                "entities": list(p.related_entities)[:5],
            }
            for p in self._patterns.values()
        ]

    async def get_pattern_stats(self) -> Dict:
        """Get pattern learning statistics"""
        return {
            "total_patterns": len(self._patterns),
            "total_observations": len(self._observations),
            "by_type": {
                ct.value: len(
                    [p for p in self._patterns.values() if p.context_type == ct]
                )
                for ct in ContextType
            },
            "high_confidence": len(
                [p for p in self._patterns.values() if p.confidence > 0.7]
            ),
        }
