"""
AURA v3 Deep User Profiling System
==================================

Deep, adaptive user profiling that goes beyond simple preferences:
- Psychological profile (Big Five traits)
- Communication preferences
- Behavioral patterns
- Emotional patterns
- Relationship mapping
- Life context understanding

This is NOT static user data - it's continuously learning and adapting.
"""

import asyncio
import logging
import random
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


# ============================================================================
# PSYCHOLOGICAL PROFILING (Big Five Inspired)
# ============================================================================


class BigFiveTrait(Enum):
    """Big Five personality traits"""

    OPENNESS = "openness"  # Curious, creative vs conventional
    CONSCIENTIOUSNESS = "conscientiousness"  # Organized vs spontaneous
    EXTRAVERSION = "extraversion"  # Social vs reserved
    AGREEABLENESS = "agreeableness"  # Cooperative vs competitive
    NEUROTICISM = "neuroticism"  # Emotional stability vs reactive


@dataclass
class PsychologicalProfile:
    """
    Psychological profile based on Big Five

    Values: -1.0 to 1.0 (continuous, not discrete)
    """

    # Core traits (learned from behavior)
    traits: Dict[BigFiveTrait, float] = field(default_factory=dict)

    # Confidence in each trait estimate
    trait_confidence: Dict[BigFiveTrait, float] = field(default_factory=dict)

    # How trait estimates change over time
    trait_stability: Dict[BigFiveTrait, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.traits:
            # Initialize with neutral values
            for trait in BigFiveTrait:
                self.traits[trait] = 0.0
                self.trait_confidence[trait] = 0.1  # Low confidence initially
                self.trait_stability[trait] = 0.95  # High stability


# ============================================================================
# COMMUNICATION PATTERNS
# ============================================================================


@dataclass
class CommunicationProfile:
    """User's communication patterns"""

    # Vocabulary richness
    avg_message_length: float = 50.0
    vocabulary_diversity: float = 0.5  # Unique words / total words

    # Formality level (-1 casual to 1 formal)
    formality_level: float = 0.0
    formality_confidence: float = 0.1

    # Question patterns
    questions_per_message: float = 0.3
    rhetorical_questions: float = 0.1

    # Emotional expression
    emoji_usage: float = 0.2  # How often uses emoji
    exclamation_usage: float = 0.3  # How often uses !

    # Response patterns
    avg_response_time_seconds: float = 30.0
    typing_indicators: bool = True

    # Preferences (explicit)
    preferred_tone: Optional[str] = None
    prefers_concise: bool = False
    prefers_explanations: bool = True


# ============================================================================
# BEHAVIORAL PATTERNS
# ============================================================================


@dataclass
class BehavioralProfile:
    """User's behavioral patterns"""

    # Time patterns
    active_hours: Dict[int, float] = field(
        default_factory=dict
    )  # hour -> activity level
    peak_activity_hour: int = 10

    # Day patterns
    weekend_vs_weekday: float = 0.0  # -1 more weekday, 1 more weekend

    # Task completion
    task_completion_rate: float = 0.7
    average_tasks_per_day: float = 5.0

    # Interruption patterns
    interruptions_per_hour: float = 0.5
    context_switch_cost: float = 0.3  # How much context matters

    # Persistence
    retry_rate: float = 0.6  # How often retries failed tasks
    gives_up_easily: bool = False


# ============================================================================
# EMOTIONAL PATTERNS
# ============================================================================


@dataclass
class EmotionalProfile:
    """User's emotional patterns"""

    # Baseline emotional state
    baseline_mood: str = "neutral"  # happy, neutral, sad, anxious

    # Mood variability
    mood_volatility: float = 0.3  # How much mood changes
    stress_indicators: List[str] = field(default_factory=list)

    # Triggers
    positive_triggers: List[str] = field(default_factory=list)
    negative_triggers: List[str] = field(default_factory=list)
    stress_triggers: List[str] = field(default_factory=list)

    # Coping patterns
    asks_for_help_when_stressed: bool = True
    withdraws_when_upset: bool = False
    humor_as_coping: bool = True


# ============================================================================
# RELATIONSHIP MAPPING
# ============================================================================


@dataclass
class Relationship:
    """User's relationship with someone"""

    person_id: str
    name: str
    relationship_type: str  # family, friend, colleague, etc.

    # Interaction patterns with this person
    mentions_count: int = 0
    sentiment_toward_them: float = 0.0  # How user feels about them

    # What user has said about them
    notes: List[str] = field(default_factory=list)

    # Importance to user
    importance: float = 0.5  # 0-1


# ============================================================================
# LIFE CONTEXT
# ============================================================================


@dataclass
class LifeContext:
    """User's life situation"""

    # Work/School
    occupation: Optional[str] = None
    work_hours: Optional[Dict[str, str]] = None  # day -> hours

    # Location
    timezone: str = "UTC"
    current_location: Optional[str] = None

    # Family
    family_status: Optional[str] = None  # single, married, etc.
    has_children: bool = False

    # Health
    sleep_pattern: Optional[str] = None  # early_bird, night_owl, etc.
    exercise_frequency: Optional[str] = None

    # Interests (learned)
    topics_interested_in: List[str] = field(default_factory=list)
    topics_avoided: List[str] = field(default_factory=list)

    # Goals (stated)
    stated_goals: List[str] = field(default_factory=list)
    implicit_goals: List[str] = field(default_factory=list)


# ============================================================================
# DEEP USER PROFILE
# ============================================================================


class DeepUserProfiler:
    """
    Deep User Profiling System

    Builds comprehensive psychological, behavioral, emotional, and contextual
    understanding of the user through continuous observation.

    NOT static - adapts over time with increasing confidence.

    Key principles:
    - Multiple hypotheses (not single interpretation)
    - Confidence tracking (knows what it knows)
    - Temporal patterns (time-based learning)
    - Explicit vs implicit (distinguishes stated vs observed)
    - Respectful boundaries (doesn't assume too much)
    """

    def __init__(self, user_id: str):
        self.user_id = user_id

        # All profiling dimensions
        self.psychology = PsychologicalProfile()
        self.communication = CommunicationProfile()
        self.behavior = BehavioralProfile()
        self.emotional = EmotionalProfile()
        self.relationships: Dict[str, Relationship] = {}
        self.context = LifeContext()

        # Learning state
        self._observations: List[Dict] = []  # Raw observations
        self._insights: Dict[str, float] = {}  # Derived insights
        self._last_update = datetime.now()

        # Adaptive parameters
        self.learning_rate = 0.05  # How fast to adapt
        self.forgetting_rate = 0.01  # How fast to forget old patterns

    # =========================================================================
    # OBSERVATION & LEARNING
    # =========================================================================

    def observe(self, observation_type: str, data: Dict[str, Any]):
        """
        Record an observation about the user

        observation_type: message, task, reaction, statement, etc.
        """
        observation = {
            "type": observation_type,
            "data": data,
            "timestamp": datetime.now(),
            "confidence": data.get("confidence", 0.5),
        }

        self._observations.append(observation)

        # Process observation based on type
        if observation_type == "message":
            self._learn_from_message(data)
        elif observation_type == "task":
            self._learn_from_task(data)
        elif observation_type == "reaction":
            self._learn_from_reaction(data)
        elif observation_type == "statement":
            self._learn_from_statement(data)

        self._last_update = datetime.now()

    def _learn_from_message(self, data: Dict):
        """Learn from a message the user sent"""
        message = data.get("message", "")

        # Update communication profile
        words = message.split()
        if words:
            # Message length
            self.communication.avg_message_length = (
                self.communication.avg_message_length * (1 - self.learning_rate)
                + len(words) * self.learning_rate
            )

            # Vocabulary diversity
            unique = len(set(words))
            diversity = unique / max(len(words), 1)
            self.communication.vocabulary_diversity = (
                self.communication.vocabulary_diversity * (1 - self.learning_rate)
                + diversity * self.learning_rate
            )

        # Formality indicators
        formal_words = ["would", "could", "please", "thank", "regards"]
        informal_words = ["lol", "haha", "yeah", "gonna", "wanna"]

        formal_count = sum(1 for w in formal_words if w in message.lower())
        informal_count = sum(1 for w in informal_words if w in message.lower())

        if formal_count + informal_count > 0:
            formality_delta = (formal_count - informal_count) / (
                formal_count + informal_count
            )
            self.communication.formality_level += (
                formality_delta * self.learning_rate * 0.5
            )

        # Emoji usage
        if any(c in message for c in ["😀", "😂", "😍", "👍", "❤️", "😊"]):
            self.communication.emoji_usage = min(
                1.0, self.communication.emoji_usage + self.learning_rate
            )

        # Question detection
        if "?" in message:
            self.communication.questions_per_message = (
                self.communication.questions_per_message * (1 - self.learning_rate)
                + 1.0 * self.learning_rate
            )

    def _learn_from_task(self, data: Dict):
        """Learn from task completion"""
        task_completed = data.get("completed", False)
        task_type = data.get("type", "unknown")

        if task_completed:
            self.behavior.task_completion_rate = (
                self.behavior.task_completion_rate * (1 - self.learning_rate)
                + 1.0 * self.learning_rate
            )

        # Learning time patterns
        hour = datetime.now().hour
        if hour not in self.behavior.active_hours:
            self.behavior.active_hours[hour] = 0.0
        self.behavior.active_hours[hour] += self.learning_rate

        # Update peak hour
        if self.behavior.active_hours:
            self.behavior.peak_activity_hour = max(
                self.behavior.active_hours.items(), key=lambda x: x[1]
            )[0]

    def _learn_from_reaction(self, data: Dict):
        """Learn from user's reaction to something"""
        reaction = data.get("reaction", "")
        context = data.get("context", "")

        # Positive/negative reactions
        positive = ["👍", "❤️", "great", "thanks", "awesome", "perfect"]
        negative = ["😕", "😤", "ugh", "annoying", "hate", "wrong"]

        if any(p in reaction.lower() for p in positive):
            self._update_emotional_trigger(context, "positive")
        elif any(n in reaction.lower() for n in negative):
            self._update_emotional_trigger(context, "negative")

    def _learn_from_statement(self, data: Dict):
        """Learn from explicit statements"""
        statement = data.get("statement", "")

        # Extract preferences
        preference_indicators = [
            ("prefer", "prefers_explanations"),
            ("like detailed", "prefers_explanations"),
            ("just do it", "prefers_concise"),
            ("short", "prefers_concise"),
        ]

        for indicator, pref_key in preference_indicators:
            if indicator in statement.lower():
                if "not" in statement.lower() or "don't" in statement.lower():
                    setattr(self.communication, pref_key, False)
                else:
                    setattr(self.communication, pref_key, True)

        # Extract interests
        interest_indicators = ["interested in", "love", "enjoy", "passionate about"]
        for indicator in interest_indicators:
            if indicator in statement.lower():
                # Extract topic (simplified)
                idx = statement.lower().find(indicator)
                topic = (
                    statement[idx + len(indicator) :].strip().split()[0]
                    if idx >= 0
                    else ""
                )
                if topic and topic not in self.context.topics_interested_in:
                    self.context.topics_interested_in.append(topic)

    def _update_emotional_trigger(self, context: str, trigger_type: str):
        """Update emotional triggers"""
        if trigger_type == "positive":
            if context not in self.emotional.positive_triggers:
                self.emotional.positive_triggers.append(context)
        elif trigger_type == "negative":
            if context not in self.emotional.negative_triggers:
                self.emotional.negative_triggers.append(context)

    # =========================================================================
    # INSIGHT GENERATION
    # =========================================================================

    def generate_insights(self) -> Dict[str, Any]:
        """Generate derived insights from profile"""
        insights = {}

        # Psychological synthesis
        insights["personality_summary"] = self._synthesize_personality()

        # Communication synthesis
        insights["communication_style"] = self._synthesize_communication()

        # Behavioral synthesis
        insights["availability"] = self._synthesize_availability()

        # Emotional synthesis
        insights["emotional_state"] = self._synthesize_emotional()

        # Recommendations
        insights["interaction_recommendations"] = self._get_recommendations()

        return insights

    def _synthesize_personality(self) -> str:
        """Generate personality summary"""
        traits = []

        if self.psychology.traits.get(BigFiveTrait.OPENNESS, 0) > 0.3:
            traits.append("creative")
        elif self.psychology.traits.get(BigFiveTrait.OPENNESS, 0) < -0.3:
            traits.append("practical")

        if self.psychology.traits.get(BigFiveTrait.CONSCIENTIOUSNESS, 0) > 0.3:
            traits.append("organized")
        elif self.psychology.traits.get(BigFiveTrait.CONSCIENTIOUSNESS, 0) < -0.3:
            traits.append("spontaneous")

        if self.psychology.traits.get(BigFiveTrait.EXTRAVERSION, 0) > 0.3:
            traits.append("outgoing")
        elif self.psychology.traits.get(BigFiveTrait.EXTRAVERSION, 0) < -0.3:
            traits.append("reserved")

        return ", ".join(traits) if traits else "balanced"

    def _synthesize_communication(self) -> str:
        """Generate communication summary"""
        if self.communication.formality_level > 0.3:
            tone = "formal"
        elif self.communication.formality_level < -0.3:
            tone = "casual"
        else:
            tone = "balanced"

        if self.communication.prefers_concise:
            length = "brief"
        elif self.communication.avg_message_length > 80:
            length = "detailed"
        else:
            length = "moderate"

        return f"{tone}, {length}"

    def _synthesize_availability(self) -> Dict:
        """Synthesize availability patterns"""
        return {
            "peak_hour": self.behavior.peak_activity_hour,
            "typical_tasks_per_day": self.behavior.average_tasks_per_day,
            "completion_rate": self.behavior.task_completion_rate,
        }

    def _synthesize_emotional(self) -> Dict:
        """Synthesize emotional state"""
        return {
            "baseline": self.emotional.baseline_mood,
            "volatility": self.emotional.mood_volatility,
            "key_triggers": {
                "positive": self.emotional.positive_triggers[:3],
                "negative": self.emotional.negative_triggers[:3],
            },
        }

    def _get_recommendations(self) -> List[str]:
        """Get interaction recommendations"""
        recs = []

        # Based on communication
        if self.communication.prefers_concise:
            recs.append("Keep responses brief and to the point")
        else:
            recs.append("Provide detailed explanations when helpful")

        # Based on time patterns
        peak_hour = self.behavior.peak_activity_hour
        if 9 <= peak_hour <= 17:
            recs.append("Most active during work hours - respect focus time")
        elif 18 <= peak_hour <= 22:
            recs.append("Most active evenings - good time for detailed chats")

        # Based on emotional
        if self.emotional.humor_as_coping:
            recs.append("Can handle light humor - use appropriately")

        return recs

    # =========================================================================
    # QUICK PROFILE ACCESS
    # =========================================================================

    def get_communication_style(self) -> Dict:
        """Quick access to communication preferences"""
        return {
            "formality": self.communication.formality_level,
            "concise": self.communication.prefers_concise,
            "explanations": self.communication.prefers_explanations,
            "emoji_usage": self.communication.emoji_usage,
            "preferred_tone": self.communication.preferred_tone,
        }

    def get_availability(self) -> Dict:
        """Quick access to availability patterns"""
        return {
            "peak_hour": self.behavior.peak_activity_hour,
            "active_hours": dict(self.behavior.active_hours),
            "completion_rate": self.behavior.task_completion_rate,
        }

    def get_context(self) -> Dict:
        """Quick access to life context"""
        return {
            "timezone": self.context.timezone,
            "interests": self.context.topics_interested_in,
            "occupation": self.context.occupation,
        }


# Global storage
_profilers: Dict[str, DeepUserProfiler] = {}


def get_user_profiler(user_id: str) -> DeepUserProfiler:
    """Get or create user profiler"""
    if user_id not in _profilers:
        _profilers[user_id] = DeepUserProfiler(user_id)
    return _profilers[user_id]
