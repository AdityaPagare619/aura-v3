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

SQLite Persistence:
- Profiles are persisted to data/aura_profile.db
- Loaded on startup, saved on significant changes
- Debounced saves to avoid excessive I/O
"""

import asyncio
import json
import logging
import os
import random
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

# ============================================================================
# SQLITE PERSISTENCE CONFIGURATION
# ============================================================================

# Database path - relative to project root
DATABASE_DIR = Path(__file__).parent.parent.parent / "data"
DATABASE_PATH = DATABASE_DIR / "aura_profile.db"

# Save thresholds
TRAIT_CHANGE_THRESHOLD = 0.05  # Save if any trait changes by more than this
INTERACTION_SAVE_INTERVAL = 10  # Save every N interactions regardless


# ============================================================================
# SQLITE PERSISTENCE LAYER
# ============================================================================


class ProfilePersistence:
    """
    SQLite persistence layer for user profiles.

    Thread-safe with connection pooling per thread.
    Handles database creation, migrations, and CRUD operations.
    """

    _instance: Optional["ProfilePersistence"] = None
    _lock = threading.Lock()
    _connections: Dict[int, sqlite3.Connection] = {}  # thread_id -> connection

    def __new__(cls) -> "ProfilePersistence":
        """Singleton pattern for persistence layer"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self._db_initialized = False
        self._initialized = True

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        thread_id = threading.get_ident()

        with self._lock:
            if (
                thread_id not in self._connections
                or self._connections[thread_id] is None
            ):
                # Ensure directory exists
                DATABASE_DIR.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(
                    str(DATABASE_PATH), timeout=30.0, check_same_thread=False
                )
                conn.row_factory = sqlite3.Row
                # Enable WAL mode for better concurrent access
                conn.execute("PRAGMA journal_mode=WAL")
                self._connections[thread_id] = conn

                # Initialize database on first connection
                if not self._db_initialized:
                    self._ensure_database_tables(conn)
                    self._db_initialized = True

            return self._connections[thread_id]

    def _ensure_database_tables(self, conn: sqlite3.Connection) -> None:
        """Create database tables if they don't exist"""
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY,
                user_id TEXT UNIQUE NOT NULL,
                openness REAL DEFAULT 0.0,
                conscientiousness REAL DEFAULT 0.0,
                extraversion REAL DEFAULT 0.0,
                agreeableness REAL DEFAULT 0.0,
                neuroticism REAL DEFAULT 0.0,
                interaction_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            )
        """)

        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id 
            ON user_profiles(user_id)
        """)

        conn.commit()
        logger.info(f"Database initialized at {DATABASE_PATH}")

    def load_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a user profile from SQLite.

        Returns None if profile doesn't exist.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT user_id, openness, conscientiousness, extraversion,
                   agreeableness, neuroticism, interaction_count, 
                   last_updated, metadata
            FROM user_profiles
            WHERE user_id = ?
        """,
            (user_id,),
        )

        row = cursor.fetchone()
        if row is None:
            logger.debug(f"No profile found for user_id={user_id}")
            return None

        # Parse metadata JSON
        try:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        except json.JSONDecodeError:
            metadata = {}

        profile_data = {
            "user_id": row["user_id"],
            "traits": {
                "openness": row["openness"],
                "conscientiousness": row["conscientiousness"],
                "extraversion": row["extraversion"],
                "agreeableness": row["agreeableness"],
                "neuroticism": row["neuroticism"],
            },
            "interaction_count": row["interaction_count"],
            "last_updated": row["last_updated"],
            "metadata": metadata,
        }

        logger.info(f"Loaded profile for user_id={user_id}")
        return profile_data

    def save_profile(
        self,
        user_id: str,
        traits: Dict[str, float],
        interaction_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save or update a user profile in SQLite.

        Uses UPSERT (INSERT OR REPLACE) for atomic operations.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        metadata_json = json.dumps(metadata or {})

        try:
            cursor.execute(
                """
                INSERT INTO user_profiles 
                    (user_id, openness, conscientiousness, extraversion,
                     agreeableness, neuroticism, interaction_count, 
                     last_updated, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    openness = excluded.openness,
                    conscientiousness = excluded.conscientiousness,
                    extraversion = excluded.extraversion,
                    agreeableness = excluded.agreeableness,
                    neuroticism = excluded.neuroticism,
                    interaction_count = excluded.interaction_count,
                    last_updated = CURRENT_TIMESTAMP,
                    metadata = excluded.metadata
            """,
                (
                    user_id,
                    traits.get("openness", 0.0),
                    traits.get("conscientiousness", 0.0),
                    traits.get("extraversion", 0.0),
                    traits.get("agreeableness", 0.0),
                    traits.get("neuroticism", 0.0),
                    interaction_count,
                    metadata_json,
                ),
            )

            conn.commit()
            logger.debug(f"Saved profile for user_id={user_id}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to save profile for user_id={user_id}: {e}")
            conn.rollback()
            return False

    def delete_profile(self, user_id: str) -> bool:
        """Delete a user profile from SQLite"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Failed to delete profile for user_id={user_id}: {e}")
            return False

    def list_profiles(self) -> List[str]:
        """List all user IDs with profiles"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT user_id FROM user_profiles")
        return [row["user_id"] for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection for current thread"""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id in self._connections and self._connections[thread_id]:
                self._connections[thread_id].close()
                del self._connections[thread_id]

    def close_all(self) -> None:
        """Close all database connections (for cleanup/shutdown)"""
        with self._lock:
            for conn in self._connections.values():
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
            self._connections.clear()
            self._db_initialized = False


# Global persistence instance
_persistence: Optional[ProfilePersistence] = None


def get_persistence() -> ProfilePersistence:
    """Get the singleton persistence instance"""
    global _persistence
    if _persistence is None:
        _persistence = ProfilePersistence()
    return _persistence


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

    Persistence:
    - Profiles are loaded from SQLite on initialization
    - Saved on significant trait changes (>0.05) or every N interactions
    - Thread-safe for concurrent access

    Key principles:
    - Multiple hypotheses (not single interpretation)
    - Confidence tracking (knows what it knows)
    - Temporal patterns (time-based learning)
    - Explicit vs implicit (distinguishes stated vs observed)
    - Respectful boundaries (doesn't assume too much)
    """

    def __init__(self, user_id: str, auto_persist: bool = True):
        self.user_id = user_id
        self._auto_persist = auto_persist
        self._persistence = get_persistence() if auto_persist else None

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

        # Persistence tracking
        self._interaction_count = 0
        self._last_saved_traits: Dict[str, float] = {}
        self._last_save_interaction_count = 0

        # Load from SQLite if exists
        self._load_from_persistence()

    def _load_from_persistence(self) -> bool:
        """Load profile from SQLite if it exists"""
        if not self._persistence:
            return False

        try:
            profile_data = self._persistence.load_profile(self.user_id)
            if profile_data is None:
                logger.debug(f"No persisted profile for {self.user_id}, starting fresh")
                return False

            # Restore Big Five traits
            traits = profile_data.get("traits", {})
            for trait in BigFiveTrait:
                trait_key = trait.value
                if trait_key in traits:
                    self.psychology.traits[trait] = traits[trait_key]

            # Restore interaction count
            self._interaction_count = profile_data.get("interaction_count", 0)
            self._last_save_interaction_count = self._interaction_count

            # Store last saved traits for change detection
            self._last_saved_traits = traits.copy()

            # Restore metadata if present
            metadata = profile_data.get("metadata", {})
            if metadata:
                self._restore_metadata(metadata)

            logger.info(
                f"Loaded profile for {self.user_id} with {self._interaction_count} interactions"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load profile for {self.user_id}: {e}")
            return False

    def _restore_metadata(self, metadata: Dict[str, Any]) -> None:
        """Restore extended profile data from metadata"""
        # Restore communication preferences
        if "communication" in metadata:
            comm = metadata["communication"]
            self.communication.formality_level = comm.get("formality_level", 0.0)
            self.communication.prefers_concise = comm.get("prefers_concise", False)
            self.communication.prefers_explanations = comm.get(
                "prefers_explanations", True
            )
            self.communication.emoji_usage = comm.get("emoji_usage", 0.2)

        # Restore interests
        if "interests" in metadata:
            self.context.topics_interested_in = metadata["interests"]

        # Restore behavior
        if "behavior" in metadata:
            beh = metadata["behavior"]
            self.behavior.peak_activity_hour = beh.get("peak_activity_hour", 10)
            self.behavior.task_completion_rate = beh.get("task_completion_rate", 0.7)

    def _get_current_traits(self) -> Dict[str, float]:
        """Get current Big Five traits as a dict"""
        return {
            trait.value: self.psychology.traits.get(trait, 0.0)
            for trait in BigFiveTrait
        }

    def _should_persist(self) -> bool:
        """Determine if we should save to persistence"""
        if not self._auto_persist:
            return False

        # Check interaction count threshold
        if (
            self._interaction_count - self._last_save_interaction_count
            >= INTERACTION_SAVE_INTERVAL
        ):
            return True

        # Check trait change threshold
        current_traits = self._get_current_traits()
        for key, value in current_traits.items():
            last_value = self._last_saved_traits.get(key, 0.0)
            if abs(value - last_value) >= TRAIT_CHANGE_THRESHOLD:
                return True

        return False

    def _persist_if_needed(self) -> None:
        """Save to persistence if thresholds are met"""
        if not self._should_persist():
            return

        self.save_to_persistence()

    def save_to_persistence(self) -> bool:
        """Force save profile to SQLite"""
        if not self._persistence:
            return False

        try:
            traits = self._get_current_traits()
            metadata = {
                "communication": {
                    "formality_level": self.communication.formality_level,
                    "prefers_concise": self.communication.prefers_concise,
                    "prefers_explanations": self.communication.prefers_explanations,
                    "emoji_usage": self.communication.emoji_usage,
                },
                "interests": self.context.topics_interested_in,
                "behavior": {
                    "peak_activity_hour": self.behavior.peak_activity_hour,
                    "task_completion_rate": self.behavior.task_completion_rate,
                },
            }

            success = self._persistence.save_profile(
                user_id=self.user_id,
                traits=traits,
                interaction_count=self._interaction_count,
                metadata=metadata,
            )

            if success:
                self._last_saved_traits = traits.copy()
                self._last_save_interaction_count = self._interaction_count
                logger.debug(f"Persisted profile for {self.user_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to persist profile for {self.user_id}: {e}")
            return False

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

        # Increment interaction count for persistence tracking
        self._interaction_count += 1

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

        # Persist if thresholds are met (debounced)
        self._persist_if_needed()

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

    def get_interaction_count(self) -> int:
        """Get total interaction count"""
        return self._interaction_count


# Global storage
_profilers: Dict[str, DeepUserProfiler] = {}


def get_user_profiler(user_id: str, auto_persist: bool = True) -> DeepUserProfiler:
    """
    Get or create user profiler with SQLite persistence.

    Args:
        user_id: Unique identifier for the user
        auto_persist: Whether to auto-save to SQLite (default True)

    Returns:
        DeepUserProfiler instance (loaded from SQLite if exists)
    """
    if user_id not in _profilers:
        _profilers[user_id] = DeepUserProfiler(user_id, auto_persist=auto_persist)
    return _profilers[user_id]


def flush_all_profiles() -> int:
    """
    Force save all in-memory profiles to SQLite.

    Returns:
        Number of profiles successfully saved
    """
    saved = 0
    for profiler in _profilers.values():
        if profiler.save_to_persistence():
            saved += 1
    return saved


def clear_profile_cache() -> None:
    """Clear in-memory profile cache (does not delete from SQLite)"""
    global _profilers
    # Save before clearing
    flush_all_profiles()
    _profilers = {}


def delete_user_profile(user_id: str) -> bool:
    """
    Delete a user profile from both memory and SQLite.

    Args:
        user_id: User ID to delete

    Returns:
        True if deleted, False otherwise
    """
    # Remove from memory
    if user_id in _profilers:
        del _profilers[user_id]

    # Remove from SQLite
    persistence = get_persistence()
    return persistence.delete_profile(user_id)
