"""
AURA v3 Self-Learning & User Profile System
AURA learns from interactions, builds mental model of user
Adapts to preferences, improves over time
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import deque, defaultdict
import hashlib

logger = logging.getLogger(__name__)


class PreferenceCategory(Enum):
    """Categories of user preferences"""

    COMMUNICATION = "communication"
    NOTIFICATIONS = "notifications"
    PRIVACY = "privacy"
    ASSISTANT = "assistant"
    SOCIAL = "social"
    SHOPPING = "shopping"
    WORK = "work"
    PERSONAL = "personal"


class LearningType(Enum):
    """Types of learning"""

    PREFERENCE = "preference"
    PATTERN = "pattern"
    INTEREST = "interest"
    BEHAVIOR = "behavior"
    FEEDBACK = "feedback"


@dataclass
class UserPreference:
    """A learned user preference"""

    id: str
    category: PreferenceCategory
    key: str
    value: Any
    confidence: float  # How confident we are
    source: str  # How we learned it
    first_observed: datetime
    last_confirmed: datetime
    confirmation_count: int = 0


@dataclass
class UserProfile:
    """Complete user profile built from learning"""

    user_id: str

    # Demographics (if known)
    name: Optional[str] = None
    occupation: Optional[str] = None
    location: Optional[str] = None

    # Personality indicators
    personality_traits: Dict[str, float] = field(default_factory=dict)

    # Communication preferences
    communication_style: str = "balanced"  # formal, casual, brief, detailed
    response_preference: str = "text"  # text, voice, emoji

    # Behavioral patterns
    active_hours: Dict[str, tuple] = field(default_factory=dict)
    weekly_patterns: Dict[int, List[str]] = field(default_factory=dict)

    # Interests (from analysis)
    interests: Dict[str, float] = field(default_factory=dict)
    professional_interests: Dict[str, float] = field(default_factory=dict)
    social_interests: Dict[str, float] = field(default_factory=dict)

    # Goals and priorities
    current_goals: List[str] = field(default_factory=list)
    priorities: Dict[str, int] = field(default_factory=dict)

    # Relationships (tracked)
    important_contacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Last updated
    last_updated: datetime = field(default_factory=datetime.now)
    profile_completeness: float = 0.0


@dataclass
class LearningEntry:
    """A single learning entry"""

    id: str
    learning_type: LearningType
    content: str
    context: Dict[str, Any]
    confidence: float
    timestamp: datetime
    validated: bool = False


class SelfLearningEngine:
    """
    Self-Learning Engine

    AURA learns from EVERY interaction:
    - Explicit feedback (user says "I like this")
    - Implicit feedback (user keeps using a feature)
    - Behavioral patterns (when user is most active)
    - Content preferences (what user engages with)
    - Social patterns (who user interacts with most)

    Builds a mental model of the USER over time.
    """

    def __init__(self, storage_path: str = "data/user_profile.json"):
        self.storage_path = storage_path

        # User profile
        self._profile: Optional[UserProfile] = None

        # Learning history
        self._learning_history: deque = deque(maxlen=1000)

        # Pending validations
        self._pending_validations: deque = deque(maxlen=100)

        # Feedback patterns
        self._positive_patterns: Dict[str, int] = defaultdict(int)
        self._negative_patterns: Dict[str, int] = defaultdict(int)

        # Confirmation tracking
        self._preference_confirmations: Dict[str, int] = defaultdict(int)

    async def initialize(self, user_id: str = "default"):
        """Initialize or load user profile"""
        self._profile = UserProfile(user_id=user_id)
        await self._load_profile()
        logger.info(f"Self-learning engine initialized for user: {user_id}")

    # =========================================================================
    # EXPLICIT LEARNING (User tells us)
    # =========================================================================

    async def learn_from_feedback(
        self,
        feedback_type: str,  # "positive", "negative", "neutral"
        topic: str,
        context: Dict[str, Any] = None,
    ):
        """Learn from explicit user feedback"""

        entry = LearningEntry(
            id=f"fb_{datetime.now().timestamp()}",
            learning_type=LearningType.FEEDBACK,
            content=f"{feedback_type}: {topic}",
            context=context or {},
            confidence=0.9,  # High confidence for explicit feedback
            timestamp=datetime.now(),
        )

        self._learning_history.append(entry)

        # Update pattern tracking
        if feedback_type == "positive":
            self._positive_patterns[topic] += 1
        elif feedback_type == "negative":
            self._negative_patterns[topic] += 1

        # Update profile
        await self._update_from_feedback(feedback_type, topic, context)

        logger.info(f"Learned feedback: {feedback_type} - {topic}")

    async def learn_preference(
        self,
        category: PreferenceCategory,
        key: str,
        value: Any,
        confidence: float = 0.5,
        source: str = "inferred",
    ):
        """Learn a user preference"""

        # Check if preference already exists
        existing = self._find_preference(category, key)

        if existing:
            # Update existing preference
            existing.value = value
            existing.confidence = min(existing.confidence + 0.1, 1.0)
            existing.last_confirmed = datetime.now()
            existing.confirmation_count += 1

            self._preference_confirmations[f"{category.value}.{key}"] += 1

        else:
            # Create new preference
            pref = UserPreference(
                id=f"pref_{category.value}_{key}",
                category=category,
                key=key,
                value=value,
                confidence=confidence,
                source=source,
                first_observed=datetime.now(),
                last_confirmed=datetime.now(),
            )

            # Add to profile
            self._profile.personality_traits[f"{category.value}.{key}"] = pref

        await self._save_profile()

    # =========================================================================
    # IMPLICIT LEARNING (AURA observes)
    # =========================================================================

    async def learn_from_behavior(
        self, behavior: str, context: Dict[str, Any] = None, count: int = 1
    ):
        """Learn from user behavior patterns"""

        entry = LearningEntry(
            id=f"beh_{datetime.now().timestamp()}",
            learning_type=LearningType.BEHAVIOR,
            content=behavior,
            context=context or {},
            confidence=0.6,  # Medium confidence for behavior
            timestamp=datetime.now(),
        )

        self._learning_history.append(entry)

        # Detect patterns
        await self._detect_pattern(behavior, context)

    async def learn_interest(
        self,
        interest: str,
        category: str = "general",
        source: str = "content",
        confidence: float = 0.5,
    ):
        """Learn a user interest from content"""

        # Update interest in profile
        if self._profile:
            if category == "professional":
                current = self._profile.professional_interests.get(interest, 0)
                self._profile.professional_interests[interest] = max(
                    current, confidence
                )
            elif category == "social":
                current = self._profile.social_interests.get(interest, 0)
                self._profile.social_interests[interest] = max(current, confidence)
            else:
                current = self._profile.interests.get(interest, 0)
                self._profile.interests[interest] = max(current, confidence)

            self._profile.last_updated = datetime.now()
            await self._save_profile()

        logger.info(f"Learned interest: {interest} ({category})")

    async def learn_communication_style(
        self,
        style: str,  # "formal", "casual", "brief", "detailed"
    ):
        """Learn user's communication style"""

        await self.learn_preference(
            PreferenceCategory.COMMUNICATION,
            "style",
            style,
            confidence=0.7,
            source="behavior_analysis",
        )

        if self._profile:
            self._profile.communication_style = style

    # =========================================================================
    # PATTERN DETECTION
    # =========================================================================

    async def _detect_pattern(self, behavior: str, context: Optional[Dict]):
        """Detect patterns in behavior"""

        # Simple pattern detection
        now = datetime.now()

        # Time-based patterns
        hour = now.hour
        day = now.weekday()

        if self._profile:
            # Update active hours
            day_name = [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ][day]

            if day_name not in self._profile.weekly_patterns:
                self._profile.weekly_patterns[day] = []

            if hour not in self._profile.weekly_patterns[day]:
                self._profile.weekly_patterns[day].append(hour)

    async def analyze_and_update_profile(self):
        """Periodically analyze learning history and update profile"""

        # Calculate profile completeness
        completeness = 0.0
        factors = [
            bool(self._profile.name),
            bool(self._profile.occupation),
            len(self._profile.interests) > 0,
            len(self._profile.active_hours) > 0,
            len(self._profile.current_goals) > 0,
        ]

        self._profile.profile_completeness = sum(factors) / len(factors)

        await self._save_profile()

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_profile(self) -> Optional[UserProfile]:
        """Get current user profile"""
        return self._profile

    def get_preferences(
        self, category: Optional[PreferenceCategory] = None
    ) -> List[UserPreference]:
        """Get learned preferences"""

        if not self._profile:
            return []

        prefs = list(self._profile.personality_traits.values())

        if category:
            prefs = [p for p in prefs if p.category == category]

        return prefs

    def get_top_interests(
        self, limit: int = 5, category: str = "general"
    ) -> List[tuple]:
        """Get user's top interests"""

        if not self._profile:
            return []

        if category == "professional":
            interests = self._profile.professional_interests
        elif category == "social":
            interests = self._profile.social_interests
        else:
            interests = self._profile.interests

        return sorted(interests.items(), key=lambda x: x[1], reverse=True)[:limit]

    def get_active_hours(self) -> Dict[str, List[int]]:
        """Get user's typical active hours"""

        if not self._profile:
            return {}

        result = {}
        day_names = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]

        for day_num, hours in self._profile.weekly_patterns.items():
            if hours:
                result[day_names[day_num]] = hours

        return result

    def get_recommendation_context(self) -> Dict[str, Any]:
        """Get context for making recommendations"""

        if not self._profile:
            return {}

        return {
            "communication_style": self._profile.communication_style,
            "top_interests": self.get_top_interests(3),
            "active_hours": self.get_active_hours(),
            "profile_completeness": self._profile.profile_completeness,
            "positive_patterns": dict(self._positive_patterns),
        }

    def _find_preference(
        self, category: PreferenceCategory, key: str
    ) -> Optional[UserPreference]:
        """Find existing preference"""

        if not self._profile:
            return None

        pref_key = f"{category.value}.{key}"
        return self._profile.personality_traits.get(pref_key)

    async def _update_from_feedback(
        self, feedback_type: str, topic: str, context: Optional[Dict]
    ):
        """Update profile based on feedback"""

        if not self._profile:
            return

        # Update based on feedback type
        if feedback_type == "positive":
            # Increase priority of this topic
            current = self._profile.priorities.get(topic, 5)
            self._profile.priorities[topic] = min(current + 1, 10)

        elif feedback_type == "negative":
            # Decrease priority
            current = self._profile.priorities.get(topic, 5)
            self._profile.priorities[topic] = max(current - 1, 0)

    async def _load_profile(self):
        """Load profile from disk"""

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            if data:
                self._profile = UserProfile(**data)
                logger.info("Loaded user profile")

        except FileNotFoundError:
            logger.info("No existing profile, starting fresh")
        except Exception as e:
            logger.error(f"Error loading profile: {e}")

    async def _save_profile(self):
        """Save profile to disk"""

        if not self._profile:
            return

        try:
            import os

            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

            with open(self.storage_path, "w") as f:
                json.dump(vars(self._profile), f, default=str)

        except Exception as e:
            logger.error(f"Error saving profile: {e}")
