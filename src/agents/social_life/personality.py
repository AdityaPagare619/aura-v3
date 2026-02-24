"""
AURA v3 Social Personality
Adaptive personality that learns user's social preferences
Integrates with AURA's learning engine for continuous improvement
"""

import asyncio
import logging
import json
import os
import random
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class SocialMood(Enum):
    """User's current social mood"""

    SOCIAL = "social"
    PRIVATE = "private"
    ACTIVE = "active"
    REFLECTIVE = "reflective"
    CONNECTED = "connected"
    ISOLATED = "isolated"


@dataclass
class SocialPreferences:
    """User's social preferences"""

    introversion_extroversion: float = 0.5

    preferred_response_length: str = "medium"
    preferred_response_style: str = "friendly"

    follow_up_frequency: str = "moderate"
    reminder_sensitivity: float = 0.5

    preferred_contact_methods: List[str] = field(default_factory=list)
    blocked_contacts: List[str] = field(default_factory=list)

    quiet_hours_start: int = 22
    quiet_hours_end: int = 8

    auto_followup_enabled: bool = True
    event_reminders_enabled: bool = True


@dataclass
class ContactPreference:
    """Preferences for a specific contact"""

    contact_name: str

    preferred_response_style: str = "friendly"
    response_priority: str = "normal"

    interaction_frequency_target: str = "weekly"

    follow_up_reminders: bool = True
    important_dates_tracking: bool = True

    last_adjusted: datetime = field(default_factory=datetime.now)


class SocialPersonality:
    """
    Adaptive social personality
    Learns user preferences and adapts behavior accordingly
    """

    def __init__(self, data_dir: str = "data/social_life/personality"):
        import os

        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self._preferences = SocialPreferences()
        self._contact_preferences: Dict[str, ContactPreference] = {}

        self._mood: SocialMood = SocialMood.SOCIAL
        self._mood_history: List[Dict] = []

        self._interaction_style = "supportive"
        self._learned_responses: Dict[str, List[str]] = {}

    async def initialize(self):
        """Initialize social personality"""
        logger.info("Initializing Social Personality...")
        await self._load_preferences()
        await self._load_contact_preferences()
        logger.info("Social Personality initialized")

    async def _load_preferences(self):
        """Load preferences from disk"""
        prefs_file = f"{self.data_dir}/preferences.json"
        try:
            if os.path.exists(prefs_file):
                with open(prefs_file, "r") as f:
                    data = json.load(f)
                    self._preferences = SocialPreferences(**data.get("preferences", {}))
                    self._interaction_style = data.get(
                        "interaction_style", "supportive"
                    )
                logger.info("Loaded social preferences")
        except Exception as e:
            logger.error(f"Error loading preferences: {e}")

    async def _save_preferences(self):
        """Save preferences to disk"""
        prefs_file = f"{self.data_dir}/preferences.json"
        try:
            data = {
                "preferences": {
                    "introversion_extroversion": self._preferences.introversion_extroversion,
                    "preferred_response_length": self._preferences.preferred_response_length,
                    "preferred_response_style": self._preferences.preferred_response_style,
                    "follow_up_frequency": self._preferences.follow_up_frequency,
                    "reminder_sensitivity": self._preferences.reminder_sensitivity,
                    "preferred_contact_methods": self._preferences.preferred_contact_methods,
                    "quiet_hours_start": self._preferences.quiet_hours_start,
                    "quiet_hours_end": self._preferences.quiet_hours_end,
                    "auto_followup_enabled": self._preferences.auto_followup_enabled,
                    "event_reminders_enabled": self._preferences.event_reminders_enabled,
                },
                "interaction_style": self._interaction_style,
            }
            with open(prefs_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")

    async def _load_contact_preferences(self):
        """Load contact-specific preferences"""
        prefs_file = f"{self.data_dir}/contact_preferences.json"
        try:
            if os.path.exists(prefs_file):
                with open(prefs_file, "r") as f:
                    data = json.load(f)
                    for cp_data in data.get("contacts", []):
                        cp_data["last_adjusted"] = datetime.fromisoformat(
                            cp_data["last_adjusted"]
                        )
                        self._contact_preferences[cp_data["contact_name"]] = (
                            ContactPreference(**cp_data)
                        )
                logger.info(
                    f"Loaded preferences for {len(self._contact_preferences)} contacts"
                )
        except Exception as e:
            logger.error(f"Error loading contact preferences: {e}")

    async def _save_contact_preferences(self):
        """Save contact-specific preferences"""
        prefs_file = f"{self.data_dir}/contact_preferences.json"
        try:
            data = {
                "contacts": [
                    {
                        **vars(cp),
                        "last_adjusted": cp.last_adjusted.isoformat(),
                    }
                    for cp in self._contact_preferences.values()
                ]
            }
            with open(prefs_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving contact preferences: {e}")

    async def learn_from_interaction(
        self,
        contact_name: str,
        message: str,
        response: str,
        user_feedback: str = None,
    ):
        """Learn from social interaction"""
        if contact_name not in self._contact_preferences:
            self._contact_preferences[contact_name] = ContactPreference(
                contact_name=contact_name
            )

        contact_pref = self._contact_preferences[contact_name]

        if user_feedback:
            await self._learn_from_feedback(contact_name, user_feedback)

        message_len = len(message.split())
        response_len = len(response.split())

        if message_len < 10 and response_len < 20:
            contact_pref.preferred_response_style = "brief"
        elif message_len > 50 or response_len > 100:
            contact_pref.preferred_response_style = "detailed"

        contact_pref.last_adjusted = datetime.now()
        await self._save_contact_preferences()

    async def _learn_from_feedback(self, contact_name: str, feedback: str):
        """Learn from user feedback"""
        feedback_lower = feedback.lower()

        positive_words = ["thanks", "great", "perfect", "love", "awesome", "good"]
        negative_words = ["weird", "off", "don't like", "annoying", "wrong", "bad"]

        if any(word in feedback_lower for word in positive_words):
            current_style = self._contact_preferences.get(contact_name)
            if current_style:
                if current_style.response_priority != "high":
                    current_style.response_priority = "high"

        elif any(word in feedback_lower for word in negative_words):
            current_style = self._contact_preferences.get(contact_name)
            if current_style:
                current_style.follow_up_reminders = False

    async def get_response_style(self, contact_name: str) -> Dict[str, Any]:
        """Get response style for a contact"""
        contact_pref = self._contact_preferences.get(contact_name)

        if contact_pref:
            return {
                "response_style": contact_pref.preferred_response_style,
                "priority": contact_pref.response_priority,
                "frequency_target": contact_pref.interaction_frequency_target,
                "follow_up_enabled": contact_pref.follow_up_reminders,
            }

        return {
            "response_style": self._preferences.preferred_response_style,
            "priority": "normal",
            "frequency_target": self._preferences.follow_up_frequency,
            "follow_up_enabled": self._preferences.auto_followup_enabled,
        }

    async def generate_response_suggestion(
        self,
        contact_name: str,
        context: str,
        response_type: str = "message",
    ) -> str:
        """Generate a response suggestion"""
        style = await self.get_response_style(contact_name)
        response_style = style.get("response_style", "friendly")

        suggestions = {
            "brief": [
                "Got it, thanks!",
                "Sounds good!",
                "Sure, will do!",
                "Thanks for letting me know!",
            ],
            "friendly": [
                "That sounds great! Let me know how it goes.",
                "Thanks for sharing! I'll keep that in mind.",
                "Appreciate you letting me know!",
                "Great to hear from you!",
            ],
            "detailed": [
                "Thank you for the detailed update. I appreciate you taking the time to share this with me.",
                "I understand. Let me think about this and get back to you with more thoughts.",
                "That's really interesting context. I wanted to make sure I understood properly.",
            ],
        }

        options = suggestions.get(response_style, suggestions["friendly"])
        return random.choice(options)

    async def get_mood(self) -> SocialMood:
        """Get current social mood"""
        return self._mood

    async def get_current_mood(self) -> Dict[str, Any]:
        """Get current mood with context"""
        return {
            "mood": self._mood.value,
            "interaction_style": self._interaction_style,
            "introversion_extroversion": self._preferences.introversion_extroversion,
        }

    async def update_mood(self, mood: SocialMood):
        """Update current social mood"""
        self._mood = mood
        self._mood_history.append(
            {
                "mood": mood.value,
                "timestamp": datetime.now().isoformat(),
            }
        )

        if len(self._mood_history) > 100:
            self._mood_history = self._mood_history[-100:]

    async def suggest_interaction(
        self, contact_name: str, relationship_data: Dict = None
    ) -> Optional[Dict[str, Any]]:
        """Suggest an interaction"""
        if not self._preferences.auto_followup_enabled:
            return None

        style = await self.get_response_style(contact_name)

        if style.get("priority") == "low":
            return None

        suggestions = []

        if relationship_data:
            days_since = relationship_data.get("days_since", 0)
            importance = relationship_data.get("importance", 0.5)

            if importance > 0.7 and days_since > 14:
                suggestions.append(
                    {
                        "type": "reach_out",
                        "contact": contact_name,
                        "reason": "important_contact_needs_attention",
                        "suggested_message": f"Just wanted to check in with {contact_name}!",
                    }
                )

            if importance > 0.5 and days_since > 30:
                suggestions.append(
                    {
                        "type": "reconnect",
                        "contact": contact_name,
                        "reason": "long_time_no_contact",
                        "suggested_message": f"Hey {contact_name}, it's been a while! How have you been?",
                    }
                )

        return suggestions[0] if suggestions else None

    async def update_preferences(self, **preferences):
        """Update social preferences"""
        for key, value in preferences.items():
            if hasattr(self._preferences, key):
                setattr(self._preferences, key, value)

        await self._save_preferences()
        logger.info("Updated social preferences")

    async def get_preferences(self) -> Dict[str, Any]:
        """Get current preferences"""
        return {
            "introversion_extroversion": self._preferences.introversion_extroversion,
            "preferred_response_length": self._preferences.preferred_response_length,
            "preferred_response_style": self._preferences.preferred_response_style,
            "follow_up_frequency": self._preferences.follow_up_frequency,
            "reminder_sensitivity": self._preferences.reminder_sensitivity,
            "quiet_hours": f"{self._preferences.quiet_hours_start}:00-{self._preferences.quiet_hours_end}:00",
            "auto_followup_enabled": self._preferences.auto_followup_enabled,
            "event_reminders_enabled": self._preferences.event_reminders_enabled,
        }

    async def is_quiet_hours(self) -> bool:
        """Check if currently in quiet hours"""
        now = datetime.now()
        current_hour = now.hour

        if self._preferences.quiet_hours_start > self._preferences.quiet_hours_end:
            return (
                current_hour >= self._preferences.quiet_hours_start
                or current_hour < self._preferences.quiet_hours_end
            )
        else:
            return (
                self._preferences.quiet_hours_start
                <= current_hour
                < self._preferences.quiet_hours_end
            )

    async def should_notify(self, contact_name: str, notification_type: str) -> bool:
        """Check if should send notification"""
        if await self.is_quiet_hours():
            return False

        style = await self.get_response_style(contact_name)

        if notification_type == "followup":
            return style.get("follow_up_enabled", True)
        elif notification_type == "event":
            return self._preferences.event_reminders_enabled
        elif notification_type == "reminder":
            return self._preferences.reminder_sensitivity > 0.3

        return True


_social_personality: Optional[SocialPersonality] = None


def get_social_personality() -> SocialPersonality:
    """Get or create social personality"""
    global _social_personality
    if _social_personality is None:
        _social_personality = SocialPersonality()
    return _social_personality
