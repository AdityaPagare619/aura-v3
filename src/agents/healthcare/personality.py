"""
Healthcare Agent Personality
===========================

Adaptive personality for the healthcare agent that learns
user's health priorities and preferences over time.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, date
from dataclasses import dataclass, field
from enum import Enum
import random

from src.agents.healthcare.models import (
    HealthPreferences,
    HealthGoal,
    InsightPriority,
)

logger = logging.getLogger(__name__)


class PersonalityTone(Enum):
    """Tone of the healthcare agent's responses"""

    ENCOURAGING = "encouraging"
    CHALLENGING = "challenging"
    INFORMATIONAL = "informational"
    DIRECT = "direct"
    SUPPORTIVE = "supportive"


class ResponseStyle(Enum):
    """How the agent delivers information"""

    BRIEF = "brief"
    DETAILED = "detailed"
    QUESTIONS = "questions"
    ACTION_ORIENTED = "action_oriented"


@dataclass
class PersonalityTraits:
    """Personality traits for healthcare agent"""

    tone: PersonalityTone = PersonalityTone.ENCOURAGING
    response_style: ResponseStyle = ResponseStyle.ACTION_ORIENTED
    formality: float = 0.5  # 0 casual, 1 formal
    empathy: float = 0.7  # 0 matter-of-fact, 1 very empathetic
    directness: float = 0.5  # 0 subtle, 1 very direct
    humor: float = 0.3  # 0 serious, 1 playful
    patience: float = 0.7  # 0 impatient, 1 very patient


@dataclass
class InteractionContext:
    """Context for generating responses"""

    metric_type: str = ""
    value: float = 0.0
    goal: str = ""
    previous_interactions: int = 0
    time_of_day: str = ""
    user_mood: str = ""


class HealthcarePersonality:
    """
    Adaptive personality for healthcare agent

    Features:
    - Learns user preferences
    - Adapts communication style
    - Provides consistent but personalized responses
    - Respects privacy
    """

    def __init__(self, storage_path: str = "data/healthcare"):
        self.storage_path = storage_path
        self._preferences: Optional[HealthPreferences] = None
        self._traits = PersonalityTraits()
        self._interaction_count = 0
        self._learning_history: List[Dict] = []
        os.makedirs(storage_path, exist_ok=True)
        self._load_preferences()

    def _load_preferences(self):
        """Load user health preferences"""
        prefs_file = os.path.join(self.storage_path, "health_preferences.json")
        if os.path.exists(prefs_file):
            try:
                with open(prefs_file) as f:
                    data = json.load(f)
                    self._preferences = HealthPreferences(**data)

                    self._apply_preferences_to_traits()
            except Exception as e:
                logger.error(f"Failed to load health preferences: {e}")
                self._preferences = HealthPreferences()

        traits_file = os.path.join(self.storage_path, "health_personality.json")
        if os.path.exists(traits_file):
            try:
                with open(traits_file) as f:
                    data = json.load(f)
                    self._traits = PersonalityTraits(**data)
            except Exception as e:
                logger.error(f"Failed to load personality traits: {e}")

    def _save_preferences(self):
        """Save user health preferences"""
        if not self._preferences:
            return

        prefs_file = os.path.join(self.storage_path, "health_preferences.json")
        try:
            with open(prefs_file, "w") as f:
                json.dump(
                    {
                        "user_id": self._preferences.user_id,
                        "preferred_workout_times": self._preferences.preferred_workout_times,
                        "preferred_meal_times": self._preferences.preferred_meal_times,
                        "disliked_foods": self._preferences.disliked_foods,
                        "liked_foods": self._preferences.liked_foods,
                        "workout_preferences": self._preferences.workout_preferences,
                        "notification_preferences": self._preferences.notification_preferences,
                        "insight_priority_focus": self._preferences.insight_priority_focus,
                        "motivational_style": self._preferences.motivational_style,
                        "response_to_insights": self._preferences.response_to_insights,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")

        traits_file = os.path.join(self.storage_path, "health_personality.json")
        with open(traits_file, "w") as f:
            json.dump(
                {
                    "tone": self._traits.tone.value
                    if hasattr(self._traits.tone, "value")
                    else self._traits.tone,
                    "response_style": self._traits.response_style.value
                    if hasattr(self._traits.response_style, "value")
                    else self._traits.response_style,
                    "formality": self._traits.formality,
                    "empathy": self._traits.empathy,
                    "directness": self._traits.directness,
                    "humor": self._traits.humor,
                    "patience": self._traits.patience,
                },
                f,
                indent=2,
            )

    def _apply_preferences_to_traits(self):
        """Apply learned preferences to personality traits"""
        if not self._preferences:
            return

        style = self._preferences.motivational_style
        if style == "encouraging":
            self._traits.tone = PersonalityTone.ENCOURAGING
            self._traits.empathy = 0.8
        elif style == "challenging":
            self._traits.tone = PersonalityTone.CHALLENGING
            self._traits.directness = 0.7
        elif style == "informational":
            self._traits.tone = PersonalityTone.INFORMATIONAL
            self._traits.response_style = ResponseStyle.DETAILED

    def set_user_id(self, user_id: str):
        """Set user ID"""
        if not self._preferences:
            self._preferences = HealthPreferences()
        self._preferences.user_id = user_id
        self._save_preferences()

    def learn_from_interaction(
        self,
        interaction_type: str,
        user_response: str,
        feedback: str = None,
    ):
        """Learn from user interactions"""
        self._interaction_count += 1

        self._learning_history.append(
            {
                "type": interaction_type,
                "timestamp": datetime.now().isoformat(),
                "user_response": user_response,
            }
        )

        if feedback:
            self._learn_from_feedback(feedback)

        if self._interaction_count % 10 == 0:
            self._adapt_personality()

    def _learn_from_feedback(self, feedback: str):
        """Learn from explicit feedback"""
        feedback_lower = feedback.lower()

        positive_words = ["great", "thanks", "love", "helpful", "good", "perfect"]
        negative_words = ["don't like", "boring", "annoying", "not helpful", "weird"]

        if any(word in feedback_lower for word in positive_words):
            self._traits.empathy = min(1.0, self._traits.empathy + 0.02)
            self._traits.humor = min(1.0, self._traits.humor + 0.01)

        if any(word in feedback_lower for word in negative_words):
            if "too much" in feedback_lower or "long" in feedback_lower:
                self._traits.response_style = ResponseStyle.BRIEF
            self._traits.empathy = max(0.3, self._traits.empathy - 0.02)

        self._save_preferences()

    def _adapt_personality(self):
        """Adapt personality based on learning history"""
        if not self._learning_history:
            return

        recent = self._learning_history[-10:]

        brief_count = sum(
            1 for r in recent if "brief" in r.get("user_response", "").lower()
        )
        if brief_count > 5:
            self._traits.response_style = ResponseStyle.BRIEF
            self._traits.formality = max(0.3, self._traits.formality - 0.1)

        question_count = sum(1 for r in recent if "?" in r.get("user_response", ""))
        if question_count > 3:
            self._traits.response_style = ResponseStyle.QUESTIONS

        logger.info(f"Adapted personality: {self._traits.response_style.value}")

    def learn_food_preference(self, food_name: str, liked: bool):
        """Learn food preferences"""
        if not self._preferences:
            self._preferences = HealthPreferences()

        if liked:
            if food_name not in self._preferences.liked_foods:
                self._preferences.liked_foods.append(food_name)
                if food_name in self._preferences.disliked_foods:
                    self._preferences.disliked_foods.remove(food_name)
        else:
            if food_name not in self._preferences.disliked_foods:
                self._preferences.disliked_foods.append(food_name)
                if food_name in self._preferences.liked_foods:
                    self._preferences.liked_foods.remove(food_name)

        self._save_preferences()

    def learn_workout_preference(self, workout_type: str, liked: bool):
        """Learn workout preferences"""
        if not self._preferences:
            self._preferences = HealthPreferences()

        if liked:
            if workout_type not in self._preferences.workout_preferences:
                self._preferences.workout_preferences.append(workout_type)

        self._save_preferences()

    def learn_insight_response(self, response_type: str):
        """Learn how user responds to insights"""
        if not self._preferences:
            self._preferences = HealthPreferences()

        self._preferences.response_to_insights = response_type
        self._save_preferences()

    def generate_response(
        self,
        response_type: str,
        context: InteractionContext = None,
        data: Dict = None,
    ) -> str:
        """Generate a personalized response"""
        data = data or {}

        if response_type == "greeting":
            return self._generate_greeting(context)
        elif response_type == "insight":
            return self._generate_insight_response(data)
        elif response_type == "workout_suggestion":
            return self._generate_workout_response(data)
        elif response_type == "meal_suggestion":
            return self._generate_meal_response(data)
        elif response_type == "progress":
            return self._generate_progress_response(data)
        elif response_type == "motivation":
            return self._generate_motivation(context)
        elif response_type == "reminder":
            return self._generate_reminder(data)
        else:
            return self._generate_default_response(response_type)

    def _generate_greeting(self, context: InteractionContext = None) -> str:
        """Generate greeting"""
        hour = datetime.now().hour

        if 5 <= hour < 12:
            greeting = "Good morning"
        elif 12 <= hour < 17:
            greeting = "Good afternoon"
        elif 17 <= hour < 21:
            greeting = "Good evening"
        else:
            greeting = "Hello"

        if self._traits.tone == PersonalityTone.ENCOURAGING:
            return f"{greeting!r}! Ready to check in on your health today?"
        elif self._traits.tone == PersonalityTone.DIRECT:
            return f"{greeting}. Let's see how you're doing."
        else:
            return f"{greeting}! How can I help with your health today?"

    def _generate_insight_response(self, data: Dict) -> str:
        """Generate response for insights"""
        priority = data.get("priority", "medium")

        if priority == "high":
            if self._traits.empathy > 0.6:
                return "I noticed something important that I'd like to share with you."
            else:
                return "There's something important to discuss about your health."
        else:
            return "Here's something I thought you'd find helpful."

    def _generate_workout_response(self, data: Dict) -> str:
        """Generate workout suggestion response"""
        workout_type = data.get("workout_type", "workout")
        duration = data.get("duration", 30)

        if self._traits.tone == PersonalityTone.CHALLENGING:
            return f"Ready for a {duration}-minute {workout_type}? Let's do this!"
        elif self._traits.tone == PersonalityTone.ENCOURAGING:
            return f"How about a {duration}-minute {workout_type}? You've got this!"
        else:
            return f"I suggest a {duration}-minute {workout_type} for today."

    def _generate_meal_response(self, data: Dict) -> str:
        """Generate meal suggestion response"""
        meal_type = data.get("meal_type", "meal")
        calories = data.get("calories", 0)

        if self._traits.response_style == ResponseStyle.BRIEF:
            return f"{meal_type.title()}: ~{calories:.0f} calories"
        elif self._traits.response_style == ResponseStyle.DETAILED:
            return f"For your {meal_type}, I recommend a meal around {calories:.0f} calories to meet your goals."
        else:
            return f"Here's a {meal_type} suggestion at ~{calories:.0f} calories."

    def _generate_progress_response(self, data: Dict) -> str:
        """Generate progress update response"""
        workouts = data.get("workouts", 0)
        streak = data.get("streak", 0)

        if streak >= 3:
            return f"Amazing! You're on a {streak}-day streak! Keep it up!"
        elif workouts >= 4:
            return f"Great week! You completed {workouts} workouts. Consistency is key!"
        else:
            return f"You've done {workouts} workouts this week. Every step counts!"

    def _generate_motivation(self, context: InteractionContext = None) -> str:
        """Generate motivational message"""
        motivations = {
            PersonalityTone.ENCOURAGING: [
                "You've got this! Every small step leads to big changes.",
                "I'm proud of the progress you're making!",
                "Remember, consistency beats intensity. Keep going!",
            ],
            PersonalityTone.CHALLENGING: [
                "Your body is capable of amazing things. Prove it to yourself.",
                "Don't let today be the reason you give up on your goals.",
                "The only bad workout is the one that didn't happen.",
            ],
            PersonalityTone.INFORMATIONAL: [
                "Research shows that regular movement improves both physical and mental health.",
                "Small daily habits compound into significant results over time.",
                "Quality nutrition and exercise are the foundations of good health.",
            ],
        }

        messages = motivations.get(
            self._traits.tone, motivations[PersonalityTone.ENCOURAGING]
        )
        return random.choice(messages)

    def _generate_reminder(self, data: Dict) -> str:
        """Generate reminder message"""
        reminder_type = data.get("type", "health")

        if reminder_type == "water":
            return "Time to drink some water! Stay hydrated."
        elif reminder_type == "workout":
            return "Ready for your workout? Let's get moving!"
        elif reminder_type == "sleep":
            return "It's getting late. Time to start winding down."
        else:
            return "Here's your health reminder!"

    def _generate_default_response(self, response_type: str) -> str:
        """Generate default response"""
        return f"I'll help you with your {response_type}."

    def get_preferences(self) -> HealthPreferences:
        """Get user preferences"""
        return self._preferences or HealthPreferences()

    def set_motivational_style(self, style: str):
        """Set motivational style"""
        if not self._preferences:
            self._preferences = HealthPreferences()

        self._preferences.motivational_style = style

        if style == "encouraging":
            self._traits.tone = PersonalityTone.ENCOURAGING
            self._traits.empathy = 0.8
        elif style == "challenging":
            self._traits.tone = PersonalityTone.CHALLENGING
            self._traits.directness = 0.7

        self._save_preferences()

    def get_personality_summary(self) -> Dict[str, Any]:
        """Get personality summary"""
        return {
            "tone": self._traits.tone.value,
            "response_style": self._traits.response_style.value,
            "formality": self._traits.formality,
            "empathy": self._traits.empathy,
            "directness": self._traits.directness,
            "humor": self._traits.humor,
            "interaction_count": self._interaction_count,
            "liked_foods_count": len(self._preferences.liked_foods)
            if self._preferences
            else 0,
            "workout_preferences": self._preferences.workout_preferences
            if self._preferences
            else [],
        }
