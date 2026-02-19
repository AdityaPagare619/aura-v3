"""
AURA AGI Core - Advanced Capabilities for True Personal AGI

This module adds:
1. Curiosity Engine - Proactive learning
2. Theory of Mind - Understanding user mental state
3. Emotional Intelligence - Detecting and responding to emotions
4. Long-term Planning - Beyond current conversation
5. Proactive Assistance - Anticipating needs before user asks
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CuriosityItem:
    """Something AURA is curious about"""

    topic: str
    importance: float  # How important to learn
    last_explored: Optional[datetime] = None
    exploration_count: int = 0


@dataclass
class UserMentalModel:
    """AURA's model of the user's mind"""

    goals: List[str] = field(default_factory=list)
    beliefs: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    emotional_state: str = "neutral"
    confidence: float = 0.5


@dataclass
class EmotionalState:
    """Detected emotional state"""

    primary: str  # happy, sad, angry, scared, surprised, neutral
    intensity: float  # 0-1
    valence: float  # -1 (negative) to +1 (positive)
    arousal: float  # 0 (calm) to 1 (excited)
    timestamp: datetime = field(default_factory=datetime.now)


class CuriosityEngine:
    """
    AURA's Curiosity - Proactive Learning System

    Like humans, AURA should be curious and want to learn:
    - What does the user need before they ask?
    - What patterns can we discover?
    - What gaps in knowledge should we fill?
    """

    def __init__(self, memory):
        self.memory = memory
        self.curiosity_queue: List[CuriosityItem] = []
        self.explored_topics = set()
        self.curiosity_threshold = 0.7

    async def assess_curiosity(self, context: Dict) -> List[CuriosityItem]:
        """
        Assess what AURA should be curious about right now
        Based on: gaps in knowledge, user patterns, unanswered questions
        """
        curiosity_items = []

        # Check for knowledge gaps
        recent_memories = self.memory.retrieve("", limit=5)
        for mem in recent_memories:
            content = mem.get("content", "")
            if "?" in content or "don't know" in content.lower():
                curiosity_items.append(
                    CuriosityItem(
                        topic=f"knowledge_gap: {content[:50]}", importance=0.8
                    )
                )

        # Check user patterns that could be improved
        user_prefs = self.memory.self_model.get_category("preference")
        for pref_key in user_prefs:
            if user_prefs[pref_key].get("confidence", 0) < 0.7:
                curiosity_items.append(
                    CuriosityItem(
                        topic=f"preference_uncertainty: {pref_key}", importance=0.9
                    )
                )

        # Sort by importance
        curiosity_items.sort(key=lambda x: x.importance, reverse=True)
        return curiosity_items[:5]

    async def explore_topic(self, topic: str) -> Dict:
        """
        Proactively explore a topic of curiosity
        Like human: "I wonder about this, let me learn more"
        """
        logger.info(f"[Curiosity] Exploring: {topic}")

        # Record exploration
        if topic not in self.explored_topics:
            self.explored_topics.add(topic)

        # This could trigger:
        # - Searching through memories for related info
        # - Asking clarifying questions
        # - Making inferences

        return {
            "topic": topic,
            "explored": True,
            "new_understanding": f"Learned about {topic}",
        }

    async def get_curiosity_questions(self) -> List[str]:
        """
        Generate questions AURA is curious about
        These could be asked proactively to user
        """
        questions = []

        # Based on low-confidence preferences
        user_prefs = self.memory.self_model.get_category("preference")
        for pref_key, data in user_prefs.items():
            if data.get("confidence", 1.0) < 0.6:
                questions.append(
                    f"I'm not sure about your preference for {pref_key}. Could you clarify?"
                )

        return questions[:3]


class TheoryOfMind:
    """
    AURA's Theory of Mind - Understanding User's Mental State

    AURA models what the user:
    - Wants (goals)
    - Believes (knowledge/assumptions)
    - Intends (plans)
    - Feels (emotions)
    """

    def __init__(self, memory):
        self.memory = memory
        self.user_model = UserMentalModel()

    async def update_model(self, user_message: str, context: Dict):
        """
        Update user mental model based on interaction
        Like human: "I think they want this because..."
        """
        # Infer goals from message
        goal_indicators = ["want", "need", "help me", "can you"]
        for indicator in goal_indicators:
            if indicator in user_message.lower():
                # Extract goal (simplified)
                self.user_model.goals.append(user_message[:100])

        # Infer preferences from choices
        preference_indicators = ["like", "prefer", "always", "never"]
        for indicator in preference_indicators:
            if indicator in user_message.lower():
                # Extract preference
                pass

        # Update confidence based on interactions
        self.user_model.confidence = min(1.0, self.user_model.confidence + 0.05)

    async def predict_user_needs(self) -> List[Dict]:
        """
        Predict what user might need based on mental model
        Like human: "Based on what they want, they'll probably need X"
        """
        predictions = []

        # Time-based predictions
        hour = datetime.now().hour
        if 8 <= hour <= 9:
            predictions.append({"need": "morning briefing", "confidence": 0.8})
        elif 18 <= hour <= 19:
            predictions.append({"need": "evening summary", "confidence": 0.7})

        # Based on goals
        if self.user_model.goals:
            for goal in self.user_model.goals[-3:]:
                predictions.append({"need": f"help with: {goal}", "confidence": 0.6})

        return predictions

    async def get_user_context(self) -> Dict:
        """Get current understanding of user"""
        return {
            "goals": self.user_model.goals[-5:],
            "confidence": self.user_model.confidence,
            "emotional_state": self.user_model.emotional_state,
        }


class EmotionalIntelligence:
    """
    AURA's Emotional Intelligence

    Detects and responds to user emotions appropriately
    """

    EMOTION_KEYWORDS = {
        "happy": ["happy", "great", "awesome", "excellent", "wonderful", "love", "yay"],
        "sad": ["sad", "unhappy", "depressed", "down", "upset", "disappointed"],
        "angry": ["angry", "mad", "furious", "annoyed", "frustrated", "irritated"],
        "scared": ["scared", "afraid", "worried", "nervous", "anxious", "fear"],
        "surprised": ["surprised", "shocked", "amazing", "unbelievable", "wow"],
    }

    def __init__(self):
        self.current_state = EmotionalState("neutral", 0.0, 0.0, 0.0)
        self.history: List[EmotionalState] = []

    async def detect_emotion(self, text: str) -> EmotionalState:
        """
        Detect emotional state from text
        Like human: "They sound happy today"
        """
        text_lower = text.lower()

        # Count emotion keywords
        emotion_counts = defaultdict(int)
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_counts[emotion] += 1

        # Determine primary emotion
        if emotion_counts:
            primary = max(emotion_counts, key=emotion_counts.get)
            intensity = min(1.0, emotion_counts[primary] / 3)
        else:
            primary = "neutral"
            intensity = 0.0

        # Calculate valence and arousal
        valence = {
            "happy": 0.8,
            "surprised": 0.3,
            "neutral": 0,
            "sad": -0.8,
            "angry": -0.7,
            "scared": -0.5,
        }.get(primary, 0)
        arousal = {
            "angry": 0.9,
            "scared": 0.8,
            "happy": 0.6,
            "surprised": 0.7,
            "sad": 0.3,
            "neutral": 0.1,
        }.get(primary, 0)

        self.current_state = EmotionalState(
            primary=primary, intensity=intensity, valence=valence, arousal=arousal
        )

        self.history.append(self.current_state)

        return self.current_state

    async def get_emotional_response(self, emotion: EmotionalState) -> str:
        """
        Generate emotionally appropriate response
        Like human: "I can tell you're happy! Let me celebrate with you"
        """
        if emotion.primary == "happy":
            return "I'm glad to hear that! That's wonderful!"
        elif emotion.primary == "sad":
            return "I'm sorry you're feeling this way. How can I help?"
        elif emotion.primary == "angry":
            return "I understand you're frustrated. Let's see how I can help."
        elif emotion.primary == "scared":
            return "It's okay. I'm here to help. What concerns you?"
        elif emotion.primary == "surprised":
            return "That sounds unexpected! Tell me more."

        return ""


class LongTermPlanner:
    """
    AURA's Long-term Planning Engine

    Plans beyond current conversation:
    - Daily goals
    - Weekly objectives
    - Habit formation
    - Remembers recurring tasks
    """

    def __init__(self, memory):
        self.memory = memory
        self.plans: Dict[str, List[Dict]] = defaultdict(list)
        self.habits: List[Dict] = []

    async def create_daily_plan(self) -> List[Dict]:
        """
        Create daily plan based on habits and goals
        Like human: "What should I do today?"
        """
        today = datetime.now().date()

        # Get habits
        habits = await self._get_habits()

        # Create plan
        plan = []
        for habit in habits:
            plan.append(
                {
                    "task": habit["action"],
                    "recurrence": habit.get("recurrence", "daily"),
                    "priority": habit.get("priority", 0.5),
                }
            )

        # Add time-based tasks
        hour = datetime.now().hour
        if 8 <= hour <= 9:
            plan.append({"task": "morning check-in", "priority": 0.8})

        return plan

    async def _get_habits(self) -> List[Dict]:
        """Get learned habits"""
        return self.habits  # Would load from memory

    async def learn_habit(self, action: str, context: Dict):
        """Learn a new habit from repeated actions"""
        # If action repeated 3+ times, make it a habit
        pass

    async def suggest_improvements(self) -> List[str]:
        """Suggest areas for user improvement based on patterns"""
        suggestions = []

        # Analyze patterns
        # If user always forgets X, suggest reminder

        return suggestions


class ProactiveAssistant:
    """
    AURA's Proactive Assistance System

    Anticipates user needs before they ask:
    - Context-aware suggestions
    - Timely reminders
    - Anticipatory actions
    """

    def __init__(self, memory, theory_of_mind, emotional_intelligence):
        self.memory = memory
        self.tom = theory_of_mind
        self.ei = emotional_intelligence
        self.suggestions_queue: List[Dict] = []

    async def generate_suggestions(self, context: Dict) -> List[Dict]:
        """
        Generate proactive suggestions
        Like human: "I think they might need this..."
        """
        suggestions = []

        # Based on time
        hour = datetime.now().hour
        if hour == 8:
            suggestions.append(
                {
                    "type": "morning_brief",
                    "message": "Good morning! Here's your day ahead.",
                    "action": "show_daily_summary",
                }
            )

        # Based on user model predictions
        predictions = await self.tom.predict_user_needs()
        for pred in predictions[:2]:
            suggestions.append(
                {
                    "type": "predicted_need",
                    "message": f"You might need: {pred['need']}",
                    "confidence": pred["confidence"],
                }
            )

        # Based on emotional state
        emotion = self.ei.current_state
        if emotion.primary == "sad" and emotion.intensity > 0.6:
            suggestions.append(
                {
                    "type": "emotional_support",
                    "message": "You seem down. Want to talk about it?",
                }
            )

        return suggestions

    async def should_proact(self, suggestion: Dict) -> bool:
        """
        Decide if should proactively offer suggestion
        Like human: "Should I say this or wait?"
        """
        # High confidence threshold
        confidence = suggestion.get("confidence", 0.5)
        if confidence < 0.7:
            return False

        # Don't overwhelm user
        if len(self.suggestions_queue) > 2:
            return False

        return True


class AGICore:
    """
    Unified AGI Core - Combines all advanced capabilities
    """

    def __init__(self, memory):
        self.memory = memory
        self.curiosity = CuriosityEngine(memory)
        self.theory_of_mind = TheoryOfMind(memory)
        self.emotional_intelligence = EmotionalIntelligence()
        self.planner = LongTermPlanner(memory)
        self.proactive = ProactiveAssistant(
            memory, self.theory_of_mind, self.emotional_intelligence
        )

    async def process(self, user_message: str, context: Dict) -> Dict:
        """
        Process with full AGI capabilities
        """
        # 1. Detect emotions
        emotion = await self.emotional_intelligence.detect_emotion(user_message)

        # 2. Update theory of mind
        await self.theory_of_mind.update_model(user_message, context)

        # 3. Assess curiosity
        curiosity_items = await self.curiosity.assess_curiosity(context)

        # 4. Generate proactive suggestions
        suggestions = await self.proactive.generate_suggestions(context)

        return {
            "emotion": emotion,
            "curiosity": curiosity_items,
            "suggestions": suggestions,
            "user_model": await self.theory_of_mind.get_user_context(),
        }


__all__ = [
    "AGICore",
    "CuriosityEngine",
    "TheoryOfMind",
    "EmotionalIntelligence",
    "LongTermPlanner",
    "ProactiveAssistant",
    "CuriosityItem",
    "UserMentalModel",
    "EmotionalState",
]
