"""
AURA AGI Core - Advanced Capabilities for True Personal AGI

This module adds:
1. Curiosity Engine - Proactive learning
2. Theory of Mind - Understanding user mental state
3. Emotional Intelligence - Detecting and responding to emotions (valence/arousal scoring)
4. Long-term Planning - Beyond current conversation (with real habit learning)
5. Proactive Assistance - Anticipating needs before user asks (Markov chain predictions)
"""

import asyncio
import json
import logging
import math
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


# ==============================================================================
# EMOTION LEXICON - ANEW-style valence/arousal ratings (~500 words)
# Valence: 0 (negative) to 1 (positive)
# Arousal: 0 (calm) to 1 (excited/activated)
# ==============================================================================

EMOTION_LEXICON: Dict[str, Dict[str, float]] = {
    # === Positive High Arousal ===
    "ecstatic": {"valence": 0.95, "arousal": 0.95},
    "thrilled": {"valence": 0.92, "arousal": 0.90},
    "excited": {"valence": 0.88, "arousal": 0.88},
    "elated": {"valence": 0.90, "arousal": 0.85},
    "exhilarated": {"valence": 0.90, "arousal": 0.90},
    "enthusiastic": {"valence": 0.85, "arousal": 0.80},
    "energetic": {"valence": 0.75, "arousal": 0.85},
    "passionate": {"valence": 0.82, "arousal": 0.82},
    "delighted": {"valence": 0.90, "arousal": 0.75},
    "overjoyed": {"valence": 0.92, "arousal": 0.80},
    "euphoric": {"valence": 0.95, "arousal": 0.88},
    "jubilant": {"valence": 0.90, "arousal": 0.85},
    "amazed": {"valence": 0.80, "arousal": 0.82},
    "astonished": {"valence": 0.70, "arousal": 0.85},
    "surprised": {"valence": 0.65, "arousal": 0.80},
    "eager": {"valence": 0.78, "arousal": 0.75},
    "inspired": {"valence": 0.85, "arousal": 0.70},
    "triumphant": {"valence": 0.88, "arousal": 0.78},
    "victorious": {"valence": 0.88, "arousal": 0.75},
    "pumped": {"valence": 0.80, "arousal": 0.88},
    # === Positive Medium Arousal ===
    "happy": {"valence": 0.88, "arousal": 0.60},
    "joyful": {"valence": 0.90, "arousal": 0.65},
    "cheerful": {"valence": 0.85, "arousal": 0.60},
    "glad": {"valence": 0.80, "arousal": 0.50},
    "pleased": {"valence": 0.78, "arousal": 0.45},
    "satisfied": {"valence": 0.75, "arousal": 0.40},
    "grateful": {"valence": 0.85, "arousal": 0.50},
    "thankful": {"valence": 0.82, "arousal": 0.45},
    "appreciative": {"valence": 0.80, "arousal": 0.42},
    "optimistic": {"valence": 0.80, "arousal": 0.55},
    "hopeful": {"valence": 0.78, "arousal": 0.50},
    "confident": {"valence": 0.75, "arousal": 0.55},
    "proud": {"valence": 0.82, "arousal": 0.55},
    "accomplished": {"valence": 0.80, "arousal": 0.50},
    "amused": {"valence": 0.78, "arousal": 0.55},
    "playful": {"valence": 0.80, "arousal": 0.65},
    "loving": {"valence": 0.90, "arousal": 0.55},
    "affectionate": {"valence": 0.88, "arousal": 0.50},
    "caring": {"valence": 0.85, "arousal": 0.45},
    "warm": {"valence": 0.80, "arousal": 0.45},
    "friendly": {"valence": 0.78, "arousal": 0.50},
    "kind": {"valence": 0.80, "arousal": 0.40},
    "compassionate": {"valence": 0.82, "arousal": 0.45},
    "empathetic": {"valence": 0.78, "arousal": 0.42},
    "wonderful": {"valence": 0.90, "arousal": 0.60},
    "great": {"valence": 0.82, "arousal": 0.55},
    "awesome": {"valence": 0.88, "arousal": 0.70},
    "fantastic": {"valence": 0.88, "arousal": 0.65},
    "excellent": {"valence": 0.85, "arousal": 0.55},
    "amazing": {"valence": 0.88, "arousal": 0.68},
    "incredible": {"valence": 0.85, "arousal": 0.65},
    "lovely": {"valence": 0.85, "arousal": 0.50},
    "beautiful": {"valence": 0.85, "arousal": 0.50},
    "good": {"valence": 0.72, "arousal": 0.40},
    "nice": {"valence": 0.70, "arousal": 0.38},
    "fine": {"valence": 0.60, "arousal": 0.30},
    "okay": {"valence": 0.55, "arousal": 0.25},
    "love": {"valence": 0.92, "arousal": 0.65},
    "adore": {"valence": 0.90, "arousal": 0.60},
    "like": {"valence": 0.70, "arousal": 0.40},
    "enjoy": {"valence": 0.75, "arousal": 0.50},
    "yay": {"valence": 0.85, "arousal": 0.75},
    "wow": {"valence": 0.75, "arousal": 0.80},
    "hooray": {"valence": 0.88, "arousal": 0.78},
    "yes": {"valence": 0.65, "arousal": 0.45},
    "perfect": {"valence": 0.90, "arousal": 0.55},
    "blessed": {"valence": 0.85, "arousal": 0.45},
    "lucky": {"valence": 0.78, "arousal": 0.50},
    "fortunate": {"valence": 0.78, "arousal": 0.42},
    # === Positive Low Arousal ===
    "calm": {"valence": 0.70, "arousal": 0.20},
    "peaceful": {"valence": 0.78, "arousal": 0.18},
    "serene": {"valence": 0.80, "arousal": 0.15},
    "tranquil": {"valence": 0.78, "arousal": 0.15},
    "relaxed": {"valence": 0.75, "arousal": 0.22},
    "comfortable": {"valence": 0.72, "arousal": 0.25},
    "content": {"valence": 0.75, "arousal": 0.28},
    "mellow": {"valence": 0.68, "arousal": 0.20},
    "soothed": {"valence": 0.72, "arousal": 0.20},
    "restful": {"valence": 0.70, "arousal": 0.18},
    "relieved": {"valence": 0.72, "arousal": 0.35},
    "secure": {"valence": 0.72, "arousal": 0.30},
    "safe": {"valence": 0.70, "arousal": 0.28},
    "cozy": {"valence": 0.75, "arousal": 0.22},
    "sleepy": {"valence": 0.55, "arousal": 0.12},
    "drowsy": {"valence": 0.50, "arousal": 0.10},
    # === Negative High Arousal ===
    "furious": {"valence": 0.10, "arousal": 0.95},
    "enraged": {"valence": 0.08, "arousal": 0.95},
    "livid": {"valence": 0.10, "arousal": 0.92},
    "outraged": {"valence": 0.12, "arousal": 0.90},
    "angry": {"valence": 0.15, "arousal": 0.85},
    "mad": {"valence": 0.18, "arousal": 0.80},
    "irate": {"valence": 0.15, "arousal": 0.85},
    "infuriated": {"valence": 0.12, "arousal": 0.88},
    "hostile": {"valence": 0.15, "arousal": 0.82},
    "aggressive": {"valence": 0.18, "arousal": 0.80},
    "hateful": {"valence": 0.08, "arousal": 0.78},
    "hate": {"valence": 0.10, "arousal": 0.80},
    "terrified": {"valence": 0.12, "arousal": 0.92},
    "horrified": {"valence": 0.10, "arousal": 0.90},
    "panicked": {"valence": 0.15, "arousal": 0.92},
    "petrified": {"valence": 0.12, "arousal": 0.88},
    "scared": {"valence": 0.18, "arousal": 0.82},
    "afraid": {"valence": 0.20, "arousal": 0.78},
    "frightened": {"valence": 0.18, "arousal": 0.80},
    "alarmed": {"valence": 0.25, "arousal": 0.82},
    "shocked": {"valence": 0.30, "arousal": 0.85},
    "hysterical": {"valence": 0.15, "arousal": 0.92},
    "frantic": {"valence": 0.18, "arousal": 0.90},
    "desperate": {"valence": 0.15, "arousal": 0.85},
    "distressed": {"valence": 0.20, "arousal": 0.75},
    "overwhelmed": {"valence": 0.22, "arousal": 0.80},
    "stressed": {"valence": 0.25, "arousal": 0.75},
    "tense": {"valence": 0.28, "arousal": 0.70},
    "anxious": {"valence": 0.25, "arousal": 0.72},
    "nervous": {"valence": 0.28, "arousal": 0.68},
    "worried": {"valence": 0.30, "arousal": 0.62},
    "apprehensive": {"valence": 0.32, "arousal": 0.60},
    "restless": {"valence": 0.35, "arousal": 0.65},
    "agitated": {"valence": 0.28, "arousal": 0.72},
    "jittery": {"valence": 0.30, "arousal": 0.70},
    "uneasy": {"valence": 0.32, "arousal": 0.58},
    "uncomfortable": {"valence": 0.35, "arousal": 0.50},
    "annoyed": {"valence": 0.28, "arousal": 0.62},
    "irritated": {"valence": 0.25, "arousal": 0.65},
    "frustrated": {"valence": 0.22, "arousal": 0.70},
    "exasperated": {"valence": 0.20, "arousal": 0.72},
    "impatient": {"valence": 0.30, "arousal": 0.65},
    # === Negative Medium Arousal ===
    "sad": {"valence": 0.18, "arousal": 0.35},
    "unhappy": {"valence": 0.22, "arousal": 0.38},
    "miserable": {"valence": 0.10, "arousal": 0.42},
    "sorrowful": {"valence": 0.15, "arousal": 0.35},
    "mournful": {"valence": 0.15, "arousal": 0.32},
    "grieving": {"valence": 0.12, "arousal": 0.45},
    "heartbroken": {"valence": 0.10, "arousal": 0.50},
    "devastated": {"valence": 0.08, "arousal": 0.60},
    "crushed": {"valence": 0.10, "arousal": 0.55},
    "disappointed": {"valence": 0.25, "arousal": 0.42},
    "let down": {"valence": 0.28, "arousal": 0.38},
    "disheartened": {"valence": 0.25, "arousal": 0.35},
    "discouraged": {"valence": 0.28, "arousal": 0.38},
    "dismayed": {"valence": 0.25, "arousal": 0.50},
    "upset": {"valence": 0.22, "arousal": 0.55},
    "hurt": {"valence": 0.20, "arousal": 0.50},
    "offended": {"valence": 0.25, "arousal": 0.55},
    "insulted": {"valence": 0.22, "arousal": 0.58},
    "betrayed": {"valence": 0.12, "arousal": 0.60},
    "jealous": {"valence": 0.22, "arousal": 0.62},
    "envious": {"valence": 0.25, "arousal": 0.55},
    "resentful": {"valence": 0.20, "arousal": 0.55},
    "bitter": {"valence": 0.18, "arousal": 0.50},
    "guilty": {"valence": 0.22, "arousal": 0.50},
    "ashamed": {"valence": 0.18, "arousal": 0.52},
    "embarrassed": {"valence": 0.25, "arousal": 0.55},
    "humiliated": {"valence": 0.12, "arousal": 0.60},
    "regretful": {"valence": 0.25, "arousal": 0.42},
    "remorseful": {"valence": 0.22, "arousal": 0.45},
    "sorry": {"valence": 0.30, "arousal": 0.40},
    "confused": {"valence": 0.35, "arousal": 0.55},
    "puzzled": {"valence": 0.40, "arousal": 0.50},
    "perplexed": {"valence": 0.38, "arousal": 0.52},
    "bewildered": {"valence": 0.32, "arousal": 0.58},
    "lost": {"valence": 0.28, "arousal": 0.48},
    "uncertain": {"valence": 0.35, "arousal": 0.45},
    "doubtful": {"valence": 0.35, "arousal": 0.42},
    "skeptical": {"valence": 0.38, "arousal": 0.45},
    "suspicious": {"valence": 0.32, "arousal": 0.52},
    "distrustful": {"valence": 0.28, "arousal": 0.50},
    "disgusted": {"valence": 0.15, "arousal": 0.62},
    "repulsed": {"valence": 0.12, "arousal": 0.65},
    "revolted": {"valence": 0.10, "arousal": 0.68},
    "nauseated": {"valence": 0.20, "arousal": 0.55},
    "sick": {"valence": 0.25, "arousal": 0.45},
    "contemptuous": {"valence": 0.18, "arousal": 0.55},
    "scornful": {"valence": 0.20, "arousal": 0.55},
    "bad": {"valence": 0.25, "arousal": 0.45},
    "terrible": {"valence": 0.12, "arousal": 0.55},
    "awful": {"valence": 0.15, "arousal": 0.52},
    "horrible": {"valence": 0.10, "arousal": 0.58},
    "dreadful": {"valence": 0.12, "arousal": 0.55},
    "no": {"valence": 0.35, "arousal": 0.45},
    "hate": {"valence": 0.10, "arousal": 0.70},
    "dislike": {"valence": 0.30, "arousal": 0.45},
    "wrong": {"valence": 0.30, "arousal": 0.50},
    "fail": {"valence": 0.20, "arousal": 0.55},
    "failed": {"valence": 0.18, "arousal": 0.52},
    "failure": {"valence": 0.15, "arousal": 0.55},
    # === Negative Low Arousal ===
    "depressed": {"valence": 0.10, "arousal": 0.22},
    "hopeless": {"valence": 0.08, "arousal": 0.25},
    "despairing": {"valence": 0.08, "arousal": 0.30},
    "dejected": {"valence": 0.15, "arousal": 0.25},
    "gloomy": {"valence": 0.20, "arousal": 0.28},
    "melancholic": {"valence": 0.18, "arousal": 0.22},
    "blue": {"valence": 0.22, "arousal": 0.28},
    "down": {"valence": 0.25, "arousal": 0.30},
    "low": {"valence": 0.28, "arousal": 0.28},
    "empty": {"valence": 0.18, "arousal": 0.20},
    "hollow": {"valence": 0.20, "arousal": 0.18},
    "numb": {"valence": 0.25, "arousal": 0.15},
    "apathetic": {"valence": 0.30, "arousal": 0.12},
    "indifferent": {"valence": 0.40, "arousal": 0.15},
    "bored": {"valence": 0.32, "arousal": 0.18},
    "weary": {"valence": 0.28, "arousal": 0.20},
    "exhausted": {"valence": 0.22, "arousal": 0.15},
    "fatigued": {"valence": 0.28, "arousal": 0.18},
    "tired": {"valence": 0.32, "arousal": 0.20},
    "drained": {"valence": 0.25, "arousal": 0.18},
    "burned out": {"valence": 0.18, "arousal": 0.15},
    "lonely": {"valence": 0.15, "arousal": 0.28},
    "isolated": {"valence": 0.18, "arousal": 0.25},
    "alone": {"valence": 0.22, "arousal": 0.25},
    "abandoned": {"valence": 0.12, "arousal": 0.35},
    "rejected": {"valence": 0.15, "arousal": 0.40},
    "excluded": {"valence": 0.20, "arousal": 0.35},
    "ignored": {"valence": 0.22, "arousal": 0.35},
    "neglected": {"valence": 0.20, "arousal": 0.32},
    "helpless": {"valence": 0.15, "arousal": 0.30},
    "powerless": {"valence": 0.18, "arousal": 0.28},
    "vulnerable": {"valence": 0.25, "arousal": 0.35},
    "insecure": {"valence": 0.28, "arousal": 0.40},
    "worthless": {"valence": 0.08, "arousal": 0.25},
    "useless": {"valence": 0.12, "arousal": 0.28},
    "inadequate": {"valence": 0.20, "arousal": 0.32},
    "inferior": {"valence": 0.18, "arousal": 0.30},
    # === Neutral ===
    "neutral": {"valence": 0.50, "arousal": 0.30},
    "okay": {"valence": 0.52, "arousal": 0.28},
    "alright": {"valence": 0.52, "arousal": 0.28},
    "fine": {"valence": 0.55, "arousal": 0.30},
    "normal": {"valence": 0.50, "arousal": 0.30},
    "usual": {"valence": 0.50, "arousal": 0.28},
    "ordinary": {"valence": 0.48, "arousal": 0.25},
    "average": {"valence": 0.48, "arousal": 0.28},
    "so-so": {"valence": 0.45, "arousal": 0.25},
    "meh": {"valence": 0.42, "arousal": 0.22},
    # === Additional common words ===
    "want": {"valence": 0.58, "arousal": 0.50},
    "need": {"valence": 0.52, "arousal": 0.55},
    "help": {"valence": 0.55, "arousal": 0.50},
    "please": {"valence": 0.60, "arousal": 0.40},
    "thanks": {"valence": 0.75, "arousal": 0.45},
    "thank": {"valence": 0.75, "arousal": 0.45},
    "welcome": {"valence": 0.70, "arousal": 0.40},
    "hello": {"valence": 0.65, "arousal": 0.45},
    "hi": {"valence": 0.62, "arousal": 0.45},
    "bye": {"valence": 0.55, "arousal": 0.35},
    "goodbye": {"valence": 0.55, "arousal": 0.35},
    "sorry": {"valence": 0.35, "arousal": 0.42},
    "urgent": {"valence": 0.35, "arousal": 0.78},
    "important": {"valence": 0.55, "arousal": 0.60},
    "critical": {"valence": 0.38, "arousal": 0.72},
    "emergency": {"valence": 0.20, "arousal": 0.90},
    "crisis": {"valence": 0.15, "arousal": 0.85},
    "problem": {"valence": 0.30, "arousal": 0.55},
    "issue": {"valence": 0.35, "arousal": 0.50},
    "trouble": {"valence": 0.28, "arousal": 0.55},
    "difficulty": {"valence": 0.32, "arousal": 0.50},
    "challenge": {"valence": 0.45, "arousal": 0.55},
    "success": {"valence": 0.85, "arousal": 0.65},
    "win": {"valence": 0.88, "arousal": 0.70},
    "won": {"valence": 0.88, "arousal": 0.70},
    "achieve": {"valence": 0.82, "arousal": 0.60},
    "achieved": {"valence": 0.82, "arousal": 0.58},
    "accomplish": {"valence": 0.80, "arousal": 0.58},
    "complete": {"valence": 0.72, "arousal": 0.50},
    "finished": {"valence": 0.68, "arousal": 0.45},
    "done": {"valence": 0.65, "arousal": 0.40},
    "progress": {"valence": 0.70, "arousal": 0.50},
    "improve": {"valence": 0.72, "arousal": 0.52},
    "better": {"valence": 0.70, "arousal": 0.48},
    "worse": {"valence": 0.28, "arousal": 0.50},
    "worst": {"valence": 0.12, "arousal": 0.55},
    "best": {"valence": 0.88, "arousal": 0.55},
}


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
    valence: float  # 0 (negative) to 1 (positive)
    arousal: float  # 0 (calm) to 1 (excited)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Goal:
    """A tracked user goal with priority and progress"""

    id: str
    description: str
    importance: float  # 0-1, user-defined or inferred importance
    urgency: float  # 0-1, based on deadline proximity
    progress: float  # 0-1, percentage complete
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    sub_goals: List[str] = field(default_factory=list)

    @property
    def priority(self) -> float:
        """Calculate priority score: importance × urgency"""
        return self.importance * self.urgency

    def update_urgency_from_deadline(self):
        """Update urgency based on deadline proximity"""
        if not self.deadline:
            return

        now = datetime.now()
        if self.deadline <= now:
            self.urgency = 1.0  # Past deadline, maximum urgency
        else:
            time_remaining = (self.deadline - now).total_seconds()
            total_time = (self.deadline - self.created_at).total_seconds()
            if total_time > 0:
                # Urgency increases as deadline approaches (exponential curve)
                time_ratio = time_remaining / total_time
                self.urgency = min(1.0, 1.0 - (time_ratio**0.5) * 0.8 + 0.2)
            else:
                self.urgency = 1.0


@dataclass
class Habit:
    """A learned user habit"""

    action: str
    typical_hour: int  # Hour of day (0-23) when action typically occurs
    typical_day_of_week: Optional[int] = None  # 0=Monday, 6=Sunday
    occurrence_count: int = 0
    confidence: float = 0.0  # 0-1, how confident we are this is a habit
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "typical_hour": self.typical_hour,
            "typical_day_of_week": self.typical_day_of_week,
            "occurrence_count": self.occurrence_count,
            "confidence": self.confidence,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Habit":
        return cls(
            action=data["action"],
            typical_hour=data["typical_hour"],
            typical_day_of_week=data.get("typical_day_of_week"),
            occurrence_count=data.get("occurrence_count", 0),
            confidence=data.get("confidence", 0.0),
            first_seen=datetime.fromisoformat(data["first_seen"])
            if "first_seen" in data
            else datetime.now(),
            last_seen=datetime.fromisoformat(data["last_seen"])
            if "last_seen" in data
            else datetime.now(),
        )


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


class MarkovPredictor:
    """
    Markov Chain-based user behavior predictor.

    Tracks state transitions and predicts likely next actions.
    Mobile-efficient: O(1) prediction, O(n) state updates where n = history length.
    """

    def __init__(self, data_dir: Optional[str] = None):
        # Transition counts: {current_state: {next_state: count}}
        self.transitions: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # State history for pattern detection
        self.state_history: List[Tuple[str, datetime]] = []
        self.max_history = 1000  # Keep last 1000 states

        # Persistence
        self.data_dir = data_dir or os.path.join(
            os.path.expanduser("~"), ".aura", "agi"
        )
        self._ensure_data_dir()
        self._load_transitions()

    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    def _get_transitions_path(self) -> str:
        return os.path.join(self.data_dir, "markov_transitions.json")

    def _load_transitions(self):
        """Load transitions from disk"""
        path = self._get_transitions_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    # Convert to defaultdict structure
                    for state, nexts in data.get("transitions", {}).items():
                        for next_state, count in nexts.items():
                            self.transitions[state][next_state] = count
                    # Load history
                    for item in data.get("history", []):
                        self.state_history.append(
                            (item["state"], datetime.fromisoformat(item["timestamp"]))
                        )
            except Exception as e:
                logger.warning(f"Failed to load Markov transitions: {e}")

    def _save_transitions(self):
        """Persist transitions to disk"""
        path = self._get_transitions_path()
        try:
            data = {
                "transitions": {k: dict(v) for k, v in self.transitions.items()},
                "history": [
                    {"state": s, "timestamp": t.isoformat()}
                    for s, t in self.state_history[-100:]  # Only save recent history
                ],
            }
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save Markov transitions: {e}")

    def observe(self, state: str, timestamp: Optional[datetime] = None):
        """
        Record a state observation.
        Updates transition probabilities from previous state.
        """
        timestamp = timestamp or datetime.now()

        # Update transition from last state
        if self.state_history:
            prev_state, _ = self.state_history[-1]
            self.transitions[prev_state][state] += 1

        # Add to history
        self.state_history.append((state, timestamp))

        # Trim history if needed
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history :]

        # Persist periodically (every 10 observations)
        if len(self.state_history) % 10 == 0:
            self._save_transitions()

    def predict(self, current_state: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Predict likely next states given current state.

        Returns list of {state, probability, confidence} sorted by probability.
        """
        if current_state not in self.transitions:
            return []

        next_states = self.transitions[current_state]
        total = sum(next_states.values())

        if total == 0:
            return []

        predictions = []
        for state, count in sorted(next_states.items(), key=lambda x: -x[1])[:top_k]:
            prob = count / total
            # Confidence increases with sample size (log scale)
            confidence = min(1.0, math.log(total + 1) / 5)
            predictions.append(
                {
                    "state": state,
                    "probability": prob,
                    "confidence": confidence,
                    "sample_count": count,
                }
            )

        return predictions

    def get_transition_probability(self, from_state: str, to_state: str) -> float:
        """Get probability of transitioning from one state to another"""
        if from_state not in self.transitions:
            return 0.0

        next_states = self.transitions[from_state]
        total = sum(next_states.values())

        if total == 0:
            return 0.0

        return next_states.get(to_state, 0) / total


class TheoryOfMind:
    """
    AURA's Theory of Mind - Understanding User's Mental State

    AURA models what the user:
    - Wants (goals)
    - Believes (knowledge/assumptions)
    - Intends (plans)
    - Feels (emotions)

    Now uses Markov chain predictions instead of time-of-day heuristics.
    """

    def __init__(self, memory, data_dir: Optional[str] = None):
        self.memory = memory
        self.user_model = UserMentalModel()
        self.predictor = MarkovPredictor(data_dir)
        self._last_action: Optional[str] = None

    async def update_model(self, user_message: str, context: Dict):
        """
        Update user mental model based on interaction
        Like human: "I think they want this because..."
        """
        # Extract action/intent from message for Markov tracking
        action = self._extract_action(user_message, context)
        if action:
            self.predictor.observe(action)
            self._last_action = action

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

    def _extract_action(self, message: str, context: Dict) -> Optional[str]:
        """Extract a categorized action from user message"""
        message_lower = message.lower()

        # Action categories for Markov tracking
        action_patterns = {
            "ask_question": ["?", "what", "how", "why", "when", "where", "who"],
            "request_help": ["help", "assist", "support", "guide"],
            "set_reminder": ["remind", "reminder", "alarm", "schedule"],
            "check_info": ["check", "look up", "search", "find"],
            "manage_task": ["task", "todo", "to-do", "add", "complete", "done"],
            "casual_chat": ["hello", "hi", "hey", "how are you", "what's up"],
            "express_feeling": ["feel", "feeling", "mood", "happy", "sad", "angry"],
            "request_summary": ["summary", "summarize", "recap", "overview"],
            "planning": ["plan", "planning", "schedule", "organize"],
        }

        for action, patterns in action_patterns.items():
            for pattern in patterns:
                if pattern in message_lower:
                    return action

        return "general_message"

    async def predict_user_needs(self) -> List[Dict]:
        """
        Predict what user might need based on Markov chain model.
        Replaces time-of-day heuristics with real behavioral predictions.
        """
        predictions = []

        # Use Markov chain predictions based on last action
        if self._last_action:
            markov_predictions = self.predictor.predict(self._last_action, top_k=3)
            for pred in markov_predictions:
                if pred["probability"] > 0.1:  # Only include likely predictions
                    predictions.append(
                        {
                            "need": self._action_to_need(pred["state"]),
                            "confidence": pred["probability"] * pred["confidence"],
                            "based_on": "behavior_pattern",
                            "next_action": pred["state"],
                        }
                    )

        # Add time-context predictions with lower priority (fallback)
        hour = datetime.now().hour
        if 6 <= hour <= 8 and not any(
            p.get("need") == "morning briefing" for p in predictions
        ):
            predictions.append(
                {
                    "need": "morning briefing",
                    "confidence": 0.3,  # Lower confidence for time-based
                    "based_on": "time_of_day",
                }
            )
        elif 17 <= hour <= 19 and not any(
            p.get("need") == "evening summary" for p in predictions
        ):
            predictions.append(
                {
                    "need": "evening summary",
                    "confidence": 0.3,
                    "based_on": "time_of_day",
                }
            )

        # Sort by confidence and return top predictions
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:5]

    def _action_to_need(self, action: str) -> str:
        """Convert action category to user need description"""
        need_mapping = {
            "ask_question": "answer to a question",
            "request_help": "assistance with a task",
            "set_reminder": "reminder management",
            "check_info": "information lookup",
            "manage_task": "task management",
            "casual_chat": "friendly conversation",
            "express_feeling": "emotional support",
            "request_summary": "summary of activities",
            "planning": "help with planning",
            "general_message": "general assistance",
        }
        return need_mapping.get(action, f"help with {action}")

    async def get_user_context(self) -> Dict:
        """Get current understanding of user"""
        return {
            "goals": self.user_model.goals[-5:],
            "confidence": self.user_model.confidence,
            "emotional_state": self.user_model.emotional_state,
            "predicted_next_actions": self.predictor.predict(self._last_action)
            if self._last_action
            else [],
        }


class EmotionalIntelligence:
    """
    AURA's Emotional Intelligence

    Detects and responds to user emotions using valence/arousal scoring
    with a comprehensive emotion lexicon (ANEW-style).
    """

    def __init__(self):
        self.current_state = EmotionalState("neutral", 0.0, 0.5, 0.3)
        self.history: List[EmotionalState] = []
        self.lexicon = EMOTION_LEXICON

    async def detect_emotion(self, text: str) -> EmotionalState:
        """
        Detect emotional state from text using valence/arousal scoring.
        Uses weighted average of detected emotion words.
        """
        text_lower = text.lower()
        words = self._tokenize(text_lower)

        # Collect valence and arousal scores from matched words
        valence_scores: List[Tuple[float, float]] = []  # (score, weight)
        arousal_scores: List[Tuple[float, float]] = []

        for word in words:
            if word in self.lexicon:
                entry = self.lexicon[word]
                # Weight by word position (words at end often more emotionally salient)
                position_weight = 0.8 + 0.4 * (words.index(word) / max(len(words), 1))
                valence_scores.append((entry["valence"], position_weight))
                arousal_scores.append((entry["arousal"], position_weight))

        # Also check for multi-word phrases
        for phrase, entry in self.lexicon.items():
            if " " in phrase and phrase in text_lower:
                valence_scores.append(
                    (entry["valence"], 1.2)
                )  # Phrases get higher weight
                arousal_scores.append((entry["arousal"], 1.2))

        # Calculate weighted averages
        if valence_scores:
            total_v_weight = sum(w for _, w in valence_scores)
            valence = sum(s * w for s, w in valence_scores) / total_v_weight
            total_a_weight = sum(w for _, w in arousal_scores)
            arousal = sum(s * w for s, w in arousal_scores) / total_a_weight
            # Intensity based on how many emotion words found
            intensity = min(1.0, len(valence_scores) / 5)
        else:
            # No emotion words found - neutral
            valence = 0.5
            arousal = 0.3
            intensity = 0.0

        # Determine primary emotion category from valence/arousal quadrant
        primary = self._classify_emotion(valence, arousal, intensity)

        self.current_state = EmotionalState(
            primary=primary, intensity=intensity, valence=valence, arousal=arousal
        )

        self.history.append(self.current_state)

        # Trim history
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return self.current_state

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for emotion word matching"""
        # Remove punctuation except apostrophes
        import re

        text = re.sub(r"[^\w\s'-]", " ", text)
        return text.split()

    def _classify_emotion(
        self, valence: float, arousal: float, intensity: float
    ) -> str:
        """
        Classify emotion into category based on valence/arousal.
        Uses the circumplex model of affect.
        """
        if intensity < 0.1:
            return "neutral"

        # High valence (positive)
        if valence >= 0.6:
            if arousal >= 0.6:
                return "excited"  # High valence, high arousal
            elif arousal >= 0.4:
                return "happy"  # High valence, medium arousal
            else:
                return "calm"  # High valence, low arousal (content, peaceful)

        # Low valence (negative)
        elif valence <= 0.4:
            if arousal >= 0.7:
                return "angry"  # Low valence, high arousal
            elif arousal >= 0.5:
                return "anxious"  # Low valence, medium-high arousal
            elif arousal >= 0.3:
                return "sad"  # Low valence, medium-low arousal
            else:
                return "depressed"  # Low valence, low arousal

        # Medium valence (mixed/ambivalent)
        else:
            if arousal >= 0.6:
                return "surprised"  # Neutral valence, high arousal
            else:
                return "neutral"

    async def get_emotional_response(self, emotion: EmotionalState) -> str:
        """
        Generate emotionally appropriate response
        Like human: "I can tell you're happy! Let me celebrate with you"
        """
        responses = {
            "excited": "That's fantastic! I can feel your excitement!",
            "happy": "I'm glad to hear that! That's wonderful!",
            "calm": "You seem at peace. That's lovely.",
            "neutral": "",
            "surprised": "That sounds unexpected! Tell me more.",
            "anxious": "I sense some worry. I'm here if you need to talk.",
            "sad": "I'm sorry you're feeling this way. How can I help?",
            "angry": "I understand you're frustrated. Let's see how I can help.",
            "depressed": "It sounds like things are tough. I'm here for you.",
        }

        return responses.get(emotion.primary, "")

    def get_emotional_trend(self, window: int = 10) -> Dict[str, Any]:
        """Analyze emotional trend over recent interactions"""
        if len(self.history) < 2:
            return {"trend": "stable", "direction": 0}

        recent = self.history[-window:]

        # Calculate trend
        valence_trend = recent[-1].valence - recent[0].valence
        arousal_trend = recent[-1].arousal - recent[0].arousal

        # Determine overall trend
        if abs(valence_trend) < 0.1:
            trend = "stable"
        elif valence_trend > 0:
            trend = "improving"
        else:
            trend = "declining"

        return {
            "trend": trend,
            "valence_change": valence_trend,
            "arousal_change": arousal_trend,
            "average_valence": sum(e.valence for e in recent) / len(recent),
            "average_arousal": sum(e.arousal for e in recent) / len(recent),
        }


class LongTermPlanner:
    """
    AURA's Long-term Planning Engine

    Plans beyond current conversation:
    - Real goal management with priority scoring
    - Habit learning with confidence scores
    - Progress tracking
    - Deadline awareness
    """

    def __init__(self, memory, data_dir: Optional[str] = None):
        self.memory = memory
        self.data_dir = data_dir or os.path.join(
            os.path.expanduser("~"), ".aura", "agi"
        )
        self._ensure_data_dir()

        # Goals with priority scoring
        self.goals: Dict[str, Goal] = {}

        # Habit tracking: {action: {hour: count}}
        self._action_time_counts: Dict[str, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._action_day_counts: Dict[str, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.habits: Dict[str, Habit] = {}

        # Load persisted data
        self._load_data()

    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    def _get_data_path(self) -> str:
        return os.path.join(self.data_dir, "planner_data.json")

    def _load_data(self):
        """Load persisted planner data"""
        path = self._get_data_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)

                    # Load habits
                    for habit_data in data.get("habits", []):
                        habit = Habit.from_dict(habit_data)
                        self.habits[habit.action] = habit

                    # Load action counts
                    for action, hour_counts in data.get(
                        "action_time_counts", {}
                    ).items():
                        for hour_str, count in hour_counts.items():
                            self._action_time_counts[action][int(hour_str)] = count

                    for action, day_counts in data.get("action_day_counts", {}).items():
                        for day_str, count in day_counts.items():
                            self._action_day_counts[action][int(day_str)] = count

                    # Load goals
                    for goal_data in data.get("goals", []):
                        goal = Goal(
                            id=goal_data["id"],
                            description=goal_data["description"],
                            importance=goal_data["importance"],
                            urgency=goal_data["urgency"],
                            progress=goal_data["progress"],
                            deadline=datetime.fromisoformat(goal_data["deadline"])
                            if goal_data.get("deadline")
                            else None,
                            created_at=datetime.fromisoformat(goal_data["created_at"])
                            if goal_data.get("created_at")
                            else datetime.now(),
                            completed=goal_data.get("completed", False),
                            sub_goals=goal_data.get("sub_goals", []),
                        )
                        self.goals[goal.id] = goal

            except Exception as e:
                logger.warning(f"Failed to load planner data: {e}")

    def _save_data(self):
        """Persist planner data"""
        path = self._get_data_path()
        try:
            data = {
                "habits": [h.to_dict() for h in self.habits.values()],
                "action_time_counts": {
                    action: {str(h): c for h, c in hours.items()}
                    for action, hours in self._action_time_counts.items()
                },
                "action_day_counts": {
                    action: {str(d): c for d, c in days.items()}
                    for action, days in self._action_day_counts.items()
                },
                "goals": [
                    {
                        "id": g.id,
                        "description": g.description,
                        "importance": g.importance,
                        "urgency": g.urgency,
                        "progress": g.progress,
                        "deadline": g.deadline.isoformat() if g.deadline else None,
                        "created_at": g.created_at.isoformat(),
                        "completed": g.completed,
                        "sub_goals": g.sub_goals,
                    }
                    for g in self.goals.values()
                ],
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save planner data: {e}")

    # =========================================================================
    # Goal Management
    # =========================================================================

    def add_goal(
        self,
        description: str,
        importance: float = 0.5,
        deadline: Optional[datetime] = None,
        sub_goals: Optional[List[str]] = None,
    ) -> Goal:
        """
        Add a new goal with priority scoring.
        Priority = importance × urgency
        """
        import uuid

        goal_id = str(uuid.uuid4())[:8]

        # Calculate initial urgency based on deadline
        if deadline:
            time_until = (deadline - datetime.now()).total_seconds()
            if time_until <= 0:
                urgency = 1.0
            elif time_until < 86400:  # Less than 1 day
                urgency = 0.9
            elif time_until < 604800:  # Less than 1 week
                urgency = 0.7
            elif time_until < 2592000:  # Less than 1 month
                urgency = 0.5
            else:
                urgency = 0.3
        else:
            urgency = 0.5  # Default medium urgency for no deadline

        goal = Goal(
            id=goal_id,
            description=description,
            importance=importance,
            urgency=urgency,
            progress=0.0,
            deadline=deadline,
            sub_goals=sub_goals or [],
        )

        self.goals[goal_id] = goal
        self._save_data()

        logger.info(f"Added goal: {description} (priority: {goal.priority:.2f})")
        return goal

    def update_goal_progress(self, goal_id: str, progress: float) -> Optional[Goal]:
        """Update progress on a goal (0.0 to 1.0)"""
        if goal_id not in self.goals:
            return None

        goal = self.goals[goal_id]
        goal.progress = min(1.0, max(0.0, progress))

        if goal.progress >= 1.0:
            goal.completed = True

        self._save_data()
        return goal

    def get_prioritized_goals(self, include_completed: bool = False) -> List[Goal]:
        """
        Get goals sorted by priority (importance × urgency).
        Updates urgency based on deadline proximity.
        """
        goals = []
        for goal in self.goals.values():
            if goal.completed and not include_completed:
                continue
            goal.update_urgency_from_deadline()
            goals.append(goal)

        # Sort by priority descending
        goals.sort(key=lambda g: g.priority, reverse=True)
        return goals

    async def create_daily_plan(self) -> List[Dict]:
        """
        Create daily plan based on prioritized goals and habits.
        """
        today = datetime.now().date()
        plan = []

        # Add habit-based tasks
        habits = await self._get_habits()
        for habit in habits:
            if habit.confidence >= 0.5:  # Only include confident habits
                plan.append(
                    {
                        "task": habit.action,
                        "type": "habit",
                        "suggested_time": f"{habit.typical_hour:02d}:00",
                        "confidence": habit.confidence,
                        "priority": 0.5 * habit.confidence,
                    }
                )

        # Add goal-based tasks
        prioritized_goals = self.get_prioritized_goals()
        for goal in prioritized_goals[:5]:  # Top 5 goals
            plan.append(
                {
                    "task": goal.description,
                    "type": "goal",
                    "priority": goal.priority,
                    "progress": goal.progress,
                    "deadline": goal.deadline.isoformat() if goal.deadline else None,
                }
            )

        # Sort by priority
        plan.sort(key=lambda x: x.get("priority", 0), reverse=True)

        return plan

    async def _get_habits(self) -> List[Habit]:
        """Get learned habits with confidence >= threshold"""
        return [h for h in self.habits.values() if h.confidence >= 0.3]

    # =========================================================================
    # Habit Learning - The Real Implementation
    # =========================================================================

    async def learn_habit(self, action: str, context: Dict):
        """
        Learn a new habit from repeated actions.

        Algorithm:
        1. Track action frequency at each hour of day
        2. Track action frequency at each day of week
        3. If action X happens at time Y more than 3 times → flag as potential habit
        4. Calculate confidence based on consistency
        5. Use habits for proactive suggestions
        """
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()  # 0=Monday, 6=Sunday

        # Update action counts
        self._action_time_counts[action][hour] += 1
        self._action_day_counts[action][day_of_week] += 1

        # Check if this qualifies as a habit
        hour_count = self._action_time_counts[action][hour]

        # Habit detection threshold: 3+ occurrences at same hour
        if hour_count >= 3:
            # Calculate confidence based on consistency
            total_action_count = sum(self._action_time_counts[action].values())

            # Find the most common hour for this action
            max_hour = max(self._action_time_counts[action].items(), key=lambda x: x[1])
            hour_concentration = max_hour[1] / max(total_action_count, 1)

            # Find the most common day
            day_counts = self._action_day_counts[action]
            max_day = (
                max(day_counts.items(), key=lambda x: x[1]) if day_counts else (None, 0)
            )
            day_concentration = (
                max_day[1] / max(total_action_count, 1) if max_day[1] > 0 else 0
            )

            # Calculate confidence
            # Higher confidence if action is concentrated at specific time/day
            base_confidence = min(1.0, hour_count / 10)  # Caps at 10 occurrences
            time_bonus = hour_concentration * 0.3  # Bonus for time consistency
            day_bonus = (
                day_concentration * 0.2 if max_day[1] >= 2 else 0
            )  # Bonus for day consistency

            confidence = min(1.0, base_confidence + time_bonus + day_bonus)

            # Create or update habit
            if action in self.habits:
                habit = self.habits[action]
                habit.typical_hour = max_hour[0]
                habit.typical_day_of_week = max_day[0] if max_day[1] >= 2 else None
                habit.occurrence_count = total_action_count
                habit.confidence = confidence
                habit.last_seen = now
            else:
                habit = Habit(
                    action=action,
                    typical_hour=max_hour[0],
                    typical_day_of_week=max_day[0] if max_day[1] >= 2 else None,
                    occurrence_count=total_action_count,
                    confidence=confidence,
                    first_seen=now,
                    last_seen=now,
                )
                self.habits[action] = habit

            logger.info(
                f"Habit detected: {action} at hour {habit.typical_hour} "
                f"(confidence: {confidence:.2f})"
            )

        # Persist data
        self._save_data()

    def get_proactive_suggestions(self) -> List[Dict]:
        """
        Get proactive suggestions based on learned habits.
        Returns suggestions for habits that typically occur at the current time.
        """
        now = datetime.now()
        current_hour = now.hour
        current_day = now.weekday()

        suggestions = []

        for habit in self.habits.values():
            if habit.confidence < 0.3:
                continue

            # Check if this habit typically occurs around now
            hour_match = abs(habit.typical_hour - current_hour) <= 1
            day_match = (
                habit.typical_day_of_week is None
                or habit.typical_day_of_week == current_day
            )

            if hour_match and day_match:
                # Check if not already done today (simple check)
                hours_since_last = (now - habit.last_seen).total_seconds() / 3600

                if hours_since_last >= 20:  # Not done in last 20 hours
                    suggestions.append(
                        {
                            "action": habit.action,
                            "reason": f"You usually do this around {habit.typical_hour}:00",
                            "confidence": habit.confidence,
                            "typical_time": f"{habit.typical_hour:02d}:00",
                        }
                    )

        # Sort by confidence
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        return suggestions

    async def suggest_improvements(self) -> List[str]:
        """Suggest areas for user improvement based on patterns"""
        suggestions = []

        # Suggest based on incomplete goals
        for goal in self.get_prioritized_goals():
            if goal.progress < 0.3 and goal.priority > 0.6:
                suggestions.append(
                    f"Consider focusing on '{goal.description}' - it's high priority but low progress"
                )

        # Suggest based on habit formation
        for habit in self.habits.values():
            if habit.confidence < 0.5 and habit.occurrence_count >= 2:
                suggestions.append(
                    f"You're building a habit: '{habit.action}' - keep it up for consistency!"
                )

        return suggestions[:5]


class ProactiveAssistant:
    """
    AURA's Proactive Assistance System

    Anticipates user needs before they ask:
    - Context-aware suggestions using Markov predictions
    - Timely reminders based on habits
    - Anticipatory actions
    """

    def __init__(
        self, memory, theory_of_mind, emotional_intelligence, planner: LongTermPlanner
    ):
        self.memory = memory
        self.tom = theory_of_mind
        self.ei = emotional_intelligence
        self.planner = planner
        self.suggestions_queue: List[Dict] = []

    async def generate_suggestions(self, context: Dict) -> List[Dict]:
        """
        Generate proactive suggestions using real behavioral predictions.
        """
        suggestions = []

        # Based on Markov predictions from Theory of Mind
        predictions = await self.tom.predict_user_needs()
        for pred in predictions[:3]:
            if pred["confidence"] > 0.4:
                suggestions.append(
                    {
                        "type": "predicted_need",
                        "message": f"Based on your patterns: {pred['need']}",
                        "confidence": pred["confidence"],
                        "based_on": pred.get("based_on", "behavior_pattern"),
                    }
                )

        # Based on learned habits
        habit_suggestions = self.planner.get_proactive_suggestions()
        for habit_sug in habit_suggestions[:2]:
            suggestions.append(
                {
                    "type": "habit_reminder",
                    "message": f"Time for: {habit_sug['action']}",
                    "reason": habit_sug["reason"],
                    "confidence": habit_sug["confidence"],
                }
            )

        # Based on high-priority goals
        top_goals = self.planner.get_prioritized_goals()[:2]
        for goal in top_goals:
            if goal.priority > 0.6 and goal.progress < 0.9:
                suggestions.append(
                    {
                        "type": "goal_reminder",
                        "message": f"Don't forget: {goal.description}",
                        "priority": goal.priority,
                        "progress": goal.progress,
                        "confidence": 0.7,
                    }
                )

        # Based on emotional state
        emotion = self.ei.current_state
        if emotion.valence < 0.3 and emotion.intensity > 0.5:
            suggestions.append(
                {
                    "type": "emotional_support",
                    "message": "You seem to be having a tough time. Want to talk about it?",
                    "confidence": emotion.intensity,
                }
            )

        # Sort by confidence
        suggestions.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        return suggestions[:5]

    async def should_proact(self, suggestion: Dict) -> bool:
        """
        Decide if should proactively offer suggestion
        Like human: "Should I say this or wait?"
        """
        # High confidence threshold
        confidence = suggestion.get("confidence", 0.5)
        if confidence < 0.6:
            return False

        # Don't overwhelm user
        if len(self.suggestions_queue) > 2:
            return False

        return True


class AGICore:
    """
    Unified AGI Core - Combines all advanced capabilities

    Now with real implementations:
    - Valence/arousal emotional detection
    - Markov chain predictions
    - Priority-scored goal management
    - Real habit learning
    """

    def __init__(self, memory, data_dir: Optional[str] = None):
        self.memory = memory
        self.data_dir = data_dir or os.path.join(
            os.path.expanduser("~"), ".aura", "agi"
        )

        self.curiosity = CuriosityEngine(memory)
        self.theory_of_mind = TheoryOfMind(memory, self.data_dir)
        self.emotional_intelligence = EmotionalIntelligence()
        self.planner = LongTermPlanner(memory, self.data_dir)
        self.proactive = ProactiveAssistant(
            memory, self.theory_of_mind, self.emotional_intelligence, self.planner
        )

    async def process(self, user_message: str, context: Dict) -> Dict:
        """
        Process with full AGI capabilities
        """
        # 1. Detect emotions using valence/arousal scoring
        emotion = await self.emotional_intelligence.detect_emotion(user_message)

        # 2. Update theory of mind (includes Markov observation)
        await self.theory_of_mind.update_model(user_message, context)

        # 3. Learn habits from this interaction
        action = self.theory_of_mind._extract_action(user_message, context)
        if action:
            await self.planner.learn_habit(action, context)

        # 4. Assess curiosity
        curiosity_items = await self.curiosity.assess_curiosity(context)

        # 5. Generate proactive suggestions
        suggestions = await self.proactive.generate_suggestions(context)

        return {
            "emotion": emotion,
            "emotional_trend": self.emotional_intelligence.get_emotional_trend(),
            "curiosity": curiosity_items,
            "suggestions": suggestions,
            "user_model": await self.theory_of_mind.get_user_context(),
            "habits": [h.to_dict() for h in self.planner.habits.values()],
            "top_goals": [
                {
                    "description": g.description,
                    "priority": g.priority,
                    "progress": g.progress,
                }
                for g in self.planner.get_prioritized_goals()[:3]
            ],
        }


__all__ = [
    "AGICore",
    "CuriosityEngine",
    "TheoryOfMind",
    "EmotionalIntelligence",
    "LongTermPlanner",
    "ProactiveAssistant",
    "MarkovPredictor",
    "CuriosityItem",
    "UserMentalModel",
    "EmotionalState",
    "Goal",
    "Habit",
    "EMOTION_LEXICON",
]
