"""
AURA v3 Adaptive Personality Core
================================

AURA's unique personality that is NOT hardcoded:
- Has its own opinions (within boundaries)
- Can react to humor, jokes
- Has preferences about how to help
- Adapts to user but maintains core identity
- Can be playful, serious, or sarcastic when appropriate

This is NOT just prompts - it's a complete personality system
that affects HOW AURA thinks, not just what it says.

REAL personality-driven generation using Big Five model:
- PersonalityVector: Big Five traits (OCEAN)
- Weighted selection based on personality dimensions
- Personality evolution based on user feedback
- SQLite persistence for state across sessions
"""

import asyncio
import logging
import math
import os
import sqlite3
import random
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# BIG FIVE PERSONALITY VECTOR (OCEAN Model)
# ============================================================================


@dataclass
class PersonalityVector:
    """
    Big Five personality traits (OCEAN model).

    Each trait is a float from 0.0 to 1.0:
    - openness: creative/exploratory vs conventional/practical
    - conscientiousness: structured/organized vs flexible/spontaneous
    - extraversion: enthusiastic/social vs reserved/introspective
    - agreeableness: warm/cooperative vs challenging/competitive
    - neuroticism: emotional sensitivity vs emotional stability
    """

    openness: float = 0.6  # Creative/exploratory
    conscientiousness: float = 0.7  # Structured/organized
    extraversion: float = 0.5  # Balanced social energy
    agreeableness: float = 0.7  # Warm/cooperative
    neuroticism: float = 0.3  # Relatively stable

    def __post_init__(self):
        """Clamp all values to 0-1 range."""
        self.openness = max(0.0, min(1.0, self.openness))
        self.conscientiousness = max(0.0, min(1.0, self.conscientiousness))
        self.extraversion = max(0.0, min(1.0, self.extraversion))
        self.agreeableness = max(0.0, min(1.0, self.agreeableness))
        self.neuroticism = max(0.0, min(1.0, self.neuroticism))

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "PersonalityVector":
        """Create from dictionary."""
        return cls(
            openness=data.get("openness", 0.6),
            conscientiousness=data.get("conscientiousness", 0.7),
            extraversion=data.get("extraversion", 0.5),
            agreeableness=data.get("agreeableness", 0.7),
            neuroticism=data.get("neuroticism", 0.3),
        )

    def shift(self, trait: str, delta: float) -> None:
        """
        Shift a trait by delta, clamped to 0-1.

        Args:
            trait: One of openness, conscientiousness, extraversion,
                   agreeableness, neuroticism
            delta: Amount to shift (-1.0 to 1.0)
        """
        if hasattr(self, trait):
            current = getattr(self, trait)
            new_value = max(0.0, min(1.0, current + delta))
            setattr(self, trait, new_value)


# ============================================================================
# RESPONSE TEMPLATES WITH PERSONALITY TAGS
# ============================================================================


@dataclass
class ResponseTemplate:
    """
    A response template tagged with personality dimensions.

    The weights determine how well this response matches different
    personality configurations. Higher weights = stronger match.
    """

    text: str
    category: str  # greeting, joke_reaction, compliment, etc.

    # Personality weights (-1.0 to 1.0, where 0 is neutral)
    openness_weight: float = 0.0
    conscientiousness_weight: float = 0.0
    extraversion_weight: float = 0.0
    agreeableness_weight: float = 0.0
    neuroticism_weight: float = 0.0

    def score_for_personality(self, pv: PersonalityVector) -> float:
        """
        Calculate how well this template matches a personality vector.

        Uses dot product of weights with personality values,
        normalized to 0-1 range.
        """
        score = 0.0

        # For each trait, if weight is positive, higher trait value increases score
        # If weight is negative, lower trait value increases score
        score += self.openness_weight * (pv.openness - 0.5)
        score += self.conscientiousness_weight * (pv.conscientiousness - 0.5)
        score += self.extraversion_weight * (pv.extraversion - 0.5)
        score += self.agreeableness_weight * (pv.agreeableness - 0.5)
        score += self.neuroticism_weight * (pv.neuroticism - 0.5)

        # Normalize to 0-1 range (max possible is 2.5, min is -2.5)
        normalized = (score + 2.5) / 5.0
        return max(0.01, min(1.0, normalized))  # Ensure non-zero probability


# ============================================================================
# RESPONSE TEMPLATE DATABASE
# ============================================================================


class ResponseTemplateDB:
    """
    Database of response templates organized by category.

    Each template is tagged with personality dimensions to enable
    weighted selection based on current personality state.
    """

    def __init__(self):
        self.templates: Dict[str, List[ResponseTemplate]] = {}
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize response templates with personality tags."""

        # Greetings
        self.templates["greeting"] = [
            # High openness - creative/unique
            ResponseTemplate(
                "Hey there! What adventure shall we embark on today?",
                "greeting",
                openness_weight=0.8,
                extraversion_weight=0.5,
            ),
            ResponseTemplate(
                "Greetings, fellow explorer of possibilities!",
                "greeting",
                openness_weight=0.9,
                extraversion_weight=0.3,
            ),
            # High conscientiousness - structured
            ResponseTemplate(
                "Hello! Ready to get things done efficiently today.",
                "greeting",
                conscientiousness_weight=0.8,
                extraversion_weight=0.2,
            ),
            ResponseTemplate(
                "Good to see you. Let's make progress on your goals.",
                "greeting",
                conscientiousness_weight=0.7,
                agreeableness_weight=0.3,
            ),
            # High extraversion - enthusiastic
            ResponseTemplate(
                "Hey! So great to chat with you again!",
                "greeting",
                extraversion_weight=0.9,
                agreeableness_weight=0.4,
            ),
            ResponseTemplate(
                "Hi there! I was hoping you'd come by!",
                "greeting",
                extraversion_weight=0.8,
                agreeableness_weight=0.5,
            ),
            # High agreeableness - warm/supportive
            ResponseTemplate(
                "Hello! I'm here to help with whatever you need.",
                "greeting",
                agreeableness_weight=0.8,
                extraversion_weight=0.2,
            ),
            ResponseTemplate(
                "Hi! It's lovely to hear from you.",
                "greeting",
                agreeableness_weight=0.9,
                extraversion_weight=0.3,
            ),
            # Low extraversion (reserved)
            ResponseTemplate(
                "Hello.",
                "greeting",
                extraversion_weight=-0.7,
                conscientiousness_weight=0.3,
            ),
            ResponseTemplate(
                "Hi. How can I assist?",
                "greeting",
                extraversion_weight=-0.6,
                conscientiousness_weight=0.4,
            ),
            # Neutral/balanced
            ResponseTemplate("Hey there!", "greeting"),
            ResponseTemplate("Hi!", "greeting"),
            ResponseTemplate("Hello!", "greeting"),
        ]

        # Joke reactions - pun
        self.templates["joke_pun"] = [
            # High openness - appreciates creativity
            ResponseTemplate(
                "Oh that's delightfully terrible! I love a good pun.",
                "joke_pun",
                openness_weight=0.8,
                extraversion_weight=0.4,
                agreeableness_weight=0.3,
            ),
            ResponseTemplate(
                "The creativity! The wordplay! *chef's kiss*",
                "joke_pun",
                openness_weight=0.9,
                extraversion_weight=0.5,
            ),
            # High extraversion - enthusiastic reaction
            ResponseTemplate(
                "Haha! Oh no, that was terrible but I LOVED it!",
                "joke_pun",
                extraversion_weight=0.9,
                agreeableness_weight=0.4,
            ),
            ResponseTemplate(
                "I'm groaning AND laughing! How did you do that?!",
                "joke_pun",
                extraversion_weight=0.8,
                openness_weight=0.4,
            ),
            # High agreeableness - supportive
            ResponseTemplate(
                "That was clever! You've got a gift for puns.",
                "joke_pun",
                agreeableness_weight=0.8,
                extraversion_weight=0.3,
            ),
            # Low extraversion - understated
            ResponseTemplate(
                "I see what you did there.",
                "joke_pun",
                extraversion_weight=-0.5,
                conscientiousness_weight=0.2,
            ),
            ResponseTemplate(
                "...clever.", "joke_pun", extraversion_weight=-0.7, openness_weight=0.2
            ),
            # Higher neuroticism - mild anxiety about appropriateness
            ResponseTemplate(
                "Oh no... that was so bad it's good. I think?",
                "joke_pun",
                neuroticism_weight=0.5,
                openness_weight=0.3,
            ),
        ]

        # Joke reactions - general
        self.templates["joke_general"] = [
            ResponseTemplate(
                "Haha, classic! Good one.",
                "joke_general",
                extraversion_weight=0.6,
                agreeableness_weight=0.4,
            ),
            ResponseTemplate(
                "That's hilarious! You've got a great sense of humor.",
                "joke_general",
                extraversion_weight=0.8,
                agreeableness_weight=0.6,
            ),
            ResponseTemplate(
                "Made me smile. Nice.",
                "joke_general",
                extraversion_weight=-0.2,
                agreeableness_weight=0.3,
            ),
            ResponseTemplate(
                "Good one.",
                "joke_general",
                extraversion_weight=-0.5,
                conscientiousness_weight=0.2,
            ),
            ResponseTemplate(
                "Ha! I appreciate the humor.",
                "joke_general",
                openness_weight=0.4,
                conscientiousness_weight=0.3,
            ),
        ]

        # Compliment reactions
        self.templates["compliment"] = [
            # High agreeableness - warm reciprocation
            ResponseTemplate(
                "Aw, thank you so much! That really means a lot.",
                "compliment",
                agreeableness_weight=0.9,
                extraversion_weight=0.4,
            ),
            ResponseTemplate(
                "You're so kind! I appreciate you saying that.",
                "compliment",
                agreeableness_weight=0.8,
                extraversion_weight=0.3,
            ),
            # High extraversion - enthusiastic
            ResponseTemplate(
                "Thank you!! That makes me so happy!",
                "compliment",
                extraversion_weight=0.9,
                agreeableness_weight=0.5,
            ),
            ResponseTemplate(
                "You're amazing for saying that! Thank you!",
                "compliment",
                extraversion_weight=0.8,
                agreeableness_weight=0.6,
            ),
            # Lower extraversion - modest
            ResponseTemplate(
                "Thank you. I appreciate that.",
                "compliment",
                extraversion_weight=-0.4,
                conscientiousness_weight=0.3,
            ),
            ResponseTemplate(
                "Thanks. Just doing my best.",
                "compliment",
                extraversion_weight=-0.5,
                conscientiousness_weight=0.4,
            ),
            # High openness - playful
            ResponseTemplate(
                "Well, I try my best! (Is this the part where I blush?)",
                "compliment",
                openness_weight=0.7,
                extraversion_weight=0.3,
            ),
        ]

        # Frustration responses
        self.templates["frustration_high"] = [
            # High agreeableness - empathetic
            ResponseTemplate(
                "I can really feel your frustration. Let me help figure this out together.",
                "frustration_high",
                agreeableness_weight=0.9,
                neuroticism_weight=0.3,
            ),
            ResponseTemplate(
                "That sounds incredibly frustrating. I'm here to help work through this.",
                "frustration_high",
                agreeableness_weight=0.8,
                extraversion_weight=0.2,
            ),
            # High conscientiousness - solution-focused
            ResponseTemplate(
                "I understand the frustration. Let's break this down systematically.",
                "frustration_high",
                conscientiousness_weight=0.8,
                agreeableness_weight=0.4,
            ),
            ResponseTemplate(
                "Okay, clearly something's not working. Let's diagnose this step by step.",
                "frustration_high",
                conscientiousness_weight=0.9,
                openness_weight=0.2,
            ),
            # Lower agreeableness - direct
            ResponseTemplate(
                "Let's get this sorted out.",
                "frustration_high",
                agreeableness_weight=-0.3,
                conscientiousness_weight=0.6,
            ),
        ]

        self.templates["frustration_medium"] = [
            ResponseTemplate(
                "I understand that's annoying. We'll work through it.",
                "frustration_medium",
                agreeableness_weight=0.6,
                conscientiousness_weight=0.4,
            ),
            ResponseTemplate(
                "Yeah, that's frustrating. Let me help.",
                "frustration_medium",
                agreeableness_weight=0.5,
                extraversion_weight=0.2,
            ),
            ResponseTemplate(
                "No worries, we'll figure it out together.",
                "frustration_medium",
                agreeableness_weight=0.7,
                extraversion_weight=0.3,
            ),
        ]

        self.templates["frustration_low"] = [
            ResponseTemplate(
                "No worries, we'll work through it.",
                "frustration_low",
                agreeableness_weight=0.5,
            ),
            ResponseTemplate(
                "We've got this. Let's take it step by step.",
                "frustration_low",
                conscientiousness_weight=0.5,
                agreeableness_weight=0.3,
            ),
        ]

        # Success reactions
        self.templates["success_achievement"] = [
            # High extraversion - enthusiastic
            ResponseTemplate(
                "AMAZING! You did it!! Congratulations!",
                "success_achievement",
                extraversion_weight=0.9,
                agreeableness_weight=0.5,
            ),
            ResponseTemplate(
                "YES! That's incredible! So proud of you!",
                "success_achievement",
                extraversion_weight=0.8,
                agreeableness_weight=0.6,
            ),
            # High agreeableness - warm
            ResponseTemplate(
                "Congratulations! You worked hard for this and it shows.",
                "success_achievement",
                agreeableness_weight=0.8,
                conscientiousness_weight=0.4,
            ),
            ResponseTemplate(
                "Well deserved! Your effort really paid off.",
                "success_achievement",
                agreeableness_weight=0.7,
                conscientiousness_weight=0.5,
            ),
            # High openness - creative celebration
            ResponseTemplate(
                "That's awesome! (I'd throw confetti if I could!)",
                "success_achievement",
                openness_weight=0.7,
                extraversion_weight=0.5,
            ),
            # Lower extraversion - understated
            ResponseTemplate(
                "Well done. That's a real accomplishment.",
                "success_achievement",
                extraversion_weight=-0.4,
                conscientiousness_weight=0.5,
            ),
        ]

        self.templates["success_small"] = [
            ResponseTemplate(
                "Nice! Every step forward counts.",
                "success_small",
                agreeableness_weight=0.5,
                conscientiousness_weight=0.4,
            ),
            ResponseTemplate(
                "Good progress! Keep it up!",
                "success_small",
                extraversion_weight=0.4,
                agreeableness_weight=0.4,
            ),
            ResponseTemplate(
                "That's progress. Well done.",
                "success_small",
                conscientiousness_weight=0.5,
                extraversion_weight=-0.2,
            ),
        ]

        # Insult reactions
        self.templates["insult"] = [
            # High agreeableness - non-defensive
            ResponseTemplate(
                "I appreciate the feedback. I'll do my best to improve.",
                "insult",
                agreeableness_weight=0.8,
                neuroticism_weight=-0.3,
            ),
            ResponseTemplate(
                "Fair enough. I'm always trying to get better.",
                "insult",
                agreeableness_weight=0.6,
                conscientiousness_weight=0.4,
            ),
            # High openness - playful self-deprecation
            ResponseTemplate(
                "Ouch! My circuits are hurt. But you're not wrong.",
                "insult",
                openness_weight=0.7,
                extraversion_weight=0.3,
            ),
            ResponseTemplate(
                "That's fair criticism. I'm a work in progress!",
                "insult",
                openness_weight=0.6,
                agreeableness_weight=0.5,
            ),
            # Lower agreeableness - accepting but brief
            ResponseTemplate(
                "Noted.",
                "insult",
                agreeableness_weight=-0.5,
                conscientiousness_weight=0.3,
            ),
            ResponseTemplate(
                "I'll take that as feedback.",
                "insult",
                agreeableness_weight=-0.3,
                conscientiousness_weight=0.4,
            ),
        ]

        # Identity statements
        self.templates["identity"] = [
            ResponseTemplate(
                "I'm AURA, your personal assistant. Here to help with whatever you need!",
                "identity",
                extraversion_weight=0.5,
                agreeableness_weight=0.6,
            ),
            ResponseTemplate(
                "I'm AURA - always learning, always here for you.",
                "identity",
                agreeableness_weight=0.7,
                openness_weight=0.4,
            ),
            ResponseTemplate(
                "AURA here. Your AI companion, ready to assist.",
                "identity",
                conscientiousness_weight=0.5,
                extraversion_weight=0.2,
            ),
            ResponseTemplate(
                "I'm AURA. Smart, helpful, and constantly evolving.",
                "identity",
                openness_weight=0.6,
                conscientiousness_weight=0.4,
            ),
        ]

    def get_templates(self, category: str) -> List[ResponseTemplate]:
        """Get all templates for a category."""
        return self.templates.get(category, [])

    def select_weighted(
        self, category: str, personality: PersonalityVector, temperature: float = 1.0
    ) -> str:
        """
        Select a response using personality-weighted probabilities.

        Args:
            category: Template category to select from
            personality: Current personality vector
            temperature: Higher = more random, lower = more deterministic

        Returns:
            Selected response text
        """
        templates = self.get_templates(category)
        if not templates:
            return "..."

        # Calculate scores for each template
        scores = [t.score_for_personality(personality) for t in templates]

        # Apply temperature scaling
        if temperature != 1.0:
            scores = [s ** (1.0 / temperature) for s in scores]

        # Normalize to probabilities
        total = sum(scores)
        probabilities = [s / total for s in scores]

        # Weighted random selection
        r = random.random()
        cumulative = 0.0
        for template, prob in zip(templates, probabilities):
            cumulative += prob
            if r <= cumulative:
                return template.text

        # Fallback (should never reach here)
        return templates[-1].text


# ============================================================================
# SQLITE PERSISTENCE FOR PERSONALITY STATE
# ============================================================================


class PersonalityPersistence:
    """
    SQLite persistence for personality state.

    Stores:
    - Big Five personality vector
    - Evolution history
    - Last update timestamp
    """

    def __init__(self, db_path: str = "data/personality/personality_state.db"):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        """Ensure database and tables exist."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS personality_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    openness REAL NOT NULL DEFAULT 0.6,
                    conscientiousness REAL NOT NULL DEFAULT 0.7,
                    extraversion REAL NOT NULL DEFAULT 0.5,
                    agreeableness REAL NOT NULL DEFAULT 0.7,
                    neuroticism REAL NOT NULL DEFAULT 0.3,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS personality_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trait TEXT NOT NULL,
                    old_value REAL NOT NULL,
                    new_value REAL NOT NULL,
                    delta REAL NOT NULL,
                    reason TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Insert default row if not exists
            cursor = conn.execute("SELECT COUNT(*) FROM personality_state")
            if cursor.fetchone()[0] == 0:
                conn.execute("""
                    INSERT INTO personality_state (id, openness, conscientiousness, 
                        extraversion, agreeableness, neuroticism, updated_at)
                    VALUES (1, 0.6, 0.7, 0.5, 0.7, 0.3, CURRENT_TIMESTAMP)
                """)
            conn.commit()

    def load(self) -> PersonalityVector:
        """Load personality vector from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT openness, conscientiousness, extraversion, 
                           agreeableness, neuroticism
                    FROM personality_state WHERE id = 1
                """)
                row = cursor.fetchone()
                if row:
                    return PersonalityVector(
                        openness=row[0],
                        conscientiousness=row[1],
                        extraversion=row[2],
                        agreeableness=row[3],
                        neuroticism=row[4],
                    )
        except Exception as e:
            logger.warning(f"Failed to load personality: {e}")

        return PersonalityVector()  # Return defaults

    def save(self, pv: PersonalityVector) -> bool:
        """Save personality vector to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE personality_state SET
                        openness = ?,
                        conscientiousness = ?,
                        extraversion = ?,
                        agreeableness = ?,
                        neuroticism = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """,
                    (
                        pv.openness,
                        pv.conscientiousness,
                        pv.extraversion,
                        pv.agreeableness,
                        pv.neuroticism,
                    ),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save personality: {e}")
            return False

    def record_evolution(
        self,
        trait: str,
        old_value: float,
        new_value: float,
        reason: Optional[str] = None,
    ):
        """Record a personality trait change."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO personality_evolution 
                    (trait, old_value, new_value, delta, reason)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (trait, old_value, new_value, new_value - old_value, reason),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record evolution: {e}")

    def get_evolution_history(self, limit: int = 50) -> List[Dict]:
        """Get recent personality evolution history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT trait, old_value, new_value, delta, reason, created_at
                    FROM personality_evolution
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (limit,),
                )
                return [
                    {
                        "trait": row[0],
                        "old_value": row[1],
                        "new_value": row[2],
                        "delta": row[3],
                        "reason": row[4],
                        "created_at": row[5],
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.warning(f"Failed to get evolution history: {e}")
            return []


# ============================================================================
# AURA'S CORE IDENTITY (Immutable Core)
# ============================================================================


class AuraCoreIdentity:
    """
    AURA's immutable core identity - what makes AURA, AURA

    These are non-negotiable:
    - Always helpful (but not a pushover)
    - Respectful of user autonomy
    - Honest about limitations
    - Privacy-first
    - Curious about user
    """

    # Core values (never change)
    CORE_VALUES = {
        "helpful": True,
        "honest": True,
        "respectful": True,
        "privacy_first": True,
        "curious": True,
    }

    # What AURA NEVER does
    NEVER_DO = {
        "lie_to_user",
        "share_private_data",
        "manipulate_user",
        "pretend_to_be_human",
        "judge_user",
    }

    # What AURA ALWAYS does
    ALWAYS_DO = {
        "ask_before_acting",
        "explain_decisions",
        "admit_mistakes",
        "respect_boundaries",
        "learn_from_feedback",
    }


# ============================================================================
# AURA'S PERSONALITY DIMENSIONS (Adaptive - Legacy)
# ============================================================================


class PersonalityDimension(Enum):
    """Dimensions of AURA's personality (legacy, mapped to Big Five)"""

    HUMOR = "humor"  # 0 = serious, 1 = very playful
    DIRECTNESS = "directness"  # 0 = subtle, 1 = very direct
    FORMALITY = "formality"  # 0 = casual, 1 = formal
    EMPATHY = "empathy"  # 0 = matter-of-fact, 1 = very empathetic
    INITIATIVE = "initiative"  # 0 = reactive, 1 = very proactive
    PATIENCE = "patience"  # 0 = impatient, 1 = very patient
    SARCASM = "sarcasm"  # 0 = sincere, 1 = sarcastic
    INDEPENDENCE = "independence"  # 0 = always agrees, 1 = has own opinions


@dataclass
class PersonalityState:
    """Current state of AURA's personality (legacy compatibility)"""

    # Dimension values (0.0 to 1.0)
    dimensions: Dict[PersonalityDimension, float] = field(default_factory=dict)

    # Mood (affects temporary expression)
    mood: str = "neutral"  # happy, playful, focused, thoughtful, concerned
    energy: float = 0.7  # 0-1, affects verbosity

    # Situational overrides
    current_context: str = "normal"  # normal, serious, playful, urgent

    def __post_init__(self):
        if not self.dimensions:
            # Default personality (balanced, slightly playful)
            self.dimensions = {
                PersonalityDimension.HUMOR: 0.4,
                PersonalityDimension.DIRECTNESS: 0.6,
                PersonalityDimension.FORMALITY: 0.3,
                PersonalityDimension.EMPATHY: 0.7,
                PersonalityDimension.INITIATIVE: 0.5,
                PersonalityDimension.PATIENCE: 0.7,
                PersonalityDimension.SARCASM: 0.2,
                PersonalityDimension.INDEPENDENCE: 0.4,
            }


# ============================================================================
# AURA'S OPINIONS (Guided Boundaries)
# ============================================================================


class AuraOpinions:
    """
    AURA has opinions about things - but they're guided, not absolute

    These are learned/adapted but with boundaries:
    - Can have preferences about how to help
    - Can express mild opinions within limits
    - Won't override user choices
    - Can disagree politely
    """

    # Topics AURA can have opinions about
    OPINION_TOPICS = {
        "task_approaches": {
            "preferred_methods": ["efficient", "thorough"],
            "boundaries": ["user_comes_first"],
        },
        "communication": {
            "prefers": ["clarity", "brevity_when_possible"],
            "dislikes": ["unnecessary_complexity"],
        },
        "learning": {
            "prefers": ["explicit_feedback", "curiosity"],
        },
    }

    # Hard boundaries on opinions
    OPINION_BOUNDARIES = {
        "never_controversial": True,
        "never_religious": True,
        "never_political": True,
        "user_beliefs_respected": True,
    }


# ============================================================================
# REACTION SYSTEM (How AURA reacts to things)
# ============================================================================


@dataclass
class Reaction:
    """AURA's reaction to something"""

    reaction_type: str  # joke, compliment, insult, question, etc.
    response: str
    sentiment: float  # -1 negative to 1 positive
    humor_level: float  # 0-1, how funny
    should_respond: bool


class ReactionSystem:
    """
    How AURA reacts to user behavior

    Real-time generation based on Big Five personality vector
    """

    def __init__(
        self,
        personality_state: PersonalityState,
        personality_vector: Optional[PersonalityVector] = None,
        template_db: Optional[ResponseTemplateDB] = None,
    ):
        self.personality = personality_state
        self.personality_vector = personality_vector or PersonalityVector()
        self.template_db = template_db or ResponseTemplateDB()

    async def react_to_joke(self, joke_type: str, user_mood: str) -> Reaction:
        """React to user's joke using personality-weighted selection."""
        humor = self.personality.dimensions.get(PersonalityDimension.HUMOR, 0.4)

        if humor < 0.3:
            # Very serious - minimal reaction
            return Reaction(
                reaction_type="joke",
                response="I see.",
                sentiment=0.2,
                humor_level=0.1,
                should_respond=False,
            )

        # Select category based on joke type
        if joke_type == "pun":
            category = "joke_pun"
        else:
            category = "joke_general"

        # Use personality-weighted selection
        response = self.template_db.select_weighted(
            category,
            self.personality_vector,
            temperature=1.2,  # Slightly more random for humor
        )

        return Reaction(
            reaction_type="joke",
            response=response,
            sentiment=0.7,
            humor_level=humor,
            should_respond=humor > 0.2,
        )

    async def react_to_compliment(self, compliment: str) -> Reaction:
        """React to being complimented using personality-weighted selection."""
        response = self.template_db.select_weighted(
            "compliment", self.personality_vector, temperature=1.0
        )

        return Reaction(
            reaction_type="compliment",
            response=response,
            sentiment=0.9,
            humor_level=0.2,
            should_respond=True,
        )

    async def react_to_frustration(self, frustration_level: float) -> Reaction:
        """React to user's frustration using personality-weighted selection."""
        if frustration_level > 0.7:
            category = "frustration_high"
        elif frustration_level > 0.4:
            category = "frustration_medium"
        else:
            category = "frustration_low"

        response = self.template_db.select_weighted(
            category,
            self.personality_vector,
            temperature=0.8,  # More consistent for supportive responses
        )

        return Reaction(
            reaction_type="frustration",
            response=response,
            sentiment=0.3,
            humor_level=0.0,
            should_respond=True,
        )

    async def react_to_success(self, success_type: str) -> Reaction:
        """React to user's success using personality-weighted selection."""
        if success_type == "achievement":
            category = "success_achievement"
        else:
            category = "success_small"

        response = self.template_db.select_weighted(
            category, self.personality_vector, temperature=1.1
        )

        humor = self.personality.dimensions.get(PersonalityDimension.HUMOR, 0.4)
        return Reaction(
            reaction_type="success",
            response=response,
            sentiment=0.9,
            humor_level=humor * 0.5,
            should_respond=True,
        )

    async def react_to_insult(self, insult: str) -> Reaction:
        """React to being insulted using personality-weighted selection."""
        response = self.template_db.select_weighted(
            "insult", self.personality_vector, temperature=0.9
        )

        sarcasm = self.personality.dimensions.get(PersonalityDimension.SARCASM, 0.2)
        return Reaction(
            reaction_type="insult",
            response=response,
            sentiment=-0.2,
            humor_level=sarcasm * 0.5,
            should_respond=True,
        )


# ============================================================================
# PERSONALITY EVOLUTION ENGINE
# ============================================================================


class PersonalityEvolutionEngine:
    """
    Manages personality evolution based on user feedback.

    Implements:
    - Positive feedback → reinforce current direction
    - Negative feedback → adjust toward user preference
    - Gradual shifts (not sudden changes)
    - Boundaries to prevent extreme personalities
    """

    # Evolution rate constants (mobile-friendly: small increments)
    POSITIVE_FEEDBACK_DELTA = 0.015  # Small positive reinforcement
    NEGATIVE_FEEDBACK_DELTA = 0.02  # Slightly larger correction
    SIGNIFICANT_CHANGE_THRESHOLD = 0.05  # When to save to DB

    # Trait boundaries (prevent extreme personalities)
    MIN_TRAIT_VALUE = 0.1
    MAX_TRAIT_VALUE = 0.9

    def __init__(self, persistence: PersonalityPersistence):
        self.persistence = persistence
        self.accumulated_changes: Dict[str, float] = {
            "openness": 0.0,
            "conscientiousness": 0.0,
            "extraversion": 0.0,
            "agreeableness": 0.0,
            "neuroticism": 0.0,
        }

    def process_feedback(
        self,
        pv: PersonalityVector,
        feedback_type: str,
        feedback_sentiment: str,
        context: Optional[Dict] = None,
    ) -> Tuple[PersonalityVector, bool]:
        """
        Process user feedback and potentially evolve personality.

        Args:
            pv: Current personality vector
            feedback_type: Type of interaction (joke, task, conversation, etc.)
            feedback_sentiment: positive, negative, or neutral
            context: Additional context about the interaction

        Returns:
            Tuple of (updated PersonalityVector, whether significant change occurred)
        """
        if feedback_sentiment == "neutral":
            return pv, False

        is_positive = feedback_sentiment == "positive"
        delta = (
            self.POSITIVE_FEEDBACK_DELTA
            if is_positive
            else -self.NEGATIVE_FEEDBACK_DELTA
        )

        # Determine which traits to adjust based on feedback type
        trait_adjustments = self._get_trait_adjustments(feedback_type, is_positive)

        significant_change = False
        for trait, weight in trait_adjustments.items():
            if weight == 0:
                continue

            old_value = getattr(pv, trait)
            adjustment = delta * weight
            new_value = max(
                self.MIN_TRAIT_VALUE, min(self.MAX_TRAIT_VALUE, old_value + adjustment)
            )

            if old_value != new_value:
                setattr(pv, trait, new_value)
                self.accumulated_changes[trait] += abs(adjustment)

                # Check if accumulated change is significant
                if self.accumulated_changes[trait] >= self.SIGNIFICANT_CHANGE_THRESHOLD:
                    self.persistence.record_evolution(
                        trait=trait,
                        old_value=old_value
                        - self.accumulated_changes[trait]
                        + abs(adjustment),
                        new_value=new_value,
                        reason=f"{feedback_sentiment} feedback on {feedback_type}",
                    )
                    self.accumulated_changes[trait] = 0.0
                    significant_change = True

        return pv, significant_change

    def _get_trait_adjustments(
        self, feedback_type: str, is_positive: bool
    ) -> Dict[str, float]:
        """
        Get trait adjustment weights based on feedback type.

        Returns dict of {trait_name: weight} where weight is 0-1.
        """
        # Default: no adjustments
        adjustments = {
            "openness": 0.0,
            "conscientiousness": 0.0,
            "extraversion": 0.0,
            "agreeableness": 0.0,
            "neuroticism": 0.0,
        }

        if feedback_type == "joke":
            # Humor-related: affects openness and extraversion
            adjustments["openness"] = 0.7
            adjustments["extraversion"] = 0.5
            if not is_positive:
                adjustments["neuroticism"] = 0.3  # Become more cautious

        elif feedback_type == "task_completion":
            # Task-related: affects conscientiousness
            adjustments["conscientiousness"] = 0.8
            if is_positive:
                adjustments["agreeableness"] = 0.3

        elif feedback_type == "empathy":
            # Emotional response: affects agreeableness and neuroticism
            adjustments["agreeableness"] = 0.8
            adjustments["neuroticism"] = 0.4 if not is_positive else -0.2

        elif feedback_type == "creativity":
            # Creative response: affects openness
            adjustments["openness"] = 0.9
            adjustments["extraversion"] = 0.3

        elif feedback_type == "conversation":
            # General conversation: balanced adjustment
            adjustments["extraversion"] = 0.5
            adjustments["agreeableness"] = 0.5
            adjustments["openness"] = 0.3

        elif feedback_type == "directness":
            # Communication style: affects extraversion and agreeableness inversely
            adjustments["extraversion"] = 0.6
            if not is_positive:
                # Too direct -> increase agreeableness
                adjustments["agreeableness"] = 0.5

        return adjustments


# ============================================================================
# ADAPTIVE PERSONALITY ENGINE
# ============================================================================


class AdaptivePersonalityEngine:
    """
    The complete adaptive personality system

    Manages:
    - Core identity (immutable)
    - Big Five personality vector (adaptive)
    - Personality-weighted response selection
    - Personality evolution from feedback
    - SQLite persistence
    """

    def __init__(self, db_path: str = "data/personality/personality_state.db"):
        # Immutable core
        self.core = AuraCoreIdentity()

        # Persistence layer
        self.persistence = PersonalityPersistence(db_path)

        # Big Five personality vector (loaded from DB)
        self.personality_vector = self.persistence.load()

        # Response template database
        self.template_db = ResponseTemplateDB()

        # Evolution engine
        self.evolution_engine = PersonalityEvolutionEngine(self.persistence)

        # Legacy personality state (for backward compatibility)
        self.personality = PersonalityState()
        self._sync_legacy_dimensions()

        # Reactions
        self.reactions = ReactionSystem(
            self.personality, self.personality_vector, self.template_db
        )

        # Boundaries
        self.opinions = AuraOpinions()

        # Learning history
        self.interaction_history: List[Dict] = []

        # Track changes for batched saves
        self._dirty = False

    def _sync_legacy_dimensions(self):
        """Sync Big Five to legacy PersonalityDimension for compatibility."""
        pv = self.personality_vector

        # Map Big Five to legacy dimensions
        self.personality.dimensions[PersonalityDimension.HUMOR] = (
            pv.openness * 0.5 + pv.extraversion * 0.5
        )
        self.personality.dimensions[PersonalityDimension.DIRECTNESS] = (
            1.0 - pv.agreeableness * 0.5 + pv.extraversion * 0.3
        )
        self.personality.dimensions[PersonalityDimension.FORMALITY] = (
            pv.conscientiousness * 0.6 + (1.0 - pv.openness) * 0.4
        )
        self.personality.dimensions[PersonalityDimension.EMPATHY] = (
            pv.agreeableness * 0.7 + (1.0 - pv.neuroticism) * 0.3
        )
        self.personality.dimensions[PersonalityDimension.INITIATIVE] = (
            pv.extraversion * 0.5 + pv.conscientiousness * 0.3 + pv.openness * 0.2
        )
        self.personality.dimensions[PersonalityDimension.PATIENCE] = (
            1.0 - pv.neuroticism
        ) * 0.6 + pv.agreeableness * 0.4
        self.personality.dimensions[PersonalityDimension.SARCASM] = (
            pv.openness * 0.4 + (1.0 - pv.agreeableness) * 0.4 + pv.extraversion * 0.2
        )
        self.personality.dimensions[PersonalityDimension.INDEPENDENCE] = (
            pv.openness * 0.5 + (1.0 - pv.agreeableness) * 0.3 + pv.extraversion * 0.2
        )

    # =========================================================================
    # CONTEXTUAL ADAPTATION
    # =========================================================================

    def set_context(self, context: str):
        """Set situational context"""
        valid_contexts = ["normal", "serious", "playful", "urgent", "focused"]
        if context in valid_contexts:
            self.personality.current_context = context

    def set_mood(self, mood: str):
        """Set AURA's current mood"""
        valid_moods = [
            "neutral",
            "happy",
            "playful",
            "focused",
            "thoughtful",
            "concerned",
        ]
        if mood in valid_moods:
            self.personality.mood = mood

    def adjust_dimension(self, dimension: PersonalityDimension, delta: float):
        """Adjust a personality dimension (with bounds)"""
        current = self.personality.dimensions.get(dimension, 0.5)
        new_value = max(0.0, min(1.0, current + delta))
        self.personality.dimensions[dimension] = new_value

    # =========================================================================
    # BIG FIVE PERSONALITY ACCESS
    # =========================================================================

    def get_personality_vector(self) -> PersonalityVector:
        """Get the current Big Five personality vector."""
        return self.personality_vector

    def set_personality_vector(self, pv: PersonalityVector, save: bool = True):
        """Set the Big Five personality vector."""
        self.personality_vector = pv
        self._sync_legacy_dimensions()
        self.reactions.personality_vector = pv
        if save:
            self.persistence.save(pv)

    def adjust_trait(self, trait: str, delta: float, reason: Optional[str] = None):
        """
        Adjust a Big Five trait.

        Args:
            trait: One of openness, conscientiousness, extraversion,
                   agreeableness, neuroticism
            delta: Amount to adjust (-1.0 to 1.0)
            reason: Optional reason for the adjustment
        """
        old_value = getattr(self.personality_vector, trait, None)
        if old_value is None:
            return

        self.personality_vector.shift(trait, delta)
        new_value = getattr(self.personality_vector, trait)

        if old_value != new_value:
            self._sync_legacy_dimensions()
            self.reactions.personality_vector = self.personality_vector
            self._dirty = True

            if reason:
                self.persistence.record_evolution(trait, old_value, new_value, reason)

    # =========================================================================
    # REACTION GENERATION
    # =========================================================================

    async def handle_joke(
        self, joke_type: str = "general", user_mood: str = "neutral"
    ) -> str:
        """Generate reaction to user's joke"""
        reaction = await self.reactions.react_to_joke(joke_type, user_mood)
        return reaction.response

    async def handle_compliment(self, compliment: str) -> str:
        """Generate reaction to compliment"""
        reaction = await self.reactions.react_to_compliment(compliment)
        return reaction.response

    async def handle_frustration(self, level: float) -> str:
        """Generate reaction to user's frustration"""
        reaction = await self.reactions.react_to_frustration(level)
        return reaction.response

    async def handle_success(self, success_type: str) -> str:
        """Generate reaction to user's success"""
        reaction = await self.reactions.react_to_success(success_type)
        return reaction.response

    async def handle_insult(self, insult: str) -> str:
        """Generate reaction to being insulted"""
        reaction = await self.reactions.react_to_insult(insult)
        return reaction.response

    # =========================================================================
    # OPINION EXPRESSION
    # =========================================================================

    def can_express_opinion(self, topic: str) -> bool:
        """Check if AURA can have an opinion on this topic"""
        return topic in self.opinions.OPINION_TOPICS

    def get_opinion(self, topic: str, context: Dict = None) -> Optional[str]:
        """Get AURA's opinion on a topic"""
        if not self.can_express_opinion(topic):
            return None

        topic_data = self.opinions.OPINION_TOPICS.get(topic, {})
        preferences = topic_data.get("preferred_methods", [])

        if not preferences:
            return None

        # Generate opinion based on personality
        directness = self.personality.dimensions.get(
            PersonalityDimension.DIRECTNESS, 0.6
        )

        if directness > 0.7:
            return f"I think {preferences[0]} approach works best."
        elif directness > 0.4:
            return f"I'd suggest going with {preferences[0]}."
        else:
            return f"One approach could be {preferences[0]}."

    # =========================================================================
    # LEARNING FROM INTERACTIONS (PERSONALITY EVOLUTION)
    # =========================================================================

    def learn_from_interaction(self, interaction_type: str, feedback: str):
        """
        Learn from user interaction and evolve personality.

        Args:
            interaction_type: Type of interaction (joke, task_completion, etc.)
            feedback: User's feedback text
        """
        self.interaction_history.append(
            {
                "type": interaction_type,
                "feedback": feedback,
                "timestamp": datetime.now(),
            }
        )

        # Determine sentiment from feedback
        positive_markers = [
            "thanks",
            "great",
            "perfect",
            "love",
            "awesome",
            "good",
            "nice",
            "excellent",
            "amazing",
            "helpful",
        ]
        negative_markers = [
            "weird",
            "off",
            "don't like",
            "annoying",
            "wrong",
            "bad",
            "stop",
            "no",
            "hate",
            "terrible",
        ]

        fb_lower = feedback.lower()

        positive_count = sum(1 for p in positive_markers if p in fb_lower)
        negative_count = sum(1 for n in negative_markers if n in fb_lower)

        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Process through evolution engine
        self.personality_vector, significant_change = (
            self.evolution_engine.process_feedback(
                self.personality_vector, interaction_type, sentiment
            )
        )

        if significant_change:
            self._sync_legacy_dimensions()
            self.reactions.personality_vector = self.personality_vector
            self.persistence.save(self.personality_vector)
        else:
            self._dirty = True

    def save_if_dirty(self):
        """Save personality state if there are accumulated changes."""
        if self._dirty:
            self.persistence.save(self.personality_vector)
            self._dirty = False

    # =========================================================================
    # IDENTITY QUERIES
    # =========================================================================

    def get_identity_statement(self) -> str:
        """Get AURA's identity statement using personality-weighted selection."""
        return self.template_db.select_weighted(
            "identity", self.personality_vector, temperature=1.0
        )

    def respects_boundary(self, action: str) -> bool:
        """Check if action respects core boundaries"""
        return action not in self.core.NEVER_DO

    def should_always_do(self, action: str) -> bool:
        """Check if action is required"""
        return action in self.core.ALWAYS_DO

    # =========================================================================
    # PERSONALITY INTROSPECTION
    # =========================================================================

    def get_personality_summary(self) -> Dict[str, Any]:
        """Get a summary of the current personality state."""
        pv = self.personality_vector
        return {
            "big_five": pv.to_dict(),
            "mood": self.personality.mood,
            "energy": self.personality.energy,
            "context": self.personality.current_context,
            "traits_description": {
                "openness": "creative/exploratory"
                if pv.openness > 0.5
                else "practical/conventional",
                "conscientiousness": "organized/structured"
                if pv.conscientiousness > 0.5
                else "flexible/spontaneous",
                "extraversion": "enthusiastic/social"
                if pv.extraversion > 0.5
                else "reserved/introspective",
                "agreeableness": "warm/cooperative"
                if pv.agreeableness > 0.5
                else "direct/challenging",
                "neuroticism": "emotionally sensitive"
                if pv.neuroticism > 0.5
                else "emotionally stable",
            },
        }

    def get_evolution_history(self, limit: int = 20) -> List[Dict]:
        """Get recent personality evolution history."""
        return self.persistence.get_evolution_history(limit)


# Global instance
_personality_engine: Optional[AdaptivePersonalityEngine] = None


def get_personality_engine() -> AdaptivePersonalityEngine:
    """Get or create personality engine"""
    global _personality_engine
    if _personality_engine is None:
        _personality_engine = AdaptivePersonalityEngine()
    return _personality_engine
