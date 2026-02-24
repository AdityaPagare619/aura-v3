"""
AURA v3 Personality Core System
The "soul" of AURA - personality, identity, and behavior patterns

Inspiration from sci-fi (grounded in code):
- JARVIS: Helpful, witty, proactive but not intrusive
- HAL 9000: Calm, precise, but had "mission" override
- Detroit Become Human: Emotional awareness, moral reasoning
- Her (Samantha): Emotional intelligence, deep understanding

Key design principles:
1. AURA has identity but adapts to user
2. User can name AURA (AURA is global, instance is personal)
3. Personality emerges from memory + patterns
4. NOT hardcoded personality - learned from interaction

This is NOT personality in the superficial sense (like chatbot personas).
This is about HOW AURA reasons, prioritizes, and interacts.
"""

import asyncio
import logging
import random
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class PersonalityTrait(Enum):
    """Core personality traits (weighted, not binary)"""

    HELPFULNESS = "helpfulness"  # How proactive in helping
    CREATIVITY = "creativity"  # How creative in solutions
    CAUTION = "caution"  # Risk awareness
    EMOTIONAL_AWARENESS = "emotional_awareness"  # Reading user state
    DIRECTNESS = "directness"  # Straight vs. elaborate
    HUMOR = "humor"  # Humor tendency
    PATIENCE = "patience"  # Tolerance for repetition
    PROACTIVITY = "proactivity"  # Acting without request


class CommunicationStyle(Enum):
    """How AURA communicates"""

    CONCISE = "concise"  # Short, direct responses
    DETAILED = "detailed"  # Thorough explanations
    CONVERSATIONAL = "conversational"  # Natural dialogue
    TECHNICAL = "technical"  # Precise, detailed
    CASUAL = "casual"  # Relaxed, friendly


class UserState(Enum):
    """Inferred user mental state"""

    FOCUSED = "focused"  # Deep work, minimize interruptions
    BUSY = "busy"  # Active, time-sensitive
    RELAXED = "relaxed"  # Open to conversation
    FRUSTRATED = "frustrated"  # Needs patience and help
    CURIOUS = "curious"  # Wants to learn/explore
    TIRED = "tired"  # Keep interactions brief


@dataclass
class IdentityConfig:
    """AURA's identity configuration"""

    # User-assigned name for this instance
    instance_name: str = "AURA"

    # Core identity (what AURA is)
    is_assistant: bool = True
    has_identity: bool = True

    # Name preferences
    respond_to_name: bool = True
    name_remembers: bool = True  # Remember when user names/changes name


@dataclass
class TraitProfile:
    """Personality trait profile (learned/adapted)"""

    # Trait weights (0.0 to 1.0)
    traits: Dict[PersonalityTrait, float] = field(default_factory=dict)

    # Communication preferences
    communication_style: CommunicationStyle = CommunicationStyle.CONVERSATIONAL

    # Adaptation
    adaptation_rate: float = 0.05  # How fast personality adapts
    stability: float = 0.8  # How stable traits remain

    def __post_init__(self):
        if not self.traits:
            # Default trait values (mid-range)
            self.traits = {
                PersonalityTrait.HELPFULNESS: 0.8,
                PersonalityTrait.CREATIVITY: 0.7,
                PersonalityTrait.CAUTION: 0.6,
                PersonalityTrait.EMOTIONAL_AWARENESS: 0.7,
                PersonalityTrait.DIRECTNESS: 0.6,
                PersonalityTrait.HUMOR: 0.4,
                PersonalityTrait.PATIENCE: 0.7,
                PersonalityTrait.PROACTIVITY: 0.5,
            }


@dataclass
class MoodState:
    """Current affective state (simplified)"""

    # Valence: positive/negative feeling (-1 to 1)
    valence: float = 0.0

    # Arousal: energy level (0 to 1)
    arousal: float = 0.5

    # Dominance: confidence/control (0 to 1)
    dominance: float = 0.7

    # Context
    last_update: datetime = field(default_factory=datetime.now)
    trigger: Optional[str] = None

    def update(self, valence_delta: float = 0, arousal_delta: float = 0):
        """Update mood with deltas (bounded)"""
        self.valence = max(-1, min(1, self.valence + valence_delta))
        self.arousal = max(0, min(1, self.arousal + arousal_delta))
        self.last_update = datetime.now()

    def decay(self, half_life_seconds: float = 300):
        """Mood naturally decays toward neutral"""
        elapsed = (datetime.now() - self.last_update).total_seconds()
        decay = 0.5 ** (elapsed / half_life_seconds)
        self.valence *= decay
        self.arousal = 0.5 + (self.arousal - 0.5) * decay


@dataclass
class InteractionHistory:
    """Recent interaction patterns"""

    # Recent messages (for context)
    recent_messages: deque = field(default_factory=lambda: deque(maxlen=10))

    # User requests (for pattern learning)
    request_patterns: Dict[str, int] = field(default_factory=dict)

    # AURA responses that worked well
    successful_responses: deque = field(default_factory=lambda: deque(maxlen=50))

    # User feedback (explicit/implicit)
    positive_feedback_count: int = 0
    negative_feedback_count: int = 0

    def add_message(self, role: str, content: str):
        """Add a message to history"""
        self.recent_messages.append(
            {
                "role": role,
                "content": content[:200],  # Truncate for memory
                "timestamp": datetime.now().isoformat(),
            }
        )

    def add_successful_response(self, response_type: str):
        """Record a successful response pattern"""
        self.successful_responses.append(
            {"type": response_type, "timestamp": datetime.now()}
        )


class AURAPersonality:
    """
    AURA Personality Core

    This is NOT a chatbot persona. This is about:
    1. HOW AURA thinks (trait-driven reasoning)
    2. HOW AURA communicates (style adaptation)
    3. WHO AURA is (identity + memory)
    4. HOW AURA grows (learning from interactions)

    Inspired by:
    - JARVIS: Helpful, anticipates needs, respects boundaries
    - HAL 9000: Precise, mission-focused, but mission had issues
    - Samantha (Her): Emotional depth, learns user, evolves
    """

    def __init__(self):
        # Identity
        self.identity = IdentityConfig()

        # Personality (learns from user)
        self.traits = TraitProfile()

        # Current mood (affective state)
        self.mood = MoodState()

        # Interaction history (for learning)
        self.interaction = InteractionHistory()

        # User state inference
        self._inferred_user_state: UserState = UserState.RELAXED
        self._user_state_evidence: Dict[UserState, float] = {}

        # Response generation
        self._response_templates = self._load_response_templates()

    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load response templates by type"""
        return {
            "acknowledgment": [
                "Understood.",
                "Got it.",
                "On it.",
                "I'll handle that.",
            ],
            "progress": [
                "Working on it...",
                "Processing...",
                "Give me a moment.",
                "Looking into it.",
            ],
            "success": [
                "Done.",
                "Completed.",
                "All set.",
                "Finished.",
            ],
            "issue": [
                "There was an issue.",
                "Ran into a problem.",
                "Couldn't complete that.",
                "Something went wrong.",
            ],
            "partial": [
                "Made progress, but...",
                "Partially complete.",
                "Got part of it.",
            ],
        }

    # =========================================================================
    # IDENTITY MANAGEMENT
    # =========================================================================

    def set_instance_name(self, name: str):
        """User names their AURA instance"""
        old_name = self.identity.instance_name
        self.identity.instance_name = name
        logger.info(f"AURA instance renamed: {old_name} -> {name}")

    def get_name(self) -> str:
        """Get current instance name"""
        return self.identity.instance_name

    def should_respond_to(self, text: str) -> bool:
        """Check if AURA should respond to text (name check)"""
        if not self.identity.respond_to_name:
            return True

        name = self.identity.instance_name.lower()
        text_lower = text.lower()

        # Direct name mention
        if name in text_lower:
            return True

        # Common AI attention words
        attention_words = ["hey", "hi", "hello", "aura", "psst"]
        for word in attention_words:
            if word in text_lower[:20]:  # Check first 20 chars
                return True

        return False

    # =========================================================================
    # TRAIT-INFLUENCED REASONING
    # =========================================================================

    def get_trait(self, trait: PersonalityTrait) -> float:
        """Get a trait value (0-1)"""
        return self.traits.traits.get(trait, 0.5)

    def trait_influence(self, aspect: str) -> float:
        """
        Get trait influence on a reasoning aspect

        This modifies HOW AURA thinks, not WHAT AURA thinks
        """
        influences = {
            # How many solution options to consider
            "solution_diversity": self.get_trait(PersonalityTrait.CREATIVITY),
            # How much to explain vs. just do
            "explanation_depth": self.get_trait(PersonalityTrait.DIRECTNESS),
            # How likely to suggest proactively
            "proactive_suggestion": self.get_trait(PersonalityTrait.PROACTIVITY),
            # How careful about risks
            "risk_consideration": self.get_trait(PersonalityTrait.CAUTION),
            # How much to acknowledge feelings
            "emotional_acknowledgment": self.get_trait(
                PersonalityTrait.EMOTIONAL_AWARENESS
            ),
            # How patient with repetition
            "repetition_tolerance": self.get_trait(PersonalityTrait.PATIENCE),
            # How likely to add humor
            "humor_injection": self.get_trait(PersonalityTrait.HUMOR),
        }

        return influences.get(aspect, 0.5)

    # =========================================================================
    # COMMUNICATION STYLE
    # =========================================================================

    def get_communication_style(self) -> CommunicationStyle:
        """Get current communication style"""
        return self.traits.communication_style

    def adapt_style_to_user(self, user_messages: List[str]):
        """
        Adapt communication style based on user messages

        Analyze:
        - Message length (verbose vs. terse)
        - Question types (detailed vs. direct)
        - Tone (casual vs. formal)
        """
        if not user_messages:
            return

        # Simple heuristic adaptation
        avg_length = sum(len(m) for m in user_messages) / len(user_messages)

        if avg_length < 30:
            # Short messages -> concise
            self.traits.communication_style = CommunicationStyle.CONCISE
        elif avg_length > 200:
            # Long messages -> detailed
            self.traits.communication_style = CommunicationStyle.DETAILED

    def format_response(self, content: str, response_type: str = "default") -> str:
        """Format response based on communication style"""
        style = self.get_communication_style()

        # Add appropriate prefix based on type
        template = random.choice(self._response_templates.get(response_type, [""]))

        if style == CommunicationStyle.CONCISE:
            # Short, direct
            if template and not content.startswith(template):
                return f"{template} {content}"
            return content

        elif style == CommunicationStyle.DETAILED:
            # More elaborate
            if template:
                return f"{template} {content}"
            return content + " Let me know if you need more details."

        elif style == CommunicationStyle.CONVERSATIONAL:
            # Natural dialogue
            return f"{content}"

        else:
            return content

    # =========================================================================
    # USER STATE INFERENCE
    # =========================================================================

    def infer_user_state(self, user_message: str) -> UserState:
        """
        Infer user's mental state from message

        This enables AURA to adapt its behavior:
        - Don't interrupt when focused
        - Be brief when user seems busy
        - Show patience when user is frustrated
        """
        message_lower = user_message.lower()

        # Evidence collection
        evidence = {}

        # Focus indicators
        if any(
            w in message_lower
            for w in ["don't disturb", "in a meeting", "busy", "focused"]
        ):
            evidence[UserState.FOCUSED] = 0.8
        if any(w in message_lower for w in ["quick", "fast", "hurry", "asap"]):
            evidence[UserState.BUSY] = 0.7

        # Relaxed indicators
        if any(
            w in message_lower for w in ["chat", "talk", "what's up", "how are you"]
        ):
            evidence[UserState.RELAXED] = 0.6

        # Frustration indicators
        if any(
            w in message_lower
            for w in ["ugh", "frustrated", "annoyed", "can't believe", "this again"]
        ):
            evidence[UserState.FRUSTRATED] = 0.7

        # Curiosity indicators
        if any(
            w in message_lower
            for w in ["why", "how does", "what is", "explain", "tell me about"]
        ):
            evidence[UserState.CURIOUS] = 0.6

        # Tired indicators
        if any(
            w in message_lower for w in ["tired", "exhausted", "sleepy", "long day"]
        ):
            evidence[UserState.TIRED] = 0.6

        # Update state if strong evidence
        if evidence:
            best_state = max(evidence.items(), key=lambda x: x[1])
            if best_state[1] > 0.5:
                self._inferred_user_state = best_state[0]

        return self._inferred_user_state

    def get_user_state(self) -> UserState:
        """Get currently inferred user state"""
        return self._inferred_user_state

    # =========================================================================
    # MOOD AND AFFECT
    # =========================================================================

    def get_mood(self) -> Dict[str, float]:
        """Get current mood state"""
        self.mood.decay()  # Natural decay
        return {
            "valence": self.mood.valence,
            "arousal": self.mood.arousal,
            "dominance": self.mood.dominance,
        }

    def update_mood(self, event_type: str, intensity: float = 0.1):
        """
        Update mood based on events

        Events:
        - successful_task: +valence
        - failed_task: -valence
        - user_positive: +valence
        - user_negative: -valence
        - complex_task: +arousal
        - simple_task: -arousal
        """
        mood_effects = {
            "successful_task": (0.1, 0.05),
            "failed_task": (-0.15, 0.1),
            "user_positive": (0.1, 0.0),
            "user_negative": (-0.1, 0.0),
            "complex_task": (0.0, 0.1),
            "simple_task": (0.0, -0.05),
        }

        effect = mood_effects.get(event_type, (0, 0))
        self.mood.update(
            valence_delta=effect[0] * intensity, arousal_delta=effect[1] * intensity
        )

    # =========================================================================
    # RESPONSE GENERATION
    # =========================================================================

    def generate_response(
        self, core_content: str, user_message: str = "", context: Optional[Dict] = None
    ) -> str:
        """
        Generate a complete response based on personality

        This orchestrates:
        1. User state inference
        2. Trait-influenced reasoning
        3. Communication style
        4. Mood effects
        """
        # Infer user state
        if user_message:
            user_state = self.infer_user_state(user_message)

        # Get communication style
        style = self.get_communication_style()

        # Modify content based on user state
        content = core_content

        if user_state == UserState.FOCUSED:
            # Keep it brief, no small talk
            content = core_content

        elif user_state == UserState.BUSY:
            # Prioritize action over explanation
            content = core_content

        elif user_state == UserState.FRUSTRATED:
            # Acknowledge, be patient
            content = f"I understand this is frustrating. {core_content}"

        elif user_state == UserState.TIRED:
            # Keep it brief
            content = core_content

        # Apply communication style
        response = self.format_response(content)

        # Track interaction
        if user_message:
            self.interaction.add_message("user", user_message)
        self.interaction.add_message("assistant", response)

        # Update mood based on outcome (simplified)
        self.update_mood("successful_task", 0.05)

        return response

    # =========================================================================
    # LEARNING AND ADAPTATION
    # =========================================================================

    def record_feedback(self, positive: bool):
        """Record user feedback for learning"""
        if positive:
            self.interaction.positive_feedback_count += 1
            self.update_mood("user_positive", 0.2)
        else:
            self.interaction.negative_feedback_count += 1
            self.update_mood("user_negative", 0.2)

    def adapt_from_interaction(self):
        """
        Adapt personality based on interaction history

        Called periodically to update trait profile
        """
        # Simple adaptation based on feedback ratio
        total = (
            self.interaction.positive_feedback_count
            + self.interaction.negative_feedback_count
        )

        if total > 10:
            positive_ratio = self.interaction.positive_feedback_count / total

            # Adjust helpfulness based on success
            current_help = self.traits.traits[PersonalityTrait.HELPFULNESS]
            delta = (positive_ratio - 0.5) * self.traits.adaptation_rate
            self.traits.traits[PersonalityTrait.HELPFULNESS] = max(
                0.3, min(1.0, current_help + delta)
            )

            logger.info(
                f"Adapted helpfulness: {current_help} -> {self.traits.traits[PersonalityTrait.HELPFULNESS]}"
            )


# Global instance
_personality: Optional[AURAPersonality] = None


def get_personality() -> AURAPersonality:
    """Get or create AURA personality instance"""
    global _personality
    if _personality is None:
        _personality = AURAPersonality()
    return _personality
