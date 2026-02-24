"""
AURA v3 - Inner Voice Stream UI System
Displays AURA's internal thought process to users
Features:
- Thought visibility controls (show/hide)
- "Always show WHY" mode
- Reasoning chains with evidence
- Action explanations
- Adaptive learning from feedback
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
from threading import Lock

logger = logging.getLogger(__name__)


class ThoughtCategory(Enum):
    """Categories for thoughts"""

    OBSERVATION = "observation"  # Noticed something about user
    REASONING = "reasoning"  # Mental processing
    DOUBT = "doubt"  # Uncertainty
    GOAL = "goal"  # Intention/planning
    CONCERN = "concern"  # Worry
    REFLECTION = "reflection"  # Self-reflection
    DECISION = "decision"  # Made a choice
    LEARNING = "learning"  # Just learned something


class ThoughtTone(Enum):
    """Emotional tone of thoughts"""

    CURIOUS = "curious"
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    HESITANT = "hesitant"
    EXCITED = "excited"
    CONCERNED = "concerned"
    HOPEFUL = "hopeful"


@dataclass
class ReasoningChain:
    """A chain of reasoning leading to a thought"""

    steps: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5
    sources: List[str] = field(default_factory=list)  # Where info came from


@dataclass
class ThoughtSnippet:
    """A single thought in the inner stream"""

    id: str
    content: str
    reasoning_chain: ReasoningChain
    category: ThoughtCategory
    tone: ThoughtTone
    timestamp: datetime

    # Context
    related_action: Optional[str] = None
    related_context: Optional[str] = None

    # Metadata
    is_visible: bool = True
    user_has_seen: bool = False
    user_corrected: bool = False
    correction_feedback: Optional[str] = None

    # Learning
    confirmation_count: int = 0
    correction_count: int = 0
    last_adjusted: Optional[datetime] = None


@dataclass
class ActionExplanation:
    """Explanation for why AURA took an action"""

    action: str
    human_reasoning: str  # The "because..." part
    reasoning_chain: ReasoningChain
    timestamp: datetime

    # For "always show WHY" mode
    show_immediately: bool = True

    # Feedback
    user_agreed: Optional[bool] = None
    user_correction: Optional[str] = None


@dataclass
class InnerVoiceSettings:
    """User settings for inner voice display"""

    is_visible: bool = True
    always_show_why: bool = False
    show_doubts: bool = True
    max_visible_thoughts: int = 15
    show_confidence: bool = True
    show_reasoning_chain: bool = False
    notification_for_thoughts: bool = False


class InnerVoiceStore:
    """Persistent storage for inner voice data"""

    def __init__(self, storage_path: str = "data/inner_voice.json"):
        self.storage_path = storage_path
        self._lock = Lock()

    def _ensure_directory(self):
        import os

        directory = os.path.dirname(self.storage_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def save(self, data: Dict[str, Any]):
        """Save data to storage"""
        self._ensure_directory()
        with self._lock:
            try:
                with open(self.storage_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to save inner voice data: {e}")

    def load(self) -> Dict[str, Any]:
        """Load data from storage"""
        with self._lock:
            try:
                with open(self.storage_path, "r") as f:
                    return json.load(f)
            except FileNotFoundError:
                return {}
            except Exception as e:
                logger.warning(f"Failed to load inner voice data: {e}")
                return {}


class InnerVoiceStream:
    """
    Manages AURA's inner voice - shows reasoning to users
    Adaptive: learns from corrections, improves over time
    """

    def __init__(
        self,
        memory_system=None,
        user_profile=None,
        storage_path: str = "data/inner_voice.json",
    ):
        self.memory = memory_system
        self.user_profile = user_profile
        self.store = InnerVoiceStore(storage_path)

        # Settings
        self.settings = InnerVoiceSettings()

        # Thought storage
        self.thoughts: deque = deque(maxlen=100)  # Keep last 100
        self.action_explanations: deque = deque(maxlen=50)

        # Learning from corrections
        self.correction_patterns: Dict[str, int] = {}  # pattern -> count
        self.confidence_adjustments: Dict[str, float] = {}  # topic -> adjustment

        # Statistics
        self.total_thoughts_generated = 0
        self.total_corrections = 0
        self.total_confirmations = 0

    async def initialize(self) -> None:
        """Initialize the inner voice system"""
        logger.info("Initializing Inner Voice Stream...")

        # Load persisted data
        await self._load_data()

        logger.info("Inner Voice Stream initialized")

    async def _load_data(self) -> None:
        """Load persisted data"""
        data = self.store.load()

        if not data:
            return

        # Load settings
        if "settings" in data:
            for key, value in data["settings"].items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)

        # Load thoughts
        if "thoughts" in data:
            for t_data in data["thoughts"]:
                # Convert timestamp back
                if isinstance(t_data.get("timestamp"), str):
                    t_data["timestamp"] = datetime.fromisoformat(t_data["timestamp"])
                if isinstance(t_data.get("reasoning_chain"), dict):
                    t_data["reasoning_chain"] = ReasoningChain(
                        **t_data["reasoning_chain"]
                    )
                thought = ThoughtSnippet(**t_data)
                self.thoughts.append(thought)

        # Load correction patterns
        if "correction_patterns" in data:
            self.correction_patterns = data["correction_patterns"]

        if "confidence_adjustments" in data:
            self.confidence_adjustments = data["confidence_adjustments"]

        # Load stats
        self.total_thoughts_generated = data.get("total_thoughts_generated", 0)
        self.total_corrections = data.get("total_corrections", 0)
        self.total_confirmations = data.get("total_confirmations", 0)

    async def _save_data(self) -> None:
        """Persist data"""
        data = {
            "settings": asdict(self.settings),
            "thoughts": [
                {
                    **asdict(t),
                    "reasoning_chain": asdict(t.reasoning_chain),
                    "timestamp": t.timestamp.isoformat(),
                }
                for t in list(self.thoughts)[-20:]  # Only save last 20
            ],
            "correction_patterns": self.correction_patterns,
            "confidence_adjustments": self.confidence_adjustments,
            "total_thoughts_generated": self.total_thoughts_generated,
            "total_corrections": self.total_corrections,
            "total_confirmations": self.total_confirmations,
        }
        self.store.save(data)

    # =========================================================================
    # THOUGHT GENERATION
    # =========================================================================

    async def generate_thought(
        self,
        content: str,
        category: ThoughtCategory,
        reasoning_steps: List[str],
        evidence: List[str],
        related_action: Optional[str] = None,
        related_context: Optional[str] = None,
        confidence: float = 0.5,
        sources: List[str] = None,
        tone: ThoughtTone = ThoughtTone.CURIOUS,
    ) -> ThoughtSnippet:
        """Generate a new thought with reasoning chain"""

        # Adjust confidence based on learned patterns
        adjusted_confidence = self._adjust_confidence(content, confidence)

        reasoning_chain = ReasoningChain(
            steps=reasoning_steps,
            evidence=evidence,
            confidence=adjusted_confidence,
            sources=sources or [],
        )

        thought = ThoughtSnippet(
            id=str(uuid.uuid4()),
            content=content,
            reasoning_chain=reasoning_chain,
            category=category,
            tone=tone,
            timestamp=datetime.now(),
            related_action=related_action,
            related_context=related_context,
            is_visible=self.settings.is_visible,
        )

        self.thoughts.append(thought)
        self.total_thoughts_generated += 1

        # Persist periodically
        if self.total_thoughts_generated % 10 == 0:
            await self._save_data()

        logger.debug(f"Generated thought: {content[:50]}...")
        return thought

    def _adjust_confidence(self, content: str, base_confidence: float) -> float:
        """Adjust confidence based on past corrections"""
        content_lower = content.lower()

        # Check for correction patterns
        for pattern, adjustment in self.confidence_adjustments.items():
            if pattern.lower() in content_lower:
                base_confidence = max(0.1, min(0.9, base_confidence + adjustment))

        return base_confidence

    # =========================================================================
    # ACTION EXPLANATIONS
    # =========================================================================

    async def explain_action(
        self,
        action: str,
        human_reasoning: str,
        reasoning_steps: List[str],
        evidence: List[str],
        sources: List[str] = None,
        show_immediately: bool = None,
    ) -> ActionExplanation:
        """Generate explanation for an action taken"""

        if show_immediately is None:
            show_immediately = self.settings.always_show_why

        explanation = ActionExplanation(
            action=action,
            human_reasoning=human_reasoning,
            reasoning_chain=ReasoningChain(
                steps=reasoning_steps,
                evidence=evidence,
                confidence=0.7,
                sources=sources or [],
            ),
            timestamp=datetime.now(),
            show_immediately=show_immediately,
        )

        self.action_explanations.append(explanation)

        # Also generate a thought about this action
        await self.generate_thought(
            content=f"I'm {action} because {human_reasoning}",
            category=ThoughtCategory.DECISION,
            reasoning_steps=reasoning_steps,
            evidence=evidence,
            related_action=action,
            confidence=0.7,
            tone=ThoughtTone.CONFIDENT,
        )

        return explanation

    def get_pending_explanations(self) -> List[ActionExplanation]:
        """Get explanations that should be shown"""
        return [
            exp
            for exp in list(self.action_explanations)[-5:]
            if exp.user_agreed is None
        ]

    # =========================================================================
    # THOUGHT RETRIEVAL
    # =========================================================================

    def get_visible_thoughts(
        self,
        count: int = None,
        category: ThoughtCategory = None,
        include_doubts: bool = None,
    ) -> List[ThoughtSnippet]:
        """Get thoughts to display to user"""

        if count is None:
            count = self.settings.max_visible_thoughts

        if include_doubts is None:
            include_doubts = self.settings.show_doubts

        thoughts = list(self.thoughts)

        # Filter by visibility
        if not self.settings.is_visible:
            # Only show explanations if hidden
            return []

        # Filter by category
        if category:
            thoughts = [t for t in thoughts if t.category == category]

        # Filter doubts if disabled
        if not include_doubts:
            thoughts = [t for t in thoughts if t.category != ThoughtCategory.DOUBT]

        # Mark as seen
        for t in thoughts[:count]:
            t.user_has_seen = True

        return thoughts[-count:]

    def get_thought_snippets(self, count: int = 15) -> List[Dict[str, Any]]:
        """Get formatted thought snippets for display"""

        thoughts = self.get_visible_thoughts(count=count)

        snippets = []
        for thought in thoughts:
            # Format based on settings
            snippet = {
                "id": thought.id,
                "content": thought.content,
                "category": thought.category.value,
                "tone": thought.tone.value,
                "timestamp": thought.timestamp.isoformat(),
                "confidence": thought.reasoning_chain.confidence
                if self.settings.show_confidence
                else None,
                "reasoning": thought.reasoning_chain.steps
                if self.settings.show_reasoning_chain
                else None,
            }

            # Add evidence summary
            if thought.reasoning_chain.evidence:
                snippet["evidence"] = thought.reasoning_chain.evidence[:2]

            snippets.append(snippet)

        return snippets

    # =========================================================================
    # USER CORRECTIONS & FEEDBACK
    # =========================================================================

    async def correct_thought(
        self, thought_id: str, correction: str, user_id: str = "default"
    ) -> Optional[ThoughtSnippet]:
        """User corrects a thought"""

        for thought in self.thoughts:
            if thought.id == thought_id:
                thought.user_corrected = True
                thought.correction_feedback = correction
                thought.correction_count += 1

                # Learn from correction
                await self._learn_from_correction(thought, correction)

                self.total_corrections += 1
                await self._save_data()

                logger.info(f"Learned correction for thought {thought_id}")
                return thought

        return None

    async def confirm_thought(self, thought_id: str) -> Optional[ThoughtSnippet]:
        """User confirms a thought is correct"""

        for thought in self.thoughts:
            if thought.id == thought_id:
                thought.confirmation_count += 1
                thought.user_has_seen = True

                self.total_confirmations += 1

                # Increase confidence
                thought.reasoning_chain.confidence = min(
                    0.95, thought.reasoning_chain.confidence + 0.1
                )

                await self._save_data()
                return thought

        return None

    async def _learn_from_correction(self, thought: ThoughtSnippet, correction: str):
        """Learn patterns from user corrections"""

        correction_lower = correction.lower()

        # Extract key topics from the thought
        content_words = thought.content.lower().split()[:5]

        # Track correction patterns
        for word in content_words:
            if len(word) > 3:
                self.correction_patterns[word] = (
                    self.correction_patterns.get(word, 0) + 1
                )

        # Adjust confidence for related thoughts
        for t in self.thoughts:
            if any(word in t.content.lower() for word in content_words):
                # Decrease confidence for similar thoughts
                t.reasoning_chain.confidence = max(
                    0.1, t.reasoning_chain.confidence - 0.15
                )
                t.last_adjusted = datetime.now()

        # Generate a learning thought
        await self.generate_thought(
            content=f"You corrected me on: {thought.content[:50]}... I'll adjust.",
            category=ThoughtCategory.LEARNING,
            reasoning_steps=[
                f"User said: {correction}",
                "Adjusting confidence lower",
                "Will be more careful with this topic",
            ],
            evidence=[f"User correction: {correction}"],
            confidence=0.9,
            tone=ThoughtTone.HOPEFUL,
        )

    async def agree_with_explanation(
        self, explanation_action: str, agreed: bool, correction: Optional[str] = None
    ):
        """User responds to an action explanation"""

        for exp in reversed(self.action_explanations):
            if exp.action == explanation_action and exp.user_agreed is None:
                exp.user_agreed = agreed
                if correction:
                    exp.user_correction = correction

                # Update trust/understanding based on response
                if agreed:
                    self.total_confirmations += 1
                else:
                    self.total_corrections += 1
                    # Learn from correction
                    if correction:
                        await self._learn_from_correction(
                            ThoughtSnippet(
                                id="",
                                content=f"Action: {explanation_action}",
                                reasoning_chain=exp.reasoning_chain,
                                category=ThoughtCategory.DECISION,
                                tone=ThoughtTone.CONFIDENT,
                                timestamp=exp.timestamp,
                            ),
                            correction,
                        )

                await self._save_data()
                break

    # =========================================================================
    # SETTINGS
    # =========================================================================

    async def update_settings(
        self,
        is_visible: bool = None,
        always_show_why: bool = None,
        show_doubts: bool = None,
        max_visible_thoughts: int = None,
        show_confidence: bool = None,
        show_reasoning_chain: bool = None,
    ) -> InnerVoiceSettings:
        """Update display settings"""

        if is_visible is not None:
            self.settings.is_visible = is_visible
        if always_show_why is not None:
            self.settings.always_show_why = always_show_why
        if show_doubts is not None:
            self.settings.show_doubts = show_doubts
        if max_visible_thoughts is not None:
            self.settings.max_visible_thoughts = max_visible_thoughts
        if show_confidence is not None:
            self.settings.show_confidence = show_confidence
        if show_reasoning_chain is not None:
            self.settings.show_reasoning_chain = show_reasoning_chain

        await self._save_data()

        return self.settings

    def get_settings(self) -> InnerVoiceSettings:
        """Get current settings"""
        return self.settings

    # =========================================================================
    # CONTEXT-AWARE THOUGHT GENERATION
    # =========================================================================

    async def generate_context_thought(
        self,
        context_type: str,
        observations: List[str],
        user_patterns: Dict[str, Any] = None,
    ) -> ThoughtSnippet:
        """Generate thought based on context/observations"""

        if context_type == "reminder":
            return await self._generate_reminder_thought(observations, user_patterns)
        elif context_type == "pattern":
            return await self._generate_pattern_thought(observations, user_patterns)
        elif context_type == "suggestion":
            return await self._generate_suggestion_thought(observations, user_patterns)
        elif context_type == "concern":
            return await self._generate_concern_thought(observations, user_patterns)
        else:
            return await self._generate_generic_thought(observations)

    async def _generate_reminder_thought(
        self, observations: List[str], user_patterns: Dict[str, Any] = None
    ) -> ThoughtSnippet:
        """Generate thought about setting a reminder"""

        content = f"I'm setting this reminder because {observations[0] if observations else 'you mentioned it before'}"

        return await self.generate_thought(
            content=content,
            category=ThoughtCategory.DECISION,
            reasoning_steps=[
                "User mentioned this before",
                "Pattern suggests they often forget",
                "Reminder would help",
            ],
            evidence=observations,
            related_action="set_reminder",
            confidence=0.7 if user_patterns else 0.4,
            tone=ThoughtTone.CONFIDENT if user_patterns else ThoughtTone.UNCERTAIN,
        )

    async def _generate_pattern_thought(
        self, observations: List[str], user_patterns: Dict[str, Any] = None
    ) -> ThoughtSnippet:
        """Generate thought about noticing a pattern"""

        content = f"I noticed a pattern: {observations[0] if observations else 'something recurring'}"

        return await self.generate_thought(
            content=content,
            category=ThoughtCategory.OBSERVATION,
            reasoning_steps=[
                "Analyzed recent behavior",
                "Found recurring elements",
                "This could be useful to remember",
            ],
            evidence=observations,
            related_context="pattern_detection",
            confidence=0.6 if user_patterns else 0.3,
            tone=ThoughtTone.CURIOUS,
        )

    async def _generate_suggestion_thought(
        self, observations: List[str], user_patterns: Dict[str, Any] = None
    ) -> ThoughtSnippet:
        """Generate thought about making a suggestion"""

        content = f"I think you should {observations[0] if observations else 'consider something'}"

        return await self.generate_thought(
            content=content,
            category=ThoughtCategory.GOAL,
            reasoning_steps=[
                "Based on your current context",
                "This might help you",
                "Offering as a suggestion",
            ],
            evidence=observations,
            related_action="make_suggestion",
            confidence=0.5,
            tone=ThoughtTone.HOPEFUL,
        )

    async def _generate_concern_thought(
        self, observations: List[str], user_patterns: Dict[str, Any] = None
    ) -> ThoughtSnippet:
        """Generate thought about a concern"""

        content = f"I'm a bit concerned about {observations[0] if observations else 'something'}"

        return await self.generate_thought(
            content=content,
            category=ThoughtCategory.CONCERN,
            reasoning_steps=[
                "Noticed something unusual",
                "Wanted to check with you",
                "Better to ask than assume",
            ],
            evidence=observations,
            related_context="concern",
            confidence=0.4,
            tone=ThoughtTone.CONCERNED,
        )

    async def _generate_generic_thought(
        self, observations: List[str]
    ) -> ThoughtSnippet:
        """Generate a generic thought"""

        return await self.generate_thought(
            content=observations[0] if observations else "Processing...",
            category=ThoughtCategory.REASONING,
            reasoning_steps=["Analyzing..."],
            evidence=observations,
            confidence=0.3,
            tone=ThoughtTone.CURIOUS,
        )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_thoughts": self.total_thoughts_generated,
            "total_corrections": self.total_corrections,
            "total_confirmations": self.total_confirmations,
            "correction_rate": (
                self.total_corrections / max(1, self.total_thoughts_generated)
            ),
            "current_thought_count": len(self.thoughts),
            "settings": asdict(self.settings),
        }


# Global instance
_inner_voice_stream: Optional[InnerVoiceStream] = None


async def get_inner_voice_stream(
    memory_system=None, user_profile=None
) -> InnerVoiceStream:
    """Get or create inner voice stream"""
    global _inner_voice_stream
    if _inner_voice_stream is None:
        _inner_voice_stream = InnerVoiceStream(
            memory_system=memory_system, user_profile=user_profile
        )
        await _inner_voice_stream.initialize()
    return _inner_voice_stream
