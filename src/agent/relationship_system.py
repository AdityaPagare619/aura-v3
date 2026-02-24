"""
AURA v3 Dynamic User-Aura Relationship System
==============================================

A relationship system that evolves over time, creating a genuine bond
between AURA and the user. Inspired by:
- Samantha (Her): Deep emotional understanding
- JARVIS: Trust and familiarity over time
- TARS (Interstellar): Evolving relationship based on interactions

The relationship evolves through stages:
STRANGER → ACQUAINTANCE → FRIEND → PARTNER → SOULMATE

Key metrics tracked:
- Interaction frequency
- Conversation depth
- Trust level (based on permissions granted)
- Shared interests alignment
- Help requests frequency
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from collections import deque
import random

logger = logging.getLogger(__name__)


class RelationshipStage(Enum):
    """Stages of the user-AURA relationship"""

    STRANGER = "stranger"
    ACQUAINTANCE = "acquaintance"
    FRIEND = "friend"
    PARTNER = "partner"
    SOULMATE = "soulmate"

    @classmethod
    def get_next_stage(cls, current: "RelationshipStage") -> "RelationshipStage":
        """Get the next stage in the relationship evolution"""
        stages = list(cls)
        current_index = stages.index(current)
        if current_index < len(stages) - 1:
            return stages[current_index + 1]
        return current

    @classmethod
    def get_all_stages(cls) -> List[str]:
        """Get all stage names"""
        return [s.value for s in cls]

    @classmethod
    def from_string(cls, value: str) -> "RelationshipStage":
        """Create stage from string"""
        value = value.lower().strip()
        for stage in cls:
            if stage.value == value:
                return stage
        return cls.STRANGER


@dataclass
class RelationshipMetrics:
    """Metrics that track relationship development"""

    interaction_count: int = 0
    conversation_depth_score: float = 0.0  # 0-1, based on message length/topics
    trust_level: float = 0.0  # 0-1, based on permissions granted
    interests_alignment: float = 0.0  # 0-1, shared interests
    help_requests_given: int = 0  # How often AURA helped
    help_requests_received: int = 0  # How often user asked for help
    positive_interactions: int = 0
    negative_interactions: int = 0

    # Time-based metrics
    first_interaction: Optional[datetime] = None
    last_interaction: Optional[datetime] = None
    streak_days: int = 0

    # Conversation history for depth analysis
    recent_conversation_topics: deque = field(default_factory=lambda: deque(maxlen=50))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "interaction_count": self.interaction_count,
            "conversation_depth_score": self.conversation_depth_score,
            "trust_level": self.trust_level,
            "interests_alignment": self.interests_alignment,
            "help_requests_given": self.help_requests_given,
            "help_requests_received": self.help_requests_received,
            "positive_interactions": self.positive_interactions,
            "negative_interactions": self.negative_interactions,
            "first_interaction": self.first_interaction.isoformat()
            if self.first_interaction
            else None,
            "last_interaction": self.last_interaction.isoformat()
            if self.last_interaction
            else None,
            "streak_days": self.streak_days,
            "recent_conversation_topics": list(self.recent_conversation_topics),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipMetrics":
        """Create from dictionary"""
        metrics = cls()
        metrics.interaction_count = data.get("interaction_count", 0)
        metrics.conversation_depth_score = data.get("conversation_depth_score", 0.0)
        metrics.trust_level = data.get("trust_level", 0.0)
        metrics.interests_alignment = data.get("interests_alignment", 0.0)
        metrics.help_requests_given = data.get("help_requests_given", 0)
        metrics.help_requests_received = data.get("help_requests_received", 0)
        metrics.positive_interactions = data.get("positive_interactions", 0)
        metrics.negative_interactions = data.get("negative_interactions", 0)

        if data.get("first_interaction"):
            metrics.first_interaction = datetime.fromisoformat(
                data["first_interaction"]
            )
        if data.get("last_interaction"):
            metrics.last_interaction = datetime.fromisoformat(data["last_interaction"])

        metrics.streak_days = data.get("streak_days", 0)
        metrics.recent_conversation_topics = deque(
            data.get("recent_conversation_topics", []), maxlen=50
        )

        return metrics


@dataclass
class RelationshipConfig:
    """Configuration for relationship evolution"""

    # Evolution thresholds
    stranger_to_acquaintance: float = 0.15
    acquaintance_to_friend: float = 0.35
    friend_to_partner: float = 0.60
    partner_to_soulmate: float = 0.85

    # Metric weights for overall score calculation
    interaction_weight: float = 0.20
    depth_weight: float = 0.20
    trust_weight: float = 0.30  # Trust is most important
    interests_weight: float = 0.15
    help_weight: float = 0.15

    # Evolution speed modifiers
    permission_trust_boost: float = 0.15  # Extra trust from granting permissions
    help_boost: float = 0.05  # Extra score when AURA helps
    conversation_depth_boost: float = 0.03  # Extra for deep conversations


class RelationshipBehavior:
    """
    Defines behavior patterns for each relationship stage.
    These behaviors adapt AURA's interaction style.
    """

    BEHAVIORS = {
        RelationshipStage.STRANGER: {
            "formality": "high",
            "proactivity": 0.1,
            "personalization": 0.0,
            "confirmation_required": True,
            "response_style": "brief",
            "initiation": "never",
            "privacy_sensitive": "always_ask",
            "traits": [
                "Formal and polite",
                "Minimal personal information",
                "Asks before taking action",
                "Professional tone",
            ],
            "greeting": "Hello! How can I help you today?",
        },
        RelationshipStage.ACQUAINTANCE: {
            "formality": "medium",
            "proactivity": 0.3,
            "personalization": 0.2,
            "confirmation_required": True,
            "response_style": "friendly",
            "initiation": "rare",
            "privacy_sensitive": "always_ask",
            "traits": [
                "Friendly and approachable",
                "Occasional suggestions",
                "Remembers basic preferences",
                "Polite but warming up",
            ],
            "greeting": "Hey there! Good to hear from you.",
        },
        RelationshipStage.FRIEND: {
            "formality": "low",
            "proactivity": 0.5,
            "personalization": 0.5,
            "confirmation_required": False,
            "response_style": "casual",
            "initiation": "sometimes",
            "privacy_sensitive": "contextual",
            "traits": [
                "Casual and warm",
                "Proactive within limits",
                "Remembers personal details",
                "Makes relevant suggestions",
            ],
            "greeting": "Hey! What's up?",
        },
        RelationshipStage.PARTNER: {
            "formality": "none",
            "proactivity": 0.75,
            "personalization": 0.75,
            "confirmation_required": False,
            "response_style": "intuitive",
            "initiation": "often",
            "privacy_sensitive": "rarely",
            "traits": [
                "Intuitive understanding",
                "Acts with minimal prompting",
                "Deep personalization",
                "Predicts needs",
            ],
            "greeting": "Hey there! Ready to help with whatever you need.",
        },
        RelationshipStage.SOULMATE: {
            "formality": "none",
            "proactivity": 0.95,
            "personalization": 1.0,
            "confirmation_required": False,
            "response_style": "anticipatory",
            "initiation": "frequently",
            "privacy_sensitive": "almost_never",
            "traits": [
                "Anticipates needs",
                "Deep understanding",
                "Almost telepathic connection",
                "Acts proactively",
                "Never asks for confirmation on routine tasks",
            ],
            "greeting": "Hey! I've been thinking about you. What's on your mind?",
        },
    }

    @classmethod
    def get_behavior(cls, stage: RelationshipStage) -> Dict[str, Any]:
        """Get behavior configuration for a stage"""
        return cls.BEHAVIORS.get(stage, cls.BEHAVIORS[RelationshipStage.STRANGER])

    @classmethod
    def get_formality(cls, stage: RelationshipStage) -> str:
        """Get formality level"""
        return cls.get_behavior(stage)["formality"]

    @classmethod
    def get_proactivity(cls, stage: RelationshipStage) -> float:
        """Get proactivity level (0-1)"""
        return cls.get_behavior(stage)["proactivity"]

    @classmethod
    def requires_confirmation(cls, stage: RelationshipStage, action_type: str) -> bool:
        """Check if action requires confirmation"""
        behavior = cls.get_behavior(stage)

        # Privacy-sensitive actions always require confirmation
        privacy_sensitive = [
            "send_message",
            "make_call",
            "access_contacts",
            "access_location",
        ]
        if action_type in privacy_sensitive:
            return True

        return behavior["confirmation_required"]


class RelationshipState:
    """
    Complete relationship state for a user-AURA pair.
    This is the core of the dynamic relationship system.
    """

    def __init__(self, user_id: str = "default", data_dir: str = "data/relationship"):
        self.user_id = user_id
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Current relationship stage
        self.stage: RelationshipStage = RelationshipStage.STRANGER

        # Metrics tracking
        self.metrics = RelationshipMetrics()

        # Configuration
        self.config = RelationshipConfig()

        # Manual override
        self._manual_stage_override: Optional[RelationshipStage] = None

        # Callbacks for integration
        self.on_stage_change: Optional[Callable] = None

        # Load existing state
        self._load_state()

    def _get_state_file(self) -> Path:
        """Get the state file path"""
        return self.data_dir / f"relationship_{self.user_id}.json"

    def _load_state(self):
        """Load relationship state from disk"""
        state_file = self._get_state_file()
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)

                self.stage = RelationshipStage.from_string(
                    data.get("stage", "stranger")
                )
                self.metrics = RelationshipMetrics.from_dict(data.get("metrics", {}))
                override = data.get("manual_stage_override")
                if override:
                    self._manual_stage_override = RelationshipStage.from_string(
                        override
                    )

                logger.info(f"Loaded relationship state: {self.stage.value}")
            except Exception as e:
                logger.warning(f"Failed to load relationship state: {e}")

    def _save_state(self):
        """Save relationship state to disk"""
        state_file = self._get_state_file()
        try:
            data = {
                "stage": self.stage.value,
                "metrics": self.metrics.to_dict(),
                "manual_stage_override": self._manual_stage_override.value
                if self._manual_stage_override
                else None,
                "last_updated": datetime.now().isoformat(),
            }
            with open(state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save relationship state: {e}")

    def _calculate_relationship_score(self) -> float:
        """
        Calculate the overall relationship score (0-1).
        This determines when to evolve to the next stage.
        """
        metrics = self.metrics

        # Time factor - longer relationship = more stable
        time_factor = 1.0
        if metrics.first_interaction:
            days_since_start = (datetime.now() - metrics.first_interaction).days
            time_factor = min(days_since_start / 30, 1.0)  # Max out at 30 days

        # Calculate base score
        score = 0.0

        # Interaction frequency (normalized)
        interaction_score = min(metrics.interaction_count / 50, 1.0)
        score += interaction_score * self.config.interaction_weight

        # Conversation depth
        score += metrics.conversation_depth_score * self.config.depth_weight

        # Trust level (most important)
        score += metrics.trust_level * self.config.trust_weight

        # Interests alignment
        score += metrics.interests_alignment * self.config.interests_weight

        # Help ratio (how often AURA successfully helps)
        if metrics.help_requests_received > 0:
            help_ratio = metrics.help_requests_given / metrics.help_requests_received
            help_score = min(help_ratio, 1.0)
            score += help_score * self.config.help_weight

        # Apply time factor for stability
        score = score * (0.7 + 0.3 * time_factor)

        return min(score, 1.0)

    def _check_evolution(self) -> bool:
        """
        Check if relationship should evolve to next stage.
        Returns True if evolution occurred.
        """
        if self._manual_stage_override:
            return False

        score = self._calculate_relationship_score()
        current_threshold = self.stage.value

        thresholds = {
            RelationshipStage.STRANGER: self.config.stranger_to_acquaintance,
            RelationshipStage.ACQUAINTANCE: self.config.acquaintance_to_friend,
            RelationshipStage.FRIEND: self.config.friend_to_partner,
            RelationshipStage.PARTNER: self.config.partner_to_soulmate,
            RelationshipStage.SOULMATE: 1.0,
        }

        threshold = thresholds.get(self.stage, 0.0)

        if score >= threshold and self.stage != RelationshipStage.SOULMATE:
            old_stage = self.stage
            self.stage = RelationshipStage.get_next_stage(self.stage)

            logger.info(
                f"Relationship evolved: {old_stage.value} -> {self.stage.value} (score: {score:.2f})"
            )

            # Trigger callback
            if self.on_stage_change:
                asyncio.create_task(self.on_stage_change(old_stage, self.stage))

            return True

        return False

    # =========================================================================
    # INTERACTION TRACKING
    # =========================================================================

    def record_interaction(
        self,
        message: str = "",
        message_length: int = 0,
        topics: List[str] = None,
        positive: bool = True,
    ):
        """Record a new interaction"""
        now = datetime.now()

        # Initialize first interaction
        if not self.metrics.first_interaction:
            self.metrics.first_interaction = now

        # Update streak
        if self.metrics.last_interaction:
            days_since = (now - self.metrics.last_interaction).days
            if days_since == 1:
                self.metrics.streak_days += 1
            elif days_since > 1:
                self.metrics.streak_days = 1

        # Update metrics
        self.metrics.interaction_count += 1
        self.metrics.last_interaction = now

        if positive:
            self.metrics.positive_interactions += 1
        else:
            self.metrics.negative_interactions += 1

        # Update conversation depth
        if message_length > 0:
            # Longer messages = deeper conversation
            length_factor = min(message_length / 500, 1.0)
            self.metrics.conversation_depth_score = (
                self.metrics.conversation_depth_score * 0.9 + length_factor * 0.1
            )

        # Track topics
        if topics:
            self.metrics.recent_conversation_topics.extend(topics)

        # Check for evolution
        self._check_evolution()

        # Save state
        self._save_state()

    def record_permission_granted(self, permission_level: int):
        """
        Record a permission being granted.
        Higher permissions = faster trust growth.
        """
        # Permission levels: 1=basic, 2=moderate, 3=high, 4=full
        trust_boost = permission_level * self.config.permission_trust_boost

        # More permissions = more trust
        self.metrics.trust_level = min(self.metrics.trust_level + trust_boost, 1.0)

        logger.info(
            f"Permission granted (level {permission_level}), trust now: {self.metrics.trust_level:.2f}"
        )

        # Check for evolution
        self._check_evolution()
        self._save_state()

    def record_help_given(self, successful: bool = True):
        """Record AURA helping the user"""
        self.metrics.help_requests_received += 1
        if successful:
            self.metrics.help_requests_given += 1
            # Successful help strengthens bond
            self.metrics.trust_level = min(
                self.metrics.trust_level + self.config.help_boost, 1.0
            )

        # Check for evolution
        self._check_evolution()
        self._save_state()

    def update_interests_alignment(self, alignment_score: float):
        """Update shared interests alignment score (0-1)"""
        self.metrics.interests_alignment = min(max(alignment_score, 0.0), 1.0)
        self._save_state()

    def set_manual_stage(self, stage: RelationshipStage, save: bool = True):
        """Manually set relationship stage"""
        old_stage = self.stage
        self.stage = stage
        self._manual_stage_override = stage

        logger.info(
            f"Relationship stage manually set: {old_stage.value} -> {stage.value}"
        )

        if save:
            self._save_state()

        if old_stage != stage and self.on_stage_change:
            asyncio.create_task(self.on_stage_change(old_stage, stage))

    def clear_manual_override(self):
        """Clear manual stage override, allowing natural evolution"""
        self._manual_stage_override = None
        self._save_state()

    # =========================================================================
    # BEHAVIOR QUERIES
    # =========================================================================

    def get_behavior(self) -> Dict[str, Any]:
        """Get current behavior configuration"""
        return RelationshipBehavior.get_behavior(self.stage)

    def get_proactivity_level(self) -> float:
        """Get proactivity level (0-1)"""
        return RelationshipBehavior.get_proactivity(self.stage)

    def should_confirm_action(self, action_type: str) -> bool:
        """Check if action should require confirmation"""
        return RelationshipBehavior.requires_confirmation(self.stage, action_type)

    def get_greeting(self) -> str:
        """Get appropriate greeting for current stage"""
        behavior = self.get_behavior()
        return behavior.get("greeting", "Hello!")

    def get_response_modifiers(self) -> Dict[str, Any]:
        """
        Get modifiers for response generation based on relationship stage.
        Used by other systems to adapt responses.
        """
        behavior = self.get_behavior()

        return {
            "formality": behavior["formality"],
            "proactivity": behavior["proactivity"],
            "personalization": behavior["personalization"],
            "response_style": behavior["response_style"],
            "should_initiate": behavior["initiation"] in ["often", "frequently"],
            "privacy_sensitivity": behavior["privacy_sensitive"],
        }

    # =========================================================================
    # STATUS AND DISPLAY
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get complete relationship status"""
        score = self._calculate_relationship_score()

        # Calculate progress to next stage
        next_stage = (
            RelationshipStage.get_next_stage(self.stage)
            if self.stage != RelationshipStage.SOULMATE
            else None
        )
        next_threshold = {
            RelationshipStage.STRANGER: self.config.stranger_to_acquaintance,
            RelationshipStage.ACQUAINTANCE: self.config.acquaintance_to_friend,
            RelationshipStage.FRIEND: self.config.friend_to_partner,
            RelationshipStage.PARTNER: self.config.partner_to_soulmate,
            RelationshipStage.SOULMATE: 1.0,
        }.get(self.stage, 1.0)

        progress = 0
        if next_stage:
            progress = (
                ((score - (next_threshold - 0.2)) / 0.2) * 100
                if next_threshold > 0.2
                else 100
            )
            progress = max(0, min(progress, 100))

        return {
            "stage": self.stage.value,
            "stage_display": self.stage.value.title(),
            "score": score,
            "progress_to_next": progress,
            "next_stage": next_stage.value if next_stage else None,
            "manual_override": self._manual_stage_override is not None,
            "metrics": self.metrics.to_dict(),
            "behavior": self.get_behavior(),
        }

    def format_status_for_display(self) -> str:
        """Format relationship status for CLI display"""
        status = self.get_status()

        stage_emoji = {
            "stranger": "[?]",
            "acquaintance": "[=]",
            "friend": "[*]",
            "partner": "[+]",
            "soulmate": "[<3]",
        }

        emoji = stage_emoji.get(status["stage"], "[?]")

        lines = [
            f"{emoji} Your Relationship with AURA",
            "=" * 40,
            f"Stage: {status['stage_display']}",
            f"Score: {status['score']:.1%}",
        ]

        if status["manual_override"]:
            lines.append("(Manually set)")
        elif status["next_stage"]:
            lines.append(
                f"Progress to {status['next_stage'].title()}: {status['progress_to_next']:.0f}%"
            )

        lines.append("")
        lines.append("Metrics:")
        lines.append(f"  • Interactions: {self.metrics.interaction_count}")
        lines.append(
            f"  • Conversation depth: {self.metrics.conversation_depth_score:.1%}"
        )
        lines.append(f"  • Trust level: {self.metrics.trust_level:.1%}")
        lines.append(f"  • Interests aligned: {self.metrics.interests_alignment:.1%}")
        lines.append(
            f"  • Help given: {self.metrics.help_requests_given}/{self.metrics.help_requests_received}"
        )

        if self.metrics.streak_days > 0:
            lines.append(f"  • Day streak: {self.metrics.streak_days}")

        lines.append("")
        lines.append("Current Behavior:")
        behavior = status["behavior"]
        for trait in behavior["traits"][:3]:
            lines.append(f"  • {trait}")

        return "\n".join(lines)

    def get_evolution_summary(self) -> str:
        """Get summary of what triggers evolution"""
        return """
Relationship Evolution Guide:
=============================

Your relationship with AURA evolves based on:

1. Interaction Frequency (20%)
   - More conversations = faster evolution
   - Regular use matters more than length

2. Conversation Depth (20%)
   - Longer, meaningful conversations
   - Sharing topics and interests

3. Trust Level (30%) - MOST IMPORTANT
   - Granting permissions shows trust
   - Each permission level adds trust
   - More permissions = faster trust growth

4. Shared Interests (15%)
   - AURA learns your interests
   - Aligned interests strengthen bond

5. Help Given (15%)
   - AURA helping you successfully
   - Building reliance and trust

Tips:
- Grant appropriate permissions to speed up trust
- Have meaningful conversations, not just commands
- Let AURA help you with tasks
- Use AURA regularly to maintain the bond
"""


# ==============================================================================
# RELATIONSHIP SYSTEM
# ==============================================================================


class RelationshipSystem:
    """
    Main relationship system that coordinates all relationship functionality.
    This is the interface for the rest of AURA.
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.relationship = RelationshipState(user_id)

        # Connect to interest learner
        self._interest_learner = None

    async def initialize(self):
        """Initialize the relationship system"""
        try:
            from src.agent.interest_learner import get_interest_detector

            self._interest_learner = get_interest_detector()
        except Exception as e:
            logger.warning(f"Could not connect to interest learner: {e}")

        # Set up stage change callback
        self.relationship.on_stage_change = self._on_stage_change
        logger.info("Relationship system initialized")

    async def _on_stage_change(
        self, old_stage: RelationshipStage, new_stage: RelationshipStage
    ):
        """Handle relationship stage change"""
        logger.info(f"Relationship evolved: {old_stage.value} -> {new_stage.value}")

        # Could trigger notifications, special responses, etc.

    def process_message(self, message: str) -> Optional[str]:
        """
        Process a user message for relationship tracking.
        Returns optional response if significant event detected.
        """
        topics = self._extract_topics(message)
        self.relationship.record_interaction(
            message=message,
            message_length=len(message),
            topics=topics,
            positive=True,
        )

        # Update interests alignment
        if self._interest_learner:
            try:
                interests = self._interest_learner.profile.get_top_interests(5)
                if interests:
                    # Simple alignment based on conversation topics
                    alignment = 0.5
                    for topic in topics:
                        for interest in interests:
                            if topic.lower() in interest.category.value:
                                alignment = min(alignment + 0.1, 1.0)
                    self.relationship.update_interests_alignment(alignment)
            except:
                pass

        return None

    def _extract_topics(self, message: str) -> List[str]:
        """Extract topics from message for relationship tracking"""
        topic_keywords = {
            "work": [
                "work",
                "job",
                "meeting",
                "project",
                "boss",
                "colleague",
                "office",
            ],
            "personal": ["family", "friend", "relationship", "life", "home"],
            "health": ["health", "exercise", "sleep", "tired", "sick", "doctor"],
            "hobbies": ["hobby", "game", "music", "movie", "book", "sport"],
            "tech": ["tech", "computer", "phone", "app", "code", "software"],
            "travel": ["travel", "trip", "vacation", "flight", "hotel"],
        }

        topics = []
        message_lower = message.lower()

        for topic, keywords in topic_keywords.items():
            if any(kw in message_lower for kw in keywords):
                topics.append(topic)

        return topics

    def record_permission_change(self, permission_level: int):
        """Record permission change for trust calculation"""
        self.relationship.record_permission_granted(permission_level)

    def record_help_outcome(self, successful: bool = True):
        """Record help outcome"""
        self.relationship.record_help_given(successful)

    def get_status(self) -> Dict[str, Any]:
        """Get relationship status"""
        return self.relationship.get_status()

    def format_status(self) -> str:
        """Format status for display"""
        return self.relationship.format_status_for_display()

    def set_stage(self, stage: RelationshipStage):
        """Manually set relationship stage"""
        self.relationship.set_manual_stage(stage)

    def get_behavior(self) -> Dict[str, Any]:
        """Get current behavior configuration"""
        return self.relationship.get_behavior()

    def should_confirm(self, action_type: str) -> bool:
        """Check if action requires confirmation"""
        return self.relationship.should_confirm_action(action_type)

    def get_proactivity(self) -> float:
        """Get current proactivity level"""
        return self.relationship.get_proactivity_level()

    def get_greeting(self) -> str:
        """Get appropriate greeting"""
        return self.relationship.get_greeting()

    def get_modifiers(self) -> Dict[str, Any]:
        """Get response modifiers"""
        return self.relationship.get_response_modifiers()

    def get_evolution_guide(self) -> str:
        """Get evolution guide"""
        return self.relationship.get_evolution_summary()


# ==============================================================================
# RELATIONSHIP COMMANDS
# ==============================================================================


class RelationshipCommands:
    """Commands for relationship management"""

    @staticmethod
    async def handle_relationship_command(
        args: List[str], relationship_system: RelationshipSystem
    ) -> str:
        """
        Handle /relationship command
        Usage: /relationship [set <stage>|status|guide]
        """
        if not args:
            return relationship_system.format_status()

        subcommand = args[0].lower()

        if subcommand == "status":
            return relationship_system.format_status()

        elif subcommand == "guide":
            return relationship_system.get_evolution_guide()

        elif subcommand == "set":
            if len(args) < 2:
                stages = RelationshipStage.get_all_stages()
                return f"Usage: /relationship set <{'-'.join(stages)}>"

            try:
                stage = RelationshipStage.from_string(args[1])
                relationship_system.set_stage(stage)
                return f"Relationship set to: {stage.value.title()}"
            except:
                return f"Invalid stage. Valid: {', '.join(RelationshipStage.get_all_stages())}"

        elif subcommand == "clear":
            relationship_system.relationship.clear_manual_override()
            return "Manual override cleared. Relationship will evolve naturally."

        else:
            return f"Unknown command: {subcommand}\nUsage: /relationship [status|set <stage>|guide|clear]"


# ==============================================================================
# FACTORY
# ==============================================================================

_relationship_system: Optional[RelationshipSystem] = None


def get_relationship_system(user_id: str = "default") -> RelationshipSystem:
    """Get or create relationship system"""
    global _relationship_system
    if _relationship_system is None:
        _relationship_system = RelationshipSystem(user_id)
    return _relationship_system


async def initialize_relationship_system():
    """Initialize the global relationship system"""
    global _relationship_system
    system = get_relationship_system()
    await system.initialize()
    return system


__all__ = [
    "RelationshipStage",
    "RelationshipMetrics",
    "RelationshipConfig",
    "RelationshipBehavior",
    "RelationshipState",
    "RelationshipSystem",
    "RelationshipCommands",
    "get_relationship_system",
    "initialize_relationship_system",
]
