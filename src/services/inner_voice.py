"""
AURA v3 - Inner Voice & UX System
Makes AURA feel like a living companion, not just a bot
Features:
- Inner voice stream (shows AURA's reasoning)
- Feelings & Trust Meter
- Character sheets (AURA + User)
- Thought bubbles
- Weekly recap rituals
- Transparent background process logging
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AuraFeeling(Enum):
    """AURA's emotional states"""

    CURIOUS = "curious"
    FOCUSED = "focused"
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    CONCERNED = "concerned"
    HAPPY = "happy"
    Worried = "worried"
    EXCITED = "excited"
    CALM = "calm"
    TIRED = "tired"
    FRUSTRATED = "frustrated"
    HOPEFUL = "hopeful"


class TrustLevel(Enum):
    """Trust level between AURA and user"""

    LEARNING = "learning"  # Just met
    GETTING_KNOW = "getting_to_know"  # Building rapport
    COMFORTABLE = "comfortable"  # Good understanding
    DEEP_BOND = "deep_bond"  # Strong connection
    PARTNER = "partner"  # Equal partnership


@dataclass
class ThoughtBubble:
    """A thought that AURA can choose to reveal or hide"""

    id: str
    content: str
    timestamp: datetime
    category: str  # "observation", "concern", "goal", "reflection"
    is_sensitive: bool = False  # "secret" thoughts
    is_revealed: bool = False
    user_response: Optional[str] = None  # user's reaction


@dataclass
class InnerThought:
    """A thought in AURA's inner monologue stream"""

    id: str
    content: str
    reasoning: str  # Why AURA thought this
    timestamp: datetime
    related_action: Optional[str] = None
    confidence: float = 0.5


@dataclass
class AuraFeelingState:
    """AURA's current feeling state"""

    primary: AuraFeeling = AuraFeeling.CALM
    secondary: AuraFeeling = AuraFeeling.CURIOUS
    intensity: float = 0.5  # 0-1 scale
    cause: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrustMeter:
    """Trust/understanding meter between AURA and user"""

    understanding_work: float = 0.0  # 0-10
    understanding_mood: float = 0.0
    understanding_goals: float = 0.0
    understanding_relationships: float = 0.0
    trust_level: TrustLevel = TrustLevel.LEARNING
    last_updated: datetime = field(default_factory=datetime.now)

    def get_overall(self) -> float:
        """Get overall understanding score"""
        return (
            self.understanding_work * 0.3
            + self.understanding_mood * 0.25
            + self.understanding_goals * 0.25
            + self.understanding_relationships * 0.2
        )

    def get_trust_description(self) -> str:
        """Get human-readable trust description"""
        overall = self.get_overall()
        if overall < 2:
            return "I'm still learning about you. Please be patient!"
        elif overall < 4:
            return "I'm getting to know you better each day."
        elif overall < 6:
            return "I feel like I'm starting to understand you."
        elif overall < 8:
            return "I have a good sense of who you are now."
        else:
            return "We understand each other really well now."


@dataclass
class CharacterSheet:
    """Character sheet - AURA's view of user and itself"""

    # User's attributes (as AURA perceives)
    user_traits: List[str] = field(default_factory=list)
    user_goals: List[Dict] = field(default_factory=list)
    user_focus_level: float = 0.5
    user_energy_level: float = 0.5
    user_risk_tolerance: float = 0.5
    user_social_load: float = 0.5

    # AURA's capabilities
    aura_capabilities: List[str] = field(default_factory=list)
    aura_currently_training_on: List[str] = field(default_factory=list)
    aura_next_upgrades: List[str] = field(default_factory=list)

    # Relationship
    days_known: int = 0
    total_interactions: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0


class InnerVoiceSystem:
    """
    AURA's inner voice - shows reasoning to users
    This is what makes AURA feel alive!
    """

    def __init__(self, neural_memory=None, user_profile=None):
        self.neural_memory = neural_memory
        self.user_profile = user_profile

        # Inner monologue stream
        self.thought_stream: List[InnerThought] = []
        self.max_thoughts = 50

        # Thought bubbles (optional reveals)
        self.thought_bubbles: List[ThoughtBubble] = []
        self.max_bubbles = 20

        # Current feeling state
        self.feeling_state = AuraFeelingState()

        # Trust meter
        self.trust_meter = TrustMeter()

        # Character sheets
        self.character_sheet = CharacterSheet()

        # Weekly recap data
        self.weekly_data: Dict[str, Any] = {}
        self.last_recap: Optional[datetime] = None

    async def initialize(self):
        """Initialize the inner voice system"""
        logger.info("Initializing Inner Voice System...")

        # Load persisted data
        await self._load_persisted_data()

        logger.info("Inner Voice System initialized")

    async def _load_persisted_data(self):
        """Load saved data from storage"""
        # This would load from neural memory
        pass

    # =========================================================================
    # INNER THOUGHTS
    # =========================================================================

    async def add_thought(
        self,
        content: str,
        reasoning: str,
        related_action: Optional[str] = None,
        confidence: float = 0.5,
    ) -> InnerThought:
        """Add a thought to the inner stream"""
        thought = InnerThought(
            id=f"thought_{datetime.now().timestamp()}",
            content=content,
            reasoning=reasoning,
            timestamp=datetime.now(),
            related_action=related_action,
            confidence=confidence,
        )

        self.thought_stream.append(thought)

        # Keep only last N thoughts
        if len(self.thought_stream) > self.max_thoughts:
            self.thought_stream = self.thought_stream[-self.max_thoughts :]

        logger.debug(f"AURA thought: {content[:50]}...")
        return thought

    def get_recent_thoughts(self, count: int = 10) -> List[InnerThought]:
        """Get recent thoughts for display"""
        return self.thought_stream[-count:]

    def format_thought_for_user(self, thought: InnerThought) -> str:
        """Format a thought in a human-readable way"""
        # Convert technical reasoning to friendly explanation
        return f"💭 {thought.content}"

    # =========================================================================
    # THOUGHT BUBBLES (Revealed Thoughts)
    # =========================================================================

    async def create_thought_bubble(
        self,
        content: str,
        category: str = "observation",
        is_sensitive: bool = False,
    ) -> ThoughtBubble:
        """Create a thought bubble AURA might reveal"""
        bubble = ThoughtBubble(
            id=f"bubble_{datetime.now().timestamp()}",
            content=content,
            timestamp=datetime.now(),
            category=category,
            is_sensitive=is_sensitive,
        )

        self.thought_bubbles.append(bubble)

        # Keep only recent bubbles
        if len(self.thought_bubbles) > self.max_bubbles:
            self.thought_bubbles = self.thought_bubbles[-self.max_bubbles :]

        return bubble

    def get_bubbles_to_reveal(self) -> List[ThoughtBubble]:
        """Get bubbles that should be revealed to user"""
        # Return non-sensitive bubbles or all if user enabled secrets
        return [
            b
            for b in self.thought_bubbles[-5:]
            if not b.is_sensitive and not b.is_revealed
        ]

    async def reveal_bubble(self, bubble_id: str) -> Optional[ThoughtBubble]:
        """Reveal a thought bubble to user"""
        for bubble in self.thought_bubbles:
            if bubble.id == bubble_id:
                bubble.is_revealed = True
                return bubble
        return None

    async def respond_to_bubble(
        self, bubble_id: str, response: str
    ) -> Optional[ThoughtBubble]:
        """User responds to a thought bubble"""
        for bubble in self.thought_bubbles:
            if bubble.id == bubble_id:
                bubble.user_response = response

                # Learn from response
                await self._learn_from_response(bubble, response)
                return bubble
        return None

    async def _learn_from_response(self, bubble: ThoughtBubble, response: str):
        """Learn from user's response to thought bubble"""
        # Adjust trust meter based on response
        response_lower = response.lower()

        if "wrong" in response_lower or "disagree" in response_lower:
            # AURA was wrong - decrease confidence
            self.trust_meter.understanding_work = max(
                0, self.trust_meter.understanding_work - 0.5
            )
            logger.info("Learned: I was wrong about something")
        elif "right" in response_lower or "agree" in response_lower:
            # AURA was correct - increase confidence
            self.trust_meter.understanding_work = min(
                10, self.trust_meter.understanding_work + 0.3
            )
            logger.info("Learned: I was right about something")

    # =========================================================================
    # FEELINGS
    # =========================================================================

    async def update_feeling(
        self,
        primary: AuraFeeling,
        secondary: Optional[AuraFeeling] = None,
        intensity: float = 0.5,
        cause: str = "",
    ):
        """Update AURA's current feeling"""
        self.feeling_state.primary = primary
        if secondary:
            self.feeling_state.secondary = secondary
        self.feeling_state.intensity = max(0, min(1, intensity))
        self.feeling_state.cause = cause
        self.feeling_state.timestamp = datetime.now()

        # Log the feeling
        logger.info(f"AURA feels: {primary.value} (intensity: {intensity})")

    def get_current_feeling(self) -> AuraFeelingState:
        """Get AURA's current feeling"""
        return self.feeling_state

    def format_feeling_for_user(self) -> str:
        """Format current feeling as a message"""
        feeling = self.feeling_state
        intensity_bar = "█" * int(feeling.intensity * 5) + "░" * (
            5 - int(feeling.intensity * 5)
        )

        msg = f"**Currently feeling:** {feeling.primary.value.title()}\n"
        msg += f"Intensity: [{intensity_bar}] {int(feeling.intensity * 100)}%\n"

        if feeling.cause:
            msg += f"**Why:** {feeling.cause}\n"

        return msg

    # =========================================================================
    # TRUST METER
    # =========================================================================

    async def update_trust_from_interaction(
        self,
        interaction_type: str,
        success: bool,
    ):
        """Update trust meter based on interaction"""
        now = datetime.now()

        # Update based on interaction type
        if interaction_type == "task_completion":
            if success:
                self.trust_meter.understanding_work = min(
                    10, self.trust_meter.understanding_work + 0.2
                )
                self.character_sheet.successful_tasks += 1
            else:
                self.trust_meter.understanding_work = max(
                    0, self.trust_meter.understanding_work - 0.1
                )
                self.character_sheet.failed_tasks += 1

        elif interaction_type == "mood_understanding":
            if success:
                self.trust_meter.understanding_mood = min(
                    10, self.trust_meter.understanding_mood + 0.3
                )

        elif interaction_type == "goal_alignment":
            if success:
                self.trust_meter.understanding_goals = min(
                    10, self.trust_meter.understanding_goals + 0.2
                )

        # Update trust level based on overall
        overall = self.trust_meter.get_overall()
        if overall < 2:
            self.trust_meter.trust_level = TrustLevel.LEARNING
        elif overall < 4:
            self.trust_meter.trust_level = TrustLevel.GETTING_KNOW
        elif overall < 6:
            self.trust_meter.trust_level = TrustLevel.COMFORTABLE
        elif overall < 8:
            self.trust_meter.trust_level = TrustLevel.DEEP_BOND
        else:
            self.trust_meter.trust_level = TrustLevel.PARTNER

        self.trust_meter.last_updated = now
        self.character_sheet.total_interactions += 1

    def get_trust_meter(self) -> TrustMeter:
        """Get current trust meter"""
        return self.trust_meter

    def format_trust_for_user(self) -> str:
        """Format trust meter as a message"""
        tm = self.trust_meter

        msg = "**🤝 Our Connection:**\n\n"
        msg += f"**Trust Level:** {tm.trust_level.value.replace('_', ' ').title()}\n\n"
        msg += f"**How well I understand you:**\n"
        msg += f"  • Your work style: {tm.understanding_work:.1f}/10\n"
        msg += f"  • Your mood patterns: {tm.understanding_mood:.1f}/10\n"
        msg += f"  • Your goals: {tm.understanding_goals:.1f}/10\n"
        msg += f"  • Your relationships: {tm.understanding_relationships:.1f}/10\n\n"
        msg += (
            f"**Overall:** {tm.get_overall():.1f}/10 - {tm.get_trust_description()}\n"
        )

        return msg

    # =========================================================================
    # CHARACTER SHEETS
    # =========================================================================

    async def update_user_traits(self, traits: List[str]):
        """Update perceived user traits"""
        self.character_sheet.user_traits = traits

    async def add_user_goal(self, goal: str, progress: float = 0.0):
        """Add a user goal"""
        self.character_sheet.user_goals.append(
            {
                "goal": goal,
                "progress": progress,
                "added_at": datetime.now().isoformat(),
            }
        )

    def get_character_sheet(self) -> CharacterSheet:
        """Get character sheet"""
        return self.character_sheet

    def format_character_sheet(self) -> str:
        """Format character sheet for display"""
        cs = self.character_sheet

        msg = "=" * 40 + "\n"
        msg += "👤 YOUR SHEET (How AURA sees you)\n"
        msg += "=" * 40 + "\n\n"

        msg += "**Traits I've noticed:**\n"
        if cs.user_traits:
            msg += "  • " + "\n  • ".join(cs.user_traits[:5]) + "\n"
        else:
            msg += "  Still learning...\n"

        msg += "\n**Your goals:**\n"
        if cs.user_goals:
            for goal in cs.user_goals[:3]:
                progress_bar = "█" * int(goal["progress"] * 5) + "░" * (
                    5 - int(goal["progress"] * 5)
                )
                msg += f"  • {goal['goal']} [{progress_bar}]\n"
        else:
            msg += "  No goals set yet.\n"

        msg += f"\n**Energy level:** {cs.user_energy_level:.0%}\n"
        msg += f"**Focus:** {cs.user_focus_level:.0%}\n"

        msg += "\n" + "=" * 40 + "\n"
        msg += "🤖 AURA'S SHEET\n"
        msg += "=" * 40 + "\n\n"

        msg += f"**Days known:** {cs.days_known}\n"
        msg += f"**Total interactions:** {cs.total_interactions}\n"
        msg += f"**Successful tasks:** {cs.successful_tasks}\n"
        msg += f"**Failed tasks:** {cs.failed_tasks}\n"

        msg += "\n**What I'm training on:**\n"
        if cs.aura_currently_training_on:
            msg += "  • " + "\n  • ".join(cs.aura_currently_training_on[:3]) + "\n"
        else:
            msg += "  Building foundations...\n"

        return msg

    # =========================================================================
    # WEEKLY RECAP
    # =========================================================================

    async def record_weekly_data(self, data: Dict[str, Any]):
        """Record data for weekly recap"""
        self.weekly_data = data

    async def should_generate_weekly_recap(self) -> bool:
        """Check if it's time for weekly recap"""
        if not self.last_recap:
            return True

        days_since = (datetime.now() - self.last_recap).days
        return days_since >= 7

    async def generate_weekly_recap(self) -> str:
        """Generate weekly recap message"""
        self.last_recap = datetime.now()

        data = self.weekly_data
        cs = self.character_sheet

        msg = "📅 **WEEKLY RECAP** 📅\n\n"
        msg += "Here's what we accomplished together this week:\n\n"

        # Top achievements
        msg += "**🏆 Top Achievements:**\n"
        if data.get("achievements"):
            for a in data["achievements"][:3]:
                msg += f"  • {a}\n"
        else:
            msg += "  • Working on goals\n"

        # Patterns noticed
        msg += "\n**🔍 Patterns I Noticed:**\n"
        if data.get("patterns"):
            for p in data["patterns"][:2]:
                msg += f"  • {p}\n"
        else:
            msg += "  • Still learning your patterns\n"

        # AURA's reflections
        msg += "\n**💭 My Thoughts:**\n"
        msg += f"  • We're {cs.days_known} days into knowing each other\n"
        msg += f"  • You've completed {cs.successful_tasks} tasks with me\n"

        # Trust update
        msg += f"\n**🤝 Our Connection:** {self.trust_meter.get_trust_description()}\n"

        # Next week goals
        msg += "\n**🎯 What I want to help with next week:**\n"
        if data.get("next_week_goals"):
            for g in data["next_week_goals"][:2]:
                msg += f"  • {g}\n"
        else:
            msg += "  • Continue supporting your goals\n"

        msg += "\n_Here's to another great week!_ 🌟\n"

        return msg

    # =========================================================================
    # PROACTIVE THOUGHTS
    # =========================================================================

    async def generate_proactive_thought(self) -> Optional[str]:
        """Generate a proactive thought bubble"""
        # This could be triggered by:
        # - Time of day (morning greeting, night reminder)
        # - Pattern detection (user seems stressed)
        # - Goal tracking (deadline approaching)

        import random

        thought_templates = [
            ("observation", False, "I noticed you opened {app} {count} times today"),
            ("concern", False, "You seem a bit tired lately. Want to take a break?"),
            ("goal", False, "Your deadline for {goal} is in {days} days"),
            ("reflection", False, "We've been working together for {days} days now"),
            (
                "observation",
                True,
                "I have a hunch you're feeling {emotion} about {topic}",
            ),
        ]

        # Select a thought based on current state
        if random.random() < 0.3:  # 30% chance of proactive thought
            category, is_sensitive, template = random.choice(thought_templates)
            content = template.format(
                app="VS Code",
                count=5,
                goal="project launch",
                days=3,
                emotion="excited",
                topic="the new feature",
            )

            bubble = await self.create_thought_bubble(content, category, is_sensitive)
            return bubble.content

        return None

    # =========================================================================
    # TRANSPARENT LOGS INTEGRATION
    # =========================================================================

    async def get_transparent_logs(self, limit: int = 10) -> Dict[str, Any]:
        """Get transparent logs for dashboard display"""
        try:
            from src.core.transparent_logger import get_transparent_logger

            logger = get_transparent_logger()
            logs = logger.get_logs(limit=limit)
            status = logger.get_status()

            return {
                "logs": [
                    {
                        "level": log.level.value,
                        "content": log.display_content or log.content,
                        "category": log.category,
                        "status": log.status,
                        "timestamp": log.timestamp.isoformat(),
                    }
                    for log in logs
                ],
                "processing": {
                    "is_processing": status.is_processing,
                    "phase": status.current_phase,
                    "message": status.get_display_message(),
                    "data_access": status.data_access,
                },
            }
        except ImportError:
            return {"logs": [], "processing": {"is_processing": False}}


# Global instance
_inner_voice: Optional[InnerVoiceSystem] = None


async def get_inner_voice() -> InnerVoiceSystem:
    """Get or create inner voice system"""
    global _inner_voice
    if _inner_voice is None:
        _inner_voice = InnerVoiceSystem()
        await _inner_voice.initialize()
    return _inner_voice
