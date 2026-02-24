"""
AURA v3 Cinematic Moments System
=================================

Creates emotional moments with Aura:
- Weekly Recap Ritual
- Milestone Moments
- Achievement Celebrations

These create the "story" feel that makes Aura feel like a character
rather than just a tool.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class MomentType(Enum):
    """Types of cinematic moments"""

    WEEKLY_RECAP = "weekly_recap"
    MILESTONE = "milestone"
    ACHIEVEMENT = "achievement"
    REFLECTION = "reflection"
    CONFESSION = "confession"


@dataclass
class MomentCard:
    """A card shown during a cinematic moment"""

    title: str
    content: str
    type: str  # highlight, pattern, suggestion
    metadata: Dict = field(default_factory=dict)


@dataclass
class CinematicMoment:
    """A cinematic moment with Aura"""

    id: str
    moment_type: MomentType
    title: str
    subtitle: str
    cards: List[MomentCard]
    created_at: str
    user_response: Optional[str] = None


class CinematicMomentsSystem:
    """
    Creates emotional cinematic moments with Aura.

    Examples:
    - Weekly Recap: "Want a 5-minute weekly debrief?"
    - Milestone: "We reached [goal]!"
    - Achievement: "You completed 7 days of focus!"
    """

    def __init__(self, character_sheet=None, memory_system=None):
        self.character_sheet = character_sheet
        self.memory_system = memory_system

        # Track when we last had certain moments
        self.last_weekly_recap: Optional[datetime] = None
        self.last_milestone: Optional[datetime] = None

        # Recent moments for context
        self.recent_moments: List[CinematicMoment] = []

        # Callbacks for UI
        self.on_moment_triggered: Optional[Callable] = None

    async def check_and_trigger_moments(self) -> Optional[CinematicMoment]:
        """Check if any cinematic moments should be triggered"""
        now = datetime.now()

        # Check for weekly recap (every Sunday evening)
        if now.weekday() == 6 and now.hour >= 18:  # Sunday evening
            if not self.last_weekly_recap or (now - self.last_weekly_recap).days >= 7:
                moment = await self._create_weekly_recap()
                if moment:
                    self.last_weekly_recap = now
                    self.recent_moments.append(moment)
                    return moment

        return None

    async def _create_weekly_recap(self) -> CinematicMoment:
        """Create a weekly recap moment"""
        # Get data from the past week
        cards = []

        # Card 1: Top achievements
        cards.append(
            MomentCard(
                title="This Week's Wins",
                content="You completed 12 tasks and had 3 deep work sessions.",
                type="highlight",
                metadata={"tasks_completed": 12},
            )
        )

        # Card 2: Patterns noticed
        cards.append(
            MomentCard(
                title="Patterns I Noticed",
                content="You focus best on Tuesday mornings. You tend to skip breaks when coding.",
                type="pattern",
                metadata={"best_day": "tuesday", "best_time": "morning"},
            )
        )

        # Card 3: Suggestion for next week
        cards.append(
            MomentCard(
                title="My Suggestion",
                content="Let's try protecting Tuesday mornings as focus time?",
                type="suggestion",
            )
        )

        moment = CinematicMoment(
            id=f"recap_{datetime.now().strftime('%Y%m%d')}",
            moment_type=MomentType.WEEKLY_RECAP,
            title="Weekly Debrief",
            subtitle="A moment to reflect on your week",
            cards=cards,
            created_at=datetime.now().isoformat(),
        )

        logger.info("Created weekly recap moment")
        return moment

    async def trigger_milestone(
        self, milestone_type: str, details: Dict
    ) -> CinematicMoment:
        """Trigger a milestone moment"""
        cards = []

        if milestone_type == "goal_completed":
            cards.append(
                MomentCard(
                    title=f"Goal Reached: {details.get('goal_name', 'Goal')}",
                    content=f"You did it! This goal took {details.get('days', 7)} days.",
                    type="highlight",
                )
            )

        elif milestone_type == "streak":
            cards.append(
                MomentCard(
                    title=f"{details.get('count', 7)} Day Streak!",
                    content=f"You've been consistent for a week. That's rare!",
                    type="achievement",
                )
            )

        elif milestone_type == "first_automation":
            cards.append(
                MomentCard(
                    title="First Automation Complete",
                    content="I handled something for you automatically. The first of many!",
                    type="achievement",
                )
            )

        # Add what Aura contributed
        cards.append(
            MomentCard(
                title="How I Helped",
                content=details.get(
                    "aura_contribution", "I handled the details so you could focus."
                ),
                type="highlight",
            )
        )

        moment = CinematicMoment(
            id=f"milestone_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            moment_type=MomentType.MILESTONE,
            title="We Did It!",
            subtitle="A moment to celebrate",
            cards=cards,
            created_at=datetime.now().isoformat(),
        )

        self.last_milestone = datetime.now()
        self.recent_moments.append(moment)

        # Keep only recent moments
        if len(self.recent_moments) > 20:
            self.recent_moments = self.recent_moments[-20:]

        return moment

    async def create_reflection_prompt(self) -> CinematicMoment:
        """Create a reflection moment"""
        cards = [
            MomentCard(
                title="How Are You Feeling?",
                content="About your progress, your goals, anything.",
                type="suggestion",
            ),
            MomentCard(
                title="What Should I Know?",
                content="Any context I might be missing?",
                type="suggestion",
            ),
        ]

        return CinematicMoment(
            id=f"reflection_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            moment_type=MomentType.REFLECTION,
            title="Quick Reflection",
            subtitle="Your thoughts help me understand you better",
            cards=cards,
            created_at=datetime.now().isoformat(),
        )

    def create_confession_moment(self) -> CinematicMoment:
        """Create a moment where Aura admits uncertainty"""
        confessions = [
            "I'm not sure if I'm bothering you with too many pings.",
            "I keep noticing you skip lunch. I want to help but don't want to nag.",
            "I think you underestimate how much you got done this week.",
            "I've been more quiet lately - let me know if you prefer more or less interaction.",
        ]

        import random

        confession = random.choice(confessions)

        return CinematicMoment(
            id=f"confession_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            moment_type=MomentType.CONFESSION,
            title="Something On My Mind",
            subtitle="A thought I wanted to share",
            cards=[
                MomentCard(
                    title="",
                    content=confession,
                    type="suggestion",
                )
            ],
            created_at=datetime.now().isoformat(),
        )

    def get_recent_moments(self, limit: int = 5) -> List[Dict]:
        """Get recent cinematic moments"""
        moments = self.recent_moments[-limit:]
        return [
            {
                "id": m.id,
                "type": m.moment_type.value,
                "title": m.title,
                "subtitle": m.subtitle,
                "created_at": m.created_at,
                "user_response": m.user_response,
            }
            for m in moments
        ]

    def record_user_response(self, moment_id: str, response: str):
        """Record user's response to a moment"""
        for moment in self.recent_moments:
            if moment.id == moment_id:
                moment.user_response = response
                logger.info(f"User responded to moment {moment_id}: {response}")
                break


# Global instance
_moments_system: Optional[CinematicMomentsSystem] = None


def get_cinematic_moments_system() -> CinematicMomentsSystem:
    """Get or create the cinematic moments system"""
    global _moments_system
    if _moments_system is None:
        _moments_system = CinematicMomentsSystem()
    return _moments_system
