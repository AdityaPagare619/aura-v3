"""
AURA v3 Social-Life Manager Subsystem
=====================================
A sub-agent that works autonomously within AURA v3

Capabilities:
- Social app analysis (WhatsApp, Instagram, etc.)
- Pattern recognition in social behavior
- Relationship insights and management
- Event/contact tracking
- 100% offline - all data stored locally

Architecture:
- Integrates with learning/ engine for adaptive behavior
- Uses memory/ system for social history storage
- Works offline on Android/Termux (4GB RAM constraint)
- Production-ready with proper error handling
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

from src.agents.social_life.social_app_analyzer import (
    SocialAppAnalyzer,
    AppDataSource,
    MessagePattern,
    ResponseMetrics,
    get_social_app_analyzer,
)

from src.agents.social_life.pattern_recognizer import (
    PatternRecognizer,
    SocialPattern,
    PatternType,
    get_pattern_recognizer,
)

from src.agents.social_life.relationship_tracker import (
    RelationshipTracker,
    Contact,
    Relationship,
    RelationshipStrength,
    Interaction,
    get_relationship_tracker,
)

from src.agents.social_life.social_insights import (
    SocialInsights,
    Insight,
    InsightType,
    get_social_insights,
)

from src.agents.social_life.event_manager import (
    EventManager,
    SocialEvent,
    EventType,
    Reminder,
    get_event_manager,
)

from src.agents.social_life.personality import (
    SocialPersonality,
    SocialPreferences,
    SocialMood,
    get_social_personality,
)


class SocialLifeAgent:
    """
    Main orchestrator for Social-Life Manager
    Coordinates all subsystems for complete social life management
    """

    def __init__(self):
        self._running = False
        self._initialized = False

        self.app_analyzer: Optional[SocialAppAnalyzer] = None
        self.pattern_recognizer: Optional[PatternRecognizer] = None
        self.relationship_tracker: Optional[RelationshipTracker] = None
        self.social_insights: Optional[SocialInsights] = None
        self.event_manager: Optional[EventManager] = None
        self.personality: Optional[SocialPersonality] = None

        self._analysis_interval = 3600
        self._last_analysis: Optional[datetime] = None

    async def initialize(self):
        """Initialize all social life subsystems"""
        if self._initialized:
            logger.warning("SocialLifeAgent already initialized")
            return

        logger.info("Initializing Social-Life Manager...")

        try:
            self.app_analyzer = get_social_app_analyzer()
            self.pattern_recognizer = get_pattern_recognizer()
            self.relationship_tracker = get_relationship_tracker()
            self.social_insights = get_social_insights()
            self.event_manager = get_event_manager()
            self.personality = get_social_personality()

            await self.app_analyzer.initialize()
            await self.pattern_recognizer.initialize()
            await self.relationship_tracker.initialize()
            await self.social_insights.initialize()
            await self.event_manager.initialize()
            await self.personality.initialize()

            self._initialized = True
            logger.info("Social-Life Manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Social-Life Manager: {e}")
            raise

    async def start(self):
        """Start the social life agent"""
        if not self._initialized:
            await self.initialize()

        self._running = True
        logger.info("Social-Life Manager started")

    async def stop(self):
        """Stop the social life agent"""
        self._running = False
        logger.info("Social-Life Manager stopped")

    async def analyze_social_data(self) -> Dict[str, Any]:
        """Run full social data analysis"""
        if not self._running:
            return {"error": "Agent not running"}

        logger.info("Running social data analysis...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "app_analysis": {},
            "patterns": {},
            "relationships": {},
            "insights": {},
            "upcoming_events": {},
        }

        try:
            if self.app_analyzer:
                results["app_analysis"] = await self.app_analyzer.analyze_all_apps()

            if self.pattern_recognizer:
                results["patterns"] = await self.pattern_recognizer.detect_patterns()

            if self.relationship_tracker:
                results[
                    "relationships"
                ] = await self.relationship_tracker.get_relationship_summary()

            if self.social_insights:
                results["insights"] = await self.social_insights.generate_insights()

            if self.event_manager:
                results[
                    "upcoming_events"
                ] = await self.event_manager.get_upcoming_events()

            self._last_analysis = datetime.now()
            logger.info("Social data analysis completed")

        except Exception as e:
            logger.error(f"Error in social data analysis: {e}")
            results["error"] = str(e)

        return results

    async def process_social_interaction(
        self, contact_name: str, message: str, platform: str
    ) -> Dict[str, Any]:
        """Process a new social interaction"""
        if not self._running:
            return {"error": "Agent not running"}

        results = {
            "contact": contact_name,
            "platform": platform,
            "processed_at": datetime.now().isoformat(),
        }

        try:
            if self.app_analyzer:
                pattern = await self.app_analyzer.analyze_message_pattern(
                    contact_name, message, platform
                )
                results["pattern"] = pattern

            if self.relationship_tracker:
                await self.relationship_tracker.record_interaction(
                    contact=contact_name,
                    message=message,
                    platform=platform,
                )
                relationship = await self.relationship_tracker.get_relationship(
                    contact_name
                )
                results["relationship"] = relationship

            if self.pattern_recognizer:
                detected = await self.pattern_recognizer.check_pattern_triggers(
                    contact_name, message
                )
                results["pattern_triggers"] = detected

            if self.event_manager:
                reminders = await self.event_manager.check_reminders(contact_name)
                results["reminders"] = reminders

            if self.personality:
                response_style = await self.personality.get_response_style(contact_name)
                results["response_style"] = response_style

        except Exception as e:
            logger.error(f"Error processing social interaction: {e}")
            results["error"] = str(e)

        return results

    async def get_social_summary(self) -> Dict[str, Any]:
        """Get comprehensive social life summary"""
        if not self._running:
            return {"error": "Agent not running"}

        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_contacts": 0,
            "important_contacts": [],
            "recent_interactions": [],
            "upcoming_events": [],
            "insights": [],
            "social_mood": None,
        }

        try:
            if self.relationship_tracker:
                rels = await self.relationship_tracker.get_all_relationships()
                summary["total_contacts"] = len(rels)
                summary["important_contacts"] = [
                    {"name": r.name, "strength": r.strength.value}
                    for r in sorted(rels, key=lambda x: x.strength.value, reverse=True)[
                        :5
                    ]
                ]
                summary[
                    "recent_interactions"
                ] = await self.relationship_tracker.get_recent_interactions(limit=10)

            if self.event_manager:
                summary[
                    "upcoming_events"
                ] = await self.event_manager.get_upcoming_events(limit=5)

            if self.social_insights:
                summary["insights"] = await self.social_insights.get_latest_insights(
                    limit=3
                )

            if self.personality:
                summary["social_mood"] = await self.personality.get_current_mood()

        except Exception as e:
            logger.error(f"Error generating social summary: {e}")
            summary["error"] = str(e)

        return summary

    async def suggest_reconnections(self) -> List[Dict[str, Any]]:
        """Suggest contacts to reconnect with"""
        if not self._running or not self.relationship_tracker:
            return []

        return await self.relationship_tracker.suggest_reconnections()

    async def get_event_reminders(self) -> List[Dict[str, Any]]:
        """Get upcoming event reminders"""
        if not self._running or not self.event_manager:
            return []

        return await self.event_manager.get_due_reminders()


_social_life_agent: Optional[SocialLifeAgent] = None


def get_social_life_agent() -> SocialLifeAgent:
    """Get or create social life agent"""
    global _social_life_agent
    if _social_life_agent is None:
        _social_life_agent = SocialLifeAgent()
    return _social_life_agent


__all__ = [
    "SocialLifeAgent",
    "get_social_life_agent",
    "SocialAppAnalyzer",
    "PatternRecognizer",
    "RelationshipTracker",
    "SocialInsights",
    "EventManager",
    "SocialPersonality",
    "AppDataSource",
    "MessagePattern",
    "ResponseMetrics",
    "SocialPattern",
    "PatternType",
    "Contact",
    "Relationship",
    "RelationshipStrength",
    "Interaction",
    "Insight",
    "InsightType",
    "SocialEvent",
    "EventType",
    "Reminder",
    "SocialPreferences",
    "SocialMood",
]
