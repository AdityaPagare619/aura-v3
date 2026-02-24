"""
AURA v3 Social Insights
Generates insights about social life
Uses pattern recognition and relationship data
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import statistics
from collections import Counter

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Types of social insights"""

    RELATIONSHIP = "relationship"
    COMMUNICATION = "communication"
    TEMPORAL = "temporal"
    EMOTIONAL = "emotional"
    OPPORTUNITY = "opportunity"
    WARNING = "warning"
    TREND = "trend"
    RECOMMENDATION = "recommendation"


@dataclass
class Insight:
    """A generated social insight"""

    id: str
    insight_type: InsightType
    title: str
    description: str

    confidence: float = 0.0
    priority: str = "medium"

    related_contacts: List[str] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)

    generated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    actionable: bool = True
    action_suggestion: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)


class SocialInsights:
    """
    Generates insights about user's social life
    Analyzes patterns, relationships, and behavior
    """

    def __init__(self, data_dir: str = "data/social_life/insights"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._insights: List[Insight] = []
        self._insight_history: List[Insight] = []

        self._max_active_insights = 50
        self._min_confidence_threshold = 0.5

    async def initialize(self):
        """Initialize social insights"""
        logger.info("Initializing Social Insights...")
        await self._load_insights()
        logger.info("Social Insights initialized")

    async def _load_insights(self):
        """Load insights from disk"""
        insights_file = self.data_dir / "insights.json"
        if insights_file.exists():
            try:
                with open(insights_file, "r") as f:
                    data = json.load(f)
                    for i_data in data.get("insights", []):
                        i_data["generated_at"] = datetime.fromisoformat(
                            i_data["generated_at"]
                        )
                        if i_data.get("expires_at"):
                            i_data["expires_at"] = datetime.fromisoformat(
                                i_data["expires_at"]
                            )
                        self._insights.append(Insight(**i_data))
                logger.info(f"Loaded {len(self._insights)} insights")
            except Exception as e:
                logger.error(f"Error loading insights: {e}")

    async def _save_insights(self):
        """Save insights to disk"""
        insights_file = self.data_dir / "insights.json"
        try:
            data = {
                "insights": [
                    {
                        **vars(i),
                        "generated_at": i.generated_at.isoformat(),
                        "expires_at": i.expires_at.isoformat()
                        if i.expires_at
                        else None,
                    }
                    for i in self._insights
                ]
            }
            with open(insights_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving insights: {e}")

    async def generate_insights(
        self,
        relationships: Dict[str, Any] = None,
        patterns: Dict[str, Any] = None,
        events: Dict[str, Any] = None,
    ) -> List[Insight]:
        """Generate insights from available data"""
        new_insights = []

        if relationships:
            new_insights.extend(await self._analyze_relationships(relationships))

        if patterns:
            new_insights.extend(await self._analyze_patterns(patterns))

        if events:
            new_insights.extend(await self._analyze_events(events))

        new_insights.extend(await self._analyze_communication_balance())
        new_insights.extend(await self._analyze_social_trends())

        for insight in new_insights:
            if insight.id not in [i.id for i in self._insights]:
                self._insights.append(insight)

        self._expire_old_insights()

        if len(self._insights) > self._max_active_insights:
            self._insights = sorted(
                self._insights, key=lambda x: x.confidence, reverse=True
            )[: self._max_active_insights]

        await self._save_insights()

        return new_insights

    async def _analyze_relationships(
        self, relationships: Dict[str, Any]
    ) -> List[Insight]:
        """Analyze relationships for insights"""
        insights = []

        total = relationships.get("total_contacts", 0)
        health_dist = relationships.get("health_distribution", {})

        if health_dist.get("at_risk", 0) > total * 0.3:
            insights.append(
                Insight(
                    id=f"insight_relationships_at_risk_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    insight_type=InsightType.WARNING,
                    title="Multiple Relationships at Risk",
                    description=f"{health_dist.get('at_risk', 0)} of your {total} relationships are at risk of fading",
                    confidence=0.8,
                    priority="high",
                    action_suggestion="Consider reaching out to neglected contacts",
                )
            )

        category_dist = relationships.get("category_distribution", {})
        if category_dist.get("work", 0) > total * 0.5:
            insights.append(
                Insight(
                    id=f"insight_work_heavy_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    insight_type=InsightType.TREND,
                    title="Work-Heavy Social Circle",
                    description="Over half your tracked contacts are work-related",
                    confidence=0.7,
                    priority="low",
                    action_suggestion="Consider nurturing non-work relationships for better work-life balance",
                )
            )

        important = relationships.get("important_contacts", [])
        for contact in important[:3]:
            if contact.get("importance", 0) > 0.8:
                insights.append(
                    Insight(
                        id=f"insight_important_{contact['name'].lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        insight_type=InsightType.RECOMMENDATION,
                        title=f"Nurture {contact['name']}",
                        description=f"{contact['name']} is an important contact",
                        confidence=0.9,
                        priority="medium",
                        related_contacts=[contact["name"]],
                        action_suggestion=f"Schedule quality time with {contact['name']}",
                    )
                )

        return insights

    async def _analyze_patterns(self, patterns: Dict[str, Any]) -> List[Insight]:
        """Analyze patterns for insights"""
        insights = []

        temporal = patterns.get("temporal", [])
        communication = patterns.get("communication", [])

        if any("night" in p.get("name", "").lower() for p in temporal):
            insights.append(
                Insight(
                    id=f"insight_night_owl_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    insight_type=InsightType.TREND,
                    title="Night Owl Pattern",
                    description="You tend to connect with people late at night",
                    confidence=0.7,
                    priority="low",
                    action_suggestion="Your peak social hours are different from typical daytime schedules",
                )
            )

        if any(p.get("name", "").lower().find("quick") >= 0 for p in communication):
            insights.append(
                Insight(
                    id=f"insight_quick_response_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    insight_type=InsightType.COMMUNICATION,
                    title="Quick Responder",
                    description="You generally respond quickly to messages",
                    confidence=0.8,
                    priority="low",
                )
            )

        return insights

    async def _analyze_events(self, events: Dict[str, Any]) -> List[Insight]:
        """Analyze events for insights"""
        insights = []

        upcoming = events.get("upcoming_events", [])
        if upcoming:
            insights.append(
                Insight(
                    id=f"insight_upcoming_events_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    insight_type=InsightType.OPPORTUNITY,
                    title=f"{len(upcoming)} Upcoming Social Events",
                    description=f"You have {len(upcoming)} events coming up",
                    confidence=0.9,
                    priority="medium",
                    action_suggestion="Review and prepare for upcoming social commitments",
                )
            )

        return insights

    async def _analyze_communication_balance(self) -> List[Insight]:
        """Analyze communication balance"""
        insights = []

        insights.append(
            Insight(
                id=f"insight_balance_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                insight_type=InsightType.EMOTIONAL,
                title="Social Balance Check",
                description="Regular social interaction is important for well-being",
                confidence=0.6,
                priority="low",
                action_suggestion="Consider your social balance across different relationship types",
            )
        )

        return insights

    async def _analyze_social_trends(self) -> List[Insight]:
        """Analyze social trends"""
        insights = []

        today = datetime.now()
        weekday = today.weekday()

        if weekday == 4:
            insights.append(
                Insight(
                    id=f"insight_friday_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    insight_type=InsightType.OPPORTUNITY,
                    title="Friday Opportunity",
                    description="It's Friday - a great time for social activities",
                    confidence=0.8,
                    priority="low",
                    action_suggestion="Consider planning a weekend social activity",
                )
            )

        return insights

    def _expire_old_insights(self):
        """Expire old insights"""
        now = datetime.now()
        expired = [i for i in self._insights if i.expires_at and i.expires_at < now]
        self._insight_history.extend(expired)
        self._insights = [
            i for i in self._insights if i.expires_at is None or i.expires_at >= now
        ]

    async def get_latest_insights(self, limit: int = 10) -> List[Dict]:
        """Get latest insights"""
        sorted_insights = sorted(
            self._insights, key=lambda x: x.generated_at, reverse=True
        )[:limit]

        return [
            {
                "id": i.id,
                "type": i.insight_type.value,
                "title": i.title,
                "description": i.description,
                "confidence": i.confidence,
                "priority": i.priority,
                "actionable": i.actionable,
                "action_suggestion": i.action_suggestion,
                "generated_at": i.generated_at.isoformat(),
            }
            for i in sorted_insights
        ]

    async def get_insights_by_type(self, insight_type: InsightType) -> List[Insight]:
        """Get insights by type"""
        return [i for i in self._insights if i.insight_type == insight_type]

    async def get_high_priority_insights(self) -> List[Insight]:
        """Get high priority insights"""
        return [
            i
            for i in self._insights
            if i.priority == "high" and i.confidence >= self._min_confidence_threshold
        ]

    async def dismiss_insight(self, insight_id: str):
        """Dismiss an insight"""
        for i in self._insights:
            if i.id == insight_id:
                self._insights.remove(i)
                self._insight_history.append(i)
                break
        await self._save_insights()

    async def get_insight_summary(self) -> Dict[str, Any]:
        """Get summary of insights"""
        type_dist = {}
        priority_dist = {}

        for insight in self._insights:
            type_dist[insight.insight_type.value] = (
                type_dist.get(insight.insight_type.value, 0) + 1
            )
            priority_dist[insight.priority] = priority_dist.get(insight.priority, 0) + 1

        return {
            "total_insights": len(self._insights),
            "type_distribution": type_dist,
            "priority_distribution": priority_dist,
            "actionable_count": sum(1 for i in self._insights if i.actionable),
            "high_confidence_count": sum(
                1 for i in self._insights if i.confidence >= 0.7
            ),
        }


_social_insights: Optional[SocialInsights] = None


def get_social_insights() -> SocialInsights:
    """Get or create social insights"""
    global _social_insights
    if _social_insights is None:
        _social_insights = SocialInsights()
    return _social_insights
