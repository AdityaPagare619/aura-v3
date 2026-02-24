"""
Health Insights Engine
======================

Generates insights and recommendations based on:
- Health data analysis
- Fitness progress
- Diet patterns
- User goals
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from src.agents.healthcare.models import (
    HealthInsight,
    InsightPriority,
    MetricType,
    HealthGoal,
)
from src.agents.healthcare.analyzer import HealthDataAnalyzer
from src.agents.healthcare.fitness_tracker import FitnessTracker

logger = logging.getLogger(__name__)


@dataclass
class InsightRule:
    """Rule for generating insights"""

    name: str
    condition: callable
    priority: InsightPriority
    category: str
    title: str
    description_template: str
    recommendation_template: str
    action_items: List[str]


class HealthInsightsEngine:
    """
    Generates health insights and recommendations

    Features:
    - Analyzes patterns in health data
    - Generates personalized insights
    - Provides actionable recommendations
    - Tracks insight history
    """

    def __init__(
        self,
        storage_path: str = "data/healthcare",
        analyzer: HealthDataAnalyzer = None,
        fitness_tracker: FitnessTracker = None,
    ):
        self.storage_path = storage_path
        self._analyzer = analyzer
        self._fitness_tracker = fitness_tracker
        self._insights: List[HealthInsight] = []
        self._insight_history: List[HealthInsight] = []
        self._goals: List[HealthGoal] = []
        os.makedirs(storage_path, exist_ok=True)
        self._load_insights()
        self._setup_rules()

    def _load_insights(self):
        """Load saved insights"""
        insights_file = os.path.join(self.storage_path, "insights.json")
        if os.path.exists(insights_file):
            try:
                with open(insights_file) as f:
                    data = json.load(f)
                    for i in data.get("insights", []):
                        insight = HealthInsight(
                            id=i.get("id", ""),
                            title=i.get("title", ""),
                            description=i.get("description", ""),
                            priority=InsightPriority(i.get("priority", "medium")),
                            category=i.get("category", ""),
                            recommendation=i.get("recommendation", ""),
                            action_items=i.get("action_items", []),
                            created_at=datetime.fromisoformat(
                                i.get("created_at", datetime.now().isoformat())
                            ),
                            is_read=i.get("is_read", False),
                            is_dismissed=i.get("is_dismissed", False),
                        )
                        self._insights.append(insight)
            except Exception as e:
                logger.error(f"Failed to load insights: {e}")

    def _save_insights(self):
        """Save insights"""
        insights_file = os.path.join(self.storage_path, "insights.json")
        try:
            insights_data = [
                {
                    "id": i.id,
                    "title": i.title,
                    "description": i.description,
                    "priority": i.priority.value,
                    "category": i.category,
                    "recommendation": i.recommendation,
                    "action_items": i.action_items,
                    "created_at": i.created_at.isoformat(),
                    "is_read": i.is_read,
                    "is_dismissed": i.is_dismissed,
                }
                for i in self._insights
            ]
            with open(insights_file, "w") as f:
                json.dump({"insights": insights_data}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save insights: {e}")

    def set_dependencies(
        self, analyzer: HealthDataAnalyzer, fitness_tracker: FitnessTracker
    ):
        """Set dependencies"""
        self._analyzer = analyzer
        self._fitness_tracker = fitness_tracker
        logger.info("Health insights dependencies set")

    def set_goals(self, goals: List[HealthGoal]):
        """Set user health goals"""
        self._goals = goals
        logger.info(f"Health goals set: {[g.value for g in goals]}")

    def _setup_rules(self):
        """Setup insight generation rules"""
        self._rules = [
            InsightRule(
                name="low_steps",
                condition=lambda: self._analyzer
                and self._analyzer.analyze_metric(MetricType.STEPS, 1).current_value
                < 5000,
                priority=InsightPriority.HIGH,
                category="fitness",
                title="Low Activity Today",
                description_template="You've taken only {steps} steps today, which is below your target of 10,000.",
                recommendation_template="Try to take a 15-minute walk to boost your daily activity.",
                action_items=[
                    "Go for a walk",
                    "Take the stairs",
                    "Set a reminder to move",
                ],
            ),
            InsightRule(
                name="low_sleep",
                condition=lambda: self._analyzer
                and self._analyzer.analyze_metric(
                    MetricType.SLEEP_DURATION, 1
                ).current_value
                < 6,
                priority=InsightPriority.HIGH,
                category="sleep",
                title="Insufficient Sleep",
                description_template="You slept only {hours:.1f} hours last night. Adults need 7-9 hours for optimal health.",
                recommendation_template="Try to establish a consistent bedtime routine.",
                action_items=[
                    "Set a bedtime",
                    "Avoid screens before bed",
                    "Keep bedroom cool",
                ],
            ),
            InsightRule(
                name="low_water",
                condition=lambda: self._analyzer
                and self._analyzer.analyze_metric(
                    MetricType.WATER_INTAKE, 1
                ).current_value
                < 1500,
                priority=InsightPriority.MEDIUM,
                category="nutrition",
                title="Stay Hydrated",
                description_template="You've consumed only {water}ml of water today. Aim for at least 2,500ml.",
                recommendation_template="Keep a water bottle nearby and set reminders to drink.",
                action_items=[
                    "Drink water with meals",
                    "Set hydration reminders",
                    "Add lemon for flavor",
                ],
            ),
            InsightRule(
                name="high_stress",
                condition=lambda: self._analyzer
                and self._analyzer.analyze_metric(
                    MetricType.STRESS_LEVEL, 1
                ).current_value
                > 7,
                priority=InsightPriority.HIGH,
                category="general",
                title="High Stress Detected",
                description_template="Your stress levels have been elevated today.",
                recommendation_template="Consider taking a break and practicing relaxation techniques.",
                action_items=[
                    "Try deep breathing",
                    "Take a walk",
                    "Practice mindfulness",
                ],
            ),
            InsightRule(
                name="workout_streak",
                condition=lambda: self._fitness_tracker
                and self._fitness_tracker.get_progress(7).streak_days >= 3,
                priority=InsightPriority.LOW,
                category="fitness",
                title="Great Workout Streak!",
                description_template="You've been working out for {streak} days in a row! Keep it up!",
                recommendation_template="Your dedication is impressive. Consistency is key to fitness!",
                action_items=["Plan next workout", "Try a new exercise"],
            ),
            InsightRule(
                name="sedentary_time",
                condition=lambda: self._analyzer
                and self._analyzer.analyze_metric(
                    MetricType.ACTIVE_MINUTES, 1
                ).current_value
                < 30,
                priority=InsightPriority.MEDIUM,
                category="fitness",
                title="Low Active Minutes",
                description_template="You've been active for only {minutes} minutes today. Aim for at least 30.",
                recommendation_template="Incorporate more movement into your day.",
                action_items=[
                    "Take walking breaks",
                    "Do stretching at desk",
                    "Park farther away",
                ],
            ),
        ]

    def generate_insights(self, force: bool = False) -> List[HealthInsight]:
        """Generate new insights based on current data"""
        new_insights = []
        today = date.today()

        active_insights = [i for i in self._insights if not i.is_dismissed]

        for rule in self._rules:
            existing = any(
                i.title == rule.title and i.created_at.date() == today
                for i in active_insights
            )

            if existing and not force:
                continue

            try:
                if rule.condition():
                    insight = self._create_insight_from_rule(rule)
                    new_insights.append(insight)
                    self._insights.append(insight)
            except Exception as e:
                logger.debug(f"Rule {rule.name} condition check failed: {e}")

        if new_insights:
            self._save_insights()
            logger.info(f"Generated {len(new_insights)} new insights")

        return new_insights

    def _create_insight_from_rule(self, rule: InsightRule) -> HealthInsight:
        """Create an insight from a rule"""
        values = self._get_template_values()

        description = rule.description_template.format(**values)
        recommendation = rule.recommendation_template.format(**values)

        return HealthInsight(
            title=rule.title,
            description=description,
            priority=rule.priority,
            category=rule.category,
            recommendation=recommendation,
            action_items=rule.action_items,
        )

    def _get_template_values(self) -> Dict[str, Any]:
        """Get values for template formatting"""
        values = {}

        if self._analyzer:
            try:
                steps = self._analyzer.analyze_metric(MetricType.STEPS, 1)
                values["steps"] = int(steps.current_value)

                sleep = self._analyzer.analyze_metric(MetricType.SLEEP_DURATION, 1)
                values["hours"] = sleep.current_value

                water = self._analyzer.analyze_metric(MetricType.WATER_INTAKE, 1)
                values["water"] = int(water.current_value)

                active = self._analyzer.analyze_metric(MetricType.ACTIVE_MINUTES, 1)
                values["minutes"] = int(active.current_value)

                stress = self._analyzer.analyze_metric(MetricType.STRESS_LEVEL, 1)
                values["stress"] = int(stress.current_value)
            except Exception as e:
                logger.debug(f"Could not get template values: {e}")

        if self._fitness_tracker:
            try:
                progress = self._fitness_tracker.get_progress(7)
                values["streak"] = progress.streak_days
            except Exception as e:
                logger.debug(f"Could not get fitness values: {e}")

        return values

    def get_active_insights(
        self, category: str = None, priority: InsightPriority = None
    ) -> List[HealthInsight]:
        """Get active (non-dismissed) insights"""
        insights = [i for i in self._insights if not i.is_dismissed]

        if category:
            insights = [i for i in insights if i.category == category]

        if priority:
            insights = [i for i in insights if i.priority == priority]

        return sorted(
            insights,
            key=lambda i: (i.is_read, -i.priority.value == "high"),
            reverse=True,
        )

    def mark_as_read(self, insight_id: str):
        """Mark an insight as read"""
        for insight in self._insights:
            if insight.id == insight_id:
                insight.is_read = True
                break
        self._save_insights()

    def dismiss_insight(self, insight_id: str):
        """Dismiss an insight"""
        for insight in self._insights:
            if insight.id == insight_id:
                insight.is_dismissed = True
                break
        self._save_insights()

    def get_unread_count(self) -> int:
        """Get count of unread insights"""
        return len([i for i in self._insights if not i.is_read and not i.is_dismissed])

    def get_daily_tip(self) -> str:
        """Get a daily health tip"""
        tips = [
            "Stay hydrated! Drink at least 8 glasses of water today.",
            "Take a 10-minute walk after meals to aid digestion.",
            "Get 7-9 hours of quality sleep for optimal health.",
            "Include protein in every meal for sustained energy.",
            "Practice deep breathing for 5 minutes to reduce stress.",
            "Stand up and stretch every hour if you have a sedentary job.",
            "Eat colorful vegetables - they contain important nutrients.",
            "Limit processed foods and added sugars.",
            "Aim for 10,000 steps today for good cardiovascular health.",
            "Start your day with a healthy breakfast.",
        ]

        day_of_year = date.today().timetuple().tm_yday
        return tips[day_of_year % len(tips)]

    def get_weekly_summary(self) -> Dict[str, Any]:
        """Get a weekly health summary"""
        summary = {
            "date": date.today().isoformat(),
            "insights_generated": len(self._insights),
            "insights_read": len([i for i in self._insights if i.is_read]),
            "active_insights": len(self.get_active_insights()),
        }

        if self._analyzer:
            summary["health_score"] = self._analyzer.get_daily_summary().score

        if self._fitness_tracker:
            progress = self._fitness_tracker.get_progress(7)
            summary["fitness"] = {
                "workouts": progress.weekly_workouts,
                "minutes": progress.weekly_minutes,
                "calories": progress.weekly_calories,
                "streak": progress.streak_days,
            }

        return summary

    def get_stats(self) -> Dict[str, Any]:
        """Get insights engine statistics"""
        return {
            "total_insights": len(self._insights),
            "active_insights": len(self.get_active_insights()),
            "unread_insights": self.get_unread_count(),
            "dismissed_insights": len([i for i in self._insights if i.is_dismissed]),
            "insights_by_category": self._count_by_category(),
        }

    def _count_by_category(self) -> Dict[str, int]:
        """Count insights by category"""
        counts = defaultdict(int)
        for insight in self._insights:
            if not insight.is_dismissed:
                counts[insight.category] += 1
        return dict(counts)
