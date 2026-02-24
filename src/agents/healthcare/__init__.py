"""
AURA Healthcare Assistant Subsystem
==================================

AURA's healthcare agent that provides:
- Health data analysis
- Diet planning
- Fitness tracking
- Health insights
- Adaptive personality

100% offline - all data stored locally on device
"""

from src.agents.healthcare.models import (
    MetricType,
    MealType,
    WorkoutType,
    HealthGoal,
    InsightPriority,
    HealthMetric,
    HealthDataSet,
    NutritionInfo,
    FoodItem,
    Meal,
    MealPlan,
    Workout,
    FitnessProgress,
    HealthInsight,
    HealthProfile,
    HealthPreferences,
)

from src.agents.healthcare.analyzer import (
    HealthDataAnalyzer,
    MetricAnalysis,
    HealthSummary,
)

from src.agents.healthcare.diet_planner import (
    DietPlanner,
    FoodDatabase,
    NutritionTarget,
    MealSuggestion,
)

from src.agents.healthcare.fitness_tracker import (
    FitnessTracker,
    ExerciseDatabase,
    Exercise,
    WorkoutSuggestion,
    ProgressMetrics,
)

from src.agents.healthcare.health_insights import HealthInsightsEngine, InsightRule

from src.agents.healthcare.personality import (
    HealthcarePersonality,
    PersonalityTone,
    ResponseStyle,
    PersonalityTraits,
    InteractionContext,
)

from src.agents.healthcare.healthcare_agent import HealthcareAgent

__all__ = [
    # Models
    "MetricType",
    "MealType",
    "WorkoutType",
    "HealthGoal",
    "InsightPriority",
    "HealthMetric",
    "HealthDataSet",
    "NutritionInfo",
    "FoodItem",
    "Meal",
    "MealPlan",
    "Workout",
    "FitnessProgress",
    "HealthInsight",
    "HealthProfile",
    "HealthPreferences",
    # Analyzer
    "HealthDataAnalyzer",
    "MetricAnalysis",
    "HealthSummary",
    # Diet Planner
    "DietPlanner",
    "FoodDatabase",
    "NutritionTarget",
    "MealSuggestion",
    # Fitness Tracker
    "FitnessTracker",
    "ExerciseDatabase",
    "Exercise",
    "WorkoutSuggestion",
    "ProgressMetrics",
    # Health Insights
    "HealthInsightsEngine",
    "InsightRule",
    # Personality
    "HealthcarePersonality",
    "PersonalityTone",
    "ResponseStyle",
    "PersonalityTraits",
    "InteractionContext",
    # Main Agent
    "HealthcareAgent",
]
