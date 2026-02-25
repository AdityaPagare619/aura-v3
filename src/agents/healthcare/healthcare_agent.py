"""
Healthcare Agent
==============

Main healthcare assistant agent that integrates all healthcare modules:
- Health data analysis
- Diet planning
- Fitness tracking
- Health insights
- Adaptive personality

100% offline - all data stored locally
"""

import logging
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field

from src.agents.coordinator import Agent, AgentType, AgentTask
from src.agents.healthcare.models import (
    HealthMetric,
    HealthDataSet,
    MetricType,
    Meal,
    MealType,
    MealPlan,
    Workout,
    WorkoutType,
    HealthProfile,
    HealthGoal,
    HealthInsight,
    InsightPriority,
)
from src.agents.healthcare.analyzer import HealthDataAnalyzer
from src.agents.healthcare.diet_planner import DietPlanner
from src.agents.healthcare.fitness_tracker import FitnessTracker
from src.agents.healthcare.health_insights import HealthInsightsEngine
from src.agents.healthcare.personality import HealthcarePersonality, InteractionContext

# Import SecureStorage for encrypted data storage
try:
    from src.core.security_layers import SecureStorage
    SECURE_STORAGE_AVAILABLE = True
except ImportError:
    SECURE_STORAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HealthTask:
    """Task data for healthcare agent"""

    action: str
    params: Dict[str, Any] = field(default_factory=dict)


class HealthcareAgent(Agent):
    """
    Healthcare Assistant Agent

    Provides:
    - Health data analysis and tracking
    - Diet planning and nutrition
    - Fitness tracking and suggestions
    - Health insights and recommendations
    - Personalized health coaching

    100% offline - works without internet
    """

    def __init__(self, coordinator, storage_path: str = "data/healthcare"):
        super().__init__(
            agent_id="healthcare_assistant",
            name="Healthcare Assistant",
            agent_type=AgentType.ANALYZER,
            coordinator=coordinator,
        )

        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

        # Initialize encrypted storage for sensitive health data
        self._secure_storage = None
        if SECURE_STORAGE_AVAILABLE:
            try:
                self._secure_storage = SecureStorage(storage_path, encrypt_by_default=True)
                logger.info("Healthcare data encryption enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize secure storage: {e}")

        self._analyzer = HealthDataAnalyzer(storage_path)
        self._diet_planner = DietPlanner(storage_path)
        self._fitness_tracker = FitnessTracker(storage_path)
        self._insights_engine = HealthInsightsEngine(
            storage_path, self._analyzer, self._fitness_tracker
        )
        self._personality = HealthcarePersonality(storage_path)

        self._profile: Optional[HealthProfile] = None
        self._load_profile()

        logger.info("Healthcare agent initialized")

    async def initialize(self):
        """Async initialization - healthcare agent uses __init__ for sync setup"""
        logger.info("Healthcare Agent async initialized")
        return self

    def _load_profile(self):
        """Load user health profile - WITH DECRYPTION"""
        profile_file = "profile.json"
        
        # Try encrypted storage first
        if self._secure_storage is not None:
            data = self._secure_storage.load(profile_file)
            if data is not None:
                try:
                    goals = [HealthGoal(g) for g in data.get("goals", [])]
                    data["goals"] = goals
                    self._profile = HealthProfile(**data)
                    self._analyzer.set_profile(self._profile)
                    self._insights_engine.set_goals(goals)
                    logger.info("Loaded encrypted health profile")
                    return
                except Exception as e:
                    logger.error(f"Failed to load encrypted health profile: {e}")
        
        # Fallback to plain JSON (for backwards compatibility)
        profile_path = os.path.join(self.storage_path, profile_file)
        if os.path.exists(profile_path):
            try:
                with open(profile_path) as f:
                    data = json.load(f)
                    goals = [HealthGoal(g) for g in data.get("goals", [])]
                    data["goals"] = goals
                    self._profile = HealthProfile(**data)
                    self._analyzer.set_profile(self._profile)
                    self._insights_engine.set_goals(goals)
            except Exception as e:
                logger.error(f"Failed to load health profile: {e}")

    def _save_profile(self):
        """Save user health profile - WITH ENCRYPTION"""
        if not self._profile:
            return

        profile_file = "profile.json"
        
        data = {
            "user_id": self._profile.user_id,
            "age": self._profile.age,
            "gender": self._profile.gender,
            "height_cm": self._profile.height_cm,
            "weight_kg": self._profile.weight_kg,
            "target_weight_kg": self._profile.target_weight_kg,
            "activity_level": self._profile.activity_level,
            "goals": [g.value for g in self._profile.goals],
            "dietary_restrictions": self._profile.dietary_restrictions,
            "allergies": self._profile.allergies,
        }
        
        # Use encrypted storage
        if self._secure_storage is not None:
            if self._secure_storage.save(profile_file, data, encrypt=True):
                logger.info("Health profile saved with encryption")
                return
            else:
                logger.warning("Failed to save encrypted profile, falling back to plain")
        
        # Fallback to plain JSON
        profile_path = os.path.join(self.storage_path, profile_file)
        try:
            with open(profile_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")

    def get_capabilities(self) -> List[str]:
        return [
            "health_tracking",
            "health_analysis",
            "diet_planning",
            "fitness_tracking",
            "nutrition_advice",
            "workout_suggestions",
            "health_insights",
            "weight_management",
            "sleep_tracking",
            "offline_health",
        ]

    async def process_task(self, task: AgentTask) -> Any:
        """Process healthcare tasks"""
        action = task.data.get("action", "help")

        try:
            if action == "log_metric":
                return await self._log_metric(task.data)
            elif action == "get_summary":
                return await self.get_daily_summary(task.data.get("date"))
            elif action == "analyze_metric":
                return await self.analyze_metric(
                    task.data.get("metric_type"), task.data.get("days", 7)
                )
            elif action == "create_meal_plan":
                return await self.create_meal_plan(task.data.get("days", 7))
            elif action == "log_meal":
                return await self.log_meal(task.data)
            elif action == "get_workout_suggestion":
                return await self.get_workout_suggestion(task.data)
            elif action == "log_workout":
                return await self.log_workout(task.data)
            elif action == "get_progress":
                return await self.get_fitness_progress(task.data.get("days", 7))
            elif action == "generate_insights":
                return await self.generate_insights()
            elif action == "get_insights":
                return await self.get_insights(task.data.get("category"))
            elif action == "set_profile":
                return await self.set_profile(task.data)
            elif action == "update_profile":
                return await self.update_profile(task.data)
            elif action == "get_health_tip":
                return self.get_daily_tip()
            elif action == "search_food":
                return await self.search_food(task.data.get("query"))
            elif action == "get_exercises":
                return await self.get_exercises(task.data)
            elif action == "help":
                return self.get_help()
            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error(f"Error processing healthcare task: {e}")
            return {"error": str(e)}

    async def _log_metric(self, data: Dict) -> Dict:
        """Log a health metric"""
        metric_type = MetricType(data.get("metric_type", "steps"))
        value = float(data.get("value", 0))
        unit = data.get("unit", "")

        is_valid, message = self._analyzer.validate_metric(metric_type, value)
        if not is_valid:
            return {"error": message}

        metric = self._analyzer.add_metric(
            metric_type=metric_type,
            value=value,
            unit=unit,
            source=data.get("source", "manual"),
        )

        self._personality.learn_from_interaction(
            "metric_log", f"logged {metric_type.value}"
        )

        self._generate_insights_if_needed()

        return {
            "status": "logged",
            "metric": {
                "type": metric.metric_type.value,
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp.isoformat(),
            },
        }

    async def get_daily_summary(self, target_date: str = None) -> Dict:
        """Get daily health summary"""
        if target_date:
            target_date = date.fromisoformat(target_date)
        else:
            target_date = date.today()

        summary = self._analyzer.get_daily_summary(target_date)

        return {
            "date": summary.date.isoformat(),
            "total_steps": summary.total_steps,
            "average_heart_rate": summary.average_heart_rate,
            "sleep_hours": summary.sleep_hours,
            "calories_consumed": summary.calories_consumed,
            "calories_burned": summary.calories_burned,
            "water_intake_ml": summary.water_intake_ml,
            "active_minutes": summary.active_minutes,
            "health_score": summary.score,
        }

    async def analyze_metric(self, metric_type: str, days: int = 7) -> Dict:
        """Analyze a health metric"""
        try:
            mt = MetricType(metric_type)
        except ValueError:
            return {"error": f"Unknown metric type: {metric_type}"}

        analysis = self._analyzer.analyze_metric(mt, days)

        return {
            "metric_type": analysis.metric_type.value,
            "current_value": analysis.current_value,
            "average_value": analysis.average_value,
            "trend": analysis.trend,
            "trend_percentage": analysis.trend_percentage,
            "anomaly_detected": analysis.anomaly_detected,
            "anomaly_message": analysis.anomaly_message,
        }

    async def create_meal_plan(self, days: int = 7) -> Dict:
        """Create a meal plan"""
        if not self._profile:
            return {"error": "Please set up your health profile first"}

        meal_plan = self._diet_planner.create_meal_plan(self._profile, days)

        return {
            "status": "created",
            "start_date": meal_plan.start_date.isoformat(),
            "end_date": meal_plan.end_date.isoformat(),
            "daily_calories": meal_plan.total_daily_calories,
            "meals": [
                {
                    "type": m.meal_type.value,
                    "name": m.name,
                    "foods": [f.name for f in m.foods],
                    "calories": m.total_nutrition.calories,
                    "protein": m.total_nutrition.protein,
                }
                for m in meal_plan.meals
            ],
        }

    async def log_meal(self, data: Dict) -> Dict:
        """Log a meal"""
        meal_type = MealType(data.get("meal_type", "lunch"))
        food_names = data.get("foods", [])
        notes = data.get("notes", "")

        meal = self._diet_planner.log_meal(meal_type, food_names, notes)

        for food_name in food_names:
            self._personality.learn_food_preference(food_name, liked=True)

        self._analyzer.add_metric(
            MetricType.CALORIES_CONSUMED,
            meal.total_nutrition.calories,
            "kcal",
        )

        return {
            "status": "logged",
            "meal_type": meal.meal_type.value,
            "calories": meal.total_nutrition.calories,
            "protein": meal.total_nutrition.protein,
        }

    async def get_workout_suggestion(self, data: Dict) -> Dict:
        """Get workout suggestion"""
        workout_type = None
        if data.get("workout_type"):
            try:
                workout_type = WorkoutType(data.get("workout_type"))
            except ValueError:
                pass

        duration = data.get("duration", 30)
        difficulty = data.get("difficulty")

        suggestion = self._fitness_tracker.suggest_workout(
            workout_type, duration, difficulty
        )

        return {
            "workout_type": suggestion.workout.workout_type.value,
            "duration": suggestion.workout.duration_minutes,
            "exercises": suggestion.workout.exercises,
            "reason": suggestion.reason,
        }

    async def log_workout(self, data: Dict) -> Dict:
        """Log a workout"""
        workout_type = WorkoutType(data.get("workout_type", "other"))
        duration = data.get("duration", 30)
        intensity = data.get("intensity", "moderate")
        calories = data.get("calories_burned", 0)
        notes = data.get("notes", "")

        workout = self._fitness_tracker.log_workout(
            workout_type=workout_type,
            duration_minutes=duration,
            intensity=intensity,
            calories_burned=calories,
            notes=notes,
        )

        self._personality.learn_workout_preference(workout_type.value, liked=True)

        self._analyzer.add_metric(
            MetricType.CALORIES_BURNED,
            workout.calories_burned,
            "kcal",
        )
        self._analyzer.add_metric(
            MetricType.ACTIVE_MINUTES,
            duration,
            "minutes",
        )

        return {
            "status": "logged",
            "workout_type": workout.workout_type.value,
            "duration": workout.duration_minutes,
            "calories": workout.calories_burned,
        }

    async def get_fitness_progress(self, days: int = 7) -> Dict:
        """Get fitness progress"""
        progress = self._fitness_tracker.get_progress(days)

        return {
            "period_days": days,
            "workouts_completed": progress.weekly_workouts,
            "total_minutes": progress.weekly_minutes,
            "total_calories": progress.weekly_calories,
            "streak_days": progress.streak_days,
            "improvement_percentage": progress.improvement_percentage,
            "goal_progress": progress.goal_progress,
        }

    async def generate_insights(self) -> List[Dict]:
        """Generate new health insights"""
        new_insights = self._insights_engine.generate_insights()

        return [
            {
                "id": i.id,
                "title": i.title,
                "description": i.description,
                "priority": i.priority.value,
                "category": i.category,
                "recommendation": i.recommendation,
                "action_items": i.action_items,
            }
            for i in new_insights
        ]

    async def get_insights(self, category: str = None) -> List[Dict]:
        """Get active insights"""
        insights = self._insights_engine.get_active_insights(category)

        return [
            {
                "id": i.id,
                "title": i.title,
                "description": i.description,
                "priority": i.priority.value,
                "category": i.category,
                "recommendation": i.recommendation,
                "action_items": i.action_items,
                "is_read": i.is_read,
            }
            for i in insights
        ]

    async def set_profile(self, data: Dict) -> Dict:
        """Set up health profile"""
        goals = [HealthGoal(g) for g in data.get("goals", ["general_wellness"])]

        self._profile = HealthProfile(
            user_id=data.get("user_id", "default"),
            age=data.get("age", 30),
            gender=data.get("gender", "unspecified"),
            height_cm=data.get("height_cm", 170),
            weight_kg=data.get("weight_kg", 70),
            target_weight_kg=data.get("target_weight_kg", 70),
            activity_level=data.get("activity_level", "moderate"),
            goals=goals,
            dietary_restrictions=data.get("dietary_restrictions", []),
            allergies=data.get("allergies", []),
        )

        self._profile.calculate_tdee()

        self._analyzer.set_profile(self._profile)
        self._insights_engine.set_goals(goals)
        self._personality.set_user_id(self._profile.user_id)

        self._save_profile()

        return {
            "status": "profile_set",
            "profile": {
                "bmi": self._profile.bmi,
                "bmr": self._profile.bmr,
                "tdee": self._profile.tdee,
            },
        }

    async def update_profile(self, data: Dict) -> Dict:
        """Update health profile"""
        if not self._profile:
            return {"error": "No profile set. Use set_profile first."}

        if "age" in data:
            self._profile.age = data["age"]
        if "weight_kg" in data:
            self._profile.weight_kg = data["weight_kg"]
        if "target_weight_kg" in data:
            self._profile.target_weight_kg = data["target_weight_kg"]
        if "activity_level" in data:
            self._profile.activity_level = data["activity_level"]

        self._profile.calculate_tdee()
        self._save_profile()

        return {
            "status": "updated",
            "tdee": self._profile.tdee,
        }

    def get_daily_tip(self) -> str:
        """Get daily health tip"""
        return self._insights_engine.get_daily_tip()

    async def search_food(self, query: str) -> List[Dict]:
        """Search for foods"""
        foods = self._diet_planner.search_food(query)

        return [
            {
                "name": f.name,
                "serving_size": f.serving_size,
                "serving_unit": f.serving_unit,
                "calories": f.nutrition.calories,
                "protein": f.nutrition.protein,
                "carbohydrates": f.nutrition.carbohydrates,
                "fat": f.nutrition.fat,
                "tags": f.tags,
            }
            for f in foods
        ]

    async def get_exercises(self, data: Dict) -> List[Dict]:
        """Get exercise suggestions"""
        muscle_groups = data.get("muscle_groups")
        equipment = data.get("equipment")

        exercises = self._fitness_tracker.get_exercise_suggestions(
            muscle_groups, equipment
        )

        return [
            {
                "name": e.name,
                "muscle_groups": e.muscle_groups,
                "equipment": e.equipment,
                "difficulty": e.difficulty,
                "calories_per_minute": e.calories_per_minute,
                "instructions": e.instructions,
            }
            for e in exercises
        ]

    def get_help(self) -> Dict:
        """Get help information"""
        return {
            "available_actions": [
                {"action": "log_metric", "description": "Log a health metric"},
                {"action": "get_summary", "description": "Get daily health summary"},
                {"action": "analyze_metric", "description": "Analyze a metric trend"},
                {"action": "create_meal_plan", "description": "Create a meal plan"},
                {"action": "log_meal", "description": "Log a meal"},
                {
                    "action": "get_workout_suggestion",
                    "description": "Get workout suggestion",
                },
                {"action": "log_workout", "description": "Log a workout"},
                {"action": "get_progress", "description": "Get fitness progress"},
                {
                    "action": "generate_insights",
                    "description": "Generate health insights",
                },
                {"action": "get_insights", "description": "Get active insights"},
                {"action": "set_profile", "description": "Set up health profile"},
                {"action": "update_profile", "description": "Update health profile"},
                {"action": "get_health_tip", "description": "Get daily health tip"},
                {"action": "search_food", "description": "Search food database"},
                {"action": "get_exercises", "description": "Get exercise suggestions"},
            ],
            "metric_types": [m.value for m in MetricType],
            "workout_types": [w.value for w in WorkoutType],
        }

    def _generate_insights_if_needed(self):
        """Generate insights if threshold reached"""
        try:
            self._insights_engine.generate_insights()
        except Exception as e:
            logger.debug(f"Could not generate insights: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get healthcare agent statistics"""
        return {
            "analyzer": self._analyzer.get_stats(),
            "diet_planner": self._diet_planner.get_stats(),
            "fitness_tracker": self._fitness_tracker.get_stats(),
            "insights": self._insights_engine.get_stats(),
            "personality": self._personality.get_personality_summary(),
        }
