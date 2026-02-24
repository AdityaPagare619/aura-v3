"""
Healthcare Agent Data Models
===========================

Data structures for health data, diet, fitness, and insights
"""

import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum


class MetricType(Enum):
    """Types of health metrics"""

    STEPS = "steps"
    HEART_RATE = "heart_rate"
    SLEEP_DURATION = "sleep_duration"
    SLEEP_QUALITY = "sleep_quality"
    CALORIES_BURNED = "calories_burned"
    CALORIES_CONSUMED = "calories_consumed"
    WATER_INTAKE = "water_intake"
    WEIGHT = "weight"
    BODY_FAT = "body_fat"
    BLOOD_PRESSURE_SYSTOLIC = "blood_pressure_systolic"
    BLOOD_PRESSURE_DIASTOLIC = "blood_pressure_diastolic"
    BLOOD_OXYGEN = "blood_oxygen"
    STRESS_LEVEL = "stress_level"
    ACTIVE_MINUTES = "active_minutes"
    STANDING_HOURS = "standing_hours"
    FLIGHTS_CLIMBED = "flights_climbed"


class MealType(Enum):
    """Types of meals"""

    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK_MORNING = "snack_morning"
    SNACK_AFTERNOON = "snack_afternoon"
    SNACK_EVENING = "snack_evening"


class WorkoutType(Enum):
    """Types of workouts"""

    RUNNING = "running"
    WALKING = "walking"
    CYCLING = "cycling"
    SWIMMING = "swimming"
    STRENGTH_TRAINING = "strength_training"
    YOGA = "yoga"
    HIIT = "hiit"
    STRETCHING = "stretching"
    SPORTS = "sports"
    OTHER = "other"


class HealthGoal(Enum):
    """User health goals"""

    WEIGHT_LOSS = "weight_loss"
    WEIGHT_GAIN = "weight_gain"
    WEIGHT_MAINTENANCE = "weight_maintenance"
    MUSCLE_GAIN = "muscle_gain"
    ENDURANCE = "endurance"
    FLEXIBILITY = "flexibility"
    STRESS_REDUCTION = "stress_reduction"
    BETTER_SLEEP = "better_sleep"
    GENERAL_WELLNESS = "general_wellness"


class InsightPriority(Enum):
    """Priority levels for health insights"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class HealthMetric:
    """Single health metric data point"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metric_type: MetricType = MetricType.STEPS
    value: float = 0.0
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "manual"  # manual, health_app, device
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthDataSet:
    """Collection of health metrics"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_id: str = ""
    date: date = field(default_factory=date.today)
    metrics: List[HealthMetric] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NutritionInfo:
    """Nutritional information for a food item"""

    calories: float = 0.0
    protein: float = 0.0
    carbohydrates: float = 0.0
    fat: float = 0.0
    fiber: float = 0.0
    sugar: float = 0.0
    sodium: float = 0.0
    cholesterol: float = 0.0
    vitamin_a: float = 0.0
    vitamin_c: float = 0.0
    calcium: float = 0.0
    iron: float = 0.0


@dataclass
class FoodItem:
    """A food item with nutritional info"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    serving_size: float = 100.0
    serving_unit: str = "g"
    nutrition: NutritionInfo = field(default_factory=NutritionInfo)
    tags: List[str] = field(default_factory=list)
    is_custom: bool = False


@dataclass
class Meal:
    """A meal with food items"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    meal_type: MealType = MealType.BREAKFAST
    foods: List[FoodItem] = field(default_factory=list)
    total_nutrition: NutritionInfo = field(default_factory=NutritionInfo)
    timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""


@dataclass
class MealPlan:
    """A complete meal plan"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_id: str = ""
    start_date: date = field(default_factory=date.today)
    end_date: date = None
    meals: List[Meal] = field(default_factory=list)
    total_daily_calories: float = 2000.0
    goal: HealthGoal = HealthGoal.GENERAL_WELLNESS
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Workout:
    """A workout session"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    workout_type: WorkoutType = WorkoutType.OTHER
    name: str = ""
    duration_minutes: int = 0
    intensity: str = "moderate"  # low, moderate, high
    calories_burned: float = 0.0
    heart_rate_avg: float = 0.0
    heart_rate_max: float = 0.0
    distance: float = 0.0
    distance_unit: str = "km"
    timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""
    exercises: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FitnessProgress:
    """Fitness progress tracking"""

    user_id: str = ""
    date: date = field(default_factory=date.today)
    workouts_completed: int = 0
    total_workout_minutes: int = 0
    total_calories_burned: float = 0.0
    workouts: List[Workout] = field(default_factory=list)
    streak_days: int = 0
    personal_bests: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthInsight:
    """A health insight with recommendation"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    priority: InsightPriority = InsightPriority.MEDIUM
    category: str = ""  # fitness, nutrition, sleep, general
    metric_related: List[MetricType] = field(default_factory=list)
    recommendation: str = ""
    action_items: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    is_read: bool = False
    is_dismissed: bool = False


@dataclass
class HealthProfile:
    """User's health profile"""

    user_id: str = ""
    age: int = 30
    gender: str = "unspecified"
    height_cm: float = 170.0
    weight_kg: float = 70.0
    target_weight_kg: float = 70.0
    activity_level: str = "moderate"  # sedentary, light, moderate, active, very_active
    goals: List[HealthGoal] = field(default_factory=list)
    dietary_restrictions: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    medical_conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    bmi: float = 0.0
    bmr: float = 0.0
    tdee: float = 0.0  # Total Daily Energy Expenditure

    def calculate_bmi(self) -> float:
        """Calculate BMI"""
        if self.height_cm > 0:
            height_m = self.height_cm / 100
            self.bmi = self.weight_kg / (height_m * height_m)
        return self.bmi

    def calculate_bmr(self) -> float:
        """Calculate BMR using Mifflin-St Jeor equation"""
        if self.gender.lower() == "male":
            self.bmr = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age + 5
        else:
            self.bmr = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age - 161
        return self.bmr

    def calculate_tdee(self) -> float:
        """Calculate TDEE based on activity level"""
        self.calculate_bmr()
        multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9,
        }
        multiplier = multipliers.get(self.activity_level, 1.55)
        self.tdee = self.bmr * multiplier
        return self.tdee


@dataclass
class HealthPreferences:
    """User's health-related preferences (learned)"""

    user_id: str = ""
    preferred_workout_times: List[str] = field(default_factory=list)
    preferred_meal_times: Dict[str, str] = field(default_factory=dict)
    disliked_foods: List[str] = field(default_factory=list)
    liked_foods: List[str] = field(default_factory=list)
    workout_preferences: List[str] = field(default_factory=list)
    notification_preferences: Dict[str, bool] = field(default_factory=dict)
    insight_priority_focus: List[str] = field(default_factory=list)
    motivational_style: str = "encouraging"  # encouraging, challenging, informational
    response_to_insights: str = "action"  # action, planning, ignore
