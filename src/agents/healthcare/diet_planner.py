"""
Diet Planner
============

Creates personalized meal plans based on:
- User preferences and dietary restrictions
- Health goals
- Nutritional requirements
- Food database

100% offline - works without internet
"""

import logging
import json
import os
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from src.agents.healthcare.models import (
    FoodItem,
    Meal,
    MealPlan,
    MealType,
    NutritionInfo,
    HealthProfile,
    HealthGoal,
)

logger = logging.getLogger(__name__)


@dataclass
class NutritionTarget:
    """Daily nutrition targets"""

    calories: float = 2000.0
    protein: float = 50.0
    carbohydrates: float = 250.0
    fat: float = 65.0
    fiber: float = 25.0
    sugar: float = 50.0
    sodium: float = 2300.0


@dataclass
class MealSuggestion:
    """A suggested meal with reasoning"""

    meal: Meal
    nutrition_match: float  # How well it matches targets
    reason: str


class FoodDatabase:
    """
    Local food database with nutritional information
    Works 100% offline
    """

    def __init__(self):
        self._foods: Dict[str, FoodItem] = {}
        self._foods_by_tag: Dict[str, List[str]] = defaultdict(list)
        self._load_default_foods()

    def _load_default_foods(self):
        """Load default food database"""
        default_foods = [
            {
                "name": "Brown Rice",
                "serving_size": 100,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 111,
                    "protein": 2.6,
                    "carbohydrates": 23,
                    "fat": 0.9,
                    "fiber": 1.8,
                },
                "tags": ["whole_grain", "vegetarian", "vegan", "gluten_free"],
            },
            {
                "name": "White Rice",
                "serving_size": 100,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 130,
                    "protein": 2.7,
                    "carbohydrates": 28,
                    "fat": 0.3,
                    "fiber": 0.4,
                },
                "tags": ["vegetarian", "vegan", "gluten_free"],
            },
            {
                "name": "Chicken Breast",
                "serving_size": 100,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 165,
                    "protein": 31,
                    "carbohydrates": 0,
                    "fat": 3.6,
                    "fiber": 0,
                },
                "tags": ["protein", "low_carb", "gluten_free"],
            },
            {
                "name": "Salmon",
                "serving_size": 100,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 208,
                    "protein": 20,
                    "carbohydrates": 0,
                    "fat": 13,
                    "fiber": 0,
                },
                "tags": ["protein", "omega3", "low_carb", "gluten_free"],
            },
            {
                "name": "Eggs",
                "serving_size": 50,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 78,
                    "protein": 6,
                    "carbohydrates": 0.6,
                    "fat": 5,
                    "fiber": 0,
                },
                "tags": ["protein", "vegetarian", "low_carb", "gluten_free"],
            },
            {
                "name": "Oatmeal",
                "serving_size": 40,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 150,
                    "protein": 5,
                    "carbohydrates": 27,
                    "fat": 3,
                    "fiber": 4,
                },
                "tags": ["whole_grain", "vegetarian", "vegan", "high_fiber"],
            },
            {
                "name": "Banana",
                "serving_size": 120,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 105,
                    "protein": 1.3,
                    "carbohydrates": 27,
                    "fat": 0.4,
                    "fiber": 3.1,
                },
                "tags": ["fruit", "vegetarian", "vegan", "gluten_free"],
            },
            {
                "name": "Apple",
                "serving_size": 180,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 95,
                    "protein": 0.5,
                    "carbohydrates": 25,
                    "fat": 0.3,
                    "fiber": 4.4,
                },
                "tags": ["fruit", "vegetarian", "vegan", "gluten_free"],
            },
            {
                "name": "Broccoli",
                "serving_size": 100,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 34,
                    "protein": 2.8,
                    "carbohydrates": 7,
                    "fat": 0.4,
                    "fiber": 2.6,
                },
                "tags": ["vegetable", "vegetarian", "vegan", "low_carb"],
            },
            {
                "name": "Spinach",
                "serving_size": 100,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 23,
                    "protein": 2.9,
                    "carbohydrates": 3.6,
                    "fat": 0.4,
                    "fiber": 2.2,
                },
                "tags": ["vegetable", "vegetarian", "vegan", "low_carb"],
            },
            {
                "name": "Greek Yogurt",
                "serving_size": 150,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 100,
                    "protein": 17,
                    "carbohydrates": 6,
                    "fat": 0.7,
                    "fiber": 0,
                },
                "tags": ["protein", "dairy", "vegetarian", "low_carb"],
            },
            {
                "name": "Almonds",
                "serving_size": 30,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 173,
                    "protein": 6,
                    "carbohydrates": 6,
                    "fat": 15,
                    "fiber": 3.5,
                },
                "tags": ["nuts", "vegetarian", "vegan", "high_protein"],
            },
            {
                "name": "Sweet Potato",
                "serving_size": 130,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 112,
                    "protein": 2,
                    "carbohydrates": 26,
                    "fat": 0.1,
                    "fiber": 3.9,
                },
                "tags": ["vegetable", "vegetarian", "vegan", "gluten_free"],
            },
            {
                "name": "Quinoa",
                "serving_size": 100,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 120,
                    "protein": 4.4,
                    "carbohydrates": 21,
                    "fat": 1.9,
                    "fiber": 2.8,
                },
                "tags": ["whole_grain", "protein", "vegetarian", "vegan"],
            },
            {
                "name": "Tofu",
                "serving_size": 100,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 76,
                    "protein": 8,
                    "carbohydrates": 1.9,
                    "fat": 4.8,
                    "fiber": 0.3,
                },
                "tags": ["protein", "vegetarian", "vegan", "low_carb"],
            },
            {
                "name": "Lentils",
                "serving_size": 100,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 116,
                    "protein": 9,
                    "carbohydrates": 20,
                    "fat": 0.4,
                    "fiber": 7.9,
                },
                "tags": ["protein", "vegetarian", "vegan", "high_fiber"],
            },
            {
                "name": "Avocado",
                "serving_size": 100,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 160,
                    "protein": 2,
                    "carbohydrates": 9,
                    "fat": 15,
                    "fiber": 7,
                },
                "tags": ["fruit", "vegetarian", "vegan", "healthy_fat"],
            },
            {
                "name": "Tuna",
                "serving_size": 100,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 132,
                    "protein": 28,
                    "carbohydrates": 0,
                    "fat": 1,
                    "fiber": 0,
                },
                "tags": ["protein", "low_carb", "gluten_free"],
            },
            {
                "name": "Cottage Cheese",
                "serving_size": 100,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 98,
                    "protein": 11,
                    "carbohydrates": 3.4,
                    "fat": 4.3,
                    "fiber": 0,
                },
                "tags": ["protein", "dairy", "vegetarian", "low_carb"],
            },
            {
                "name": "Whole Wheat Bread",
                "serving_size": 30,
                "serving_unit": "g",
                "nutrition": {
                    "calories": 81,
                    "protein": 4,
                    "carbohydrates": 14,
                    "fat": 1.1,
                    "fiber": 2,
                },
                "tags": ["whole_grain", "vegetarian", "high_fiber"],
            },
        ]

        for food_data in default_foods:
            nutrition = NutritionInfo(**food_data.get("nutrition", {}))
            food = FoodItem(
                name=food_data["name"],
                serving_size=food_data["serving_size"],
                serving_unit=food_data["serving_unit"],
                nutrition=nutrition,
                tags=food_data.get("tags", []),
            )
            self._foods[food_data["name"].lower()] = food
            for tag in food_data.get("tags", []):
                self._foods_by_tag[tag].append(food_data["name"].lower())

    def search_foods(self, query: str, tags: List[str] = None) -> List[FoodItem]:
        """Search foods by name or tags"""
        results = []

        query_lower = query.lower()
        for name, food in self._foods.items():
            if query_lower in name:
                results.append(food)

        if tags:
            for tag in tags:
                for food_name in self._foods_by_tag.get(tag, []):
                    food = self._foods.get(food_name)
                    if food and food not in results:
                        results.append(food)

        return results[:10]

    def get_food(self, name: str) -> Optional[FoodItem]:
        """Get a specific food by name"""
        return self._foods.get(name.lower())

    def get_random_foods(
        self, count: int = 5, tags: List[str] = None
    ) -> List[FoodItem]:
        """Get random foods, optionally filtered by tags"""
        foods = list(self._foods.values())

        if tags:
            foods = [f for f in foods if any(tag in f.tags for tag in tags)]

        return random.sample(foods, min(count, len(foods)))


class DietPlanner:
    """
    Creates personalized meal plans

    Features:
    - Considers user profile and goals
    - Respects dietary restrictions
    - Balances nutrition
    - Learns user preferences
    """

    def __init__(self, storage_path: str = "data/healthcare"):
        self.storage_path = storage_path
        self._food_database = FoodDatabase()
        self._meal_history: List[MealPlan] = []
        self._user_preferences: Dict[str, Any] = {}
        os.makedirs(storage_path, exist_ok=True)
        self._load_preferences()

    def _load_preferences(self):
        """Load user preferences"""
        prefs_file = os.path.join(self.storage_path, "diet_preferences.json")
        if os.path.exists(prefs_file):
            with open(prefs_file) as f:
                self._user_preferences = json.load(f)

    def _save_preferences(self):
        """Save user preferences"""
        prefs_file = os.path.join(self.storage_path, "diet_preferences.json")
        with open(prefs_file, "w") as f:
            json.dump(self._user_preferences, f, indent=2)

    def get_nutrition_targets(self, profile: HealthProfile) -> NutritionTarget:
        """Calculate nutrition targets from profile"""
        tdee = profile.calculate_tdee()

        target_calories = tdee

        for goal in profile.goals:
            if goal == HealthGoal.WEIGHT_LOSS:
                target_calories -= 500
            elif goal == HealthGoal.WEIGHT_GAIN:
                target_calories += 300
            elif goal == HealthGoal.MUSCLE_GAIN:
                target_calories += 200

        target_calories = max(1200, min(target_calories, 4000))

        protein_ratio = 0.3 if HealthGoal.MUSCLE_GAIN in profile.goals else 0.25
        fat_ratio = 0.3
        carb_ratio = 1 - protein_ratio - fat_ratio

        return NutritionTarget(
            calories=target_calories,
            protein=(target_calories * protein_ratio) / 4,
            carbohydrates=(target_calories * carb_ratio) / 4,
            fat=(target_calories * fat_ratio) / 9,
            fiber=25,
        )

    def create_meal_plan(
        self,
        profile: HealthProfile,
        days: int = 7,
        preferences: Dict[str, Any] = None,
    ) -> MealPlan:
        """Create a meal plan"""
        targets = self.get_nutrition_targets(profile)
        preferences = preferences or {}

        restrictions = set(profile.dietary_restrictions + profile.allergies)

        start_date = date.today()
        meals = []

        calories_per_meal = {
            MealType.BREAKFAST: targets.calories * 0.25,
            MealType.LUNCH: targets.calories * 0.35,
            MealType.DINNER: targets.calories * 0.30,
            MealType.SNACK_MORNING: targets.calories * 0.05,
            MealType.SNACK_AFTERNOON: targets.calories * 0.025,
            MealType.SNACK_EVENING: targets.calories * 0.025,
        }

        for day in range(days):
            current_date = start_date + timedelta(days=day)

            for meal_type in MealType:
                if meal_type == MealType.SNACK_MORNING:
                    meal = self._create_snack(meal_type, "morning", restrictions, 150)
                elif meal_type == MealType.SNACK_AFTERNOON:
                    meal = self._create_snack(meal_type, "afternoon", restrictions, 100)
                elif meal_type == MealType.SNACK_EVENING:
                    meal = self._create_snack(meal_type, "evening", restrictions, 100)
                elif meal_type == MealType.BREAKFAST:
                    meal = self._create_meal(
                        meal_type,
                        "breakfast",
                        restrictions,
                        calories_per_meal[meal_type],
                    )
                elif meal_type == MealType.LUNCH:
                    meal = self._create_meal(
                        meal_type, "lunch", restrictions, calories_per_meal[meal_type]
                    )
                else:
                    meal = self._create_meal(
                        meal_type, "dinner", restrictions, calories_per_meal[meal_type]
                    )

                meal.timestamp = datetime.combine(current_date, datetime.min.time())
                meals.append(meal)

        meal_plan = MealPlan(
            user_id=profile.user_id,
            start_date=start_date,
            end_date=start_date + timedelta(days=days - 1),
            meals=meals,
            total_daily_calories=targets.calories,
            goal=profile.goals[0] if profile.goals else HealthGoal.GENERAL_WELLNESS,
            preferences=preferences,
        )

        self._meal_history.append(meal_plan)
        logger.info(f"Created meal plan for {days} days")

        return meal_plan

    def _create_meal(
        self,
        meal_type: MealType,
        time_of_day: str,
        restrictions: set,
        target_calories: float,
    ) -> Meal:
        """Create a single meal"""
        tags = []

        if "vegetarian" in restrictions or "vegan" in restrictions:
            tags.extend(["vegetarian", "vegan"])

        if "gluten_free" in restrictions:
            tags.append("gluten_free")

        if time_of_day == "breakfast":
            tags.extend(["vegetarian"])
            foods = self._food_database.get_random_foods(3, tags)
            if not foods:
                foods = [
                    self._food_database.get_food("oatmeal"),
                    self._food_database.get_food("eggs"),
                    self._food_database.get_food("banana"),
                ]
        elif time_of_day == "lunch":
            foods = self._food_database.get_random_foods(3, tags)
            if not foods:
                foods = [
                    self._food_database.get_food("brown rice"),
                    self._food_database.get_food("chicken breast"),
                    self._food_database.get_food("broccoli"),
                ]
        else:
            foods = self._food_database.get_random_foods(3, tags)
            if not foods:
                foods = [
                    self._food_database.get_food("salmon"),
                    self._food_database.get_food("sweet potato"),
                    self._food_database.get_food("spinach"),
                ]

        foods = [f for f in foods if f]

        total_nutrition = NutritionInfo()
        for food in foods:
            total_nutrition.calories += food.nutrition.calories
            total_nutrition.protein += food.nutrition.protein
            total_nutrition.carbohydrates += food.nutrition.carbohydrates
            total_nutrition.fat += food.nutrition.fat
            total_nutrition.fiber += food.nutrition.fiber

        scale = (
            target_calories / total_nutrition.calories
            if total_nutrition.calories > 0
            else 1
        )
        total_nutrition.calories *= scale
        total_nutrition.protein *= scale
        total_nutrition.carbohydrates *= scale
        total_nutrition.fat *= scale
        total_nutrition.fiber *= scale

        return Meal(
            name=f"{time_of_day.title()} Meal",
            meal_type=meal_type,
            foods=foods,
            total_nutrition=total_nutrition,
        )

    def _create_snack(
        self,
        meal_type: MealType,
        time_of_day: str,
        restrictions: set,
        target_calories: float,
    ) -> Meal:
        """Create a snack"""
        foods = self._food_database.get_random_foods(2, ["vegetarian"])
        if not foods:
            foods = [
                self._food_database.get_food("almonds"),
                self._food_database.get_food("apple"),
                self._food_database.get_food("banana"),
            ]

        foods = [f for f in foods if f][:2]

        total_nutrition = NutritionInfo()
        for food in foods:
            total_nutrition.calories += food.nutrition.calories
            total_nutrition.protein += food.nutrition.protein
            total_nutrition.carbohydrates += food.nutrition.carbohydrates
            total_nutrition.fat += food.nutrition.fat

        scale = (
            target_calories / total_nutrition.calories
            if total_nutrition.calories > 0
            else 1
        )
        total_nutrition.calories *= scale
        total_nutrition.protein *= scale
        total_nutrition.carbohydrates *= scale
        total_nutrition.fat *= scale

        return Meal(
            name=f"{time_of_day.title()} Snack",
            meal_type=meal_type,
            foods=foods,
            total_nutrition=total_nutrition,
        )

    def log_meal(
        self, meal_type: MealType, food_names: List[str], notes: str = ""
    ) -> Meal:
        """Log a meal that was eaten"""
        foods = []
        for name in food_names:
            food = self._food_database.get_food(name)
            if food:
                foods.append(food)

        total_nutrition = NutritionInfo()
        for food in foods:
            total_nutrition.calories += food.nutrition.calories
            total_nutrition.protein += food.nutrition.protein
            total_nutrition.carbohydrates += food.nutrition.carbohydrates
            total_nutrition.fat += food.nutrition.fat
            total_nutrition.fiber += food.nutrition.fiber

        meal = Meal(
            name=f"Logged {meal_type.value}",
            meal_type=meal_type,
            foods=foods,
            total_nutrition=total_nutrition,
            notes=notes,
        )

        self._learn_from_meal(meal)
        logger.info(
            f"Logged meal: {meal_type.value} with {total_nutrition.calories:.0f} cal"
        )

        return meal

    def _learn_from_meal(self, meal: Meal):
        """Learn from logged meal"""
        liked_foods = self._user_preferences.get("liked_foods", [])
        for food in meal.foods:
            if food.name not in liked_foods:
                liked_foods.append(food.name)

        self._user_preferences["liked_foods"] = liked_foods[:20]
        self._save_preferences()

    def get_meal_suggestion(
        self, meal_type: MealType, target_calories: float
    ) -> MealSuggestion:
        """Get a meal suggestion"""
        time_map = {
            MealType.BREAKFAST: "breakfast",
            MealType.LUNCH: "lunch",
            MealType.DINNER: "dinner",
        }

        time_of_day = time_map.get(meal_type, "lunch")
        meal = self._create_meal(meal_type, time_of_day, set(), target_calories)

        nutrition_match = 1.0 - (
            abs(meal.total_nutrition.calories - target_calories) / target_calories
        )

        return MealSuggestion(
            meal=meal,
            nutrition_match=max(0, nutrition_match),
            reason=f"Provides ~{meal.total_nutrition.calories:.0f} calories with {meal.total_nutrition.protein:.0f}g protein",
        )

    def search_food(self, query: str) -> List[FoodItem]:
        """Search for foods"""
        return self._food_database.search_foods(query)

    def add_custom_food(self, food: FoodItem):
        """Add a custom food to the database"""
        # Add to the food database
        self._food_database.add_food(food)
        logger.info(f"Added custom food: {food.name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get diet planner statistics"""
        return {
            "foods_in_database": len(self._food_database._foods),
            "meal_plans_created": len(self._meal_history),
            "user_preferences": self._user_preferences,
        }
