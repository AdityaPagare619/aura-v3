"""
Fitness Tracker
===============

Tracks workouts, suggests exercises, and monitors fitness progress.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from src.agents.healthcare.models import (
    Workout,
    WorkoutType,
    FitnessProgress,
    HealthProfile,
)

logger = logging.getLogger(__name__)


@dataclass
class Exercise:
    """A single exercise"""

    name: str
    muscle_groups: List[str]
    equipment: List[str]
    difficulty: str  # beginner, intermediate, advanced
    instructions: List[str]
    calories_per_minute: float


@dataclass
class WorkoutSuggestion:
    """A suggested workout"""

    workout: Workout
    match_score: float
    reason: str


@dataclass
class ProgressMetrics:
    """Fitness progress metrics"""

    weekly_workouts: int
    weekly_minutes: int
    weekly_calories: float
    streak_days: int
    improvement_percentage: float
    goal_progress: float


class ExerciseDatabase:
    """
    Local exercise database
    100% offline
    """

    def __init__(self):
        self._exercises: Dict[str, Exercise] = {}
        self._load_default_exercises()

    def _load_default_exercises(self):
        """Load default exercise database"""
        default_exercises = [
            Exercise(
                name="Push-ups",
                muscle_groups=["chest", "triceps", "shoulders"],
                equipment=["none"],
                difficulty="beginner",
                instructions=[
                    "Start in plank position",
                    "Lower body until chest nearly touches floor",
                    "Push back up to starting position",
                ],
                calories_per_minute=7.0,
            ),
            Exercise(
                name="Squats",
                muscle_groups=["quadriceps", "glutes", "hamstrings"],
                equipment=["none"],
                difficulty="beginner",
                instructions=[
                    "Stand with feet shoulder-width apart",
                    "Lower body as if sitting in a chair",
                    "Keep knees over toes",
                    "Return to standing",
                ],
                calories_per_minute=8.0,
            ),
            Exercise(
                name="Plank",
                muscle_groups=["core", "shoulders", "back"],
                equipment=["none"],
                difficulty="beginner",
                instructions=[
                    "Start in push-up position",
                    "Lower to forearms",
                    "Keep body in straight line",
                    "Hold position",
                ],
                calories_per_minute=5.0,
            ),
            Exercise(
                name="Lunges",
                muscle_groups=["quadriceps", "glutes", "hamstrings"],
                equipment=["none"],
                difficulty="beginner",
                instructions=[
                    "Step forward with one leg",
                    "Lower until both knees are 90 degrees",
                    "Push back to starting position",
                    "Alternate legs",
                ],
                calories_per_minute=6.0,
            ),
            Exercise(
                name="Burpees",
                muscle_groups=["full_body"],
                equipment=["none"],
                difficulty="intermediate",
                instructions=[
                    "Start standing",
                    "Drop to squat with hands on floor",
                    "Jump feet back to plank",
                    "Do a push-up",
                    "Jump feet forward",
                    "Jump up with arms overhead",
                ],
                calories_per_minute=10.0,
            ),
            Exercise(
                name="Mountain Climbers",
                muscle_groups=["core", "shoulders", "legs"],
                equipment=["none"],
                difficulty="intermediate",
                instructions=[
                    "Start in plank position",
                    "Drive one knee toward chest",
                    "Quickly switch legs",
                    "Continue alternating",
                ],
                calories_per_minute=9.0,
            ),
            Exercise(
                name="Jumping Jacks",
                muscle_groups=["full_body"],
                equipment=["none"],
                difficulty="beginner",
                instructions=[
                    "Stand with feet together, arms at sides",
                    "Jump while spreading legs and raising arms overhead",
                    "Return to starting position",
                    "Repeat",
                ],
                calories_per_minute=8.0,
            ),
            Exercise(
                name="High Knees",
                muscle_groups=["legs", "core"],
                equipment=["none"],
                difficulty="beginner",
                instructions=[
                    "Stand in place",
                    "Run lifting knees high",
                    "Pump arms as you run",
                    "Continue for time",
                ],
                calories_per_minute=9.0,
            ),
            Exercise(
                name="Dumbbell Rows",
                muscle_groups=["back", "biceps"],
                equipment=["dumbbells"],
                difficulty="beginner",
                instructions=[
                    "Place one knee and hand on bench",
                    "Hold dumbbell in other hand",
                    "Pull dumbbell to hip",
                    "Lower with control",
                ],
                calories_per_minute=6.0,
            ),
            Exercise(
                name="Dumbbell Press",
                muscle_groups=["chest", "triceps", "shoulders"],
                equipment=["dumbbells"],
                difficulty="beginner",
                instructions=[
                    "Lie on bench with dumbbells at chest",
                    "Press dumbbells up",
                    "Lower with control",
                ],
                calories_per_minute=6.0,
            ),
            Exercise(
                name="Running",
                muscle_groups=["legs", "core"],
                equipment=["none"],
                difficulty="beginner",
                instructions=[
                    "Start at comfortable pace",
                    "Maintain steady breathing",
                    "Keep core engaged",
                    "Land midfoot",
                ],
                calories_per_minute=10.0,
            ),
            Exercise(
                name="Walking",
                muscle_groups=["legs"],
                equipment=["none"],
                difficulty="beginner",
                instructions=[
                    "Start at brisk pace",
                    "Swing arms naturally",
                    "Maintain good posture",
                ],
                calories_per_minute=4.0,
            ),
            Exercise(
                name="Yoga Sun Salutation",
                muscle_groups=["full_body"],
                equipment=["yoga_mat"],
                difficulty="beginner",
                instructions=[
                    "Start in mountain pose",
                    "Raise arms overhead",
                    "Forward fold",
                    "Halfway lift",
                    "Plank",
                    "Chaturanga",
                    "Upward dog",
                    "Downward dog",
                ],
                calories_per_minute=4.0,
            ),
            Exercise(
                name="Cycling",
                muscle_groups=["legs", "glutes"],
                equipment=["bike"],
                difficulty="beginner",
                instructions=[
                    "Set comfortable resistance",
                    "Maintain steady cadence",
                    "Keep core engaged",
                ],
                calories_per_minute=8.0,
            ),
            Exercise(
                name="Jump Rope",
                muscle_groups=["full_body"],
                equipment=["jump_rope"],
                difficulty="intermediate",
                instructions=[
                    "Hold rope handles at hip height",
                    "Jump just off ground",
                    "Land softly",
                    "Keep jumps small",
                ],
                calories_per_minute=12.0,
            ),
        ]

        for exercise in default_exercises:
            self._exercises[exercise.name.lower()] = exercise

    def get_exercises(
        self, muscle_groups: List[str] = None, equipment: List[str] = None
    ) -> List[Exercise]:
        """Get exercises filtered by criteria"""
        results = list(self._exercises.values())

        if muscle_groups:
            results = [
                e for e in results if any(mg in e.muscle_groups for mg in muscle_groups)
            ]

        if equipment:
            results = [e for e in results if any(eq in e.equipment for eq in equipment)]

        return results

    def get_exercise(self, name: str) -> Optional[Exercise]:
        """Get a specific exercise"""
        return self._exercises.get(name.lower())

    def get_random_exercises(
        self, count: int = 5, difficulty: str = None
    ) -> List[Exercise]:
        """Get random exercises"""
        exercises = list(self._exercises.values())

        if difficulty:
            exercises = [e for e in exercises if e.difficulty == difficulty]

        return exercises[:count]


class FitnessTracker:
    """
    Tracks workouts and monitors fitness progress

    Features:
    - Log workouts
    - Track progress
    - Suggest exercises
    - Create workout plans
    """

    def __init__(self, storage_path: str = "data/healthcare"):
        self.storage_path = storage_path
        self._exercise_db = ExerciseDatabase()
        self._workouts: List[Workout] = []
        self._progress_history: List[FitnessProgress] = []
        os.makedirs(storage_path, exist_ok=True)
        self._load_data()

    def _load_data(self):
        """Load workout data"""
        workouts_file = os.path.join(self.storage_path, "workouts.json")
        if os.path.exists(workouts_file):
            try:
                with open(workouts_file) as f:
                    data = json.load(f)
                    for w in data.get("workouts", []):
                        workout = Workout(
                            id=w.get("id", ""),
                            workout_type=WorkoutType(w.get("workout_type", "other")),
                            name=w.get("name", ""),
                            duration_minutes=w.get("duration_minutes", 0),
                            intensity=w.get("intensity", "moderate"),
                            calories_burned=w.get("calories_burned", 0),
                            heart_rate_avg=w.get("heart_rate_avg", 0),
                            heart_rate_max=w.get("heart_rate_max", 0),
                            distance=w.get("distance", 0),
                            timestamp=datetime.fromisoformat(
                                w.get("timestamp", datetime.now().isoformat())
                            ),
                            notes=w.get("notes", ""),
                        )
                        self._workouts.append(workout)
            except Exception as e:
                logger.error(f"Failed to load workouts: {e}")

    def _save_data(self):
        """Save workout data"""
        workouts_file = os.path.join(self.storage_path, "workouts.json")
        try:
            workouts_data = [
                {
                    "id": w.id,
                    "workout_type": w.workout_type.value,
                    "name": w.name,
                    "duration_minutes": w.duration_minutes,
                    "intensity": w.intensity,
                    "calories_burned": w.calories_burned,
                    "heart_rate_avg": w.heart_rate_avg,
                    "heart_rate_max": w.heart_rate_max,
                    "distance": w.distance,
                    "timestamp": w.timestamp.isoformat(),
                    "notes": w.notes,
                }
                for w in self._workouts
            ]
            with open(workouts_file, "w") as f:
                json.dump({"workouts": workouts_data}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save workouts: {e}")

    def log_workout(
        self,
        workout_type: WorkoutType,
        duration_minutes: int,
        intensity: str = "moderate",
        calories_burned: float = 0.0,
        heart_rate_avg: float = 0.0,
        heart_rate_max: float = 0.0,
        distance: float = 0.0,
        notes: str = "",
        exercises: List[Dict[str, Any]] = None,
    ) -> Workout:
        """Log a completed workout"""

        if calories_burned == 0:
            calories_burned = self._estimate_calories(
                workout_type, duration_minutes, intensity
            )

        workout = Workout(
            workout_type=workout_type,
            name=f"{workout_type.value.replace('_', ' ').title()} Workout",
            duration_minutes=duration_minutes,
            intensity=intensity,
            calories_burned=calories_burned,
            heart_rate_avg=heart_rate_avg,
            heart_rate_max=heart_rate_max,
            distance=distance,
            notes=notes,
            exercises=exercises or [],
        )

        self._workouts.append(workout)
        self._save_data()

        logger.info(
            f"Logged workout: {workout.name} - {duration_minutes} min, {calories_burned:.0f} cal"
        )

        return workout

    def _estimate_calories(
        self, workout_type: WorkoutType, duration: int, intensity: str
    ) -> float:
        """Estimate calories burned"""
        base_calories_per_minute = {
            WorkoutType.RUNNING: 10.0,
            WorkoutType.WALKING: 4.0,
            WorkoutType.CYCLING: 8.0,
            WorkoutType.SWIMMING: 9.0,
            WorkoutType.STRENGTH_TRAINING: 6.0,
            WorkoutType.YOGA: 4.0,
            WorkoutType.HIIT: 11.0,
            WorkoutType.STRETCHING: 3.0,
            WorkoutType.SPORTS: 8.0,
            WorkoutType.OTHER: 5.0,
        }

        intensity_multiplier = {"low": 0.7, "moderate": 1.0, "high": 1.3}

        base = base_calories_per_minute.get(workout_type, 5.0)
        multiplier = intensity_multiplier.get(intensity, 1.0)

        return base * duration * multiplier

    def get_workouts_for_date(self, target_date: date) -> List[Workout]:
        """Get all workouts for a specific date"""
        return [w for w in self._workouts if w.timestamp.date() == target_date]

    def get_workout_trend(self, days: int = 7) -> List[Workout]:
        """Get workouts for the last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        return [w for w in self._workouts if w.timestamp >= cutoff]

    def get_progress(self, days: int = 7) -> ProgressMetrics:
        """Get fitness progress"""
        workouts = self.get_workout_trend(days)

        weekly_workouts = len(workouts)
        weekly_minutes = sum(w.duration_minutes for w in workouts)
        weekly_calories = sum(w.calories_burned for w in workouts)

        streak = self._calculate_streak()

        cutoff = datetime.now() - timedelta(days=days)
        older_cutoff = cutoff - timedelta(days=days)
        older_workouts = [
            w for w in self._workouts if older_cutoff <= w.timestamp < cutoff
        ]

        if older_workouts:
            older_minutes = sum(w.duration_minutes for w in older_workouts)
            if older_minutes > 0:
                improvement = ((weekly_minutes - older_minutes) / older_minutes) * 100
            else:
                improvement = 100.0 if weekly_minutes > 0 else 0.0
        else:
            improvement = 100.0 if weekly_workouts > 0 else 0.0

        goal_progress = min(1.0, weekly_workouts / (days / 2))

        return ProgressMetrics(
            weekly_workouts=weekly_workouts,
            weekly_minutes=weekly_minutes,
            weekly_calories=weekly_calories,
            streak_days=streak,
            improvement_percentage=improvement,
            goal_progress=goal_progress,
        )

    def _calculate_streak(self) -> int:
        """Calculate current workout streak"""
        if not self._workouts:
            return 0

        sorted_workouts = sorted(
            self._workouts, key=lambda w: w.timestamp, reverse=True
        )

        streak = 0
        check_date = date.today()

        for workout in sorted_workouts:
            workout_date = workout.timestamp.date()

            if workout_date == check_date:
                streak += 1
                check_date -= timedelta(days=1)
            elif workout_date < check_date:
                if workout_date == check_date - timedelta(days=1):
                    streak += 1
                    check_date = workout_date
                else:
                    break

        return streak

    def suggest_workout(
        self,
        workout_type: WorkoutType = None,
        duration: int = 30,
        difficulty: str = None,
    ) -> WorkoutSuggestion:
        """Suggest a workout"""
        if workout_type:
            exercises = self._exercise_db.get_exercises()
            exercises = [
                e for e in exercises if e.difficulty in ["beginner", "intermediate"]
            ]
        else:
            exercises = self._exercise_db.get_random_exercises(difficulty=difficulty)

        if not exercises:
            exercises = self._exercise_db.get_random_exercises(5)

        exercise_list = [
            {
                "name": e.name,
                "sets": 3 if difficulty != "beginner" else 2,
                "reps": "12-15" if e.difficulty == "beginner" else "8-12",
            }
            for e in exercises[:5]
        ]

        total_calories = sum(
            e.calories_per_minute * duration / len(exercise_list)
            for e in exercises[: len(exercise_list)]
        )

        workout = Workout(
            workout_type=workout_type or WorkoutType.OTHER,
            name="Custom Workout",
            duration_minutes=duration,
            exercises=exercise_list,
        )

        return WorkoutSuggestion(
            workout=workout,
            match_score=0.8,
            reason=f"Full body workout with {len(exercise_list)} exercises targeting all major muscle groups",
        )

    def get_exercise_suggestions(
        self, muscle_groups: List[str] = None, equipment: List[str] = None
    ) -> List[Exercise]:
        """Get exercise suggestions"""
        return self._exercise_db.get_exercises(muscle_groups, equipment)

    def get_personal_bests(self) -> Dict[str, Any]:
        """Get personal best records"""
        if not self._workouts:
            return {}

        by_type = defaultdict(list)
        for w in self._workouts:
            by_type[w.workout_type].append(w)

        pbs = {}

        for wtype, workouts in by_type.items():
            if workouts:
                longest = max(workouts, key=lambda w: w.duration_minutes)
                most_calories = max(workouts, key=lambda w: w.calories_burned)

                pbs[wtype.value] = {
                    "longest_workout": {
                        "duration": longest.duration_minutes,
                        "date": longest.timestamp.isoformat(),
                    },
                    "most_calories": {
                        "calories": most_calories.calories_burned,
                        "date": most_calories.timestamp.isoformat(),
                    },
                }

        return pbs

    def get_stats(self) -> Dict[str, Any]:
        """Get fitness tracker statistics"""
        return {
            "total_workouts": len(self._workouts),
            "total_minutes": sum(w.duration_minutes for w in self._workouts),
            "total_calories": sum(w.calories_burned for w in self._workouts),
            "current_streak": self._calculate_streak(),
            "personal_bests": self.get_personal_bests(),
        }
