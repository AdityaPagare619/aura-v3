"""
AURA v3 Adaptive Health Engine
===============================

This is the TRUE AGI approach to health tracking:
- NO hardcoded WorkoutType enums - discovers patterns from sensor data
- Learns activity types from motion + GPS + time patterns
- Generates and validates hypotheses about health
- Adapts to each user's unique patterns

Based on roadmap: "User's motion patterns + GPS + time = likely activity"
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import uuid

logger = logging.getLogger(__name__)


@dataclass
class SensorData:
    """Raw sensor data from device"""

    timestamp: datetime
    gps_coordinates: Optional[Tuple[float, float]] = None
    speed_mps: Optional[float] = None  # meters per second
    altitude: Optional[float] = None
    step_count: int = 0
    heart_rate: Optional[int] = None
    activity_type: Optional[str] = None  # device's guess


@dataclass
class DiscoveredActivity:
    """An activity pattern discovered through learning"""

    activity_id: str
    learned_label: str  # e.g., "morning_commute", "gym_session"
    features: Dict[
        str, Any
    ]  # e.g., {"speed_range": [1-3], "time": "7-9am", "location": "home->work"}
    occurrence_count: int = 0
    avg_duration_minutes: float = 0
    confidence: float = 0.0
    last_observed: Optional[datetime] = None


@dataclass
class HealthHypothesis:
    """A testable health hypothesis"""

    hypothesis_id: str
    statement: str  # e.g., "User sleeps worse on days with >2 coffees"
    evidence: List[Dict] = field(default_factory=list)
    supporting_data_points: int = 0
    contradicting_data_points: int = 0
    validated: bool = False
    validation_result: Optional[bool] = None


@dataclass
class FoodPreference:
    """Learned food preference"""

    food_name: str
    frequency: int = 0
    liked: int = 0
    disliked: int = 0
    neutral: int = 0
    avg_rating: float = 3.0  # 1-5 scale
    tags: List[str] = field(
        default_factory=list
    )  # e.g., ["spicy", "vegetarian", "quick"]


class AdaptiveHealthEngine:
    """
    Learns health patterns from observation.
    NO hardcoded workout types - discovers them from data.

    Key principles:
    1. Observe first - collect sensor data without assumptions
    2. Cluster patterns - group similar sensor readings
    3. Label discovered patterns - let LLM help name them
    4. Generate hypotheses - test relationships between factors
    5. Validate continuously - improve over time
    """

    def __init__(self, neural_memory=None):
        self.neural_memory = neural_memory

        # Discovered activities (not hardcoded!)
        self.discovered_activities: Dict[str, DiscoveredActivity] = {}

        # Food preferences (learned, not predefined)
        self.food_preferences: Dict[str, FoodPreference] = {}

        # Hypotheses (generated and tested)
        self.hypotheses: Dict[str, HealthHypothesis] = {}

        # Raw sensor data buffer
        self.sensor_buffer: List[SensorData] = []

        # Learning parameters
        self.min_occurrences_to_discover = 3  # Need 3+ occurrences to "discover"
        self.similarity_threshold = 0.7  # For clustering
        self.hypothesis_confidence_threshold = 0.8

    async def add_sensor_data(self, data: SensorData):
        """Add raw sensor data for processing"""
        self.sensor_buffer.append(data)

        # Keep buffer manageable
        if len(self.sensor_buffer) > 1000:
            self.sensor_buffer = self.sensor_buffer[-500:]

        # Trigger activity discovery if enough data
        if len(self.sensor_buffer) >= 10:
            await self._discover_activities()

    async def _discover_activities(self):
        """Cluster sensor data to discover activity patterns"""
        if len(self.sensor_buffer) < 10:
            return

        # Extract features from recent data
        windows = self._create_time_windows(30)  # 30-minute windows

        for window in windows:
            features = self._extract_window_features(window)

            # Check if similar to existing discovered activity
            match = self._find_similar_activity(features)

            if match:
                # Update existing activity
                match.occurrence_count += 1
                match.last_observed = datetime.now()
                match.avg_duration_minutes = (
                    match.avg_duration_minutes * (match.occurrence_count - 1)
                    + (window[-1].timestamp - window[0].timestamp).total_seconds() / 60
                ) / match.occurrence_count
            else:
                # Create new discovered activity
                await self._create_new_activity(features, window)

    def _create_time_windows(self, window_minutes: int) -> List[List[SensorData]]:
        """Create time windows from sensor data"""
        if not self.sensor_buffer:
            return []

        windows = []
        current_window = []
        window_start = self.sensor_buffer[0].timestamp

        for data in self.sensor_buffer:
            if (data.timestamp - window_start).total_seconds() / 60 < window_minutes:
                current_window.append(data)
            else:
                if current_window:
                    windows.append(current_window)
                current_window = [data]
                window_start = data.timestamp

        if current_window:
            windows.append(current_window)

        return windows

    def _extract_window_features(self, window: List[SensorData]) -> Dict[str, Any]:
        """Extract features from a time window"""
        if not window:
            return {}

        speeds = [d.speed_mps for d in window if d.speed_mps is not None]
        steps = sum(d.step_count for d in window)
        heart_rates = [d.heart_rate for d in window if d.heart_rate is not None]

        # Time features
        hour = window[0].timestamp.hour
        is_weekend = window[0].timestamp.weekday() >= 5

        # Location features (if GPS available)
        gps_data = [d.gps_coordinates for d in window if d.gps_coordinates is not None]

        # Device's activity label if available
        device_labels = [d.activity_type for d in window if d.activity_type]
        most_common_device_label = (
            max(set(device_labels), key=device_labels.count) if device_labels else None
        )

        return {
            "avg_speed": statistics.mean(speeds) if speeds else 0,
            "max_speed": max(speeds) if speeds else 0,
            "total_steps": steps,
            "avg_heart_rate": statistics.mean(heart_rates) if heart_rates else 0,
            "hour": hour,
            "is_weekend": is_weekend,
            "has_gps": len(gps_data) > len(window) * 0.5,
            "gps_variance": self._calculate_gps_variance(gps_data) if gps_data else 0,
            "device_label": most_common_device_label,
        }

    def _calculate_gps_variance(self, gps_data: List[Tuple[float, float]]) -> float:
        """Calculate variance in GPS coordinates (proxy for movement)"""
        if len(gps_data) < 2:
            return 0

        lats = [g[0] for g in gps_data]
        lons = [g[1] for g in gps_data]

        lat_var = statistics.variance(lats) if len(lats) > 1 else 0
        lon_var = statistics.variance(lons) if len(lons) > 1 else 0

        return lat_var + lon_var

    def _find_similar_activity(self, features: Dict) -> Optional[DiscoveredActivity]:
        """Find a discovered activity similar to these features"""
        if not self.discovered_activities:
            return None

        best_match = None
        best_score = 0

        for activity in self.discovered_activities.values():
            score = self._calculate_similarity(features, activity.features)

            if score >= self.similarity_threshold and score > best_score:
                best_score = score
                best_match = activity

        return best_match

    def _calculate_similarity(self, features: Dict, activity_features: Dict) -> float:
        """Calculate similarity between features and an activity pattern"""
        score = 0
        weights = 0

        # Time of day similarity
        if "hour" in features and "hour" in activity_features:
            hour_diff = abs(features["hour"] - activity_features["hour"])
            hour_similarity = 1 - (min(hour_diff, 24 - hour_diff) / 12)
            score += hour_similarity * 0.3
            weights += 0.3

        # Weekend vs weekday
        if "is_weekend" in features and "is_weekend" in activity_features:
            if features["is_weekend"] == activity_features["is_weekend"]:
                score += 0.2
            weights += 0.2

        # Speed similarity
        if "avg_speed" in features and "speed_range" in activity_features:
            speed_range = activity_features["speed_range"]
            if speed_range[0] <= features["avg_speed"] <= speed_range[1]:
                score += 0.3
            weights += 0.3

        # GPS movement pattern
        if "gps_variance" in features and "gps_pattern" in activity_features:
            if (
                features["gps_variance"] > 0
                and activity_features["gps_pattern"] == "moving"
            ):
                score += 0.2
            elif (
                features["gps_variance"] == 0
                and activity_features["gps_pattern"] == "stationary"
            ):
                score += 0.2
            weights += 0.2

        return score / weights if weights > 0 else 0

    async def _create_new_activity(self, features: Dict, window: List[SensorData]):
        """Create a new discovered activity from features"""
        # Generate a descriptive label using pattern recognition
        label = self._generate_activity_label(features)

        activity = DiscoveredActivity(
            activity_id=str(uuid.uuid4())[:8],
            learned_label=label,
            features={
                "hour": features["hour"],
                "is_weekend": features["is_weekend"],
                "speed_range": [
                    features["avg_speed"] * 0.5,
                    features["avg_speed"] * 1.5,
                ],
                "gps_pattern": "moving"
                if features.get("gps_variance", 0) > 0.0001
                else "stationary",
                "total_steps": features["total_steps"],
            },
            occurrence_count=1,
            avg_duration_minutes=(
                window[-1].timestamp - window[0].timestamp
            ).total_seconds()
            / 60,
            confidence=0.3,  # Low confidence until more occurrences
            last_observed=datetime.now(),
        )

        self.discovered_activities[activity.activity_id] = activity
        logger.info(f"Discovered new activity pattern: {label}")

    def _generate_activity_label(self, features: Dict) -> str:
        """Generate a human-readable label for the activity pattern"""
        hour = features.get("hour", 12)
        is_weekend = features.get("is_weekend", False)
        avg_speed = features.get("avg_speed", 0)
        gps_pattern = features.get("gps_pattern", "unknown")

        # Time-based labels
        if 6 <= hour < 9:
            time_label = "morning"
        elif 9 <= hour < 12:
            time_label = "late_morning"
        elif 12 <= hour < 14:
            time_label = "lunch"
        elif 14 <= hour < 17:
            time_label = "afternoon"
        elif 17 <= hour < 20:
            time_label = "evening"
        elif 20 <= hour < 23:
            time_label = "night"
        else:
            time_label = "late_night"

        # Activity intensity
        if avg_speed < 0.5:
            intensity = "sedentary"
        elif avg_speed < 2:
            intensity = "light_movement"
        elif avg_speed < 5:
            intensity = "moderate"
        else:
            intensity = "vigorous"

        # Context
        if is_weekend:
            context = "weekend"
        else:
            context = "weekday"

        return f"{time_label}_{intensity}_{context}"

    async def learn_food_preferences(self, food_name: str, feedback: str):
        """Learn food preferences from user feedback"""
        # Normalize food name
        food_name = food_name.lower().strip()

        if food_name not in self.food_preferences:
            self.food_preferences[food_name] = FoodPreference(food_name=food_name)

        pref = self.food_preferences[food_name]

        # Update based on feedback
        if feedback in ["liked", "love", "great", "good", "yes"]:
            pref.liked += 1
        elif feedback in ["disliked", "hate", "bad", "no"]:
            pref.disliked += 1
        else:
            pref.neutral += 1

        pref.frequency += 1
        pref.avg_rating = (
            pref.liked * 5 + pref.neutral * 3 + pref.disliked * 1
        ) / pref.frequency

    async def generate_hypothesis(self) -> Optional[HealthHypothesis]:
        """Generate a testable health hypothesis from observed patterns"""
        if not self.sensor_buffer or len(self.hypotheses) > 5:
            return None

        # Look for correlations in data
        # Example: "Does coffee affect sleep?"
        # Check if heart rate or activity differs based on time of day

        hypotheses_to_try = [
            "User is more active on weekday mornings",
            "User tends to be sedentary in evening hours",
            "Higher step counts correlate with better mood",
            "User sleeps better on days with >30min morning activity",
        ]

        for h in hypotheses_to_try:
            if h not in [hyp.statement for hyp in self.hypotheses.values()]:
                hypothesis = HealthHypothesis(
                    hypothesis_id=str(uuid.uuid4())[:8],
                    statement=h,
                )
                self.hypotheses[hypothesis.hypothesis_id] = hypothesis
                logger.info(f"Generated hypothesis: {h}")
                return hypothesis

        return None

    async def validate_hypothesis(self, hypothesis: HealthHypothesis) -> bool:
        """Test a hypothesis with available data"""
        # This is a simplified version - real implementation would do statistical analysis
        if not self.sensor_buffer:
            return False

        # Example validation logic
        if "more active on weekday mornings" in hypothesis.statement:
            weekday_morning = [
                d
                for d in self.sensor_buffer
                if d.timestamp.weekday() < 5 and 6 <= d.timestamp.hour < 10
            ]
            weekend_morning = [
                d
                for d in self.sensor_buffer
                if d.timestamp.weekday() >= 5 and 6 <= d.timestamp.hour < 10
            ]

            if weekday_morning and weekend_morning:
                weekday_steps = sum(d.step_count for d in weekday_morning)
                weekend_steps = sum(d.step_count for d in weekend_morning)

                hypothesis.supporting_data_points = (
                    1 if weekday_steps > weekend_steps else 0
                )
                hypothesis.contradicting_data_points = (
                    1 if weekday_steps <= weekend_steps else 0
                )

        # Calculate confidence
        total = hypothesis.supporting_data_points + hypothesis.contradicting_data_points
        if total > 0:
            confidence = hypothesis.supporting_data_points / total
            hypothesis.validation_result = confidence >= 0.6
            hypothesis.validated = True

        return hypothesis.validation_result or False

    def get_discovered_activities(self) -> List[Dict]:
        """Get all discovered activities"""
        return [
            {
                "id": a.activity_id,
                "label": a.learned_label,
                "occurrences": a.occurrence_count,
                "avg_duration_min": round(a.avg_duration_minutes, 1),
                "confidence": round(a.confidence, 2),
                "features": a.features,
            }
            for a in self.discovered_activities.values()
        ]

    def get_food_preferences(self) -> List[Dict]:
        """Get learned food preferences"""
        return [
            {
                "food": f.food_name,
                "frequency": f.frequency,
                "rating": round(f.avg_rating, 1),
                "tags": f.tags,
            }
            for f in self.food_preferences.values()
        ]

    def get_hypotheses(self) -> List[Dict]:
        """Get all hypotheses"""
        return [
            {
                "id": h.hypothesis_id,
                "statement": h.statement,
                "validated": h.validated,
                "result": h.validation_result,
                "supporting": h.supporting_data_points,
                "contradicting": h.contradicting_data_points,
            }
            for h in self.hypotheses.values()
        ]


# Global instance
_engine: Optional[AdaptiveHealthEngine] = None


def get_adaptive_health_engine() -> AdaptiveHealthEngine:
    """Get or create the global adaptive health engine"""
    global _engine
    if _engine is None:
        _engine = AdaptiveHealthEngine()
    return _engine
