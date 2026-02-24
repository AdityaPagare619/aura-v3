"""
Health Data Analyzer
====================

Parses and analyzes health metrics from various sources:
- Manual user input
- Health apps (via app discovery)
- Connected devices

Provides trend analysis, anomaly detection, and data validation.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

from src.agents.healthcare.models import (
    HealthMetric,
    HealthDataSet,
    MetricType,
    HealthProfile,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricAnalysis:
    """Analysis result for a single metric"""

    metric_type: MetricType
    current_value: float
    average_value: float
    trend: str  # increasing, decreasing, stable
    trend_percentage: float
    min_value: float
    max_value: float
    standard_deviation: float
    anomaly_detected: bool
    anomaly_message: str = ""
    comparison_to_goal: float = 0.0  # -1 to 1, how close to goal


@dataclass
class HealthSummary:
    """Overall health summary"""

    date: date
    total_steps: float
    average_heart_rate: float
    sleep_hours: float
    calories_consumed: float
    calories_burned: float
    water_intake_ml: float
    active_minutes: int
    score: float  # 0-100 overall health score


class HealthDataAnalyzer:
    """
    Analyzes health data from various sources

    Features:
    - Parse metrics from different formats
    - Validate data quality
    - Detect trends
    - Identify anomalies
    - Calculate health scores
    """

    def __init__(self, storage_path: str = "data/healthcare"):
        self.storage_path = storage_path
        self._health_data: Dict[str, HealthDataSet] = {}
        self._metric_history: Dict[MetricType, List[HealthMetric]] = defaultdict(list)
        self._profile: Optional[HealthProfile] = None
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, "metrics"), exist_ok=True)
        self._load_data()

    def _load_data(self):
        """Load stored health data"""
        metrics_file = os.path.join(self.storage_path, "metrics", "history.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file) as f:
                    data = json.load(f)
                    for metric_data in data.get("metrics", []):
                        metric = HealthMetric(
                            id=metric_data.get("id", ""),
                            metric_type=MetricType(
                                metric_data.get("metric_type", "steps")
                            ),
                            value=metric_data.get("value", 0),
                            unit=metric_data.get("unit", ""),
                            timestamp=datetime.fromisoformat(
                                metric_data.get("timestamp", datetime.now().isoformat())
                            ),
                            source=metric_data.get("source", "manual"),
                        )
                        self._metric_history[metric.metric_type].append(metric)
            except Exception as e:
                logger.error(f"Failed to load health data: {e}")

        profile_file = os.path.join(self.storage_path, "profile.json")
        if os.path.exists(profile_file):
            try:
                with open(profile_file) as f:
                    data = json.load(f)
                    self._profile = HealthProfile(**data)
            except Exception as e:
                logger.error(f"Failed to load health profile: {e}")

    def _save_data(self):
        """Save health data"""
        metrics_file = os.path.join(self.storage_path, "metrics", "history.json")
        try:
            metrics_list = []
            for metric_type, metrics in self._metric_history.items():
                for m in metrics:
                    metrics_list.append(
                        {
                            "id": m.id,
                            "metric_type": m.metric_type.value,
                            "value": m.value,
                            "unit": m.unit,
                            "timestamp": m.timestamp.isoformat(),
                            "source": m.source,
                        }
                    )
            with open(metrics_file, "w") as f:
                json.dump({"metrics": metrics_list}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save health data: {e}")

        if self._profile:
            profile_file = os.path.join(self.storage_path, "profile.json")
            with open(profile_file, "w") as f:
                json.dump(
                    {
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
                    },
                    f,
                    indent=2,
                )

    def set_profile(self, profile: HealthProfile):
        """Set user health profile"""
        self._profile = profile
        profile.calculate_tdee()
        self._save_data()
        logger.info(f"Health profile set for user {profile.user_id}")

    def add_metric(
        self,
        metric_type: MetricType,
        value: float,
        unit: str = "",
        source: str = "manual",
        timestamp: datetime = None,
    ) -> HealthMetric:
        """Add a new health metric"""
        metric = HealthMetric(
            metric_type=metric_type,
            value=value,
            unit=unit,
            source=source,
            timestamp=timestamp or datetime.now(),
        )

        self._metric_history[metric_type].append(metric)
        self._save_data()

        logger.info(f"Added metric: {metric_type.value} = {value} {unit}")
        return metric

    def add_metrics_batch(self, metrics: List[HealthMetric]):
        """Add multiple metrics at once"""
        for metric in metrics:
            self._metric_history[metric.metric_type].append(metric)
        self._save_data()
        logger.info(f"Added {len(metrics)} metrics")

    def get_metrics_for_date(
        self, target_date: date, metric_type: MetricType = None
    ) -> List[HealthMetric]:
        """Get all metrics for a specific date"""
        results = []
        metric_types = [metric_type] if metric_type else list(MetricType)

        for mtype in metric_types:
            for metric in self._metric_history.get(mtype, []):
                if metric.timestamp.date() == target_date:
                    results.append(metric)

        return results

    def get_metric_trend(
        self, metric_type: MetricType, days: int = 7
    ) -> List[HealthMetric]:
        """Get metric trend over specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        return [
            m
            for m in self._metric_history.get(metric_type, [])
            if m.timestamp >= cutoff
        ]

    def analyze_metric(self, metric_type: MetricType, days: int = 7) -> MetricAnalysis:
        """Analyze a specific metric"""
        metrics = self.get_metric_trend(metric_type, days)

        if not metrics:
            return MetricAnalysis(
                metric_type=metric_type,
                current_value=0.0,
                average_value=0.0,
                trend="stable",
                trend_percentage=0.0,
                min_value=0.0,
                max_value=0.0,
                standard_deviation=0.0,
                anomaly_detected=False,
            )

        values = [m.value for m in metrics]
        current_value = values[-1] if values else 0.0
        average_value = statistics.mean(values)
        min_value = min(values)
        max_value = max(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0

        if len(values) >= 3:
            recent_avg = statistics.mean(values[-3:])
            older_avg = statistics.mean(values[:3])
            if older_avg > 0:
                trend_percentage = ((recent_avg - older_avg) / older_avg) * 100
            else:
                trend_percentage = 0.0

            if abs(trend_percentage) < 5:
                trend = "stable"
            elif trend_percentage > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
        else:
            trend = "stable"
            trend_percentage = 0.0

        anomaly_detected = False
        anomaly_message = ""
        if std_dev > 0 and abs(current_value - average_value) > 2 * std_dev:
            anomaly_detected = True
            if current_value > average_value:
                anomaly_message = f"Unusually high {metric_type.value} detected"
            else:
                anomaly_message = f"Unusually low {metric_type.value} detected"

        comparison_to_goal = self._calculate_goal_comparison(metric_type, current_value)

        return MetricAnalysis(
            metric_type=metric_type,
            current_value=current_value,
            average_value=average_value,
            trend=trend,
            trend_percentage=trend_percentage,
            min_value=min_value,
            max_value=max_value,
            standard_deviation=std_dev,
            anomaly_detected=anomaly_detected,
            anomaly_message=anomaly_message,
            comparison_to_goal=comparison_to_goal,
        )

    def _calculate_goal_comparison(
        self, metric_type: MetricType, current_value: float
    ) -> float:
        """Calculate how close current value is to goal (-1 to 1)"""
        if not self._profile:
            return 0.0

        targets = {
            MetricType.STEPS: 10000,
            MetricType.SLEEP_DURATION: 8.0,
            MetricType.WATER_INTAKE: 2500,
            MetricType.ACTIVE_MINUTES: 30,
        }

        target = targets.get(metric_type)
        if target is None:
            return 0.0

        if current_value >= target:
            return 1.0
        elif current_value <= 0:
            return -1.0
        else:
            return (current_value / target) * 2 - 1

    def get_daily_summary(self, target_date: date = None) -> HealthSummary:
        """Get health summary for a day"""
        target_date = target_date or date.today()
        metrics = self.get_metrics_for_date(target_date)

        total_steps = 0
        heart_rates = []
        sleep_hours = 0
        calories_consumed = 0
        calories_burned = 0
        water_intake = 0
        active_minutes = 0

        for metric in metrics:
            if metric.metric_type == MetricType.STEPS:
                total_steps += metric.value
            elif metric.metric_type == MetricType.HEART_RATE:
                heart_rates.append(metric.value)
            elif metric.metric_type == MetricType.SLEEP_DURATION:
                sleep_hours = max(sleep_hours, metric.value)
            elif metric.metric_type == MetricType.CALORIES_CONSUMED:
                calories_consumed += metric.value
            elif metric.metric_type == MetricType.CALORIES_BURNED:
                calories_burned += metric.value
            elif metric.metric_type == MetricType.WATER_INTAKE:
                water_intake += metric.value
            elif metric.metric_type == MetricType.ACTIVE_MINUTES:
                active_minutes += int(metric.value)

        avg_hr = statistics.mean(heart_rates) if heart_rates else 0.0
        score = self._calculate_health_score(
            total_steps, sleep_hours, water_intake, active_minutes
        )

        return HealthSummary(
            date=target_date,
            total_steps=total_steps,
            average_heart_rate=avg_hr,
            sleep_hours=sleep_hours,
            calories_consumed=calories_consumed,
            calories_burned=calories_burned,
            water_intake_ml=water_intake,
            active_minutes=active_minutes,
            score=score,
        )

    def _calculate_health_score(
        self, steps: float, sleep: float, water: float, active: int
    ) -> float:
        """Calculate overall health score (0-100)"""
        score = 0.0

        if steps >= 10000:
            score += 25
        elif steps >= 5000:
            score += 15
        elif steps >= 3000:
            score += 10
        else:
            score += 5

        if 7 <= sleep <= 9:
            score += 25
        elif 6 <= sleep < 7 or 9 < sleep <= 10:
            score += 20
        elif 5 <= sleep < 6:
            score += 10
        else:
            score += 5

        if water >= 2500:
            score += 25
        elif water >= 2000:
            score += 20
        elif water >= 1500:
            score += 15
        else:
            score += 10

        if active >= 30:
            score += 25
        elif active >= 20:
            score += 20
        elif active >= 10:
            score += 15
        else:
            score += 10

        return min(100.0, score)

    def get_all_analysis(self, days: int = 7) -> Dict[str, MetricAnalysis]:
        """Get analysis for all tracked metrics"""
        results = {}
        for metric_type in MetricType:
            if self._metric_history.get(metric_type):
                results[metric_type.value] = self.analyze_metric(metric_type, days)
        return results

    def validate_metric(
        self, metric_type: MetricType, value: float
    ) -> Tuple[bool, str]:
        """Validate a metric value"""
        validations = {
            MetricType.STEPS: (0, 100000, "steps"),
            MetricType.HEART_RATE: (30, 250, "bpm"),
            MetricType.SLEEP_DURATION: (0, 24, "hours"),
            MetricType.CALORIES_CONSUMED: (0, 10000, "kcal"),
            MetricType.CALORIES_BURNED: (0, 5000, "kcal"),
            MetricType.WATER_INTAKE: (0, 10000, "ml"),
            MetricType.WEIGHT: (20, 500, "kg"),
            MetricType.BLOOD_PRESSURE_SYSTOLIC: (60, 250, "mmHg"),
            MetricType.BLOOD_PRESSURE_DIASTOLIC: (40, 150, "mmHg"),
            MetricType.BLOOD_OXYGEN: (50, 100, "%"),
            MetricType.ACTIVE_MINUTES: (0, 1440, "minutes"),
        }

        if metric_type not in validations:
            return True, ""

        min_val, max_val, unit = validations[metric_type]

        if value < min_val:
            return False, f"Value too low. Minimum is {min_val} {unit}"
        if value > max_val:
            return False, f"Value too high. Maximum is {max_val} {unit}"

        return True, ""

    def parse_health_export(self, data: Dict) -> List[HealthMetric]:
        """Parse health data from various export formats"""
        metrics = []

        if "apple_health" in data:
            for item in data.get("apple_health", []):
                metric_type = self._map_apple_health_type(item.get("type", ""))
                if metric_type:
                    metrics.append(
                        HealthMetric(
                            metric_type=metric_type,
                            value=item.get("value", 0),
                            unit=item.get("unit", ""),
                            timestamp=datetime.fromisoformat(item.get("date", "")),
                            source="apple_health",
                        )
                    )

        elif "google_fit" in data:
            for bucket in data.get("google_fit", []):
                for point in bucket.get("point", []):
                    metric_type = self._map_google_fit_type(
                        point.get("dataTypeName", "")
                    )
                    if metric_type:
                        metrics.append(
                            HealthMetric(
                                metric_type=metric_type,
                                value=point.get("value", [{}])[0].get("fpVal", 0),
                                timestamp=datetime.fromisoformat(
                                    point.get("startTime", "")
                                ),
                                source="google_fit",
                            )
                        )

        return metrics

    def _map_apple_health_type(self, apple_type: str) -> Optional[MetricType]:
        """Map Apple Health export types to MetricType"""
        mapping = {
            "HKQuantityTypeIdentifierStepCount": MetricType.STEPS,
            "HKQuantityTypeIdentifierHeartRate": MetricType.HEART_RATE,
            "HKQuantityTypeIdentifierSleepAnalysis": MetricType.SLEEP_DURATION,
            "HKQuantityTypeIdentifierActiveEnergyBurned": MetricType.CALORIES_BURNED,
            "HKQuantityTypeIdentifierDietaryEnergyConsumed": MetricType.CALORIES_CONSUMED,
            "HKQuantityTypeIdentifierDietaryWater": MetricType.WATER_INTAKE,
            "HKQuantityTypeIdentifierBodyMass": MetricType.WEIGHT,
            "HKQuantityTypeIdentifierDistanceWalkingRunning": MetricType.STEPS,
        }
        return mapping.get(apple_type)

    def _map_google_fit_type(self, fit_type: str) -> Optional[MetricType]:
        """Map Google Fit types to MetricType"""
        mapping = {
            "com.google.step_count.delta": MetricType.STEPS,
            "com.google.heart_rate.bpm": MetricType.HEART_RATE,
            "com.google.sleep.segment": MetricType.SLEEP_DURATION,
            "com.google.calories.expended": MetricType.CALORIES_BURNED,
            "com.google.calories.consumed": MetricType.CALORIES_CONSUMED,
            "com.google.water": MetricType.WATER_INTAKE,
            "com.google.weight": MetricType.WEIGHT,
            "com.google.activity.segment": MetricType.ACTIVE_MINUTES,
        }
        return mapping.get(fit_type)

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            "total_metrics": sum(len(m) for m in self._metric_history.values()),
            "metrics_by_type": {
                mt.value: len(self._metric_history[mt])
                for mt in MetricType
                if mt in self._metric_history
            },
            "date_range": {
                "oldest": (
                    min(
                        m.timestamp
                        for m in self._metric_history.get(MetricType.STEPS, [])
                    )
                    if self._metric_history.get(MetricType.STEPS)
                    else None
                ),
                "newest": (
                    max(
                        m.timestamp
                        for m in self._metric_history.get(MetricType.STEPS, [])
                    )
                    if self._metric_history.get(MetricType.STEPS)
                    else None
                ),
            },
            "has_profile": self._profile is not None,
        }
