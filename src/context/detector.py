"""
AURA Context Detector
Detects time, location, activity, and provides context-aware behavior hints

Unified with ContextProvider: Uses real sensor data from ContextProvider
instead of stub methods. This is the behavior/decision layer on top of
ContextProvider's data-gathering layer.
"""

from datetime import datetime, time
from enum import Enum
from typing import Dict, Optional, Any, List, TYPE_CHECKING
import asyncio
import json
import os
import logging

if TYPE_CHECKING:
    from src.context.context_provider import ContextProvider

logger = logging.getLogger(__name__)


class TimeOfDay(Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


class DayType(Enum):
    WEEKDAY = "weekday"
    WEEKEND = "weekend"


class LocationType(Enum):
    HOME = "home"
    WORK = "work"
    COMMUTING = "commuting"
    UNKNOWN = "unknown"


class ActivityType(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    IN_CALL = "in_call"
    DRIVING = "driving"
    SLEEPING = "sleeping"


class BehaviorMode(Enum):
    PROFESSIONAL = "professional"
    PERSONAL = "personal"
    QUIET = "quiet"
    DRIVING = "driving"


class ContextDetector:
    """
    High-level context detector that provides behavior modes and decision hints.

    Unified architecture: Uses ContextProvider for real sensor data instead of
    duplicating data-gathering logic. ContextDetector focuses on:
    - Behavior mode classification (professional, personal, quiet, driving)
    - Action confirmation decisions
    - Config management (work hours, known locations)

    ContextProvider handles the low-level data gathering from Termux/Android APIs.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        context_provider: Optional["ContextProvider"] = None,
    ):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "..", "config", "context_config.json"
        )
        self._load_config()
        self._last_location: Optional[Dict] = None
        self._last_activity: Optional[Dict] = None
        self._cached_context: Optional[Dict] = None
        self._cache_timestamp: float = 0
        self._cache_ttl: int = 60

        # UNIFIED: Use ContextProvider for real data
        self._context_provider = context_provider

    def _load_config(self):
        self.config = {
            "work_hours": {"start": 9, "end": 18},
            "sleep_hours": {"start": 23, "end": 7},
            "home_location": None,
            "work_location": None,
            "driving_speed_threshold": 5.0,
            "idle_time_threshold": 300,
        }
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    self.config.update(json.load(f))
            except (json.JSONDecodeError, IOError):
                pass

    def _get_time_of_day(self, dt: datetime) -> TimeOfDay:
        hour = dt.hour
        if 5 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= hour < 21:
            return TimeOfDay.EVENING
        else:
            return TimeOfDay.NIGHT

    def _get_day_type(self, dt: datetime) -> DayType:
        if dt.weekday() < 5:
            return DayType.WEEKDAY
        return DayType.WEEKEND

    def _is_work_hours(self, dt: datetime) -> bool:
        work_start = self.config["work_hours"]["start"]
        work_end = self.config["work_hours"]["end"]
        hour = dt.hour
        if work_start <= work_end:
            return work_start <= hour < work_end
        else:
            return hour >= work_start or hour < work_end

    def _is_sleep_hours(self, dt: datetime) -> bool:
        sleep_start = self.config["sleep_hours"]["start"]
        sleep_end = self.config["sleep_hours"]["end"]
        hour = dt.hour
        if sleep_start <= sleep_end:
            return sleep_start <= hour < sleep_end
        else:
            return hour >= sleep_start or hour < sleep_end

    async def _detect_location(self) -> LocationType:
        try:
            location = await self._get_android_location()
            if location is None:
                return LocationType.UNKNOWN
            self._last_location = location
            if self._is_at_home(location):
                return LocationType.HOME
            if self._is_at_work(location):
                return LocationType.WORK
            if self._is_commuting(location):
                return LocationType.COMMUTING
            return LocationType.UNKNOWN
        except Exception:
            return LocationType.UNKNOWN

    async def _get_android_location(self) -> Optional[Dict]:
        """Get location data from ContextProvider (unified architecture)."""
        if self._context_provider is None:
            # Lazy import to avoid circular imports
            from src.context.context_provider import get_context_provider

            self._context_provider = get_context_provider()

        try:
            ctx = await self._context_provider.get_current_context()
            if ctx.location and ctx.location.latitude is not None:
                return {
                    "lat": ctx.location.latitude,
                    "lng": ctx.location.longitude,
                    "speed": ctx.location.speed or 0,
                    "accuracy": ctx.location.accuracy,
                    "place_name": ctx.location.place_name,
                }
        except Exception as e:
            logger.warning(f"Error getting location from ContextProvider: {e}")

        return None

    def _is_at_home(self, location: Dict) -> bool:
        home_loc = self.config.get("home_location")
        if not home_loc or not location:
            return False
        distance = self._calculate_distance(
            location.get("lat", 0),
            location.get("lng", 0),
            home_loc.get("lat", 0),
            home_loc.get("lng", 0),
        )
        return distance < 0.1

    def _is_at_work(self, location: Dict) -> bool:
        work_loc = self.config.get("work_location")
        if not work_loc or not location:
            return False
        distance = self._calculate_distance(
            location.get("lat", 0),
            location.get("lng", 0),
            work_loc.get("lat", 0),
            work_loc.get("lng", 0),
        )
        return distance < 0.1

    def _is_commuting(self, location: Dict) -> bool:
        if not location:
            return False
        speed = location.get("speed", 0)
        return speed > self.config["driving_speed_threshold"]

    def _calculate_distance(
        self, lat1: float, lng1: float, lat2: float, lng2: float
    ) -> float:
        from math import radians, sin, cos, sqrt, atan2

        R = 6371
        dlat = radians(lat2 - lat1)
        dlng = radians(lng2 - lng1)
        a = (
            sin(dlat / 2) ** 2
            + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
        )
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    async def _detect_activity(self) -> ActivityType:
        try:
            sensor_data = await self._get_sensor_data()
            if sensor_data is None:
                return ActivityType.IDLE
            self._last_activity = sensor_data
            if self._is_sleeping(sensor_data):
                return ActivityType.SLEEPING
            if self._is_driving(sensor_data):
                return ActivityType.DRIVING
            if self._is_in_call(sensor_data):
                return ActivityType.IN_CALL
            if self._is_active(sensor_data):
                return ActivityType.ACTIVE
            return ActivityType.IDLE
        except Exception:
            return ActivityType.IDLE

    async def _get_sensor_data(self) -> Optional[Dict]:
        """Get sensor/activity data from ContextProvider (unified architecture)."""
        if self._context_provider is None:
            # Lazy import to avoid circular imports
            from src.context.context_provider import get_context_provider

            self._context_provider = get_context_provider()

        try:
            ctx = await self._context_provider.get_current_context()
            data = {}

            # Activity data
            if ctx.activity:
                data["activity"] = ctx.activity.activity_type
                data["speed"] = ctx.location.speed if ctx.location else 0

            # Device data
            if ctx.device:
                data["screen_on"] = ctx.device.screen_on
                data["battery_level"] = ctx.device.battery_level
                data["battery_state"] = ctx.device.battery_state

            # Social/call data (in_call not directly available, but could be inferred)
            data["in_call"] = False  # Would need call state from ContextProvider

            return data if data else None

        except Exception as e:
            logger.warning(f"Error getting sensor data from ContextProvider: {e}")

        return None

    def _is_sleeping(self, sensor_data: Dict) -> bool:
        now = datetime.now()
        if not self._is_sleep_hours(now):
            return False
        activity = sensor_data.get("activity", "unknown")
        screen_on = sensor_data.get("screen_on", False)
        return activity in ["still", "unknown"] and not screen_on

    def _is_driving(self, sensor_data: Dict) -> bool:
        activity = sensor_data.get("activity", "unknown")
        speed = sensor_data.get("speed", 0)
        return (
            activity == "in_vehicle" or speed > self.config["driving_speed_threshold"]
        )

    def _is_in_call(self, sensor_data: Dict) -> bool:
        return sensor_data.get("in_call", False)

    def _is_active(self, sensor_data: Dict) -> bool:
        activity = sensor_data.get("activity", "unknown")
        return activity in ["walking", "running", "on_bicycle", "on_foot"]

    async def detect(self) -> Dict:
        import time as time_module

        current_time = time_module.time()
        if (
            self._cached_context is not None
            and current_time - self._cache_timestamp < self._cache_ttl
        ):
            return self._cached_context
        now = datetime.now()
        time_of_day = self._get_time_of_day(now)
        day_type = self._get_day_type(now)
        is_work_hours = self._is_work_hours(now)
        is_sleep_hours = self._is_sleep_hours(now)
        location = await self._detect_location()
        activity = await self._detect_activity()
        context = {
            "timestamp": now.isoformat(),
            "time": {
                "time_of_day": time_of_day.value,
                "day_type": day_type.value,
                "is_work_hours": is_work_hours,
                "is_sleep_hours": is_sleep_hours,
                "hour": now.hour,
                "minute": now.minute,
            },
            "location": {
                "type": location.value,
                "details": self._last_location,
            },
            "activity": {
                "type": activity.value,
                "details": self._last_activity,
            },
            "behavior_hints": {
                "professional_mode": False,
                "personal_mode": False,
                "quiet_mode": False,
                "driving_mode": False,
            },
        }
        context["behavior_hints"]["professional_mode"] = (
            is_work_hours and location == LocationType.WORK
        )
        context["behavior_hints"]["personal_mode"] = (
            time_of_day in [TimeOfDay.EVENING, TimeOfDay.MORNING]
            and location == LocationType.HOME
            and not is_work_hours
        )
        context["behavior_hints"]["quiet_mode"] = is_sleep_hours
        context["behavior_hints"]["driving_mode"] = activity == ActivityType.DRIVING
        self._cached_context = context
        self._cache_timestamp = current_time
        return context

    def get_behavior_mode(self, context: Dict) -> str:
        hints = context.get("behavior_hints", {})
        if hints.get("driving_mode", False):
            return BehaviorMode.DRIVING.value
        if hints.get("quiet_mode", False):
            return BehaviorMode.QUIET.value
        if hints.get("professional_mode", False):
            return BehaviorMode.PROFESSIONAL.value
        if hints.get("personal_mode", False):
            return BehaviorMode.PERSONAL.value
        behavior_mode = "personal"
        time_info = context.get("time", {})
        location_info = context.get("location", {})
        if (
            time_info.get("is_work_hours", False)
            and location_info.get("type") == "work"
        ):
            return BehaviorMode.PROFESSIONAL.value
        return BehaviorMode.PERSONAL.value

    def should_ask_confirmation(self, action: str, context: Dict) -> bool:
        HIGH_RISK_ACTIONS = [
            "send_message",
            "make_call",
            "send_email",
            "post_social",
            "delete_file",
            "modify_settings",
            "make_payment",
            "share_location",
            "open_app",
        ]
        MEDIUM_RISK_ACTIONS = [
            "read_message",
            "search_web",
            "open_url",
            "create_reminder",
            "add_calendar_event",
        ]
        behavior_mode = self.get_behavior_mode(context)
        if behavior_mode == BehaviorMode.DRIVING.value:
            return action not in ["read_message", "get_directions", "play_music"]
        if behavior_mode == BehaviorMode.QUIET.value:
            return action in HIGH_RISK_ACTIONS + MEDIUM_RISK_ACTIONS
        if behavior_mode == BehaviorMode.PROFESSIONAL.value:
            personal_actions = ["post_social", "play_music", "personal_email"]
            if action in personal_actions:
                return True
        if action in HIGH_RISK_ACTIONS:
            return True
        return False

    def update_config(self, key: str, value: Any):
        self.config[key] = value
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def set_home_location(self, lat: float, lng: float):
        self.update_config("home_location", {"lat": lat, "lng": lng})

    def set_work_location(self, lat: float, lng: float):
        self.update_config("work_location", {"lat": lat, "lng": lng})

    def set_work_hours(self, start: int, end: int):
        self.update_config("work_hours", {"start": start, "end": end})

    def set_context_provider(self, provider: "ContextProvider"):
        """
        Set the ContextProvider instance for unified data access.

        This allows explicit wiring when you want to share a ContextProvider
        instance across multiple components.
        """
        self._context_provider = provider


# Global instance for convenience
_context_detector: Optional[ContextDetector] = None


def get_context_detector() -> ContextDetector:
    """
    Get or create the global ContextDetector instance.

    The ContextDetector will automatically connect to the global ContextProvider
    when first used.
    """
    global _context_detector
    if _context_detector is None:
        _context_detector = ContextDetector()
    return _context_detector
