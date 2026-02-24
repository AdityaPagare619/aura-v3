"""
AURA v3 Mobile Power & Resource Manager
======================================

Mobile-specific power management:
- Battery-aware processing
- Thermal throttling
- Screen state detection
- Doze mode handling
- Background processing optimization
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class PowerMode(Enum):
    """Power modes based on battery state"""

    FULL_POWER = "full_power"  # Charging - maximum performance
    BALANCED = "balanced"  # 50-100% - normal operation
    POWER_SAVE = "power_save"  # 20-50% - reduced processing
    ULTRA_POWER_SAVE = "ultra_save"  # <20% - minimal operations
    CRITICAL = "critical"  # <10% - essential only


class ScreenState(Enum):
    """Device screen state"""

    ON = "on"
    OFF = "off"
    LOCKED = "locked"
    DOZE = "doze"


@dataclass
class PowerConfig:
    """Power management configuration"""

    full_power_threshold: int = 50
    balanced_threshold: int = 20
    power_save_threshold: int = 10
    critical_threshold: int = 5

    max_tokens_full: int = 2048
    max_tokens_balanced: int = 1024
    max_tokens_save: int = 512
    max_tokens_ultra: int = 256

    tick_rate_full: float = 1.0
    tick_rate_balanced: float = 2.0
    tick_rate_save: float = 5.0
    tick_rate_ultra: float = 30.0

    enable_aggressive_power_save: bool = False
    thermal_throttle_threshold: int = 42
    max_cpu_percent: int = 70
    background_task_enabled: bool = True
    proactive_features_enabled: bool = True


class MobilePowerManager:
    """
    Mobile-specific power management

    Key responsibilities:
    - Monitor battery level and charging state
    - Detect screen state
    - Adjust processing based on power mode
    - Handle Doze mode and background restrictions
    - Optimize for Android's power requirements
    """

    def __init__(self, config: Optional[PowerConfig] = None):
        self.config = config or PowerConfig()

        # State
        self._battery_level: int = 100
        self._is_charging: bool = False
        self._screen_state: ScreenState = ScreenState.ON
        self._power_mode: PowerMode = PowerMode.FULL_POWER

        # Monitoring
        self._last_battery_check: datetime = datetime.now()
        self._last_screen_check: datetime = datetime.now()

    async def start(self):
        """Start power management"""
        logger.info("Starting mobile power manager...")
        # Initial battery/screen check
        await self._check_battery()
        await self._check_screen_state()

    async def _check_battery(self):
        """Check battery status via Termux"""
        try:
            # Use Termux battery-status API
            result = await self._run_termux_cmd("termux-battery-status")
            if result:
                import json

                data = json.loads(result)
                self._battery_level = int(data.get("percentage", 100))
                self._is_charging = data.get("plugged", "") != ""
        except Exception as e:
            logger.warning(f"Battery check failed: {e}")

        self._update_power_mode()
        self._last_battery_check = datetime.now()

    async def _check_screen_state(self):
        """Check screen state"""
        try:
            # Check screen state via dumpsys
            result = await self._run_termux_cmd(
                "dumpsys power | grep 'ScreenOn' || echo 'unknown'"
            )
            if "true" in result.lower():
                self._screen_state = ScreenState.ON
            else:
                self._screen_state = ScreenState.OFF
        except:
            self._screen_state = ScreenState.ON  # Default assumption

        self._last_screen_check = datetime.now()

    def _update_power_mode(self):
        """Update power mode based on battery"""
        if self._is_charging:
            self._power_mode = PowerMode.FULL_POWER
        elif self._battery_level >= self.config.full_power_threshold:
            self._power_mode = PowerMode.BALANCED
        elif self._battery_level >= self.config.power_save_threshold:
            if self.config.enable_aggressive_power_save:
                self._power_mode = PowerMode.ULTRA_POWER_SAVE
            else:
                self._power_mode = PowerMode.POWER_SAVE
        elif self._battery_level >= self.config.critical_threshold:
            self._power_mode = PowerMode.ULTRA_POWER_SAVE
        else:
            self._power_mode = PowerMode.CRITICAL

        logger.info(
            f"Power mode: {self._power_mode.value} (battery: {self._battery_level}%, charging: {self._is_charging})"
        )

    def set_aggressive_power_save(self, enabled: bool):
        """Enable or disable aggressive power save mode"""
        self.config.enable_aggressive_power_save = enabled
        logger.info(f"Aggressive power save: {enabled}")
        self._update_power_mode()

    def set_background_tasks(self, enabled: bool):
        """Enable or disable background tasks"""
        self.config.background_task_enabled = enabled
        logger.info(f"Background tasks: {enabled}")

    def set_proactive_features(self, enabled: bool):
        """Enable or disable proactive features"""
        self.config.proactive_features_enabled = enabled
        logger.info(f"Proactive features: {enabled}")

    def get_battery_status(self) -> dict:
        """Get detailed battery status"""
        return {
            "battery_level": self._battery_level,
            "is_charging": self._is_charging,
            "power_mode": self._power_mode.value,
            "screen_state": self._screen_state.value,
            "background_enabled": self.config.background_task_enabled,
            "proactive_enabled": self.config.proactive_features_enabled,
        }

    async def _run_termux_cmd(self, cmd: str) -> str:
        """Run a Termux command"""
        import subprocess

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=5
            )
            return result.stdout if result.returncode == 0 else ""
        except:
            return ""

    def get_power_mode(self) -> PowerMode:
        """Get current power mode"""
        return self._power_mode

    def get_processing_limits(self) -> Dict[str, Any]:
        """Get processing limits based on power mode"""
        limits = {
            PowerMode.FULL_POWER: {
                "max_tokens": self.config.max_tokens_full,
                "tick_rate": self.config.tick_rate_full,
                "allow_background": self.config.background_task_enabled,
                "allow_proactive": self.config.proactive_features_enabled,
            },
            PowerMode.BALANCED: {
                "max_tokens": self.config.max_tokens_balanced,
                "tick_rate": self.config.tick_rate_balanced,
                "allow_background": self.config.background_task_enabled,
                "allow_proactive": self.config.proactive_features_enabled,
            },
            PowerMode.POWER_SAVE: {
                "max_tokens": self.config.max_tokens_save,
                "tick_rate": self.config.tick_rate_save,
                "allow_background": False,
                "allow_proactive": False,
            },
            PowerMode.ULTRA_POWER_SAVE: {
                "max_tokens": self.config.max_tokens_ultra,
                "tick_rate": self.config.tick_rate_ultra,
                "allow_background": False,
                "allow_proactive": False,
            },
            PowerMode.CRITICAL: {
                "max_tokens": 128,
                "tick_rate": self.config.tick_rate_ultra * 2,
                "allow_background": False,
                "allow_proactive": False,
                "essential_only": True,
            },
        }

        mode_limits = limits.get(self._power_mode, {})

        if self._screen_state == ScreenState.OFF:
            mode_limits["tick_rate"] = mode_limits.get("tick_rate", 1.0) * 4
            mode_limits["allow_background"] = False
            mode_limits["allow_proactive"] = False

        return mode_limits

    def should_process(self, task_priority: int) -> bool:
        """
        Decide if a task should run based on power mode

        Args:
            task_priority: Priority 1-10 (10 = critical)
        """
        if (
            not self.config.background_task_enabled
            and self._screen_state == ScreenState.OFF
        ):
            return task_priority >= 9

        if self._power_mode == PowerMode.FULL_POWER:
            return True
        elif self._power_mode == PowerMode.BALANCED:
            return task_priority >= 3
        elif self._power_mode == PowerMode.POWER_SAVE:
            return task_priority >= 7
        elif self._power_mode == PowerMode.ULTRA_POWER_SAVE:
            return task_priority >= 9
        else:
            return task_priority >= 10

    def optimize_for_screen_off(self):
        """Optimize power when screen is turned off"""
        if self._screen_state == ScreenState.OFF:
            self.config.background_task_enabled = False
            self.config.proactive_features_enabled = False
            logger.info(
                "Optimized for screen-off: disabled background and proactive tasks"
            )

    def optimize_for_screen_on(self):
        """Restore normal power when screen is turned on"""
        self.config.background_task_enabled = True
        self.config.proactive_features_enabled = True
        logger.info("Restored normal power settings for screen-on")

    async def periodic_check(self):
        """Periodic power/screen check (called by background task)"""
        previous_screen_state = self._screen_state

        await self._check_battery()
        await self._check_screen_state()

        if previous_screen_state != self._screen_state:
            if self._screen_state == ScreenState.OFF:
                self.optimize_for_screen_off()
            else:
                self.optimize_for_screen_on()


# ============================================================================
# MOBILE SENSOR MANAGER
# ============================================================================


class SensorType(Enum):
    """Available mobile sensors"""

    GPS = "gps"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    ACCELEROMETER = "accelerometer"
    PROXIMITY = "proximity"
    LIGHT = "light"
    BATTERY = "battery"


@dataclass
class SensorReading:
    """Sensor data reading"""

    sensor: SensorType
    value: Any
    timestamp: datetime
    accuracy: int = 0


class MobileSensorManager:
    """
    Mobile sensor integration

    Manages access to mobile sensors for contextual awareness:
    - Location (GPS)
    - Camera (photos)
    - Microphone (voice)
    - Environment sensors
    """

    def __init__(self):
        self._available_sensors: Dict[SensorType, bool] = {}
        self._sensor_cache: Dict[SensorType, SensorReading] = {}
        self._cache_ttl = 60  # Cache TTL in seconds

    async def initialize(self):
        """Initialize sensor availability check"""
        # Check which sensors are available
        await self._check_sensor_availability()

    async def _check_sensor_availability(self):
        """Check available sensors via Termux"""
        # GPS
        try:
            result = await self._run_cmd("termux-location -last")
            self._available_sensors[SensorType.GPS] = result.success
        except:
            self._available_sensors[SensorType.GPS] = False

        # Camera (check if camera app exists)
        self._available_sensors[SensorType.CAMERA] = True  # Assume available

    async def _run_cmd(self, cmd: str):
        """Run command"""
        import subprocess

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            return type(
                "obj",
                (object,),
                {"success": result.returncode == 0, "stdout": result.stdout},
            )()
        except:
            return type("obj", (object,), {"success": False, "stdout": ""})()

    def is_available(self, sensor: SensorType) -> bool:
        """Check if sensor is available"""
        return self._available_sensors.get(sensor, False)

    async def get_location(self) -> Optional[SensorReading]:
        """Get current GPS location"""
        if not self.is_available(SensorType.GPS):
            return None

        # Check cache
        cached = self._sensor_cache.get(SensorType.GPS)
        if cached and (datetime.now() - cached.timestamp).seconds < self._cache_ttl:
            return cached

        try:
            result = await self._run_cmd("termux-location -last")
            if result.success:
                import json

                data = json.loads(result.stdout)
                reading = SensorReading(
                    sensor=SensorType.GPS,
                    value={
                        "latitude": data.get("latitude"),
                        "longitude": data.get("longitude"),
                    },
                    timestamp=datetime.now(),
                    accuracy=data.get("accuracy", 0),
                )
                self._sensor_cache[SensorType.GPS] = reading
                return reading
        except Exception as e:
            logger.error(f"Location error: {e}")

        return None

    async def take_photo(self, camera_id: int = 0) -> Optional[str]:
        """Take a photo using the camera"""
        if not self.is_available(SensorType.CAMERA):
            return None

        import os

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"/sdcard/DCIM/AURA/photo_{timestamp}.jpg"

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        result = await self._run_cmd(f"termux-camera-photo -c {camera_id} {path}")
        if result.success:
            return path
        return None


# Global instances
_power_manager: Optional[MobilePowerManager] = None
_sensor_manager: Optional[MobileSensorManager] = None


def get_power_manager() -> MobilePowerManager:
    """Get power manager instance"""
    global _power_manager
    if _power_manager is None:
        _power_manager = MobilePowerManager()
    return _power_manager


def get_sensor_manager() -> MobileSensorManager:
    """Get sensor manager instance"""
    global _sensor_manager
    if _sensor_manager is None:
        _sensor_manager = MobileSensorManager()
    return _sensor_manager
