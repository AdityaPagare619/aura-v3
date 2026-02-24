"""
AURA v3 Real-Time Context System
Handles live sensor data, device state, location, and activity detection
Mobile-optimized: Works on Android/Termux with offline-first approach

This is the "nervous system" - real-time environmental awareness
"""

import asyncio
import logging
import json
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class ContextSource(Enum):
    """Sources of context data"""

    GPS = "gps"
    NETWORK = "network"
    BLUETOOTH = "bluetooth"
    SENSORS = "sensors"
    SYSTEM = "system"
    APPLICATIONS = "applications"
    NOTIFICATIONS = "notifications"
    CALLS = "calls"


@dataclass
class LocationContext:
    """Real-time location context"""

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy: float = 0.0
    altitude: Optional[float] = None
    location_type: Optional[str] = None  # "gps", "network", "last_known"
    timestamp: Optional[datetime] = None
    place_name: Optional[str] = None  # Cached location name
    speed: Optional[float] = None  # m/s
    bearing: Optional[float] = None  # degrees


@dataclass
class DeviceContext:
    """Device state context"""

    battery_level: int = 100
    battery_state: str = "unknown"  # "charging", "discharging", "full", "not_charging"
    screen_on: bool = False
    screen_brightness: int = 0
    volume_level: int = 50
    do_not_disturb: bool = False
    airplane_mode: bool = False
    wifi_connected: bool = False
    mobile_data: bool = False
    bluetooth_on: bool = False
    location_enabled: bool = False
    timestamp: Optional[datetime] = None


@dataclass
class ActivityContext:
    """Detected user activity"""

    activity_type: str = (
        "unknown"  # "still", "walking", "running", "cycling", "driving"
    )
    confidence: float = 0.0
    duration: int = 0  # seconds
    transitions: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None


@dataclass
class SocialContext:
    """Social presence context"""

    nearby_devices: List[str] = field(default_factory=list)
    connected_devices: List[str] = field(default_factory=list)
    people_nearby: List[str] = field(default_factory=list)  # From contacts via BT
    meeting_status: str = "unknown"  # "in_meeting", "free", "busy"
    calendar_event: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class EnvironmentalContext:
    """Environmental conditions"""

    noise_level: str = "quiet"  # "quiet", "moderate", "loud"
    light_level: str = "unknown"  # "dark", "dim", "bright"
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class FullContext:
    """Complete real-time context snapshot"""

    timestamp: datetime
    location: Optional[LocationContext] = None
    device: Optional[DeviceContext] = None
    activity: Optional[ActivityContext] = None
    social: Optional[SocialContext] = None
    environmental: Optional[EnvironmentalContext] = None

    # Composite states
    is_home: bool = False
    is_work: bool = False
    is_traveling: bool = False
    is_exercising: bool = False
    is_driving: bool = False
    is_sleeping: bool = False
    is_busy: bool = False

    # Raw sensor data for LLM
    raw_data: Dict[str, Any] = field(default_factory=dict)


class ContextProvider:
    """
    Real-time context provider - gathers live data from device
    Mobile-optimized: Caches expensive operations, batches sensor reads
    """

    def __init__(self, storage_path: str = "data/context"):
        self.storage_path = storage_path
        self._running = False

        # Context caches
        self._location_cache: Optional[LocationContext] = None
        self._device_cache: Optional[DeviceContext] = None
        self._activity_cache: Optional[ActivityContext] = None
        self._social_cache: Optional[SocialContext] = None

        # Cache TTL (seconds)
        self._location_ttl = 60
        self._device_ttl = 10
        self._activity_ttl = 30

        # Known places (learned/defined)
        self._known_locations: Dict[str, Dict] = {}

        # Subscribers
        self._subscribers: List[Callable] = []

        # Background tasks
        self._context_task: Optional[asyncio.Task] = None

        # History for pattern detection
        self._context_history: deque = deque(maxlen=100)

        os.makedirs(storage_path, exist_ok=True)

    async def start(self):
        """Start context monitoring"""
        if self._running:
            return

        self._running = True
        logger.info("Context provider starting...")

        # Start background context gathering
        self._context_task = asyncio.create_task(self._context_loop())

        logger.info("Context provider started")

    async def stop(self):
        """Stop context monitoring"""
        self._running = False

        if self._context_task:
            self._context_task.cancel()
            try:
                await self._context_task
            except asyncio.CancelledError:
                pass

        logger.info("Context provider stopped")

    async def _context_loop(self):
        """Background loop to gather context"""
        while self._running:
            try:
                # Gather all context
                await self._gather_all_context()

                # Notify subscribers
                await self._notify_subscribers()

                # Sleep before next update
                await asyncio.sleep(10)  # Update every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Context loop error: {e}")
                await asyncio.sleep(30)  # Back off on error

    async def _gather_all_context(self):
        """Gather context from all sources"""

        # Gather device context (fast, always do)
        self._device_cache = await self._get_device_context()

        # Gather location (may be slow)
        self._location_cache = await self._get_location_context()

        # Gather activity (inferred from sensors)
        self._activity_cache = await self._get_activity_context()

        # Gather social context
        self._social_cache = await self._get_social_context()

        # Determine composite states
        full_context = await self._build_full_context()

        # Store in history
        self._context_history.append(full_context)

    async def _get_device_context(self) -> DeviceContext:
        """Get device state context"""
        ctx = DeviceContext(timestamp=datetime.now())

        try:
            # Read from Termux API or system files
            battery_path = "/sys/class/power_supply/battery/capacity"
            if os.path.exists(battery_path):
                with open(battery_path) as f:
                    ctx.battery_level = int(f.read().strip())

            # Battery state
            status_path = "/sys/class/power_supply/battery/status"
            if os.path.exists(status_path):
                with open(status_path) as f:
                    status = f.read().strip().lower()
                    if "charging" in status:
                        ctx.battery_state = "charging"
                    elif "full" in status:
                        ctx.battery_state = "full"
                    else:
                        ctx.battery_state = "discharging"

            # Screen state
            screen_path = "/sys/class/backlight/panel/brightness"
            if os.path.exists(screen_path):
                with open(screen_path) as f:
                    brightness = int(f.read().strip())
                    ctx.screen_on = brightness > 0
                    ctx.screen_brightness = brightness

            # Check WiFi
            wifi_path = "/data/misc/wifi/wpa_supplicant.conf"
            ctx.wifi_connected = os.path.exists(wifi_path)

            # Airplane mode
            airplane_path = "/sys/devices/system/radio/radio0/hci0/state"
            ctx.airplane_mode = not os.path.exists(airplane_path)

        except Exception as e:
            logger.warning(f"Error reading device context: {e}")

        return ctx

    async def _get_location_context(self) -> Optional[LocationContext]:
        """Get location context - tries GPS first, falls back to network"""

        # Check cache first
        if self._location_cache:
            age = (datetime.now() - self._location_cache.timestamp).total_seconds()
            if age < self._location_ttl:
                return self._location_cache

        ctx = LocationContext(timestamp=datetime.now())

        try:
            # Try to get GPS location via termux-location
            result = await self._run_termux_command("termux-location -l gps")
            if result:
                data = json.loads(result)
                ctx.latitude = data.get("latitude")
                ctx.longitude = data.get("longitude")
                ctx.accuracy = data.get("accuracy", 0)
                ctx.altitude = data.get("altitude")
                ctx.location_type = "gps"

                # Calculate speed and bearing if available
                ctx.speed = data.get("speed")
                ctx.bearing = data.get("bearing")

                # Try to get place name
                ctx.place_name = await self._reverse_geocode(
                    ctx.latitude, ctx.longitude
                )

                self._location_cache = ctx
                return ctx

            # Fallback to network location
            result = await self._run_termux_command("termux-location -l network")
            if result:
                data = json.loads(result)
                ctx.latitude = data.get("latitude")
                ctx.longitude = data.get("longitude")
                ctx.accuracy = data.get("accuracy", 0)
                ctx.location_type = "network"

                self._location_cache = ctx
                return ctx

        except Exception as e:
            logger.warning(f"Error getting location: {e}")

        return None

    async def _get_activity_context(self) -> ActivityContext:
        """Infer activity from sensors and context"""

        # Check cache
        if self._activity_cache:
            age = (datetime.now() - self._activity_cache.timestamp).total_seconds()
            if age < self._activity_ttl:
                return self._activity_cache

        ctx = ActivityContext(timestamp=datetime.now())

        # Infer from location
        if self._location_cache:
            speed = self._location_cache.speed
            if speed is not None:
                if speed < 1:
                    ctx.activity_type = "still"
                    ctx.confidence = 0.8
                elif speed < 3:
                    ctx.activity_type = "walking"
                    ctx.confidence = 0.7
                elif speed < 8:
                    ctx.activity_type = "running"
                    ctx.confidence = 0.7
                elif speed < 15:
                    ctx.activity_type = "cycling"
                    ctx.confidence = 0.6
                else:
                    ctx.activity_type = "driving"
                    ctx.confidence = 0.8

        # Infer from time patterns
        hour = datetime.now().hour
        if 22 <= hour or hour < 6:
            if ctx.activity_type == "still":
                ctx.activity_type = "sleeping"
                ctx.confidence = 0.9

        # Infer from app usage (if available)
        # Could check for driving apps, fitness apps, etc.

        self._activity_cache = ctx
        return ctx

    async def _get_social_context(self) -> SocialContext:
        """Get social presence context"""

        ctx = SocialContext(timestamp=datetime.now())

        try:
            # Check Bluetooth devices
            result = await self._run_termux_command("termux-bluetooth-connected")
            if result:
                lines = result.strip().split("\n")
                ctx.connected_devices = [l.strip() for l in lines if l.strip()]

            # Check for nearby Bluetooth devices
            result = await self._run_termux_command("termux-bluetooth-scan")
            if result:
                data = json.loads(result)
                ctx.nearby_devices = data.get("devices", [])

            # Check calendar for meeting status
            # Would need calendar API access

        except Exception as e:
            logger.warning(f"Error getting social context: {e}")

        return ctx

    async def _build_full_context(self) -> FullContext:
        """Build complete context snapshot"""

        ctx = FullContext(
            timestamp=datetime.now(),
            location=self._location_cache,
            device=self._device_cache,
            activity=self._activity_cache,
            social=self._social_cache,
        )

        # Determine composite states
        if self._location_cache and self._location_cache.place_name:
            ctx.is_home = "home" in self._location_cache.place_name.lower()
            ctx.is_work = any(
                w in self._location_cache.place_name.lower()
                for w in ["office", "work", "company", "building"]
            )

        if self._activity_cache:
            ctx.is_driving = self._activity_cache.activity_type == "driving"
            ctx.is_exercising = self._activity_cache.activity_type in [
                "running",
                "cycling",
                "walking",
            ]
            ctx.is_sleeping = self._activity_cache.activity_type == "sleeping"

        if self._device_cache:
            ctx.is_busy = (
                self._device_cache.do_not_disturb or ctx.is_driving or ctx.is_in_meeting
            )

        # Store raw data for LLM
        ctx.raw_data = {
            "battery": self._device_cache.battery_level if self._device_cache else None,
            "screen": self._device_cache.screen_on if self._device_cache else None,
            "location": {
                "lat": self._location_cache.latitude if self._location_cache else None,
                "lng": self._location_cache.longitude if self._location_cache else None,
                "place": self._location_cache.place_name
                if self._location_cache
                else None,
                "accuracy": self._location_cache.accuracy
                if self._location_cache
                else None,
            }
            if self._location_cache
            else None,
            "activity": self._activity_cache.activity_type
            if self._activity_cache
            else None,
            "time": ctx.timestamp.isoformat(),
        }

        return ctx

    async def get_current_context(self) -> FullContext:
        """Get current full context"""
        if self._context_history:
            return self._context_history[-1]
        return await self._build_full_context()

    async def get_context_for_llm(self) -> str:
        """
        Get context formatted for LLM consumption
        Returns natural language description
        """
        ctx = await self.get_current_context()

        parts = []

        # Time
        parts.append(f"Time: {ctx.timestamp.strftime('%I:%M %p')}")

        # Location
        if ctx.location and ctx.location.place_name:
            parts.append(f"Location: {ctx.location.place_name}")
        elif ctx.location and ctx.location.latitude:
            parts.append(
                f"Location: {ctx.location.latitude:.4f}, {ctx.location.longitude:.4f}"
            )

        # Activity
        if ctx.activity:
            parts.append(f"Activity: {ctx.activity.activity_type}")

        # Device state
        if ctx.device:
            parts.append(f"Battery: {ctx.device.battery_level}%")
            if ctx.device.battery_state == "charging":
                parts.append("(charging)")
            parts.append(f"Screen: {'on' if ctx.device.screen_on else 'off'}")

        # Composite states
        states = []
        if ctx.is_home:
            states.append("at home")
        if ctx.is_work:
            states.append("at work")
        if ctx.is_driving:
            states.append("driving")
        if ctx.is_exercising:
            states.append("exercising")
        if ctx.is_sleeping:
            states.append("sleeping")
        if ctx.is_busy:
            states.append("busy")

        if states:
            parts.append(f"State: {', '.join(states)}")

        return "; ".join(parts)

    async def _run_termux_command(self, cmd: str) -> Optional[str]:
        """Run a Termux command"""
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode == 0:
                return stdout.decode().strip()
        except asyncio.TimeoutError:
            logger.warning(f"Command timed out: {cmd}")
        except Exception as e:
            logger.warning(f"Command failed: {cmd} - {e}")
        return None

    async def _reverse_geocode(self, lat: float, lng: float) -> Optional[str]:
        """Reverse geocode to place name - offline if possible"""
        # For offline, would use cached tiles or local database
        # For now, return None (would need network)

        # Check known locations
        for name, loc in self._known_locations.items():
            if self._is_near(lat, lng, loc["lat"], loc["lng"], loc.get("radius", 100)):
                return name

        return None

    def _is_near(
        self, lat1: float, lng1: float, lat2: float, lng2: float, radius_m: float
    ) -> bool:
        """Simple distance check (approximate)"""
        import math

        dlat = abs(lat1 - lat2)
        dlng = abs(lng1 - lng2)
        # Rough approximation: 1 degree ~ 111km
        distance = math.sqrt(dlat**2 + dlng**2) * 111000
        return distance < radius_m

    def add_known_location(
        self, name: str, lat: float, lng: float, radius_m: float = 100
    ):
        """Add a known location (home, work, gym, etc.)"""
        self._known_locations[name] = {"lat": lat, "lng": lng, "radius": radius_m}
        logger.info(f"Added known location: {name}")

    def subscribe(self, callback: Callable):
        """Subscribe to context changes"""
        self._subscribers.append(callback)

    async def _notify_subscribers(self):
        """Notify all subscribers of context change"""
        if self._context_history:
            ctx = self._context_history[-1]
            for callback in self._subscribers:
                try:
                    await callback(ctx)
                except Exception as e:
                    logger.error(f"Subscriber error: {e}")

    async def get_history(self, duration_minutes: int = 60) -> List[FullContext]:
        """Get context history"""
        cutoff = datetime.now() - timedelta(minutes=duration_minutes)
        return [c for c in self._context_history if c.timestamp >= cutoff]

    async def detect_patterns(self) -> Dict[str, Any]:
        """Detect patterns in context history"""
        if len(self._context_history) < 10:
            return {"status": "insufficient_data"}

        # Detect location patterns
        locations = [
            c.location.place_name
            for c in self._context_history
            if c.location and c.location.place_name
        ]

        # Detect time patterns
        times = [c.timestamp.hour for c in self._context_history]

        # Detect activity patterns
        activities = [
            c.activity.activity_type for c in self._context_history if c.activity
        ]

        return {
            "locations": list(set(locations)) if locations else None,
            "common_times": times,
            "activities": list(set(activities)) if activities else None,
            "samples": len(self._context_history),
        }


# Global instance
_context_provider: Optional[ContextProvider] = None


def get_context_provider() -> ContextProvider:
    """Get or create context provider"""
    global _context_provider
    if _context_provider is None:
        _context_provider = ContextProvider()
    return _context_provider
