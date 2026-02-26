"""
AURA v3 Context Package
Real-time context detection from device sensors and state

Unified Architecture:
- ContextProvider: Low-level data gathering from Termux/Android APIs
- ContextDetector: High-level behavior classification and decision hints

ContextDetector now uses ContextProvider for real sensor data instead of
duplicating data-gathering logic with stub methods.
"""

from src.context.context_provider import (
    ContextProvider,
    LocationContext,
    DeviceContext,
    ActivityContext,
    SocialContext,
    EnvironmentalContext,
    FullContext,
    ContextSource,
    get_context_provider,
)

from src.context.detector import (
    ContextDetector,
    TimeOfDay,
    DayType,
    LocationType,
    ActivityType,
    BehaviorMode,
    get_context_detector,
)

__all__ = [
    # Data layer (ContextProvider)
    "ContextProvider",
    "LocationContext",
    "DeviceContext",
    "ActivityContext",
    "SocialContext",
    "EnvironmentalContext",
    "FullContext",
    "ContextSource",
    "get_context_provider",
    # Behavior layer (ContextDetector)
    "ContextDetector",
    "TimeOfDay",
    "DayType",
    "LocationType",
    "ActivityType",
    "BehaviorMode",
    "get_context_detector",
]
