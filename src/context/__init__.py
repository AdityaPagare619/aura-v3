"""
AURA v3 Context Package
Real-time context detection from device sensors and state
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

__all__ = [
    "ContextProvider",
    "LocationContext",
    "DeviceContext",
    "ActivityContext",
    "SocialContext",
    "EnvironmentalContext",
    "FullContext",
    "ContextSource",
    "get_context_provider",
]
