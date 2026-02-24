"""
AURA v3 Mobile Module
=====================

Mobile-optimized components for Aura including:
- Aura Space Server (local HTTP server for UI)
- Mobile UI templates
- Widget/Bubble integration
- Overlay support
"""

from src.mobile.aura_space_server import (
    AuraSpaceServer,
    AuraPersona,
    AuraState,
    get_aura_space_server,
)

__all__ = [
    "AuraSpaceServer",
    "AuraPersona",
    "AuraState",
    "get_aura_space_server",
]
