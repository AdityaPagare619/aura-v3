"""
Extensibility Package for AURA v3
"""

from src.extensibility.plugin_system import (
    PluginBase,
    PluginRegistry,
    PluginMetadata,
    PluginState,
    PluginAPI,
    SimpleEventBus,
    get_plugin_registry,
    init_plugin_system,
)

__all__ = [
    "PluginBase",
    "PluginRegistry",
    "PluginMetadata",
    "PluginState",
    "PluginAPI",
    "SimpleEventBus",
    "get_plugin_registry",
    "init_plugin_system",
]
