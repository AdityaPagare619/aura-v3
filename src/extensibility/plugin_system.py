"""
Plugin System for AURA v3
Provides extensibility through a modular plugin architecture.

Features:
- Plugin base class with lifecycle hooks
- Plugin registry for discovery and management
- Auto-loading from plugins/ directory
- Safe API for accessing AURA internals
"""

import os
import sys
import logging
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PluginState(Enum):
    """Lifecycle states for plugins"""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    UNLOADING = "unloading"


@dataclass
class PluginMetadata:
    """Metadata about a plugin"""

    name: str
    version: str
    author: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    loaded_at: Optional[datetime] = None


class PluginBase(ABC):
    """
    Base class for all AURA plugins.

    Subclass this to create custom plugins with lifecycle hooks.
    """

    metadata: PluginMetadata
    state: PluginState = PluginState.UNLOADED

    def __init__(self):
        self._api: Optional["PluginAPI"] = None
        self._commands: Dict[str, Callable] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._tools: Dict[str, Dict] = {}

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass

    def on_load(self, api: "PluginAPI") -> bool:
        """
        Called when plugin is loaded.
        Return True to indicate successful loading.
        """
        self._api = api
        self.state = PluginState.LOADED
        return True

    def on_unload(self) -> bool:
        """
        Called when plugin is unloaded.
        Return True to allow unloading.
        """
        self.state = PluginState.UNLOADING
        return True

    def on_activate(self) -> bool:
        """Called when plugin is activated"""
        self.state = PluginState.ACTIVE
        return True

    def on_deactivate(self) -> bool:
        """Called when plugin is deactivated (but not unloaded)"""
        self.state = PluginState.LOADED
        return True

    def on_message(self, message: Dict[str, Any]) -> Optional[Dict]:
        """
        Hook into message processing.
        Can modify or intercept messages.
        """
        return None

    def on_tool_call(self, tool_name: str, parameters: Dict) -> Optional[Dict]:
        """
        Hook into tool call execution.
        Can modify parameters or result.
        """
        return None

    def on_think(self, context: Dict[str, Any]) -> Optional[Dict]:
        """
        Hook into agent thought process.
        Can provide suggestions or modify context.
        """
        return None

    def on_response(self, response: Dict[str, Any]) -> Optional[Dict]:
        """
        Hook into response generation.
        Can modify the final response.
        """
        return None

    def register_command(self, name: str, handler: Callable):
        """Register a custom command"""
        self._commands[name] = handler

    def register_hook(self, event: str, handler: Callable):
        """Register a hook for an event"""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(handler)

    def register_tool(self, name: str, definition: Dict, handler: Callable):
        """Register a custom tool"""
        self._tools[name] = {"definition": definition, "handler": handler}

    def get_commands(self) -> Dict[str, Callable]:
        """Get registered commands"""
        return self._commands.copy()

    def get_hooks(self) -> Dict[str, List[Callable]]:
        """Get registered hooks"""
        return self._hooks.copy()

    def get_tools(self) -> Dict[str, Dict]:
        """Get registered tools"""
        return self._tools.copy()


class PluginAPI:
    """
    Safe API for plugins to access AURA internals.
    Provides controlled access to prevent misuse.
    """

    def __init__(self, aura_instance: Any = None):
        self._aura = aura_instance
        self._memory: Optional[Any] = None
        self._tool_registry: Optional[Any] = None
        self._config: Dict[str, Any] = {}
        self._event_bus: Optional[Any] = None

    def set_memory(self, memory_system):
        """Set reference to memory system"""
        self._memory = memory_system

    def set_tool_registry(self, registry):
        """Set reference to tool registry"""
        self._tool_registry = registry

    def set_config(self, config: Dict):
        """Set configuration"""
        self._config = config

    def set_event_bus(self, event_bus):
        """Set event bus for pub/sub"""
        self._event_bus = event_bus

    @property
    def memory(self):
        """Access memory system (read-only operations)"""
        return self._memory

    @property
    def tools(self):
        """Access tool registry"""
        return self._tool_registry

    @property
    def config(self) -> Dict:
        """Access configuration (read-only)"""
        return self._config.copy()

    def emit_event(self, event_type: str, data: Dict):
        """Emit an event to the event bus"""
        if self._event_bus:
            self._event_bus.emit(event_type, data)

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event"""
        if self._event_bus:
            self._event_bus.subscribe(event_type, callback)

    def log_info(self, message: str):
        """Log info message"""
        logger.info(f"[Plugin] {message}")

    def log_warning(self, message: str):
        """Log warning message"""
        logger.warning(f"[Plugin] {message}")

    def log_error(self, message: str):
        """Log error message"""
        logger.error(f"[Plugin] {message}")

    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference from config"""
        return self._config.get("preferences", {}).get(key, default)

    def store_data(self, key: str, value: Any):
        """Store persistent data for this plugin"""
        if not hasattr(self, "_plugin_data"):
            self._plugin_data = {}
        self._plugin_data[key] = value

    def retrieve_data(self, key: str, default: Any = None) -> Any:
        """Retrieve persistent data"""
        return getattr(self, "_plugin_data", {}).get(key, default)


class PluginRegistry:
    """
    Central registry for managing plugins.
    Handles discovery, loading, and lifecycle management.
    """

    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins: Dict[str, PluginBase] = {}
        self._plugins_dir = Path(plugins_dir)
        self._api = PluginAPI()
        self._loaded_modules: Dict[str, Any] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}

    def set_api_context(self, **context):
        """Set context for PluginAPI"""
        for key, value in context.items():
            if hasattr(self._api, f"set_{key}"):
                getattr(self._api, f"set_{key}")(value)

    def discover_plugins(self) -> List[str]:
        """Discover available plugins in plugins directory"""
        if not self._plugins_dir.exists():
            logger.warning(f"Plugins directory not found: {self._plugins_dir}")
            return []

        plugins = []
        for file in self._plugins_dir.glob("*.py"):
            if file.name.startswith("_") or file.name.startswith("example_"):
                continue
            plugins.append(file.stem)

        logger.info(f"Discovered plugins: {plugins}")
        return plugins

    def load_plugin(self, module_name: str) -> bool:
        """Load a plugin by module name"""
        try:
            if module_name in self.plugins:
                logger.warning(f"Plugin already loaded: {module_name}")
                return True

            module = importlib.import_module(f"plugins.{module_name}")

            plugin_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, PluginBase) and obj is not PluginBase:
                    plugin_class = obj
                    break

            if not plugin_class:
                logger.error(f"No PluginBase subclass found in {module_name}")
                return False

            plugin = plugin_class()
            metadata = plugin.get_metadata()
            plugin.metadata = metadata  # Store metadata on plugin instance

            if not plugin.on_load(self._api):
                logger.error(f"Plugin {module_name} failed to load")
                plugin.state = PluginState.ERROR
                return False

            self.plugins[metadata.name] = plugin
            self._loaded_modules[module_name] = module

            for event, handlers in plugin.get_hooks().items():
                self._register_plugin_hooks(metadata.name, event, handlers)

            logger.info(f"Loaded plugin: {metadata.name} v{metadata.version}")
            return True

        except Exception as e:
            logger.error(f"Failed to load plugin {module_name}: {e}")
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin not found: {plugin_name}")
            return False

        plugin = self.plugins[plugin_name]

        if not plugin.on_unload():
            logger.error(f"Plugin {plugin_name} refused to unload")
            return False

        del self.plugins[plugin_name]

        for module_name, module in list(self._loaded_modules.items()):
            if module_name.startswith(plugin_name):
                del self._loaded_modules[module_name]

        logger.info(f"Unloaded plugin: {plugin_name}")
        return True

    def activate_plugin(self, plugin_name: str) -> bool:
        """Activate a loaded plugin"""
        if plugin_name not in self.plugins:
            return False

        plugin = self.plugins[plugin_name]
        return plugin.on_activate()

    def deactivate_plugin(self, plugin_name: str) -> bool:
        """Deactivate a plugin (keeps it loaded)"""
        if plugin_name not in self.plugins:
            return False

        plugin = self.plugins[plugin_name]
        return plugin.on_deactivate()

    def get_plugin(self, name: str) -> Optional[PluginBase]:
        """Get a plugin by name"""
        return self.plugins.get(name)

    def list_plugins(self, state: Optional[PluginState] = None) -> List[str]:
        """List loaded plugins, optionally filtered by state"""
        if state:
            return [name for name, p in self.plugins.items() if p.state == state]
        return list(self.plugins.keys())

    def load_all(self) -> int:
        """Auto-load all discovered plugins"""
        discovered = self.discover_plugins()
        loaded = 0

        for module_name in discovered:
            if self.load_plugin(module_name):
                loaded += 1

        logger.info(f"Auto-loaded {loaded}/{len(discovered)} plugins")
        return loaded

    def _register_plugin_hooks(
        self, plugin_name: str, event: str, handlers: List[Callable]
    ):
        """Register hooks from a plugin"""
        if event not in self._event_handlers:
            self._event_handlers[event] = []

        for handler in handlers:
            self._event_handlers[event].append(
                {"plugin": plugin_name, "handler": handler}
            )

    def trigger_hook(self, event: str, *args, **kwargs) -> List[Any]:
        """Trigger all handlers for an event"""
        results = []

        if event in self._event_handlers:
            for entry in self._event_handlers[event]:
                try:
                    result = entry["handler"](*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Hook error in {entry['plugin']}: {e}")

        return results

    def process_message(self, message: Dict) -> Optional[Dict]:
        """Process message through plugin hooks"""
        for plugin in self.plugins.values():
            if plugin.state == PluginState.ACTIVE:
                result = plugin.on_message(message)
                if result:
                    message = result
        return message

    def process_tool_call(self, tool_name: str, parameters: Dict) -> Optional[Dict]:
        """Process tool call through plugin hooks"""
        result = {"parameters": parameters, "modified": False}

        for plugin in self.plugins.values():
            if plugin.state == PluginState.ACTIVE:
                hook_result = plugin.on_tool_call(tool_name, parameters)
                if hook_result:
                    result = hook_result
                    result["modified"] = True

        return result

    def process_think(self, context: Dict) -> Optional[Dict]:
        """Process think context through plugin hooks"""
        suggestions = []

        for plugin in self.plugins.values():
            if plugin.state == PluginState.ACTIVE:
                result = plugin.on_think(context)
                if result:
                    suggestions.append(result)

        if suggestions:
            return {"suggestions": suggestions}
        return None

    def process_response(self, response: Dict) -> Optional[Dict]:
        """Process response through plugin hooks"""
        for plugin in self.plugins.values():
            if plugin.state == PluginState.ACTIVE:
                result = plugin.on_response(response)
                if result:
                    response = result
        return response

    def register_plugin_tools(self, tool_registry):
        """Register all plugin tools with tool registry"""
        for plugin in self.plugins.values():
            for name, tool_data in plugin.get_tools().items():
                definition = tool_data["definition"]
                handler = tool_data["handler"]

                tool_registry.register(definition, handler)
                logger.info(
                    f"Registered tool '{name}' from plugin '{plugin.metadata.name}'"
                )

    def get_plugin_status(self) -> Dict[str, Any]:
        """Get status of all plugins"""
        return {
            name: {
                "state": p.state.value,
                "metadata": {
                    "name": p.metadata.name,
                    "version": p.metadata.version,
                    "author": p.metadata.author,
                    "description": p.metadata.description,
                },
                "commands": list(p.get_commands().keys()),
                "tools": list(p.get_tools().keys()),
                "hooks": list(p.get_hooks().keys()),
            }
            for name, p in self.plugins.items()
        }


class SimpleEventBus:
    """Simple event bus for plugin communication"""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from an event"""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)

    def emit(self, event_type: str, data: Dict):
        """Emit an event to all subscribers"""
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")


_global_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def init_plugin_system(
    plugins_dir: str = "plugins",
    aura_instance: Any = None,
    memory_system: Any = None,
    tool_registry: Any = None,
    config: Dict = None,
) -> PluginRegistry:
    """
    Initialize the plugin system with all dependencies.

    Args:
        plugins_dir: Directory containing plugin files
        aura_instance: Reference to main AURA instance
        memory_system: Reference to memory system
        tool_registry: Reference to tool registry
        config: Configuration dictionary

    Returns:
        Initialized PluginRegistry
    """
    registry = PluginRegistry(plugins_dir)

    registry.set_api_context(
        aura=aura_instance,
        memory=memory_system,
        tool_registry=tool_registry,
        config=config or {},
        event_bus=SimpleEventBus(),
    )

    global _global_registry
    _global_registry = registry

    return registry
