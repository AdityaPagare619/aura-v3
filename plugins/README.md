# AURA v3 Plugin System

A modular plugin system that allows developers to extend AURA v3 functionality by dropping Python files into the `plugins/` directory.

## Quick Start

1. Create your plugin in the `plugins/` directory
2. Subclass `PluginBase` and implement required methods
3. AURA auto-discovers and loads your plugin

## Writing a Plugin

### Basic Template

```python
from src.extensibility.plugin_system import PluginBase, PluginMetadata


class MyPlugin(PluginBase):
    """Description of what my plugin does"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            author="Your Name",
            description="What this plugin does",
            dependencies=[],
            tags=["category"],
        )
    
    def on_load(self, api) -> bool:
        """Called when plugin is loaded"""
        # Register commands, hooks, tools here
        self.register_command("my_command", self.handle_command)
        return True
```

### Plugin Lifecycle

| Hook | When Called | Return |
|------|-------------|--------|
| `on_load(api)` | Plugin is loaded | `bool` - success |
| `on_unload()` | Plugin is unloaded | `bool` - allow unload |
| `on_activate()` | Plugin is activated | `bool` - success |
| `on_deactivate()` | Plugin is deactivated | `bool` - success |

### Core Hooks

#### `on_message(message)`
Called for every message. Can modify or intercept messages.

```python
def on_message(self, message: Dict) -> Optional[Dict]:
    content = message.get("content", "")
    
    if "keyword" in content.lower():
        message["_triggered"] = True
    
    return None  # Return modified message if needed
```

#### `on_tool_call(tool_name, parameters)`
Called before/after tool execution. Can modify parameters or results.

```python
def on_tool_call(self, tool_name: str, parameters: Dict) -> Optional[Dict]:
    if tool_name == "send_message":
        # Add signature to messages
        parameters["message"] += " - Sent via AURA"
    
    return {"parameters": parameters}  # Return modifications
```

#### `on_think(context)`
Hook into agent thought process. Provide suggestions.

```python
def on_think(self, context: Dict) -> Optional[Dict]:
    return {
        "suggestions": [
            {"type": "action", "content": "Check weather"}
        ]
    }
```

#### `on_response(response)`
Modify agent responses before sending.

```python
def on_response(self, response: Dict) -> Optional[Dict]:
    response["metadata"] = response.get("metadata", {})
    response["metadata"]["plugin_modified"] = True
    return response
```

## Plugin API

Access AURA internals through the `PluginAPI` object (passed to `on_load`):

```python
def on_load(self, api) -> bool:
    self._api = api
    
    # Access configuration
    config = self._api.config
    user_pref = self._api.get_user_preference("theme", "dark")
    
    # Logging
    self._api.log_info("Plugin loaded!")
    
    # Emit events
    self._api.emit_event("custom_event", {"data": "value"})
    
    return True
```

### Available API Methods

| Method | Description |
|--------|-------------|
| `api.config` | Get configuration (read-only dict) |
| `api.memory` | Access memory system |
| `api.tools` | Access tool registry |
| `api.get_user_preference(key, default)` | Get user preference |
| `api.log_info/warning/error(msg)` | Logging |
| `api.emit_event(type, data)` | Emit event |
| `api.subscribe(type, callback)` | Subscribe to events |

## Registering Features

### Custom Commands

```python
def on_load(self, api) -> bool:
    self.register_command("hello", self.handle_hello)
    return True

def handle_hello(self, args: str = "") -> Dict:
    return {"success": True, "message": f"Hello, {args}!"}
```

### Custom Tools

```python
from src.tools.registry import ToolDefinition

def on_load(self, api) -> bool:
    tool_def = ToolDefinition(
        name="my_tool",
        description="What it does",
        parameters={
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "Description"}
            },
            "required": ["param"]
        },
        category="my_category",
        risk_level="low",
    )
    
    self.register_tool("my_tool", tool_def, self.handle_tool)
    return True

def handle_tool(self, **params) -> Dict:
    return {"success": True, "result": params}
```

### Event Hooks

```python
def on_load(self, api) -> bool:
    self.register_hook("message", self.on_message)
    self.register_hook("think", self.on_think)
    return True
```

## Example Plugins

### Weather Plugin (`example_weather.py`)

- Fetches weather from public API
- Offline fallback mode
- Registers `/weather` command
- Hooks into messages for weather queries

### Reminder Plugin (`example_reminder.py`)

- Natural language parsing ("remind me to X in 30 minutes")
- Recurring reminders (daily, weekly, monthly)
- Commands: `/remind`, `/reminders`, `/snooze`, `/complete`

## Integration

### Manual Loading

```python
from src.extensibility.plugin_system import init_plugin_system

registry = init_plugin_system(
    plugins_dir="plugins",
    tool_registry=your_tool_registry,
    config=your_config,
)

# Load specific plugin
registry.load_plugin("my_plugin")

# Or load all
registry.load_all()
```

### With Tool Registry

```python
from src.tools.registry import ToolRegistry

registry = init_plugin_system(
    plugins_dir="plugins",
    tool_registry=tool_registry,
)

registry.load_all()
registry.register_plugin_tools(tool_registry)
```

## Best Practices

1. **Always implement `get_metadata()`** - Required for plugin identification
2. **Return `bool` from lifecycle methods** - Indicates success/failure
3. **Clean up in `on_unload()`** - Release resources, save state
4. **Use the API for logging** - Integrates with AURA's logging
5. **Handle errors gracefully** - Don't crash the main system
6. **Version your plugins** - Use semantic versioning
7. **Document dependencies** - List required plugins or modules

## Troubleshooting

### Plugin Not Loading

- Check the plugin inherits from `PluginBase`
- Ensure `get_metadata()` returns `PluginMetadata`
- Verify `on_load()` returns `True`
- Check logs for import errors

### Commands Not Working

- Register commands in `on_load()`, not `__init__`
- Return proper dict with `success` key

### Tools Not Appearing

- Call `register_plugin_tools()` after loading plugins
- Ensure tool definitions have required fields

## File Structure

```
aura-v3/
├── plugins/
│   ├── __init__.py
│   ├── example_weather.py
│   ├── example_reminder.py
│   └── README.md
└── src/
    └── extensibility/
        └── plugin_system.py
```
