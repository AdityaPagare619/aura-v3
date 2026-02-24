"""
Tool Registry with JSON Schema Generation
Critical for AURA v3 - provides proper tool definitions to LLM

Based on OpenClaw's approach:
- Tools defined as JSON schemas
- LLM knows exact parameters, types, descriptions
- Enables precise tool calling
"""

import json
import logging
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """JSON Schema definition for a tool"""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    examples: List[Dict] = field(default_factory=list)
    category: str = "general"
    risk_level: str = "low"  # low, medium, high, critical
    requires_approval: bool = False
    handler: Optional[Callable] = None
    privacy_tier: str = "public"  # public, sensitive, private
    privacy_category: str = "general"  # Maps to privacy category


class ToolRegistry:
    """
    Registry of available tools with JSON Schema definitions

    Key innovation: Generates proper JSON schemas that LLM can understand
    """

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable] = {}  # FIX: Store handlers separately
        self._register_core_tools()

    def _register_core_tools(self):
        """Register core tools"""
        self.register(
            ToolDefinition(
                name="open_app",
                description="Open an application on the device",
                parameters={
                    "type": "object",
                    "properties": {
                        "app_name": {
                            "type": "string",
                            "description": "Name of the app to open",
                        }
                    },
                    "required": ["app_name"],
                },
                category="android",
                risk_level="medium",
            )
        )

        self.register(
            ToolDefinition(
                name="close_app",
                description="Close an application",
                parameters={
                    "type": "object",
                    "properties": {
                        "app_name": {
                            "type": "string",
                            "description": "Name of the app to close",
                        }
                    },
                    "required": ["app_name"],
                },
                category="android",
                risk_level="medium",
            )
        )

        self.register(
            ToolDefinition(
                name="tap_screen",
                description="Tap a specific position on screen",
                parameters={
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "X coordinate"},
                        "y": {"type": "number", "description": "Y coordinate"},
                    },
                    "required": ["x", "y"],
                },
                category="android",
                risk_level="medium",
            )
        )

        self.register(
            ToolDefinition(
                name="swipe_screen",
                description="Swipe on screen in a direction",
                parameters={
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right"],
                            "description": "Direction to swipe",
                        },
                        "distance": {
                            "type": "number",
                            "description": "Distance in pixels",
                        },
                    },
                    "required": ["direction"],
                },
                category="android",
                risk_level="medium",
            )
        )

        self.register(
            ToolDefinition(
                name="type_text",
                description="Type text into the current input field",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to type"}
                    },
                    "required": ["text"],
                },
                category="android",
                risk_level="medium",
            )
        )

        self.register(
            ToolDefinition(
                name="get_current_app",
                description="Get the currently active application",
                parameters={"type": "object", "properties": {}},
                category="android",
                risk_level="low",
            )
        )

        self.register(
            ToolDefinition(
                name="take_screenshot",
                description="Take a screenshot",
                parameters={
                    "type": "object",
                    "properties": {
                        "save_path": {
                            "type": "string",
                            "description": "Path to save screenshot",
                        }
                    },
                },
                category="android",
                risk_level="low",
                privacy_tier="private",
                privacy_category="screenshots",
            )
        )

        self.register(
            ToolDefinition(
                name="get_notifications",
                description="Get recent notifications",
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "number",
                            "description": "Max notifications to return",
                        }
                    },
                },
                category="android",
                risk_level="low",
                privacy_tier="sensitive",
                privacy_category="messages",
            )
        )

        self.register(
            ToolDefinition(
                name="send_message",
                description="Send a message via a messaging app",
                parameters={
                    "type": "object",
                    "properties": {
                        "recipient": {
                            "type": "string",
                            "description": "Message recipient",
                        },
                        "message": {"type": "string", "description": "Message text"},
                    },
                    "required": ["recipient", "message"],
                },
                category="communication",
                risk_level="high",
                requires_approval=True,
                privacy_tier="private",
                privacy_category="messages",
            )
        )

        self.register(
            ToolDefinition(
                name="make_call",
                description="Make a phone call",
                parameters={
                    "type": "object",
                    "properties": {
                        "number": {
                            "type": "string",
                            "description": "Phone number to call",
                        }
                    },
                    "required": ["number"],
                },
                category="communication",
                risk_level="critical",
                requires_approval=True,
            )
        )

    def register(self, tool: ToolDefinition, handler: Optional[Callable] = None):
        """Register a tool with its definition and optional handler"""
        self.tools[tool.name] = tool
        if handler:
            self._handlers[tool.name] = handler
            tool.handler = handler  # Also set on the definition
        logger.info(f"Registered tool: {tool.name} ({tool.category})")

    def register_handler(self, name: str, handler: Callable):
        """Register a handler for an existing tool"""
        self._handlers[name] = handler
        if name in self.tools:
            self.tools[name].handler = handler
        logger.info(f"Bound handler to tool: {name}")

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get tool handler by name"""
        # FIX: First check _handlers dict, then fall back to tool.handler
        if name in self._handlers:
            return self._handlers[name]
        tool = self.tools.get(name)
        return tool.handler if tool else None

    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())

    def list_tools_by_category(self, category: str) -> List[str]:
        """List tools in a specific category"""
        return [name for name, tool in self.tools.items() if tool.category == category]

    def get_json_schemas(self) -> str:
        """Get all tool schemas as formatted JSON for LLM"""
        schemas = {"tools": []}

        for name, tool in self.tools.items():
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            schemas["tools"].append(schema)

        return json.dumps(schemas, indent=2)

    def get_tool_definitions(self) -> Dict[str, ToolDefinition]:
        """Get all tool definitions"""
        return self.tools.copy()

    def get_tool_definition(self, name: str) -> Optional[ToolDefinition]:
        """Get specific tool definition"""
        return self.tools.get(name)

    def get_tools_by_risk(self, risk_level: str) -> List[str]:
        """Get tools filtered by risk level"""
        return [
            name for name, tool in self.tools.items() if tool.risk_level == risk_level
        ]

    def get_approval_required_tools(self) -> List[str]:
        """Get list of tools requiring user approval"""
        return [name for name, tool in self.tools.items() if tool.requires_approval]


class ToolExecutor:
    """
    Executes tool calls with proper error handling
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.execution_history: List[Dict] = []

    async def execute(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool with given parameters"""

        start_time = datetime.now()

        try:
            tool = self.registry.get_tool(tool_name)
            if not tool:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not found",
                    "available_tools": self.registry.list_tools(),
                }

            # Execute with timeout
            if asyncio.iscoroutinefunction(tool):
                result = await asyncio.wait_for(tool(**parameters), timeout=30.0)
            else:
                result = tool(**parameters)

            # Record execution
            self._record_execution(
                tool_name=tool_name,
                parameters=parameters,
                result=result,
                success=result.get("success", True),
                duration=(datetime.now() - start_time).total_seconds(),
            )

            return result

        except asyncio.TimeoutError:
            error_result = {
                "success": False,
                "error": "Operation timed out after 30 seconds",
            }
            self._record_execution(
                tool_name=tool_name,
                parameters=parameters,
                result=error_result,
                success=False,
                duration=30.0,
            )
            return error_result

        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_execution(
                tool_name=tool_name,
                parameters=parameters,
                result=error_result,
                success=False,
                duration=(datetime.now() - start_time).total_seconds(),
            )
            logger.error(f"Tool execution error: {tool_name}: {e}")
            return error_result

    def _record_execution(
        self,
        tool_name: str,
        parameters: Dict,
        result: Dict,
        success: bool,
        duration: float,
    ):
        """Record execution for learning"""
        self.execution_history.append(
            {
                "tool": tool_name,
                "parameters": parameters,
                "result": result,
                "success": success,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep only last 1000 executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        if not self.execution_history:
            return {"total": 0, "success_rate": 0, "avg_duration": 0}

        total = len(self.execution_history)
        successes = sum(1 for e in self.execution_history if e["success"])
        avg_duration = sum(e["duration"] for e in self.execution_history) / total

        # Per-tool stats
        tool_stats = {}
        for e in self.execution_history:
            tool = e["tool"]
            if tool not in tool_stats:
                tool_stats[tool] = {"total": 0, "success": 0}
            tool_stats[tool]["total"] += 1
            if e["success"]:
                tool_stats[tool]["success"] += 1

        return {
            "total": total,
            "success_rate": successes / total,
            "avg_duration": avg_duration,
            "tool_stats": tool_stats,
        }


def create_android_tool_handlers(android_tools) -> Dict[str, Callable]:
    """Create handler functions that wrap AndroidTools methods"""
    return {
        "open_app": lambda **params: android_tools.open_app(params.get("app_name", "")),
        "close_app": lambda **params: android_tools.close_app(
            params.get("app_name", "")
        ),
        "tap_screen": lambda **params: android_tools.tap(
            params.get("x", 0), params.get("y", 0)
        ),
        "swipe_screen": lambda **params: android_tools.swipe(
            params.get("direction", "up"), params.get("distance", 500)
        ),
        "type_text": lambda **params: android_tools.type_text(params.get("text", "")),
        "press_key": lambda **params: android_tools.press_key(
            params.get("key", "back")
        ),
        "get_current_app": lambda **params: android_tools.get_current_app(),
        "take_screenshot": lambda **params: android_tools.take_screenshot(
            params.get("save_path", "/sdcard/screenshot.png")
        ),
        "get_notifications": lambda **params: android_tools.get_notifications(
            params.get("limit", 10)
        ),
    }


def bind_tool_handlers(registry: ToolRegistry, handlers: Dict[str, Callable]):
    """Bind handlers to tools in the registry"""
    for name, handler in handlers.items():
        registry.register_handler(name, handler)
    logger.info(f"Bound {len(handlers)} tool handlers")
