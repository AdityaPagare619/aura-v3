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


class ToolRegistry:
    """
    Registry of available tools with JSON Schema definitions

    Key innovation: Generates proper JSON schemas that LLM can understand
    """

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_core_tools()

    def _register_core_tools(self):
        """Register the core toolset with JSON schemas"""

        # ==================== COMMUNICATION TOOLS ====================

        self.register(
            ToolDefinition(
                name="send_whatsapp_message",
                description="Send a WhatsApp message to a contact",
                category="communication",
                risk_level="medium",
                requires_approval=True,
                parameters={
                    "type": "object",
                    "properties": {
                        "contact": {
                            "type": "string",
                            "description": "Name or phone number of the contact",
                        },
                        "message": {
                            "type": "string",
                            "description": "Message text to send",
                        },
                    },
                    "required": ["contact", "message"],
                },
                examples=[
                    {"contact": "Papa", "message": "On my way!"},
                    {"contact": "+1234567890", "message": "Meeting at 3pm"},
                ],
            )
        )

        self.register(
            ToolDefinition(
                name="make_phone_call",
                description="Make a phone call to a contact",
                category="communication",
                risk_level="high",
                requires_approval=True,
                parameters={
                    "type": "object",
                    "properties": {
                        "contact": {
                            "type": "string",
                            "description": "Name or phone number to call",
                        },
                        "delay_seconds": {
                            "type": "integer",
                            "description": "Optional delay before calling (learned from patterns)",
                            "minimum": 0,
                            "maximum": 60,
                            "default": 0,
                        },
                    },
                    "required": ["contact"],
                },
                examples=[
                    {"contact": "Papa"},
                    {"contact": "Emergency", "delay_seconds": 5},
                ],
            )
        )

        self.register(
            ToolDefinition(
                name="send_sms",
                description="Send an SMS message",
                category="communication",
                risk_level="medium",
                requires_approval=True,
                parameters={
                    "type": "object",
                    "properties": {
                        "phone_number": {
                            "type": "string",
                            "description": "Phone number to send SMS to",
                        },
                        "message": {"type": "string", "description": "Message text"},
                    },
                    "required": ["phone_number", "message"],
                },
                examples=[{"phone_number": "+1234567890", "message": "Running late!"}],
            )
        )

        # ==================== APP CONTROL TOOLS ====================

        self.register(
            ToolDefinition(
                name="open_app",
                description="Open an application on the device",
                category="app_control",
                risk_level="low",
                parameters={
                    "type": "object",
                    "properties": {
                        "app_name": {
                            "type": "string",
                            "description": "Name of the app to open (e.g., whatsapp, phone, messages, camera)",
                        },
                        "app_package": {
                            "type": "string",
                            "description": "Optional package name if name is ambiguous",
                        },
                    },
                    "required": ["app_name"],
                },
                examples=[{"app_name": "whatsapp"}, {"app_name": "camera"}],
            )
        )

        self.register(
            ToolDefinition(
                name="close_app",
                description="Close an application",
                category="app_control",
                risk_level="low",
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
                examples=[{"app_name": "whatsapp"}],
            )
        )

        self.register(
            ToolDefinition(
                name="explore_app",
                description="Explore an app's interface - click, scroll, read (MEMORIZED for future)",
                category="app_control",
                risk_level="low",
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "click",
                                "scroll_up",
                                "scroll_down",
                                "get_text",
                                "screenshot",
                            ],
                            "description": "Action to perform",
                        },
                        "target": {
                            "type": "string",
                            "description": "Description of target (e.g., 'send button', 'menu', 'search field')",
                        },
                        "coordinates": {
                            "type": "object",
                            "description": "Optional specific coordinates",
                            "properties": {
                                "x": {"type": "integer"},
                                "y": {"type": "integer"},
                            },
                        },
                        "text_input": {
                            "type": "string",
                            "description": "Text to type (for text input action)",
                        },
                    },
                    "required": ["action"],
                },
                examples=[
                    {"action": "click", "target": "send button"},
                    {"action": "scroll_down"},
                    {"action": "get_text", "target": "message content"},
                ],
            )
        )

        # ==================== SCREEN INTERACTION ====================

        self.register(
            ToolDefinition(
                name="tap_screen",
                description="Tap at specific screen coordinates",
                category="screen",
                risk_level="low",
                parameters={
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "X coordinate"},
                        "y": {"type": "integer", "description": "Y coordinate"},
                    },
                    "required": ["x", "y"],
                },
                examples=[{"x": 500, "y": 1000}],
            )
        )

        self.register(
            ToolDefinition(
                name="swipe_screen",
                description="Swipe on screen",
                category="screen",
                risk_level="low",
                parameters={
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right"],
                            "description": "Swipe direction",
                        },
                        "distance": {
                            "type": "integer",
                            "description": "Distance in pixels",
                            "default": 500,
                        },
                        "duration": {
                            "type": "integer",
                            "description": "Duration in milliseconds",
                            "default": 300,
                        },
                    },
                    "required": ["direction"],
                },
                examples=[{"direction": "up"}, {"direction": "left", "distance": 800}],
            )
        )

        self.register(
            ToolDefinition(
                name="type_text",
                description="Type text on screen (into focused field)",
                category="screen",
                risk_level="low",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to type"},
                        "enter_key": {
                            "type": "boolean",
                            "description": "Press enter after typing",
                            "default": False,
                        },
                    },
                    "required": ["text"],
                },
                examples=[
                    {"text": "Hello there!"},
                    {"text": "Search query", "enter_key": True},
                ],
            )
        )

        self.register(
            ToolDefinition(
                name="press_key",
                description="Press a hardware/system key",
                category="screen",
                risk_level="low",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "enum": [
                                "back",
                                "home",
                                "recent",
                                "volume_up",
                                "volume_down",
                                "power",
                            ],
                            "description": "Key to press",
                        }
                    },
                    "required": ["key"],
                },
                examples=[{"key": "back"}, {"key": "home"}],
            )
        )

        # ==================== INFORMATION TOOLS ====================

        self.register(
            ToolDefinition(
                name="get_current_app",
                description="Get information about currently open app",
                category="information",
                risk_level="low",
                parameters={"type": "object", "properties": {}},
                examples=[],
            )
        )

        self.register(
            ToolDefinition(
                name="take_screenshot",
                description="Take a screenshot of current screen",
                category="information",
                risk_level="low",
                parameters={
                    "type": "object",
                    "properties": {
                        "save_path": {
                            "type": "string",
                            "description": "Optional path to save screenshot",
                        }
                    },
                },
                examples=[{}],
            )
        )

        self.register(
            ToolDefinition(
                name="read_screen",
                description="Read text visible on screen using OCR",
                category="information",
                risk_level="low",
                parameters={
                    "type": "object",
                    "properties": {
                        "region": {
                            "type": "string",
                            "enum": ["all", "top", "bottom", "center"],
                            "description": "Region to read",
                            "default": "all",
                        }
                    },
                },
                examples=[{}, {"region": "top"}],
            )
        )

        self.register(
            ToolDefinition(
                name="get_notifications",
                description="Get recent notifications",
                category="information",
                risk_level="low",
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of notifications to fetch",
                            "default": 10,
                            "maximum": 50,
                        }
                    },
                },
                examples=[{"limit": 5}],
            )
        )

        self.register(
            ToolDefinition(
                name="get_contacts",
                description="Get contacts from phone",
                category="information",
                risk_level="medium",
                parameters={
                    "type": "object",
                    "properties": {
                        "search": {
                            "type": "string",
                            "description": "Search term for contact name",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum contacts to return",
                            "default": 20,
                        },
                    },
                },
                examples=[{"search": "Dad"}, {"limit": 10}],
            )
        )

        # ==================== UTILITY TOOLS ====================

        self.register(
            ToolDefinition(
                name="wait",
                description="Wait for specified seconds",
                category="utility",
                risk_level="low",
                parameters={
                    "type": "object",
                    "properties": {
                        "seconds": {
                            "type": "integer",
                            "description": "Seconds to wait",
                            "minimum": 1,
                            "maximum": 60,
                        }
                    },
                    "required": ["seconds"],
                },
                examples=[{"seconds": 5}],
            )
        )

        self.register(
            ToolDefinition(
                name="set_reminder",
                description="Set a reminder/alarm",
                category="utility",
                risk_level="medium",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Reminder title"},
                        "time_offset_minutes": {
                            "type": "integer",
                            "description": "Minutes from now to trigger",
                        },
                        "message": {
                            "type": "string",
                            "description": "Reminder message",
                        },
                    },
                    "required": ["title", "time_offset_minutes"],
                },
                examples=[
                    {
                        "title": "Call back",
                        "time_offset_minutes": 30,
                        "message": "Call Papa back",
                    }
                ],
            )
        )

        self.register(
            ToolDefinition(
                name="get_time",
                description="Get current time and date",
                category="utility",
                risk_level="low",
                parameters={"type": "object", "properties": {}},
                examples=[],
            )
        )

        self.register(
            ToolDefinition(
                name="get_location",
                description="Get current location (if available)",
                category="utility",
                risk_level="low",
                parameters={"type": "object", "properties": {}},
                examples=[],
            )
        )

    def register(self, tool: ToolDefinition):
        """Register a tool with its definition"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} ({tool.category})")

    def bind_handlers(self, handler_object: Any):
        """
        Bind handler methods from a handler object to registered tools.

        This connects tool definitions to actual implementations.
        Handler methods should match tool names (with underscores, e.g., 'send_message').

        Args:
            handler_object: Object with methods matching tool names (e.g., ToolHandlers instance)
        """
        bound_count = 0
        for tool_name, tool_def in self.tools.items():
            # Convert tool_name to method name (already snake_case)
            method_name = tool_name
            handler = getattr(handler_object, method_name, None)

            if handler is not None and callable(handler):
                tool_def.handler = handler
                bound_count += 1
                logger.debug(f"Bound handler for tool: {tool_name}")
            else:
                logger.debug(f"No handler found for tool: {tool_name}")

        logger.info(f"Bound {bound_count}/{len(self.tools)} tool handlers")

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get tool handler by name"""
        tool = self.tools.get(name)
        return tool.handler if tool else None

    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())

    def __len__(self) -> int:
        """Return number of registered tools"""
        return len(self.tools)

    def items(self):
        """Return tool items for iteration (dict-like interface)"""
        return self.tools.items()

    def keys(self):
        """Return tool names (dict-like interface)"""
        return self.tools.keys()

    def values(self):
        """Return tool definitions (dict-like interface)"""
        return self.tools.values()

    def __iter__(self):
        """Iterate over tool names"""
        return iter(self.tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool exists"""
        return name in self.tools

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
