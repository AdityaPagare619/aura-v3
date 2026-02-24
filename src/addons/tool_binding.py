"""
AURA v3 Adaptive Tool Binding System
Automatically binds discovered apps to LLM-executable tools

This is what makes AURA "find ways" - converting app capabilities
into tools the LLM can invoke

Key pattern from OpenClaw:
- Tools defined as JSON schemas
- LLM knows exact parameters, types
- Enables precise tool calling
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ToolCapability(Enum):
    """What a tool can do"""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    COMMAND_EXEC = "command_exec"
    API_CALL = "api_call"
    NOTIFICATION = "notification"
    VIBRATE = "vibrate"
    CAMERA = "camera"
    LOCATION = "location"
    CONTACTS = "contacts"
    CALL = "call"
    SMS = "sms"
    APP_LAUNCH = "app_launch"
    BROWSER = "browser"
    SEARCH = "search"


@dataclass
class ToolDefinition:
    """
    JSON Schema definition for a tool

    This is the bridge between discovered apps and LLM
    """

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    examples: List[Dict] = field(default_factory=list)
    category: str = "general"
    risk_level: str = "low"  # low, medium, high, critical

    # Execution
    handler: Optional[Callable] = None
    requires_approval: bool = False

    # Capability mapping
    capabilities: List[ToolCapability] = field(default_factory=list)
    app_source: Optional[str] = None

    # Usage tracking
    use_count: int = 0
    success_rate: float = 1.0
    avg_execution_time_ms: float = 0.0


@dataclass
class ToolInvocation:
    """Record of a tool invocation"""

    tool_name: str
    parameters: Dict[str, Any]
    timestamp: datetime
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class AdaptiveToolBinder:
    """
    Adaptive Tool Binding System

    Converts discovered apps into LLM-executable tools
    with automatic fallback chains

    Key features:
    - Dynamic tool registration
    - Capability-based matching
    - Usage learning (which tools work best)
    - Automatic fallback chains
    - Risk-based approval
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._capability_to_tools: Dict[ToolCapability, List[str]] = {}
        self._invocation_history: List[ToolInvocation] = []

        # Learning state
        self._success_patterns: Dict[str, float] = {}  # tool -> success rate

        # Register core tools
        self._register_core_tools()

    def _register_core_tools(self):
        """Register core AURA tools"""
        core_tools = [
            ToolDefinition(
                name="open_app",
                description="Open an application on the device by package name",
                parameters={
                    "type": "object",
                    "properties": {
                        "package_name": {
                            "type": "string",
                            "description": "Android package name (e.g., com.whatsapp)",
                        }
                    },
                    "required": ["package_name"],
                },
                category="system",
                risk_level="medium",
                capabilities=[ToolCapability.APP_LAUNCH],
            ),
            ToolDefinition(
                name="list_apps",
                description="List installed applications on the device",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Filter apps by name pattern",
                        }
                    },
                },
                category="system",
                risk_level="low",
                capabilities=[ToolCapability.APP_LAUNCH],
            ),
            ToolDefinition(
                name="find_files",
                description="Find files on the device by name pattern",
                parameters={
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory to search in",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "File name pattern (e.g., *.jpg)",
                        },
                    },
                    "required": ["pattern"],
                },
                category="filesystem",
                risk_level="low",
                capabilities=[ToolCapability.FILE_READ],
            ),
            ToolDefinition(
                name="read_file",
                description="Read contents of a text file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Full path to the file",
                        }
                    },
                    "required": ["path"],
                },
                category="filesystem",
                risk_level="medium",
                capabilities=[ToolCapability.FILE_READ],
            ),
            ToolDefinition(
                name="search_content",
                description="Search for text within files",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory to search in",
                        },
                        "query": {
                            "type": "string",
                            "description": "Text to search for",
                        },
                    },
                    "required": ["query"],
                },
                category="filesystem",
                risk_level="low",
                capabilities=[ToolCapability.SEARCH],
            ),
            ToolDefinition(
                name="send_notification",
                description="Send a notification to the user",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Notification title",
                        },
                        "content": {
                            "type": "string",
                            "description": "Notification message",
                        },
                        "urgency": {
                            "type": "string",
                            "enum": ["low", "normal", "high"],
                            "default": "normal",
                        },
                    },
                    "required": ["title", "content"],
                },
                category="notification",
                risk_level="low",
                requires_approval=False,
                capabilities=[ToolCapability.NOTIFICATION],
            ),
            ToolDefinition(
                name="vibrate",
                description="Vibrate the device",
                parameters={
                    "type": "object",
                    "properties": {
                        "duration": {
                            "type": "integer",
                            "description": "Duration in milliseconds",
                            "default": 500,
                        }
                    },
                },
                category="notification",
                risk_level="low",
                capabilities=[ToolCapability.VIBRATE],
            ),
            ToolDefinition(
                name="get_location",
                description="Get current GPS location",
                parameters={"type": "object"},
                category="system",
                risk_level="medium",
                requires_approval=True,
                capabilities=[ToolCapability.LOCATION],
            ),
            ToolDefinition(
                name="take_photo",
                description="Take a photo using the camera",
                parameters={
                    "type": "object",
                    "properties": {
                        "save_path": {
                            "type": "string",
                            "description": "Where to save the photo",
                        }
                    },
                },
                category="media",
                risk_level="medium",
                requires_approval=True,
                capabilities=[ToolCapability.CAMERA],
            ),
            ToolDefinition(
                name="list_images",
                description="List images in a directory",
                parameters={
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory to list images from",
                            "default": "/sdcard/DCIM",
                        }
                    },
                },
                category="media",
                risk_level="low",
                capabilities=[ToolCapability.FILE_READ],
            ),
        ]

        for tool in core_tools:
            self.register_tool(tool)

    def register_tool(self, tool: ToolDefinition):
        """Register a new tool"""
        self._tools[tool.name] = tool

        # Update capability map
        for cap in tool.capabilities:
            if cap not in self._capability_to_tools:
                self._capability_to_tools[cap] = []
            if tool.name not in self._capability_to_tools[cap]:
                self._capability_to_tools[cap].append(tool.name)

        logger.info(f"Registered tool: {tool.name}")

    def register_app_tools(self, app_entry) -> List[ToolDefinition]:
        """
        Register tools from a discovered app

        Converts app capabilities into LLM-usable tools
        """
        tools = []

        # Map app capabilities to tool definitions
        cap_to_tool = {
            "file_access": ["find_files", "read_file", "list_images"],
            "camera_access": ["take_photo"],
            "location_access": ["get_location"],
            "notifications": ["send_notification"],
        }

        for cap in app_entry.metadata.capabilities:
            tool_names = cap_to_tool.get(cap.value, [])
            for tool_name in tool_names:
                if tool_name in self._tools:
                    tool = self._tools[tool_name]
                    tool.app_source = app_entry.id
                    tools.append(tool)

        return tools

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name"""
        return self._tools.get(name)

    def get_tools_by_capability(
        self, capability: ToolCapability
    ) -> List[ToolDefinition]:
        """Get all tools that provide a capability"""
        tool_names = self._capability_to_tools.get(capability, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_all_tools(self) -> List[ToolDefinition]:
        """Get all registered tools"""
        return list(self._tools.values())

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get tools formatted for LLM consumption

        Returns JSON schema format for tool definitions
        """
        tools = []
        for tool in self._tools.values():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        return tools

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        handler: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool with fallback support

        This is where the "finding ways" happens:
        1. Try the requested tool
        2. If fails, try alternatives
        3. Learn from success/failure
        """
        start_time = datetime.now()

        tool = self.get_tool(tool_name)
        if not tool:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        # Use provided handler or tool's handler
        exec_handler = handler or tool.handler
        if not exec_handler:
            return {"success": False, "error": f"No handler for tool: {tool_name}"}

        try:
            # Execute the tool
            result = await exec_handler(parameters)

            # Record invocation
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            invocation = ToolInvocation(
                tool_name=tool_name,
                parameters=parameters,
                timestamp=datetime.now(),
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )
            self._invocation_history.append(invocation)

            # Update tool stats
            tool.use_count += 1
            tool.avg_execution_time_ms = (
                tool.avg_execution_time_ms * (tool.use_count - 1) + execution_time
            ) / tool.use_count

            return {"success": True, "result": result}

        except Exception as e:
            logger.error(f"Tool execution error: {e}")

            # Record failure
            invocation = ToolInvocation(
                tool_name=tool_name,
                parameters=parameters,
                timestamp=datetime.now(),
                success=False,
                error=str(e),
            )
            self._invocation_history.append(invocation)

            # Try fallback
            fallback_result = await self._try_fallback(tool, parameters)
            if fallback_result:
                return fallback_result

            return {"success": False, "error": str(e)}

    async def _try_fallback(
        self, failed_tool: ToolDefinition, parameters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Try fallback tools when primary tool fails

        This implements the "finding ways" behavior
        """
        # Get tools with same capabilities
        for cap in failed_tool.capabilities:
            alternative_tools = self.get_tools_by_capability(cap)

            for alt_tool in alternative_tools:
                if alt_tool.name == failed_tool.name:
                    continue

                # Try alternative
                try:
                    if alt_tool.handler:
                        result = await alt_tool.handler(parameters)
                        logger.info(
                            f"Fallback success: {failed_tool.name} -> {alt_tool.name}"
                        )
                        return {
                            "success": True,
                            "result": result,
                            "fallback": alt_tool.name,
                        }
                except Exception as e:
                    logger.debug(f"Fallback {alt_tool.name} also failed: {e}")
                    continue

        return None

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        stats = {
            "total_invocations": len(self._invocation_history),
            "tools": {},
        }

        for tool_name, tool in self._tools.items():
            stats["tools"][tool_name] = {
                "use_count": tool.use_count,
                "success_rate": tool.success_rate,
                "avg_execution_time_ms": tool.avg_execution_time_ms,
            }

        return stats


# Global instance
_tool_binder: Optional[AdaptiveToolBinder] = None


def get_tool_binder() -> AdaptiveToolBinder:
    """Get or create tool binder instance"""
    global _tool_binder
    if _tool_binder is None:
        _tool_binder = AdaptiveToolBinder()
    return _tool_binder
