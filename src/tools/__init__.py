"""
AURA Tools Module

Provides the tool registry and executor for AURA's ReAct agent.

Usage:
    # Get a fully initialized registry with bound handlers
    registry = await get_initialized_registry()

    # Or manually:
    from src.tools import ToolRegistry
    from src.tools.handlers import get_tool_handlers

    registry = ToolRegistry()
    handlers = await get_tool_handlers()
    registry.bind_handlers(handlers)
"""

from .registry import ToolRegistry, ToolExecutor, ToolDefinition
from .base import BaseTool, ToolResult, ToolMetadata


async def get_initialized_registry() -> ToolRegistry:
    """
    Create a ToolRegistry with all handlers bound.

    This is the recommended way to get a ready-to-use tool registry.
    The registry has all core tools registered with their handlers wired up.

    Returns:
        ToolRegistry with handlers bound to all tools
    """
    from .handlers import get_tool_handlers

    registry = ToolRegistry()
    handlers = await get_tool_handlers()
    registry.bind_handlers(handlers)

    return registry


__all__ = [
    "ToolRegistry",
    "ToolExecutor",
    "ToolDefinition",
    "BaseTool",
    "ToolResult",
    "ToolMetadata",
    "get_initialized_registry",
]
