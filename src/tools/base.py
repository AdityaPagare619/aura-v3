"""
Base classes for AURA tools
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from enum import Enum


class ToolCategory(Enum):
    COMMUNICATION = "communication"
    APP_CONTROL = "app_control"
    SCREEN = "screen"
    INFORMATION = "information"
    UTILITY = "utility"


@dataclass
class ToolMetadata:
    name: str
    category: ToolCategory
    risk_level: str = "low"
    requires_approval: bool = False
    description: str = ""


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseTool:
    """Base class for all AURA tools"""

    def __init__(self, name: str, metadata: ToolMetadata):
        self.name = name
        self.metadata = metadata

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool"""
        raise NotImplementedError

    def validate_params(self, **kwargs) -> bool:
        """Validate parameters"""
        return True


__all__ = ["BaseTool", "ToolResult", "ToolMetadata", "ToolCategory"]
