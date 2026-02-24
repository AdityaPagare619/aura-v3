"""
AURA v3 Addons Package
Dynamic addon system for extending AURA capabilities
"""

__all__ = [
    "AppDiscovery",
    "AppEntry",
    "AppMetadata",
    "AppCapability",
    "AppCategory",
    "CapabilityGap",
    "get_app_discovery",
    "TermuxBridge",
    "get_termux_bridge",
    "CommandResult",
    "AppControl",
    "FileSystem",
    "MediaControl",
    "NotificationControl",
    "AdaptiveToolBinder",
    "ToolDefinition",
    "ToolCapability",
    "ToolInvocation",
    "get_tool_binder",
    "CapabilityGapHandler",
    "CapabilityType",
    "Strategy",
    "GapResolution",
    "get_capability_gap_handler",
    "KnowledgeGraph",
    "AppNode",
    "RelationshipEdge",
    "RelationshipType",
    "ValidityInterval",
    "TopologyMapper",
    "QueryEngine",
    "get_knowledge_graph",
    "get_topology_mapper",
    "get_query_engine",
    "initialize_from_app_discovery",
]

from src.addons.discovery import (
    AppDiscovery,
    AppEntry,
    AppMetadata,
    AppCapability,
    AppCategory,
    CapabilityGap,
    get_app_discovery,
)
from src.addons.termux_bridge import (
    TermuxBridge,
    get_termux_bridge,
    CommandResult,
    AppControl,
    FileSystem,
    MediaControl,
    NotificationControl,
)
from src.addons.tool_binding import (
    AdaptiveToolBinder,
    ToolDefinition,
    ToolCapability,
    ToolInvocation,
    get_tool_binder,
)
from src.addons.capability_gap import (
    CapabilityGapHandler,
    CapabilityType,
    Strategy,
    GapResolution,
    get_capability_gap_handler,
)
from src.memory.knowledge_graph import (
    KnowledgeGraph,
    AppNode,
    RelationshipEdge,
    RelationshipType,
    ValidityInterval,
    TopologyMapper,
    QueryEngine,
    get_knowledge_graph,
    get_topology_mapper,
    get_query_engine,
    initialize_from_app_discovery,
)
