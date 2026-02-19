"""
AURA v3 - Next-Generation Personal Mobile AGI Assistant
100% Offline | Privacy-First | Self-Aware | Learning

A modular, privacy-first AGI assistant optimized for mobile devices.
"""

__version__ = "3.0.0"
__author__ = "AURA Team"

from .agent.loop import ReActAgent, AgentFactory
from .tools.registry import ToolRegistry, ToolExecutor
from .memory import HierarchicalMemory
from .learning.engine import LearningEngine
from .context.detector import ContextDetector
from .security.permissions import SecurityLayer, PermissionLevel
from .llm import LLMRunner, MockLLM
from .session.manager import SessionManager
from .channels.voice import VoiceChannel

__all__ = [
    "ReActAgent",
    "AgentFactory",
    "ToolRegistry",
    "ToolExecutor",
    "HierarchicalMemory",
    "LearningEngine",
    "ContextDetector",
    "SecurityLayer",
    "PermissionLevel",
    "LLMRunner",
    "MockLLM",
    "SessionManager",
    "VoiceChannel",
]
