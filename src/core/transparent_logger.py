"""
AURA v3 - Transparent Background Process Logger
Shows users what's happening in the background without overwhelming them.

Features:
- Log levels: THOUGHT, ACTION, RESULT, SYSTEM
- User-facing log viewer via Telegram
- Configurable verbosity (minimal/normal/detailed)
- Privacy-respecting data access display
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
from threading import Lock
from pathlib import Path

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels for transparent logging"""

    THOUGHT = "thought"  # Agent thinking/reasoning
    ACTION = "action"  # Tool calls being executed
    RESULT = "result"  # Outcomes of actions
    SYSTEM = "system"  # Internal system operations


class VerbosityLevel(Enum):
    """How much detail to show users"""

    MINIMAL = "minimal"  # Just key highlights
    NORMAL = "normal"  # Balanced view
    DETAILED = "detailed"  # Full technical details


class LogPrivacy(Enum):
    """Privacy levels for log entries"""

    PUBLIC = "public"  # Can be shown to users
    SENSITIVE = "sensitive"  # Generic description only
    PRIVATE = "private"  # Never show to users


@dataclass
class LogEntry:
    """A single log entry"""

    id: str
    level: LogLevel
    content: str
    timestamp: datetime

    # Context
    category: str = ""  # e.g., "memory", "llm", "tool"
    tool_name: Optional[str] = None
    action_type: Optional[str] = None

    # Privacy & display
    privacy: LogPrivacy = LogPrivacy.PUBLIC
    user_visible: bool = True
    display_content: str = ""  # User-friendly version

    # Data access info (privacy-respecting)
    data_categories: List[str] = field(
        default_factory=list
    )  # e.g., ["messages", "calendar"]
    data_summary: str = ""  # e.g., "your recent messages"

    # Status
    status: str = "pending"  # pending, running, completed, failed
    duration_ms: Optional[int] = None

    # Related entries
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)


@dataclass
class VerbositySettings:
    """User's verbosity preferences"""

    level: VerbosityLevel = VerbosityLevel.NORMAL

    # What to show
    show_thoughts: bool = True
    show_actions: bool = True
    show_results: bool = True
    show_system: bool = False

    # Display options
    max_entries: int = 10
    show_timestamps: bool = True
    show_duration: bool = True
    use_emojis: bool = True


@dataclass
class ProcessingStatus:
    """Current processing status for UI display"""

    is_processing: bool = False
    current_phase: str = ""  # "thinking", "accessing_data", "executing", "responding"
    thought_preview: str = ""  # What's AURA thinking about
    tool_status: Optional[Dict[str, str]] = None  # tool_name -> status
    data_access: List[str] = field(default_factory=list)  # What's being accessed
    started_at: Optional[datetime] = None

    def get_display_message(self) -> str:
        """Get user-friendly status message"""
        if not self.is_processing:
            return ""

        messages = {
            "thinking": "🤔 AURA is thinking...",
            "accessing_data": f"📂 Accessing {', '.join(self.data_access[-2:]) if self.data_access else 'data'}...",
            "executing": "⚡ Executing actions...",
            "responding": "✍️ Formulating response...",
        }

        return messages.get(self.current_phase, "🔄 Processing...")

    def get_progress_indicator(self) -> str:
        """Get animated progress indicator"""
        if not self.is_processing:
            return ""

        phase_indicators = {
            "thinking": "💭",
            "accessing_data": "📂",
            "executing": "⚡",
            "responding": "✍️",
        }

        return phase_indicators.get(self.current_phase, "🔄")


class TransparentLogger:
    """
    Transparent background process logger.
    Shows users what AURA is doing without overwhelming them.
    """

    def __init__(self, storage_path: str = "data/transparent_logs.json"):
        self.storage_path = storage_path

        # Log storage
        self.logs: deque = deque(maxlen=500)  # Keep last 500
        self._lock = Lock()

        # Settings per user
        self.verbosity_settings: Dict[
            int, VerbositySettings
        ] = {}  # user_id -> settings

        # Current processing status
        self.processing_status = ProcessingStatus()

        # Active operation tracking
        self.active_operations: Dict[str, LogEntry] = {}

        # Statistics
        self.stats = {
            "total_logs": 0,
            "by_level": {level.value: 0 for level in LogLevel},
        }

    # =========================================================================
    # LOGGING METHODS
    # =========================================================================

    def _generate_id(self) -> str:
        """Generate unique log ID"""
        import uuid

        return f"log_{uuid.uuid4().hex[:8]}"

    async def log_thought(
        self,
        content: str,
        reasoning: str = "",
        category: str = "general",
        privacy: LogPrivacy = LogPrivacy.PUBLIC,
        data_categories: List[str] = None,
    ) -> LogEntry:
        """Log an agent thought"""
        import uuid

        # Generate user-friendly display
        display = self._generate_display(content, LogLevel.THOUGHT, privacy)

        entry = LogEntry(
            id=self._generate_id(),
            level=LogLevel.THOUGHT,
            content=content,
            timestamp=datetime.now(),
            category=category,
            privacy=privacy,
            user_visible=privacy != LogPrivacy.PRIVATE,
            display_content=display,
            data_categories=data_categories or [],
            data_summary=self._summarize_data_access(data_categories)
            if data_categories
            else "",
            status="completed",
        )

        await self._add_entry(entry)

        # Update processing status
        if self.processing_status.is_processing:
            self.processing_status.thought_preview = content[:100]

        return entry

    async def log_action_start(
        self,
        action_type: str,
        tool_name: Optional[str] = None,
        category: str = "tool",
        data_categories: List[str] = None,
        privacy: LogPrivacy = LogPrivacy.PUBLIC,
    ) -> LogEntry:
        """Log start of an action/tool execution"""

        content = f"Starting: {action_type}"
        if tool_name:
            content += f" ({tool_name})"

        display = self._generate_display(content, LogLevel.ACTION, privacy)

        entry = LogEntry(
            id=self._generate_id(),
            level=LogLevel.ACTION,
            content=content,
            timestamp=datetime.now(),
            category=category,
            tool_name=tool_name,
            action_type=action_type,
            privacy=privacy,
            user_visible=privacy != LogPrivacy.PRIVATE,
            display_content=display,
            data_categories=data_categories or [],
            data_summary=self._summarize_data_access(data_categories)
            if data_categories
            else "",
            status="running",
        )

        await self._add_entry(entry)

        # Track active operation
        self.active_operations[entry.id] = entry

        # Update processing status
        if self.processing_status.is_processing:
            if self.processing_status.tool_status is None:
                self.processing_status.tool_status = {}
            self.processing_status.tool_status[tool_name or action_type] = "running"

        return entry

    async def log_action_complete(
        self,
        entry_id: str,
        result_summary: str = "",
        success: bool = True,
        privacy: LogPrivacy = LogPrivacy.PUBLIC,
    ) -> Optional[LogEntry]:
        """Log completion of an action"""

        if entry_id not in self.active_operations:
            return None

        entry = self.active_operations.pop(entry_id)
        entry.status = "completed" if success else "failed"

        # Calculate duration
        if entry.duration_ms is None:
            entry.duration_ms = int(
                (datetime.now() - entry.timestamp).total_seconds() * 1000
            )

        # Add result
        if result_summary:
            entry.content += f" → {result_summary}"
            entry.display_content = self._generate_display(
                entry.content, LogLevel.RESULT, privacy
            )

        # Add as result entry
        result_entry = LogEntry(
            id=self._generate_id(),
            level=LogLevel.RESULT,
            content=result_summary or f"Completed: {entry.content}",
            timestamp=datetime.now(),
            category=entry.category,
            tool_name=entry.tool_name,
            privacy=privacy,
            user_visible=privacy != LogPrivacy.PRIVATE,
            display_content=self._generate_display(
                result_summary or "Done", LogLevel.RESULT, privacy
            ),
            status="completed" if success else "failed",
            duration_ms=entry.duration_ms,
            parent_id=entry_id,
        )

        await self._add_entry(result_entry)

        # Update processing status
        if self.processing_status.tool_status and entry.tool_name:
            self.processing_status.tool_status[entry.tool_name] = "completed"

        return result_entry

    async def log_system(
        self,
        content: str,
        category: str = "system",
    ) -> LogEntry:
        """Log internal system operation"""

        entry = LogEntry(
            id=self._generate_id(),
            level=LogLevel.SYSTEM,
            content=content,
            timestamp=datetime.now(),
            category=category,
            privacy=LogPrivacy.PRIVATE,  # System logs are private by default
            user_visible=False,
            display_content="",
            status="completed",
        )

        await self._add_entry(entry)
        return entry

    async def _add_entry(self, entry: LogEntry):
        """Add entry to logs"""
        with self._lock:
            self.logs.append(entry)
            self.stats["total_logs"] += 1
            self.stats["by_level"][entry.level.value] += 1

    # =========================================================================
    # PROCESSING STATUS
    # =========================================================================

    def start_processing(self, phase: str = "thinking", data_access: List[str] = None):
        """Mark that AURA is processing something"""
        self.processing_status.is_processing = True
        self.processing_status.current_phase = phase
        self.processing_status.started_at = datetime.now()
        if data_access:
            self.processing_status.data_access = data_access

    def update_phase(self, phase: str):
        """Update current processing phase"""
        self.processing_status.current_phase = phase

    def add_data_access(self, data_type: str):
        """Add data being accessed"""
        if data_type not in self.processing_status.data_access:
            self.processing_status.data_access.append(data_type)

    def stop_processing(self):
        """Mark processing as complete"""
        self.processing_status.is_processing = False
        self.processing_status.current_phase = ""
        self.processing_status.thought_preview = ""
        self.processing_status.tool_status = None
        self.processing_status.data_access = []
        self.processing_status.started_at = None

    def get_status(self) -> ProcessingStatus:
        """Get current processing status"""
        return self.processing_status

    # =========================================================================
    # LOG RETRIEVAL
    # =========================================================================

    def get_logs(
        self,
        user_id: int = None,
        levels: List[LogLevel] = None,
        categories: List[str] = None,
        limit: int = 10,
        since: datetime = None,
    ) -> List[LogEntry]:
        """Get logs with filters"""

        # Get verbosity settings
        settings = self._get_verbosity(user_id)

        # Filter by verbosity if not specified
        if levels is None:
            levels = self._get_allowed_levels(settings)

        # Filter logs
        filtered = []
        for entry in reversed(self.logs):
            # Skip non-visible
            if not entry.user_visible:
                continue

            # Filter by level
            if levels and entry.level not in levels:
                continue

            # Filter by category
            if categories and entry.category not in categories:
                continue

            # Filter by time
            if since and entry.timestamp < since:
                continue

            filtered.append(entry)

            if len(filtered) >= limit:
                break

        return filtered

    def get_thoughts(self, user_id: int = None, limit: int = 10) -> List[LogEntry]:
        """Get only thought-level logs"""
        return self.get_logs(user_id, levels=[LogLevel.THOUGHT], limit=limit)

    def get_actions(self, user_id: int = None, limit: int = 10) -> List[LogEntry]:
        """Get only action-level logs"""
        return self.get_logs(
            user_id, levels=[LogLevel.ACTION, LogLevel.RESULT], limit=limit
        )

    def format_for_telegram(
        self,
        logs: List[LogEntry],
        settings: VerbositySettings = None,
    ) -> str:
        """Format logs for Telegram display"""

        if settings is None:
            settings = VerbositySettings()

        if not logs:
            return "📝 No recent activity to show."

        lines = ["📋 *Recent Activity:*\n"]

        for entry in logs:
            # Format based on level
            if entry.level == LogLevel.THOUGHT:
                emoji = "💭" if settings.use_emojis else ""
                line = f"{emoji} {entry.display_content or entry.content}"
            elif entry.level == LogLevel.ACTION:
                if entry.status == "running":
                    emoji = "⚡" if settings.use_emojis else ""
                    line = f"{emoji} 🔄 {entry.display_content or entry.content}..."
                else:
                    emoji = "✅" if settings.use_emojis else ""
                    line = f"{emoji} {entry.display_content or entry.content}"
            elif entry.level == LogLevel.RESULT:
                emoji = "📊" if settings.use_emojis else ""
                line = f"{emoji} {entry.display_content or entry.content}"
            else:
                continue  # Skip system logs

            # Add duration if available
            if settings.show_duration and entry.duration_ms:
                if entry.duration_ms > 1000:
                    line += f" ({entry.duration_ms / 1000:.1f}s)"
                else:
                    line += f" ({entry.duration_ms}ms)"

            lines.append(line)

        # Add data access info if available
        data_access_entries = [e for e in logs if e.data_categories]
        if data_access_entries and settings.level != VerbosityLevel.MINIMAL:
            data_types = set()
            for e in data_access_entries:
                data_types.update(e.data_categories)

            if data_types:
                lines.append(f"\n📂 *Accessing:* {', '.join(sorted(data_types))}")

        return "\n".join(lines)

    # =========================================================================
    # SETTINGS
    # =========================================================================

    def _get_verbosity(self, user_id: int = None) -> VerbositySettings:
        """Get verbosity settings for user"""
        if user_id and user_id in self.verbosity_settings:
            return self.verbosity_settings[user_id]
        return VerbositySettings()

    def _get_allowed_levels(self, settings: VerbositySettings) -> List[LogLevel]:
        """Get allowed log levels based on verbosity"""
        levels = []

        if settings.show_thoughts:
            levels.append(LogLevel.THOUGHT)
        if settings.show_actions:
            levels.append(LogLevel.ACTION)
        if settings.show_results:
            levels.append(LogLevel.RESULT)
        if settings.show_system:
            levels.append(LogLevel.SYSTEM)

        return levels

    def set_verbosity(
        self,
        user_id: int,
        level: VerbosityLevel = None,
        show_thoughts: bool = None,
        show_actions: bool = None,
        show_results: bool = None,
        show_system: bool = None,
    ) -> VerbositySettings:
        """Update verbosity settings for user"""

        if user_id not in self.verbosity_settings:
            self.verbosity_settings[user_id] = VerbositySettings()

        settings = self.verbosity_settings[user_id]

        if level is not None:
            settings.level = level

            # Auto-adjust based on level
            if level == VerbosityLevel.MINIMAL:
                settings.show_thoughts = True
                settings.show_actions = True
                settings.show_results = False
                settings.show_system = False
                settings.max_entries = 5
            elif level == VerbosityLevel.DETAILED:
                settings.show_thoughts = True
                settings.show_actions = True
                settings.show_results = True
                settings.show_system = True
                settings.max_entries = 20

        if show_thoughts is not None:
            settings.show_thoughts = show_thoughts
        if show_actions is not None:
            settings.show_actions = show_actions
        if show_results is not None:
            settings.show_results = show_results
        if show_system is not None:
            settings.show_system = show_system

        return settings

    def clear_logs(self, user_id: int = None):
        """Clear log history"""
        with self._lock:
            if user_id:
                # Just reset stats for user
                pass
            else:
                self.logs.clear()
                self.stats = {
                    "total_logs": 0,
                    "by_level": {level.value: 0 for level in LogLevel},
                }

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _generate_display(
        self,
        content: str,
        level: LogLevel,
        privacy: LogPrivacy,
    ) -> str:
        """Generate user-friendly display version"""

        if privacy == LogPrivacy.PRIVATE:
            return ""

        if privacy == LogPrivacy.SENSITIVE:
            # Generic descriptions
            if level == LogLevel.THOUGHT:
                return "Analyzing your request..."
            elif level == LogLevel.ACTION:
                return "Accessing your data..."
            elif level == LogLevel.RESULT:
                return "Processing complete"

        # Return original for public
        return content

    def _summarize_data_access(self, data_categories: List[str]) -> str:
        """Create privacy-respecting summary of data access"""

        if not data_categories:
            return ""

        # Map to friendly names
        friendly_names = {
            "messages": "your messages",
            "calendar": "your calendar",
            "contacts": "your contacts",
            "location": "your location",
            "photos": "your photos",
            "files": "your files",
            "memory": "stored memories",
            "preferences": "your preferences",
            "health": "health data",
            "social": "social data",
        }

        names = [friendly_names.get(c, c) for c in data_categories]

        if len(names) == 1:
            return names[0]
        elif len(names) == 2:
            return f"{names[0]} and {names[1]}"
        else:
            return f"{names[0]}, {names[1]}, and {len(names) - 2} more"

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            **self.stats,
            "current_logs": len(self.logs),
            "is_processing": self.processing_status.is_processing,
        }


# Global instance
_transparent_logger: Optional[TransparentLogger] = None


def get_transparent_logger() -> TransparentLogger:
    """Get or create transparent logger instance"""
    global _transparent_logger
    if _transparent_logger is None:
        _transparent_logger = TransparentLogger()
    return _transparent_logger
