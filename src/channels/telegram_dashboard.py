"""
AURA v3 - Telegram Mission Control Dashboard
Rich dashboard accessible from Telegram showing AURA's state and inner life.

Commands:
/dashboard or /mission - Main dashboard
/inner - Show AURA's inner voice/thoughts
/status - Quick status overview
/mood - AURA's current emotional state
"""

import asyncio
import logging
import psutil
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random

logger = logging.getLogger(__name__)


class DashboardSection(Enum):
    """Dashboard sections"""

    OVERVIEW = "overview"
    SYSTEM = "system"
    MEMORY = "memory"
    TOOLS = "tools"
    MOOD = "mood"
    INNER_VOICE = "inner_voice"


@dataclass
class SystemStatus:
    """System status data"""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    disk_percent: float = 0.0
    uptime_hours: float = 0.0
    is_online: bool = True
    llm_loaded: bool = False
    model_name: str = "Not loaded"


@dataclass
class MoodStatus:
    """AURA's mood status"""

    primary: str = "calm"
    secondary: str = "curious"
    intensity: float = 0.5
    cause: str = ""
    trust_score: float = 0.0
    trust_phase: str = "learning"
    is_processing: bool = False


@dataclass
class ToolStatus:
    """Tool execution status"""

    current_tool: Optional[str] = None
    tool_history: List[str] = field(default_factory=list)
    recent_results: List[str] = field(default_factory=list)
    success_rate: float = 1.0


@dataclass
class MemoryStatus:
    """Memory system status"""

    total_memories: int = 0
    episodic_count: int = 0
    semantic_count: int = 0
    recent_recall_count: int = 0
    memory_usage_mb: float = 0.0


@dataclass
class InnerVoiceStatus:
    """Inner voice/thinking status"""

    current_thought: str = ""
    reasoning: str = ""
    confidence: float = 0.5
    consideration: str = ""
    recent_thoughts: List[str] = field(default_factory=list)


class DashboardRenderer:
    """
    Renders AURA's state as formatted Telegram messages.
    Creates rich, visually appealing dashboards.
    """

    # Emoji mappings
    MOOD_EMOJIS = {
        "curious": "🧐",
        "focused": "🎯",
        "confident": "💪",
        "uncertain": "🤔",
        "concerned": "😟",
        "happy": "😊",
        "worried": "😰",
        "excited": "🤩",
        "calm": "😌",
        "tired": "😴",
        "frustrated": "😤",
        "hopeful": "🌟",
    }

    STATUS_EMOJIS = {
        "online": "🟢",
        "offline": "🔴",
        "processing": "⚡",
        "idle": "💤",
        "error": "❌",
    }

    def __init__(self, aura_instance=None):
        self.aura = aura_instance
        self._system_cache = {}
        self._cache_ttl = 5  # seconds

    async def _get_system_status(self) -> SystemStatus:
        """Get current system status"""
        now = datetime.now()

        # Check cache
        if self._system_cache.get("timestamp"):
            age = (now - self._system_cache["timestamp"]).total_seconds()
            if age < self._cache_ttl:
                return self._system_cache["data"]

        status = SystemStatus()

        try:
            # CPU
            status.cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory
            mem = psutil.virtual_memory()
            status.memory_percent = mem.percent
            status.memory_used_gb = mem.used / (1024**3)
            status.memory_total_gb = mem.total / (1024**3)

            # Disk
            disk = psutil.disk_usage("/")
            status.disk_percent = disk.percent

            # Uptime
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            status.uptime_hours = (now - boot_time).total_seconds() / 3600

            # Network
            try:
                import socket

                socket.create_connection(("8.8.8.8", 53), timeout=1)
                status.is_online = True
            except:
                status.is_online = False

        except Exception as e:
            logger.warning(f"Error getting system status: {e}")

        # LLM status
        if self.aura and hasattr(self.aura, "llm"):
            try:
                status.llm_loaded = self.aura.llm.is_loaded()
                if hasattr(self.aura.llm, "config") and self.aura.llm.config:
                    status.model_name = getattr(
                        self.aura.llm.config, "model_type", "Unknown"
                    )
            except:
                pass

        # Cache the result
        self._system_cache = {"timestamp": now, "data": status}

        return status

    async def _get_mood_status(self) -> MoodStatus:
        """Get current mood status"""
        status = MoodStatus()

        try:
            # Try to get from feelings meter
            if self.aura and hasattr(self.aura, "feelings_meter"):
                fm = self.aura.feelings_meter
                feeling = fm.get_current_feeling()
                status.primary = feeling.primary.value
                status.secondary = feeling.secondary.value
                status.intensity = feeling.intensity
                status.cause = feeling.cause

                stats = fm.get_statistics()
                status.trust_score = stats.get("trust_overall", 0.0)
                status.trust_phase = stats.get("trust_phase", "learning")

        except Exception as e:
            logger.warning(f"Error getting mood status: {e}")
            # Use defaults

        # Try inner voice for processing status
        try:
            if self.aura and hasattr(self.aura, "inner_voice"):
                iv = self.aura.inner_voice
                transparent_logs = await iv.get_transparent_logs(limit=1)
                status.is_processing = transparent_logs.get("processing", {}).get(
                    "is_processing", False
                )
        except:
            pass

        return status

    async def _get_tool_status(self) -> ToolStatus:
        """Get tool execution status"""
        status = ToolStatus()

        try:
            # Try to get from transparent logger
            if self.aura:
                from src.core.transparent_logger import get_transparent_logger

                logger_instance = get_transparent_logger()
                actions = logger_instance.get_actions(limit=5)

                status.tool_history = [
                    a.display_content or a.content[:50] for a in actions[:5]
                ]

                if actions:
                    status.current_tool = (
                        actions[0].display_content or actions[0].content[:30]
                    )

                # Calculate success rate
                completed = [a for a in actions if a.status in ["completed", "success"]]
                if actions:
                    status.success_rate = len(completed) / len(actions)

        except Exception as e:
            logger.warning(f"Error getting tool status: {e}")

        return status

    async def _get_memory_status(self) -> MemoryStatus:
        """Get memory system status"""
        status = MemoryStatus()

        try:
            if self.aura and hasattr(self.aura, "memory"):
                mem = self.aura.memory

                # Get stats if available
                if hasattr(mem, "get_stats"):
                    stats = mem.get_stats()
                    status.total_memories = stats.get("total_memories", 0)

                if hasattr(mem, "episodic"):
                    status.episodic_count = (
                        len(mem.episodic) if hasattr(mem, "episodic") else 0
                    )

                if hasattr(mem, "semantic"):
                    status.semantic_count = (
                        len(mem.semantic) if hasattr(mem, "semantic") else 0
                    )

        except Exception as e:
            logger.warning(f"Error getting memory status: {e}")

        return status

    async def _get_inner_voice_status(self) -> InnerVoiceStatus:
        """Get inner voice/thinking status"""
        status = InnerVoiceStatus()

        try:
            if self.aura and hasattr(self.aura, "inner_voice"):
                iv = self.aura.inner_voice

                # Get recent thoughts
                thoughts = iv.get_recent_thoughts(count=3)
                status.recent_thoughts = [t.content[:60] for t in thoughts]

                if thoughts:
                    latest = thoughts[-1]
                    status.current_thought = latest.content
                    status.reasoning = latest.reasoning
                    status.confidence = latest.confidence

                # Get transparent logs for current consideration
                logs_data = await iv.get_transparent_logs(limit=3)
                processing = logs_data.get("processing", {})

                if processing.get("is_processing"):
                    status.consideration = processing.get("message", "Processing...")

        except Exception as e:
            logger.warning(f"Error getting inner voice status: {e}")
            # Use sample data for demo
            status.current_thought = "Analyzing your request..."
            status.reasoning = "Understanding context and intent"
            status.confidence = 0.75
            status.consideration = "Checking available tools"
            status.recent_thoughts = [
                "User sent a message",
                "Classifying intent",
                "Preparing response",
            ]

        return status

    # =========================================================================
    # RENDERING METHODS
    # =========================================================================

    def _format_progress_bar(self, value: float, length: int = 5) -> str:
        """Create a progress bar"""
        filled = int(value * length)
        return "█" * filled + "░" * (length - filled)

    def _format_percentage(self, value: float) -> str:
        """Format percentage with emoji indicator"""
        if value >= 80:
            return f"🔴 {value:.1f}%"
        elif value >= 50:
            return f"🟡 {value:.1f}%"
        else:
            return f"🟢 {value:.1f}%"

    async def render_dashboard(self) -> str:
        """Render the main dashboard"""
        # Gather all status data in parallel
        system, mood, tools, memory, inner_voice = await asyncio.gather(
            self._get_system_status(),
            self._get_mood_status(),
            self._get_tool_status(),
            self._get_memory_status(),
            self._get_inner_voice_status(),
        )

        # Build dashboard
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║            🎯 AURA MISSION CONTROL v3                    ║",
            "╠══════════════════════════════════════════════════════════╣",
        ]

        # Status row
        status_icon = (
            self.STATUS_EMOJIS["processing"]
            if mood.is_processing
            else self.STATUS_EMOJIS["online"]
        )
        lines.append(
            f"║  {status_icon} Status: {'Processing' if mood.is_processing else 'Ready'} | Uptime: {system.uptime_hours:.1f}h        ║"
        )

        lines.append("╠══════════════════════════════════════════════════════════╣")
        lines.append("║  📊 SYSTEM          💾 MEMORY         🧠 MOOD            ║")

        # System card
        sys_bar = self._format_progress_bar(system.memory_percent / 100)
        lines.append(
            f"║  CPU: {self._format_percentage(system.cpu_percent):<12} RAM: [{sys_bar}] {system.memory_percent:.0f}%    ║"
        )

        # Memory card
        lines.append(
            f"║  Memories: {memory.total_memories:<5} Episodic: {memory.episodic_count:<4} Semantic: {memory.semantic_count:<4}  ║"
        )

        # Mood card
        mood_emoji = self.MOOD_EMOJIS.get(mood.primary, "😐")
        mood_bar = self._format_progress_bar(mood.intensity)
        lines.append(
            f"║  {mood_emoji} {mood.primary.capitalize():<10} [{mood_bar}] {mood.intensity:.0%}  Trust: {mood.trust_score:.1f}/10    ║"
        )

        lines.append("╠══════════════════════════════════════════════════════════╣")

        # Inner voice section
        lines.append("║  💭 INNER VOICE                                           ║")

        if inner_voice.current_thought:
            thought_preview = (
                inner_voice.current_thought[:40] + "..."
                if len(inner_voice.current_thought) > 40
                else inner_voice.current_thought
            )
            lines.append(f"║  → {thought_preview:<44} ║")

            conf_bar = self._format_progress_bar(inner_voice.confidence)
            lines.append(
                f"║  Confidence: [{conf_bar}] {inner_voice.confidence:.0%}                      ║"
            )
        else:
            lines.append(f"║  → Idle... waiting for input                            ║")

        if inner_voice.consideration:
            lines.append(f"║  ⏳ {inner_voice.consideration[:40]:<44} ║")

        lines.append("╠══════════════════════════════════════════════════════════╣")

        # Quick actions hint
        lines.append("║  /inner - Thoughts  |  /mood - Feelings  |  /status    ║")
        lines.append("╚══════════════════════════════════════════════════════════╝")

        return "\n".join(lines)

    async def render_status_card(self) -> str:
        """Render quick status card"""
        system = await self._get_system_status()
        mood = await self._get_mood_status()

        card = f"""
🔴 *SYSTEM STATUS*

*Resources:*
• CPU: {self._format_percentage(system.cpu_percent)}
• RAM: {self._format_percentage(system.memory_percent)}
• Disk: {self._format_percentage(system.disk_percent)}
• Uptime: {system.uptime_hours:.1f} hours

*Network:*
• {"🟢 Online" if system.is_online else "🔴 Offline"}

*AI:*
• LLM: {"✅ " + system.model_name if system.llm_loaded else "❌ Not loaded"}
• Processing: {"⚡ Yes" if mood.is_processing else "💤 No"}
"""
        return card.strip()

    async def render_mood_card(self) -> str:
        """Render mood/feelings card"""
        mood = await self._get_mood_status()

        # Format trust meter
        trust_bar = self._format_progress_bar(mood.trust_score / 10)

        # Get phase description
        phase_descriptions = {
            "introduction": "We're just getting started!",
            "learning": "I'm learning about you...",
            "understanding": "I'm starting to understand you.",
            "comfortable": "We have a good rapport!",
            "partnership": "We really understand each other!",
        }
        phase_desc = phase_descriptions.get(mood.trust_phase, "Building connection...")

        card = f"""
😊 *AURA'S MOOD*

*Current State:*
{self.MOOD_EMOJIS.get(mood.primary, "😐")} *{mood.primary.capitalize()}* (intensity: {mood.intensity:.0%})

{self.MOOD_EMOJIS.get(mood.secondary, "✨")} Secondary: {mood.secondary.capitalize()}

{f"📝 *Why:* {mood.cause}" if mood.cause else ""}

*Trust Level:*
[{trust_bar}] {mood.trust_score:.1f}/10
_{phase_desc}_
"""
        return card.strip()

    async def render_inner_voice(self) -> str:
        """Render inner voice/thoughts visualization"""
        inner_voice = await self._get_inner_voice_status()
        mood = await self._get_mood_status()

        lines = [
            "💭 *AURA'S INNER WORLD*",
            "",
            "*🎯 Current Thought:*",
        ]

        if inner_voice.current_thought:
            lines.append(f"  {inner_voice.current_thought}")
        else:
            lines.append("  (Thinking about nothing specific)")

        lines.append("")
        lines.append("*🔍 Reasoning:*")
        lines.append(f"  {inner_voice.reasoning or 'Processing context...'}")

        # Confidence visualization
        conf_bar = self._format_progress_bar(inner_voice.confidence, length=10)
        confidence_label = (
            "Very unsure"
            if inner_voice.confidence < 0.3
            else "Somewhat confident"
            if inner_voice.confidence < 0.7
            else "Very confident"
        )

        lines.append("")
        lines.append(f"*📊 Confidence:* [{conf_bar}] {confidence_label}")

        if inner_voice.consideration:
            lines.append("")
            lines.append(f"*⏳ Considering:* {inner_voice.consideration}")

        # Recent thought stream
        if inner_voice.recent_thoughts:
            lines.append("")
            lines.append("*📝 Recent Thoughts:*")
            for i, thought in enumerate(inner_voice.recent_thoughts, 1):
                lines.append(f"  {i}. {thought[:50]}...")

        # Current mood context
        lines.append("")
        lines.append(
            f"*💡 Mood Context:* Feeling {mood.primary} ({mood.intensity:.0%} intensity)"
        )

        return "\n".join(lines)

    async def render_tools_card(self) -> str:
        """Render tools status card"""
        tools = await self._get_tool_status()

        lines = [
            "🔧 *TOOLS & ACTIONS*",
            "",
        ]

        if tools.current_tool:
            lines.append(f"*🔄 Current:* {tools.current_tool}")
        else:
            lines.append("*🔄 Current:* Idle")

        # Success rate
        success_bar = self._format_progress_bar(tools.success_rate, length=10)
        lines.append(f"*📈 Success Rate:* [{success_bar}] {tools.success_rate:.0%}")

        if tools.tool_history:
            lines.append("")
            lines.append("*📜 Recent Tools:*")
            for tool in tools.tool_history[:5]:
                lines.append(f"  • {tool[:50]}")

        return "\n".join(lines)

    async def render_memory_card(self) -> str:
        """Render memory status card"""
        memory = await self._get_memory_status()

        card = f"""
💾 *MEMORY STATUS*

*Storage:*
• Total Memories: {memory.total_memories:,}
• Episodic: {memory.episodic_count:,}
• Semantic: {memory.semantic_count:,}

*Recent Activity:*
• Recall Operations: {memory.recent_recall_count:,}

*Usage:*
• {memory.memory_usage_mb:.1f} MB estimated
"""
        return card.strip()

    # =========================================================================
    # COMMAND HANDLERS
    # =========================================================================

    async def handle_dashboard(self) -> str:
        """Handle /dashboard or /mission command"""
        return await self.render_dashboard()

    async def handle_inner(self) -> str:
        """Handle /inner command"""
        return await self.render_inner_voice()

    async def handle_status(self) -> str:
        """Handle /status command"""
        return await self.render_status_card()

    async def handle_mood(self) -> str:
        """Handle /mood command"""
        return await self.render_mood_card()

    async def handle_tools(self) -> str:
        """Handle /tools command"""
        return await self.render_tools_card()

    async def handle_memory_cmd(self) -> str:
        """Handle /memory command"""
        return await self.render_memory_card()


# =========================================================================
# TELEGRAM BOT INTEGRATION
# =========================================================================


async def register_dashboard_commands(bot_handler) -> None:
    """Register dashboard commands with the Telegram bot"""

    renderer = DashboardRenderer(bot_handler.aura)

    # Add command handlers
    bot_handler.dispatcher.add_handler(
        CommandHandler("dashboard", lambda u, c: _dashboard_cmd(u, c, renderer))
    )
    bot_handler.dispatcher.add_handler(
        CommandHandler("mission", lambda u, c: _dashboard_cmd(u, c, renderer))
    )
    bot_handler.dispatcher.add_handler(
        CommandHandler("inner", lambda u, c: _inner_cmd(u, c, renderer))
    )
    bot_handler.dispatcher.add_handler(
        CommandHandler("status", lambda u, c: _status_cmd(u, c, renderer))
    )
    bot_handler.dispatcher.add_handler(
        CommandHandler("mood", lambda u, c: _mood_cmd(u, c, renderer))
    )
    bot_handler.dispatcher.add_handler(
        CommandHandler("tools", lambda u, c: _tools_cmd(u, c, renderer))
    )
    bot_handler.dispatcher.add_handler(
        CommandHandler("memory", lambda u, c: _memory_cmd(u, c, renderer))
    )


async def _dashboard_cmd(update, context, renderer: DashboardRenderer):
    """Handle /dashboard command"""
    await update.message.reply_text(
        await renderer.handle_dashboard(), parse_mode="Markdown"
    )


async def _inner_cmd(update, context, renderer: DashboardRenderer):
    """Handle /inner command"""
    await update.message.reply_text(
        await renderer.handle_inner(), parse_mode="Markdown"
    )


async def _status_cmd(update, context, renderer: DashboardRenderer):
    """Handle /status command"""
    await update.message.reply_text(
        await renderer.handle_status(), parse_mode="Markdown"
    )


async def _mood_cmd(update, context, renderer: DashboardRenderer):
    """Handle /mood command"""
    await update.message.reply_text(await renderer.handle_mood(), parse_mode="Markdown")


async def _tools_cmd(update, context, renderer: DashboardRenderer):
    """Handle /tools command"""
    await update.message.reply_text(
        await renderer.handle_tools(), parse_mode="Markdown"
    )


async def _memory_cmd(update, context, renderer: DashboardRenderer):
    """Handle /memory command"""
    await update.message.reply_text(
        await renderer.handle_memory_cmd(), parse_mode="Markdown"
    )


# =========================================================================
# STANDALONE DEMO
# =========================================================================


async def demo_dashboard():
    """Demo the dashboard renderer"""
    renderer = DashboardRenderer()

    print("=" * 60)
    print("AURA MISSION CONTROL DASHBOARD - DEMO")
    print("=" * 60)

    print("\n📊 MAIN DASHBOARD:")
    print(await renderer.handle_dashboard())

    print("\n😊 MOOD STATUS:")
    print(await renderer.handle_mood())

    print("\n💭 INNER VOICE:")
    print(await renderer.handle_inner())

    print("\n🔧 TOOLS STATUS:")
    print(await renderer.handle_tools())

    print("\n💾 MEMORY STATUS:")
    print(await renderer.handle_memory_cmd())


if __name__ == "__main__":
    asyncio.run(demo_dashboard())
