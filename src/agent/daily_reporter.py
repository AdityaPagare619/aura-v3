"""
AURA v3 Proactive Daily/Weekly Reporter
========================================

Provides useful, actionable insights through scheduled reports:
- Daily Digest: Morning/evening summary
- Weekly Review: Week in review
- Insight Report: When AURA discovers something useful
- Alert Report: Important notifications

Key features:
- Proactive delivery - sends even if not asked
- Respects user preference - can disable anytime
- Smart timing - learns when user prefers reports
- Compact format - not overwhelming
- Actionable - includes suggestions, not just info
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import json
import os
import uuid

logger = logging.getLogger(__name__)


class ReportType(Enum):
    DAILY_DIGEST = "daily_digest"
    WEEKLY_REVIEW = "weekly_review"
    INSIGHT_REPORT = "insight_report"
    ALERT_REPORT = "alert_report"


class ReportTime(Enum):
    MORNING = "morning"
    EVENING = "evening"
    WEEKLY = "weekly"


class ContentCategory(Enum):
    INTEREST_UPDATES = "interest_updates"
    LIFE_UPDATES = "life_updates"
    DATA_INSIGHTS = "data_insights"
    SECURITY_ALERTS = "security_alerts"
    MEMORY_HIGHLIGHTS = "memory_highlights"


class ReportPreference(Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    SMART = "smart"


@dataclass
class ReportPreferences:
    enabled: bool = True
    smart_timing: bool = True
    morning_time: str = "08:00"
    evening_time: str = "20:00"
    weekly_day: str = "sunday"
    weekly_time: str = "10:00"
    delivery_telegram: bool = True
    report_types: List[str] = field(default_factory=lambda: ["daily", "weekly"])
    max_highlights: int = 5
    include_suggestions: bool = True
    mood_check: bool = True


@dataclass
class ReportHighlight:
    category: ContentCategory
    title: str
    content: str
    importance: float
    action_suggestion: Optional[str] = None
    source: str = ""


@dataclass
class ReportData:
    report_type: ReportType
    user_id: Optional[int] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    greeting: str = ""
    quick_stats: Dict[str, Any] = field(default_factory=dict)
    highlights: List[ReportHighlight] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    mood_info: Optional[Dict[str, Any]] = None

    generated_at: datetime = field(default_factory=datetime.now)


class DailyReporter:
    """
    Proactive daily/weekly report system for AURA

    Generates useful insights and delivers them on schedule
    or on-demand via /report command
    """

    def __init__(self, aura_instance=None):
        self.aura = aura_instance
        self.storage_path = "data/daily_reporter"
        self.state = self._load_state()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False

    def _load_state(self) -> Dict:
        """Load reporter state from disk"""
        os.makedirs(self.storage_path, exist_ok=True)
        state_file = os.path.join(self.storage_path, "state.json")

        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load reporter state: {e}")

        return {
            "preferences": {},
            "user_patterns": {},
            "last_reports": {},
            "report_history": [],
        }

    def _save_state(self):
        """Save reporter state to disk"""
        state_file = os.path.join(self.storage_path, "state.json")
        os.makedirs(self.storage_path, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def _get_preferences(self, user_id: int) -> ReportPreferences:
        """Get or create report preferences for user"""
        user_id_str = str(user_id)

        if user_id_str not in self.state["preferences"]:
            self.state["preferences"][user_id_str] = {
                "enabled": True,
                "smart_timing": True,
                "morning_time": "08:00",
                "evening_time": "20:00",
                "weekly_day": "sunday",
                "weekly_time": "10:00",
                "delivery_telegram": True,
                "report_types": ["daily", "weekly"],
                "max_highlights": 5,
                "include_suggestions": True,
                "mood_check": True,
            }
            self._save_state()

        return ReportPreferences(**self.state["preferences"][user_id_str])

    def _save_preferences(self, user_id: int, prefs: ReportPreferences):
        """Save user preferences"""
        user_id_str = str(user_id)
        self.state["preferences"][user_id_str] = asdict(prefs)
        self._save_state()

    async def get_user_interaction_stats(
        self, user_id: int, period_hours: int = 24
    ) -> Dict[str, Any]:
        """Get user interaction statistics for a period"""
        stats = {
            "total_interactions": 0,
            "tasks_completed": 0,
            "tasks_pending": 0,
            "reminders_set": 0,
            "active_hours": [],
            "top_intents": [],
            "mood_trend": "neutral",
        }

        try:
            from src.channels.telegram_bot import StateManager

            state_manager = StateManager()
            profile = await state_manager.get_user_profile(user_id)

            stats["total_interactions"] = profile.interaction_count

            if hasattr(self.aura, "task_engine"):
                pending = await self._get_pending_tasks(user_id)
                stats["tasks_pending"] = len(pending)

            stats["active_hours"] = (
                profile.peak_hours[-5:] if profile.peak_hours else []
            )
            stats["top_intents"] = (
                profile.common_intents[-3:] if profile.common_intents else []
            )

        except Exception as e:
            logger.warning(f"Error getting interaction stats: {e}")

        return stats

    async def _get_pending_tasks(self, user_id: int) -> List[Dict]:
        """Get pending tasks for user"""
        tasks = []
        try:
            if hasattr(self.aura, "task_engine"):
                engine = self.aura.task_engine
                if hasattr(engine, "get_user_tasks"):
                    tasks = await engine.get_user_tasks(user_id)
        except Exception as e:
            logger.warning(f"Error getting tasks: {e}")
        return tasks

    async def _get_interest_updates(
        self, user_id: int, period_hours: int = 24
    ) -> List[ReportHighlight]:
        """Get interest-related updates"""
        highlights = []

        try:
            if hasattr(self.aura, "interest_detector"):
                detector = self.aura.interest_detector
                if hasattr(detector, "get_recent_discoveries"):
                    discoveries = await detector.get_recent_discoveries(period_hours)

                    for discovery in discoveries[:3]:
                        highlights.append(
                            ReportHighlight(
                                category=ContentCategory.INTEREST_UPDATES,
                                title=discovery.get("title", "New Interest Found"),
                                content=discovery.get("description", ""),
                                importance=discovery.get("confidence", 0.5),
                                action_suggestion=discovery.get("suggestion"),
                                source="interest_learner",
                            )
                        )
        except Exception as e:
            logger.debug(f"No interest updates: {e}")

        return highlights

    async def _get_life_updates(
        self, user_id: int, period_hours: int = 24
    ) -> List[ReportHighlight]:
        """Get calendar, reminder, and task updates"""
        highlights = []

        try:
            if hasattr(self.aura, "life_tracker"):
                tracker = self.aura.life_tracker
                if hasattr(tracker, "get_upcoming_events"):
                    events = await tracker.get_upcoming_events(
                        user_id, hours=period_hours
                    )

                    for event in events[:2]:
                        highlights.append(
                            ReportHighlight(
                                category=ContentCategory.LIFE_UPDATES,
                                title=event.get("title", "Upcoming Event"),
                                content=event.get("description", ""),
                                importance=event.get("importance", 0.5),
                                action_suggestion=event.get("action_suggestion"),
                                source="life_tracker",
                            )
                        )

            if hasattr(self.aura, "task_engine"):
                engine = self.aura.task_engine
                pending = await self._get_pending_tasks(user_id)

                if pending:
                    urgent = [
                        t for t in pending if t.get("priority") in ["high", "critical"]
                    ]
                    if urgent:
                        highlights.append(
                            ReportHighlight(
                                category=ContentCategory.LIFE_UPDATES,
                                title=f"{len(urgent)} Urgent Task(s)",
                                content=urgent[0].get("name", "Tasks pending"),
                                importance=0.8,
                                action_suggestion="Review and complete urgent tasks",
                                source="task_engine",
                            )
                        )

        except Exception as e:
            logger.debug(f"No life updates: {e}")

        return highlights

    async def _get_data_insights(
        self, user_id: int, period_hours: int = 24
    ) -> List[ReportHighlight]:
        """Get data insights and pattern discoveries"""
        highlights = []

        try:
            if hasattr(self.aura, "proactive_engine"):
                engine = self.aura.proactive_engine
                if hasattr(engine, "get_recent_insights"):
                    insights = await engine.get_recent_insights(
                        user_id, hours=period_hours
                    )

                    for insight in insights[:2]:
                        highlights.append(
                            ReportHighlight(
                                category=ContentCategory.DATA_INSIGHTS,
                                title=insight.get("title", "Pattern Detected"),
                                content=insight.get("description", ""),
                                importance=insight.get("significance", 0.5),
                                action_suggestion=insight.get("action"),
                                source="proactive_engine",
                            )
                        )
        except Exception as e:
            logger.debug(f"No data insights: {e}")

        return highlights

    async def _get_memory_highlights(
        self, user_id: int, period_hours: int = 24
    ) -> List[ReportHighlight]:
        """Get memory highlights from recent interactions"""
        highlights = []

        try:
            if hasattr(self.aura, "memory"):
                memory = self.aura.memory
                if hasattr(memory, "get_important_memories"):
                    memories = await memory.get_important_memories(
                        user_id, hours=period_hours, limit=3
                    )

                    for memory in memories:
                        highlights.append(
                            ReportHighlight(
                                category=ContentCategory.MEMORY_HIGHLIGHTS,
                                title=memory.get("title", "Memory"),
                                content=memory.get("content", ""),
                                importance=memory.get("importance", 0.5),
                                source="memory_system",
                            )
                        )
        except Exception as e:
            logger.debug(f"No memory highlights: {e}")

        return highlights

    async def _get_security_alerts(self, user_id: int) -> List[ReportHighlight]:
        """Get any security alerts"""
        highlights = []

        try:
            if hasattr(self.aura, "security"):
                security = self.aura.security
                if hasattr(security, "get_recent_alerts"):
                    alerts = await security.get_recent_alerts(user_id)

                    for alert in alerts[:2]:
                        highlights.append(
                            ReportHighlight(
                                category=ContentCategory.SECURITY_ALERTS,
                                title=alert.get("title", "Security Alert"),
                                content=alert.get("description", ""),
                                importance=0.9,
                                action_suggestion=alert.get("action"),
                                source="security",
                            )
                        )
        except Exception as e:
            logger.debug(f"No security alerts: {e}")

        return highlights

    async def _get_aura_mood(self) -> Dict[str, Any]:
        """Get AURA's current mood state"""
        mood_info = {
            "emotion": "focused",
            "intensity": 0.6,
            "cause": "Ready to help",
            "trust_phase": "comfortable",
        }

        try:
            if hasattr(self.aura, "feelings_meter"):
                meter = self.aura.feelings_meter
                if hasattr(meter, "get_current_state"):
                    state = meter.get_current_state()
                    mood_info["emotion"] = state.get("emotion", "focused")
                    mood_info["intensity"] = state.get("intensity", 0.6)
                    mood_info["cause"] = state.get("cause", "Ready to help")
                    mood_info["trust_phase"] = state.get("trust_phase", "comfortable")
        except Exception as e:
            logger.debug(f"Could not get mood: {e}")

        return mood_info

    def _generate_greeting(
        self, report_type: ReportType, user_name: Optional[str] = None
    ) -> str:
        """Generate appropriate greeting based on report type"""
        hour = datetime.now().hour
        time_period = (
            "morning" if 6 <= hour < 12 else "evening" if 12 <= hour < 18 else "night"
        )

        greetings = {
            ReportType.DAILY_DIGEST: {
                "morning": f"Good morning{', ' + user_name if user_name else ''}! Here's your daily digest.",
                "evening": f"Good evening{', ' + user_name if user_name else ''}! Here's your evening summary.",
                "night": f"Hey{', ' + user_name if user_name else ''}! Here's your late-night update.",
            },
            ReportType.WEEKLY_REVIEW: f"Weekly review{', ' + user_name if user_name else ''}! Here's what happened this week.",
            ReportType.INSIGHT_REPORT: f"Thought you might find this interesting{', ' + user_name if user_name else ''}...",
            ReportType.ALERT_REPORT: f"Quick alert{', ' + user_name if user_name else ''}:",
        }

        if report_type == ReportType.DAILY_DIGEST:
            return greetings[ReportType.DAILY_DIGEST].get(
                time_period, greetings[ReportType.DAILY_DIGEST]["morning"]
            )

        return greetings.get(
            report_type, f"Here's your report{', ' + user_name if user_name else ''}!"
        )

    def _format_highlight(self, highlight: ReportHighlight) -> str:
        """Format a single highlight for the report"""
        emoji_map = {
            ContentCategory.INTEREST_UPDATES: "📰",
            ContentCategory.LIFE_UPDATES: "📅",
            ContentCategory.DATA_INSIGHTS: "💡",
            ContentCategory.SECURITY_ALERTS: "🔒",
            ContentCategory.MEMORY_HIGHLIGHTS: "🧠",
        }

        emoji = emoji_map.get(highlight.category, "•")
        importance_bar = (
            "🔴"
            if highlight.importance > 0.7
            else "🟡"
            if highlight.importance > 0.4
            else "⚪"
        )

        formatted = f"{emoji} *{highlight.title}*\n   {highlight.content}"

        if highlight.action_suggestion:
            formatted += f"\n   → {highlight.action_suggestion}"

        return formatted

    def _format_suggestions(self, suggestions: List[str]) -> str:
        """Format suggestions section"""
        if not suggestions:
            return ""

        formatted = "*💡 Suggestions*\n"
        for i, suggestion in enumerate(suggestions[:2], 1):
            formatted += f"{i}. {suggestion}\n"

        return formatted

    async def generate_report(
        self,
        user_id: int,
        report_type: ReportType = ReportType.DAILY_DIGEST,
        period_hours: int = 24,
    ) -> str:
        """Generate a complete report"""
        prefs = self._get_preferences(user_id)

        if not prefs.enabled:
            return "Reports are currently disabled. Use /report settings to enable."

        period_end = datetime.now()
        period_start = period_end - timedelta(hours=period_hours)

        data = ReportData(
            report_type=report_type,
            user_id=user_id,
            period_start=period_start,
            period_end=period_end,
        )

        try:
            from src.channels.telegram_bot import StateManager

            state_manager = StateManager()
            profile = await state_manager.get_user_profile(user_id)
            user_name = profile.display_name or None
        except:
            user_name = None

        data.greeting = self._generate_greeting(report_type, user_name)

        data.quick_stats = await self.get_user_interaction_stats(user_id, period_hours)

        all_highlights = []

        all_highlights.extend(await self._get_security_alerts(user_id))
        all_highlights.extend(await self._get_life_updates(user_id, period_hours))
        all_highlights.extend(await self._get_interest_updates(user_id, period_hours))
        all_highlights.extend(await self._get_data_insights(user_id, period_hours))
        all_highlights.extend(await self._get_memory_highlights(user_id, period_hours))

        all_highlights.sort(key=lambda h: h.importance, reverse=True)
        data.highlights = all_highlights[: prefs.max_highlights]

        if prefs.include_suggestions:
            suggestions = []
            for h in data.highlights[:3]:
                if h.action_suggestion:
                    suggestions.append(h.action_suggestion)

            if not suggestions:
                suggestions = [
                    "Check your pending tasks for anything urgent",
                    "Review your calendar for tomorrow",
                ]

            data.suggestions = suggestions[:2]

        if prefs.mood_check:
            data.mood_info = await self._get_aura_mood()

        return self._format_report(data, prefs)

    def _format_report(self, data: ReportData, prefs: ReportPreferences) -> str:
        """Format report data into final message"""
        report_lines = []

        report_lines.append(data.greeting)
        report_lines.append("")

        period_str = "Today"
        if data.report_type == ReportType.WEEKLY_REVIEW:
            period_str = "This Week"

        report_lines.append(f"📊 *{period_str} Summary*")

        stats = data.quick_stats
        stats_lines = []
        if stats.get("total_interactions", 0) > 0:
            stats_lines.append(f"• {stats['total_interactions']} interactions")
        if stats.get("tasks_completed", 0) > 0:
            stats_lines.append(f"• {stats['tasks_completed']} tasks completed")
        if stats.get("tasks_pending", 0) > 0:
            stats_lines.append(f"• {stats['tasks_pending']} tasks pending")

        if stats_lines:
            report_lines.append("".join(stats_lines))
        else:
            report_lines.append("• No significant activity to report")

        report_lines.append("")

        if data.highlights:
            report_lines.append("*🎯 Highlights*")
            for h in data.highlights[:5]:
                report_lines.append(self._format_highlight(h))
                report_lines.append("")

        if data.suggestions:
            report_lines.append("")
            report_lines.append(self._format_suggestions(data.suggestions))

        if data.mood_info and prefs.mood_check:
            report_lines.append("")
            emoji_map = {
                "curious": "🤔",
                "focused": "🎯",
                "confident": "💪",
                "uncertain": "❓",
                "concerned": "😟",
                "happy": "😊",
                "worried": "😰",
                "excited": "🤩",
                "calm": "😌",
                "tired": "😴",
                "frustrated": "😤",
                "hopeful": "🌟",
                "grateful": "🙏",
                "confused": "😕",
            }
            emoji = emoji_map.get(data.mood_info.get("emotion", "focused"), "🎯")
            trust_phase = data.mood_info.get("trust_phase", "comfortable")

            report_lines.append(f"{emoji} *AURA's Status:*")
            report_lines.append(
                f"   Feeling: {data.mood_info.get('emotion', 'focused')}"
            )
            report_lines.append(f"   Trust level: {trust_phase}")

        return "\n".join(report_lines)

    async def send_report(
        self,
        user_id: int,
        report_type: ReportType = ReportType.DAILY_DIGEST,
        telegram_bot=None,
    ) -> bool:
        """Send report to user via Telegram"""
        prefs = self._get_preferences(user_id)

        if not prefs.enabled or not prefs.delivery_telegram:
            return False

        try:
            report_content = await self.generate_report(user_id, report_type)

            if telegram_bot and hasattr(telegram_bot, "application"):
                bot = telegram_bot.application.bot
                await bot.send_message(
                    chat_id=user_id, text=report_content, parse_mode="Markdown"
                )
                return True

        except Exception as e:
            logger.error(f"Error sending report: {e}")

        return False

    async def handle_report_command(self, update, context) -> str:
        """Handle /report command"""
        user_id = update.effective_user.id
        args = context.args if hasattr(context, "args") and context.args else []

        if not args:
            return await self.generate_report(user_id)

        subcommand = args[0].lower()

        if subcommand == "today":
            return await self.generate_report(user_id, ReportType.DAILY_DIGEST, 24)

        if subcommand == "weekly":
            return await self.generate_report(user_id, ReportType.WEEKLY_REVIEW, 168)

        if subcommand == "settings":
            return self._format_settings_menu(user_id)

        if subcommand == "enable":
            prefs = self._get_preferences(user_id)
            prefs.enabled = True
            self._save_preferences(user_id, prefs)
            return "✅ Reports enabled! You'll receive daily and weekly digests."

        if subcommand == "disable":
            prefs = self._get_preferences(user_id)
            prefs.enabled = False
            self._save_preferences(user_id, prefs)
            return "❌ Reports disabled. Use /report enable to re-enable."

        if subcommand == "morning":
            if len(args) > 1:
                time_str = args[1]
                prefs = self._get_preferences(user_id)
                prefs.morning_time = time_str
                self._save_preferences(user_id, prefs)
                return f"✅ Morning report time set to {time_str}"
            return f"Morning report currently at {self._get_preferences(user_id).morning_time}"

        if subcommand == "evening":
            if len(args) > 1:
                time_str = args[1]
                prefs = self._get_preferences(user_id)
                prefs.evening_time = time_str
                self._save_preferences(user_id, prefs)
                return f"✅ Evening report time set to {time_str}"
            return f"Evening report currently at {self._get_preferences(user_id).evening_time}"

        return "Usage: /report [today|weekly|settings|enable|disable|morning <time>|evening <time>]"

    def _format_settings_menu(self, user_id: int) -> str:
        """Format settings menu for user"""
        prefs = self._get_preferences(user_id)

        status = "✅ Enabled" if prefs.enabled else "❌ Disabled"

        return f"""
*📊 Report Settings*

*Status:* {status}

*Schedule:*
• Morning: {prefs.morning_time}
• Evening: {prefs.evening_time}
• Weekly: {prefs.weekly_day} {prefs.weekly_time}

*Options:*
• Smart timing: {"✅" if prefs.smart_timing else "❌"}
• Suggestions: {"✅" if prefs.include_suggestions else "❌"}
• Mood check: {"✅" if prefs.mood_check else "❌"}
• Telegram delivery: {"✅" if prefs.delivery_telegram else "❌"}

*Commands:*
/report enable - Enable reports
/report disable - Disable reports
/report morning <HH:MM> - Set morning time
/report evening <HH:MM> - Set evening time
/report today - Get today's report
/report weekly - Get weekly review
"""

    def _learn_user_pattern(self, user_id: int, delivery_success: bool):
        """Learn from delivery success/failure to optimize timing"""
        user_id_str = str(user_id)

        if user_id_str not in self.state["user_patterns"]:
            self.state["user_patterns"][user_id_str] = {
                "successful_times": [],
                "failed_times": [],
                "preferred_times": defaultdict(list),
                "total_sent": 0,
                "total_failed": 0,
            }

        pattern = self.state["user_patterns"][user_id_str]
        current_hour = datetime.now().hour

        if delivery_success:
            pattern["successful_times"].append(current_hour)
            pattern["preferred_times"][current_hour].append(1)
            pattern["total_sent"] += 1
        else:
            pattern["failed_times"].append(current_hour)
            pattern["total_failed"] += 1

        for hour, scores in pattern["preferred_times"].items():
            avg = sum(scores) / len(scores) if scores else 0
            pattern["preferred_times"][hour] = avg

        self._save_state()

    def _should_send_report(self, user_id: int, report_type: ReportType) -> bool:
        """Determine if report should be sent based on smart timing"""
        prefs = self._get_preferences(user_id)

        if not prefs.enabled:
            return False

        if not prefs.smart_timing:
            return self._is_scheduled_time(report_type, prefs)

        user_id_str = str(user_id)
        pattern = self.state.get("user_patterns", {}).get(user_id_str, {})

        current_hour = datetime.now().hour
        preferred = pattern.get("preferred_times", {})

        if preferred:
            best_hour = max(preferred.keys(), key=lambda h: preferred.get(h, 0))

            if report_type == ReportType.DAILY_DIGEST:
                morning_hour = int(prefs.morning_time.split(":")[0])
                evening_hour = int(prefs.evening_time.split(":")[0])

                if abs(current_hour - best_hour) <= 2:
                    return True
                if (
                    abs(current_hour - morning_hour) <= 1
                    or abs(current_hour - evening_hour) <= 1
                ):
                    return True

        return self._is_scheduled_time(report_type, prefs)

    def _is_scheduled_time(
        self, report_type: ReportType, prefs: ReportPreferences
    ) -> bool:
        """Check if current time matches schedule"""
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute
        current_time = current_hour * 60 + current_minute

        if report_type == ReportType.DAILY_DIGEST:
            morning = prefs.morning_time.split(":")
            morning_minutes = int(morning[0]) * 60 + int(morning[1])

            evening = prefs.evening_time.split(":")
            evening_minutes = int(evening[0]) * 60 + int(evening[1])

            return (
                abs(current_time - morning_minutes) < 30
                or abs(current_time - evening_minutes) < 30
            )

        elif report_type == ReportType.WEEKLY_REVIEW:
            current_day = datetime.now().strftime(
                "%sunday monday tuesday wednesday thursday friday saturday".split()[
                    datetime.now().weekday()
                ]
            )
            return current_day == prefs.weekly_day.lower()

        return False

    async def start_scheduler(self, telegram_bot=None):
        """Start the report scheduler"""
        if self._running:
            return

        self._running = True
        self._telegram_bot = telegram_bot

        async def scheduler_loop():
            while self._running:
                try:
                    await self._check_and_send_reports()
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")

                await asyncio.sleep(60)

        self._scheduler_task = asyncio.create_task(scheduler_loop())
        logger.info("Daily reporter scheduler started")

    async def stop_scheduler(self):
        """Stop the report scheduler"""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Daily reporter scheduler stopped")

    async def _check_and_send_reports(self):
        """Check if any reports need to be sent"""
        current_hour = datetime.now().hour

        morning_hour = 8
        evening_hour = 20

        is_morning = current_hour == morning_hour
        is_evening = current_hour == evening_hour

        if not (is_morning or is_evening):
            return

        report_type = ReportType.DAILY_DIGEST
        if is_evening:
            report_type = ReportType.DAILY_DIGEST

        user_ids = list(self.state.get("preferences", {}).keys())

        for user_id_str in user_ids:
            try:
                user_id = int(user_id_str)

                if self._should_send_report(user_id, report_type):
                    success = await self.send_report(
                        user_id, report_type, self._telegram_bot
                    )
                    self._learn_user_pattern(user_id, success)

            except Exception as e:
                logger.error(f"Error sending scheduled report to {user_id_str}: {e}")


def get_daily_reporter(aura_instance=None) -> DailyReporter:
    """Get singleton daily reporter instance"""
    if not hasattr(get_daily_reporter, "_instance"):
        get_daily_reporter._instance = DailyReporter(aura_instance)
    return get_daily_reporter._instance
