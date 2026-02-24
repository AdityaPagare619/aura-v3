"""
Example Plugin: Advanced Reminder
Demonstrates advanced reminder patterns with scheduling and recurring events.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from src.extensibility.plugin_system import PluginBase, PluginMetadata

logger = logging.getLogger(__name__)


@dataclass
class Reminder:
    """Represents a reminder"""

    id: str
    message: str
    time: datetime
    repeat: Optional[str] = None
    completed: bool = False
    created_at: datetime = field(default_factory=datetime.now)


class ReminderPlugin(PluginBase):
    """
    Advanced reminder plugin with scheduling and patterns.

    Features:
    - Natural language reminder parsing
    - Recurring reminders (daily, weekly, monthly)
    - Snooze functionality
    - Reminder persistence
    - Hooks into message processing
    """

    def __init__(self):
        super().__init__()
        self._reminders: Dict[str, Reminder] = {}
        self._next_id = 1
        self._snoozed: Dict[str, datetime] = {}

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="reminder",
            version="1.0.0",
            author="AURA Team",
            description="Advanced reminder system with scheduling and recurring events",
            dependencies=[],
            tags=["productivity", "reminders", "scheduling"],
        )

    def on_load(self, api) -> bool:
        """Initialize plugin"""
        if not super().on_load(api):
            return False

        self.register_command("remind", self.handle_remind_command)
        self.register_command("reminders", self.handle_list_reminders)
        self.register_command("snooze", self.handle_snooze_command)
        self.register_command("complete", self.handle_complete_command)
        self.register_command("delete_reminder", self.handle_delete_command)

        self.register_hook("message", self.on_message)

        self._load_from_storage()

        logger.info("Reminder plugin loaded successfully")
        return True

    def on_unload(self) -> bool:
        """Save reminders before unloading"""
        self._save_to_storage()
        return super().on_unload()

    def on_message(self, message: Dict[str, Any]) -> Optional[Dict]:
        """Intercept messages to extract reminder requests"""
        content = message.get("content", "")

        reminder_data = self.parse_reminder_from_text(content)
        if reminder_data:
            message["_reminder_detected"] = True
            message["_reminder_data"] = reminder_data

        return None

    def handle_remind_command(self, args: str = "") -> Dict[str, Any]:
        """Handle /remind command"""
        if not args:
            return {
                "success": False,
                "error": "Usage: /remind <message> <time>",
                "examples": [
                    "/remind Meeting in 30 minutes",
                    "/remind Call mom at 5pm",
                    "/remind Weekly standup every monday 10am",
                ],
            }

        parsed = self.parse_reminder_from_text(args)
        if not parsed:
            return {
                "success": False,
                "error": "Could not parse reminder. Try: 'remind me to [message] at [time]'",
            }

        reminder = Reminder(
            id=str(self._next_id),
            message=parsed["message"],
            time=parsed["time"],
            repeat=parsed.get("repeat"),
        )

        self._reminders[reminder.id] = reminder
        self._next_id += 1
        self._save_to_storage()

        repeat_str = f" (repeats {parsed['repeat']})" if parsed.get("repeat") else ""

        return {
            "success": True,
            "message": f"✅ Reminder set for {self.format_time(reminder.time)}{repeat_str}",
            "reminder_id": reminder.id,
            "reminder": {
                "id": reminder.id,
                "message": reminder.message,
                "time": reminder.time.isoformat(),
                "repeat": reminder.repeat,
            },
        }

    def handle_list_reminders(self, args: str = "") -> Dict[str, Any]:
        """Handle /reminders command - list all reminders"""
        pending = [r for r in self._reminders.values() if not r.completed]
        if not pending:
            return {
                "success": True,
                "message": "No pending reminders",
                "reminders": [],
            }

        lines = ["📋 **Pending Reminders:**\n"]
        for r in sorted(pending, key=lambda x: x.time):
            repeat_str = f" 🔄 {r.repeat}" if r.repeat else ""
            lines.append(
                f"{r.id}. **{r.message}**\n"
                f"   ⏰ {self.format_time(r.time)}{repeat_str}\n"
            )

        return {
            "success": True,
            "message": "".join(lines),
            "reminders": [
                {
                    "id": r.id,
                    "message": r.message,
                    "time": r.time.isoformat(),
                    "repeat": r.repeat,
                }
                for r in pending
            ],
        }

    def handle_snooze_command(self, args: str = "") -> Dict[str, Any]:
        """Handle /snooze command"""
        parts = args.strip().split()

        if len(parts) < 2:
            return {
                "success": False,
                "error": "Usage: /snooze <id> <minutes>",
            }

        try:
            reminder_id = parts[0]
            minutes = int(parts[1])

            if reminder_id not in self._reminders:
                return {"success": False, "error": f"Reminder {reminder_id} not found"}

            reminder = self._reminders[reminder_id]
            reminder.time = datetime.now() + timedelta(minutes=minutes)

            return {
                "success": True,
                "message": f"⏰ Snoozed reminder for {minutes} minutes",
                "new_time": self.format_time(reminder.time),
            }

        except ValueError:
            return {"success": False, "error": "Invalid number of minutes"}

    def handle_complete_command(self, args: str = "") -> Dict[str, Any]:
        """Handle /complete command"""
        if not args.strip():
            return {"success": False, "error": "Usage: /complete <id>"}

        reminder_id = args.strip()

        if reminder_id not in self._reminders:
            return {"success": False, "error": f"Reminder {reminder_id} not found"}

        reminder = self._reminders[reminder_id]
        reminder.completed = True

        if reminder.repeat:
            new_time = self._calculate_next_occurrence(reminder)
            new_reminder = Reminder(
                id=str(self._next_id),
                message=reminder.message,
                time=new_time,
                repeat=reminder.repeat,
            )
            self._reminders[new_reminder.id] = new_reminder
            self._next_id += 1

            return {
                "success": True,
                "message": f"✅ Completed! Next reminder set for {self.format_time(new_time)}",
            }

        return {
            "success": True,
            "message": "✅ Reminder completed and removed",
        }

    def handle_delete_command(self, args: str = "") -> Dict[str, Any]:
        """Handle /delete_reminder command"""
        if not args.strip():
            return {"success": False, "error": "Usage: /delete_reminder <id>"}

        reminder_id = args.strip()

        if reminder_id not in self._reminders:
            return {"success": False, "error": f"Reminder {reminder_id} not found"}

        del self._reminders[reminder_id]
        self._save_to_storage()

        return {
            "success": True,
            "message": "🗑️ Reminder deleted",
        }

    def parse_reminder_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse reminder from natural language text"""
        text = text.lower().strip()

        patterns = [
            r"(?:remind me to|reminder:|remind me|remind)\s+(.+?)\s+(?:at|in|on|every)\s+(.+)$",
            r"(?:remind me to|remind)\s+(.+?)\s+(?:every)\s+(.+)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                message = match.group(1).strip()
                time_spec = match.group(2).strip() if match.lastindex >= 2 else ""

                parsed_time = self.parse_time_spec(time_spec)
                if parsed_time:
                    return {
                        "message": message,
                        "time": parsed_time["time"],
                        "repeat": parsed_time.get("repeat"),
                    }

        time_only = self.parse_time_spec(text)
        if time_only:
            return {
                "message": text,
                "time": time_only["time"],
                "repeat": time_only.get("repeat"),
            }

        return None

    def parse_time_spec(self, time_spec: str) -> Optional[Dict[str, Any]]:
        """Parse time specification into datetime"""
        now = datetime.now()
        time_spec = time_spec.lower().strip()

        minute_match = re.search(r"(\d+)\s*(?:minute|min|m)", time_spec)
        hour_match = re.search(r"(\d+)\s*(?:hour|hr|h)", time_spec)
        day_match = re.search(r"(\d+)\s*(?:day|d)", time_spec)

        if minute_match or hour_match or day_match:
            delta = timedelta(
                minutes=int(minute_match.group(1)) if minute_match else 0,
                hours=int(hour_match.group(1)) if hour_match else 0,
                days=int(day_match.group(1)) if day_match else 0,
            )
            return {"time": now + delta, "repeat": self._extract_repeat(time_spec)}

        time_patterns = [
            (r"(\d{1,2}):(\d{2})\s*(am|pm)?", "time"),
            (r"(\d{1,2})\s*(am|pm)", "ampm"),
            (r"noon", "noon"),
            (r"midnight", "midnight"),
        ]

        for pattern, ptype in time_patterns:
            match = re.search(pattern, time_spec)
            if match:
                if ptype == "noon":
                    return {
                        "time": now.replace(hour=12, minute=0, second=0),
                        "repeat": self._extract_repeat(time_spec),
                    }
                elif ptype == "midnight":
                    return {
                        "time": now.replace(hour=0, minute=0, second=0),
                        "repeat": self._extract_repeat(time_spec),
                    }
                elif ptype == "time" or ptype == "ampm":
                    hour = int(match.group(1))
                    minute = int(match.group(2)) if ptype == "time" else 0
                    ampm = match.group(3) if match.lastindex >= 3 else None

                    if ampm == "pm" and hour != 12:
                        hour += 12
                    elif ampm == "am" and hour == 12:
                        hour = 0

                    result_time = now.replace(hour=hour, minute=minute, second=0)
                    if result_time < now:
                        result_time += timedelta(days=1)

                    return {
                        "time": result_time,
                        "repeat": self._extract_repeat(time_spec),
                    }

        day_names = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }

        for day_name, day_num in day_names.items():
            if day_name in time_spec:
                days_ahead = (day_num - now.weekday() + 7) % 7
                if days_ahead == 0:
                    days_ahead = 7

                return {
                    "time": now + timedelta(days=days_ahead),
                    "repeat": self._extract_repeat(time_spec),
                }

        return None

    def _extract_repeat(self, text: str) -> Optional[str]:
        """Extract repeat pattern from text"""
        text = text.lower()

        if "every day" in text or "daily" in text:
            return "daily"
        elif "every week" in text or "weekly" in text:
            return "weekly"
        elif "every month" in text or "monthly" in text:
            return "monthly"
        elif "every monday" in text:
            return "weekly_monday"

        return None

    def _calculate_next_occurrence(self, reminder: Reminder) -> datetime:
        """Calculate next occurrence for recurring reminder"""
        now = datetime.now()

        if reminder.repeat == "daily":
            return reminder.time + timedelta(days=1)
        elif reminder.repeat == "weekly":
            return reminder.time + timedelta(weeks=1)
        elif reminder.repeat == "monthly":
            return reminder.time + timedelta(days=30)

        return now + timedelta(days=1)

    def format_time(self, dt: datetime) -> str:
        """Format datetime for display"""
        now = datetime.now()

        if dt.date() == now.date():
            return f"Today at {dt.strftime('%I:%M %p')}"
        elif dt.date() == (now + timedelta(days=1)).date():
            return f"Tomorrow at {dt.strftime('%I:%M %p')}"
        else:
            return dt.strftime("%b %d at %I:%M %p")

    def _load_from_storage(self):
        """Load reminders from persistent storage"""
        try:
            if self._api:
                stored = self._api.retrieve_data("reminders")
                if stored:
                    for r in stored:
                        self._reminders[r["id"]] = Reminder(
                            id=r["id"],
                            message=r["message"],
                            time=datetime.fromisoformat(r["time"]),
                            repeat=r.get("repeat"),
                            completed=r.get("completed", False),
                        )
                    self._next_id = (
                        max(int(r["id"]) for r in self._reminders.keys()) + 1
                    )
        except Exception as e:
            logger.warning(f"Could not load reminders: {e}")

    def _save_to_storage(self):
        """Save reminders to persistent storage"""
        try:
            if self._api:
                data = [
                    {
                        "id": r.id,
                        "message": r.message,
                        "time": r.time.isoformat(),
                        "repeat": r.repeat,
                        "completed": r.completed,
                    }
                    for r in self._reminders.values()
                ]
                self._api.store_data("reminders", data)
        except Exception as e:
            logger.warning(f"Could not save reminders: {e}")

    def get_pending_reminders(self) -> List[Reminder]:
        """Get all pending reminders that are due"""
        now = datetime.now()
        return [
            r for r in self._reminders.values() if not r.completed and r.time <= now
        ]


def get_plugin():
    """Factory function to create plugin instance"""
    return ReminderPlugin()
