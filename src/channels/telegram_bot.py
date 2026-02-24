"""
AURA Telegram Bot - Friday-Like Personal AI Assistant

All configuration via Telegram commands - NO external web needed!

Commands:
/start - Start AURA
/stop - Stop AURA
/status - Check status
/setup - Interactive setup wizard
/test - Test if everything works
/config - Show current config
/setmodel <model> - Set LLM model
/setvoice <voice> - Set voice engine
/setlevel <L1-L4> - Set permission level
/models - List available models
/voices - List available TTS
/memory - Show memory stats
/interests - Show discovered interests
/clear - Clear conversation
/help - Show all commands
/remember <fact> - Store information
/remind <time> <note> - Set reminder
/search <query> - Search history
/summarize - Summarize conversation
/speak - Convert to voice
/quiet - Disable voice
/style <formal/casual> - Adjust style
/stats - Usage statistics

Plus: Voice messages, Photos, Files, Inline queries, Custom keyboards
"""

import asyncio
import logging
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import re

# Telegram imports
try:
    from telegram import (
        Update,
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        KeyboardButton,
        ReplyKeyboardMarkup,
        ReplyKeyboardRemove,
        ChatAction,
        InputFile,
    )
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        filters,
        ContextTypes,
        InlineQueryHandler,
        CallbackQueryHandler,
        ConversationHandler,
    )
    from telegram.error import TelegramError

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import personality system
from src.agent.personality_system import (
    PersonalitySystem,
    PersonalityCommands,
    get_personality_system,
)


# ============== ENUMS & DATA CLASSES ==============


class Intent(Enum):
    CHAT = "chat"
    QUERY = "query"
    ACTION = "action"
    REMINDER = "reminder"
    NOTE = "note"
    COMMAND = "command"
    CLARIFICATION = "clarification"


class FormalityLevel(Enum):
    FORMAL = "formal"
    FRIENDLY = "friendly"
    CASUAL = "casual"


class ResponseLength(Enum):
    SHORT = "short"
    BALANCED = "balanced"
    LONG = "long"


class Mood(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    WORRIED = "worried"


class FactType(Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    NAME = "name"


@dataclass
class UserProfile:
    user_id: int
    display_name: str = ""
    timezone: str = "UTC"
    language: str = "en"
    prefers_voice: bool = False
    formality_level: FormalityLevel = FormalityLevel.FRIENDLY
    response_length: ResponseLength = ResponseLength.BALANCED
    uses_emoji: bool = True
    peak_hours: List[int] = field(default_factory=list)
    common_intents: List[str] = field(default_factory=list)
    last_interaction: Optional[datetime] = None
    facts: Dict[str, str] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_count: int = 0
    voice_enabled: bool = True


@dataclass
class MessageContext:
    user_id: int
    message_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    intents: List[Intent] = field(default_factory=list)
    mood: Mood = Mood.NEUTRAL
    source: str = "telegram"


# ============== STATE MANAGEMENT ==============


class StateManager:
    """Persist user state across sessions"""

    def __init__(self, storage_path: str = "data/telegram_state.json"):
        self.storage_path = storage_path
        self.state = self._load_state()
        self._lock = asyncio.Lock()

    def _load_state(self) -> dict:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        return {
            "user_profiles": {},
            "conversation_history": {},
            "reminders": [],
            "custom_commands": {},
        }

    def _save_state(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    async def save_user_profile(self, profile: UserProfile):
        async with self._lock:
            self.state["user_profiles"][profile.user_id] = asdict(profile)
            self._save_state()

    async def get_user_profile(self, user_id: int) -> UserProfile:
        if str(user_id) in self.state["user_profiles"]:
            data = self.state["user_profiles"][str(user_id)]
            if "formality_level" in data and isinstance(data["formality_level"], str):
                data["formality_level"] = FormalityLevel(data["formality_level"])
            if "response_length" in data and isinstance(data["response_length"], str):
                data["response_length"] = ResponseLength(data["response_length"])
            return UserProfile(**data)
        return UserProfile(user_id=user_id)

    async def add_to_history(self, user_id: int, role: str, content: str):
        key = str(user_id)
        if key not in self.state["conversation_history"]:
            self.state["conversation_history"][key] = []
        self.state["conversation_history"][key].append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        )
        # Keep last 100 messages
        if len(self.state["conversation_history"][key]) > 100:
            self.state["conversation_history"][key] = self.state[
                "conversation_history"
            ][key][-100:]
        self._save_state()

    async def get_history(self, user_id: int, limit: int = 20) -> List[Dict]:
        key = str(user_id)
        return self.state.get("conversation_history", {}).get(key, [])[-limit:]


# ============== PERSONALIZATION ENGINE ==============


class PersonalizationEngine:
    """Learn from users and adapt responses"""

    def __init__(self, state_manager: StateManager):
        self.state = state_manager

    async def learn_from_interaction(
        self, user_id: int, user_message: str, response: str, intent: Intent
    ):
        profile = await self.state.get_user_profile(user_id)

        # Learn response length preference
        if len(user_message) < 50 and len(response) > 500:
            profile.response_length = ResponseLength.SHORT
        elif len(user_message) > 200 and len(response) < 100:
            profile.response_length = ResponseLength.LONG

        # Learn peak hours
        current_hour = datetime.now().hour
        if current_hour not in profile.peak_hours:
            profile.peak_hours.append(current_hour)

        # Track common intents
        intent_str = intent.value
        if intent_str not in profile.common_intents:
            profile.common_intents.append(intent_str)

        profile.last_interaction = datetime.now()
        profile.interaction_count += 1

        await self.state.save_user_profile(profile)

    async def adapt_response(self, response: str, user_id: int) -> str:
        profile = await self.state.get_user_profile(user_id)

        # Adjust length
        if profile.response_length == ResponseLength.SHORT:
            sentences = response.split(".")
            response = ". ".join(sentences[:2]) + "."
        elif profile.response_length == ResponseLength.LONG:
            pass  # Keep full response

        # Handle emoji
        if not profile.uses_emoji:
            response = re.sub(r"[\U0001F300-\U0001F9FF]", "", response)

        # Add personalization
        if profile.display_name:
            response = response.replace("hey", f"hey {profile.display_name}")
            response = response.replace("Hello", f"Hello {profile.display_name}")

        return response


# ============== MOOD DETECTOR ==============


class MoodDetector:
    """Detect user mood from messages"""

    # Sentiment indicators
    HAPPY_EMOJIS = ["😊", "😄", "🙂", "😃", "🎉", "❤️", "🥰", "😍"]
    SAD_EMOJIS = ["😢", "😭", "😞", "😔", "😟", "😣"]
    ANGRY_EMOJIS = ["😡", "😠", "🤬", "😤", "😾"]
    EXCITED_EMOJIS = ["🎉", "🤩", "😎", "🔥", "⚡", "🚀"]
    WORRIED_EMOJIS = ["😰", "😨", "🤔", "😐"]

    WORDS_POSITIVE = [
        "great",
        "amazing",
        "wonderful",
        "awesome",
        "love",
        "happy",
        "excellent",
        "good",
        "fantastic",
    ]
    WORDS_NEGATIVE = [
        "bad",
        "terrible",
        "awful",
        "hate",
        "sad",
        "angry",
        "frustrated",
        "annoyed",
        "upset",
    ]
    WORDS_EXCITED = ["excited", "wow", "incredible", "unbelievable", "can't wait"]
    WORDS_WORRIED = ["worry", "concerned", "nervous", "anxious", "scared", "afraid"]

    def detect(self, text: str) -> Mood:
        text_lower = text.lower()

        # Check emojis
        for emoji in self.HAPPY_EMOJIS:
            if emoji in text:
                return Mood.HAPPY
        for emoji in self.SAD_EMOJIS:
            if emoji in text:
                return Mood.SAD
        for emoji in self.ANGRY_EMOJIS:
            if emoji in text:
                return Mood.ANGRY
        for emoji in self.EXCITED_EMOJIS:
            if emoji in text:
                return Mood.EXCITED
        for emoji in self.WORRIED_EMOJIS:
            if emoji in text:
                return Mood.WORRIED

        # Check words
        for word in self.WORDS_EXCITED:
            if word in text_lower:
                return Mood.EXCITED
        for word in self.WORDS_WORRIED:
            if word in text_lower:
                return Mood.WORRIED
        for word in self.WORDS_NEGATIVE:
            if word in text_lower:
                return Mood.SAD
        for word in self.WORDS_POSITIVE:
            if word in text_lower:
                return Mood.HAPPY

        return Mood.NEUTRAL

    def adapt_for_mood(self, response: str, mood: Mood) -> str:
        if mood == Mood.ANGRY:
            response = f"I understand this might be frustrating. {response}"
        elif mood == Mood.SAD:
            response = f"I'm sorry you're feeling this way. {response}"
        elif mood == Mood.EXCITED:
            response = f"That's wonderful! {response}"
        elif mood == Mood.WORRIED:
            response = f"I want to help you with this. {response}"

        return response


# ============== INTENT CLASSIFIER ==============


class IntentClassifier:
    """Classify user intent"""

    def classify(self, text: str, profile: UserProfile) -> Intent:
        text_lower = text.lower()

        # Check commands
        if text.startswith("/"):
            return Intent.COMMAND

        # Check reminders
        if any(
            word in text_lower
            for word in ["remind me", "reminder", "remind me to", "set a reminder"]
        ):
            return Intent.REMINDER

        # Check note-taking
        if any(
            word in text_lower
            for word in ["remember", "note that", "don't forget", "store"]
        ):
            return Intent.NOTE

        # Check actions
        if any(
            word in text_lower
            for word in [
                "do this",
                "make it",
                "create",
                "run",
                "execute",
                "start",
                "stop",
            ]
        ):
            return Intent.ACTION

        # Check queries
        if any(
            word in text_lower
            for word in ["what", "how", "when", "where", "why", "who", "which", "?"]
        ):
            return Intent.QUERY

        # Default to chat
        return Intent.CHAT

    def parse_command(self, text: str) -> Tuple[str, str]:
        """Parse command and arguments"""
        if text.startswith("/"):
            parts = text[1:].split(" ", 1)
            cmd = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            return cmd, args
        return "", ""


# ============== MAIN BOT CLASS ==============


class AuraTelegramBot:
    """
    Self-contained Telegram bot for AURA - Friday-like experience
    All setup/config via commands - no external web!
    """

    # Conversation states
    SETUP_TOKEN, SETUP_MODEL, SETUP_VOICE = range(3)

    def __init__(self, aura_instance, config_path: str = "config/config.yaml"):
        self.aura = aura_instance
        self.config_path = config_path
        self.config = self._load_config()
        self.user_sessions: Dict[int, str] = {}

        # New components
        self.state_manager = StateManager()
        self.personalization = PersonalizationEngine(self.state_manager)
        self.mood_detector = MoodDetector()
        self.intent_classifier = IntentClassifier()

        # Message queues per user
        self.queues: Dict[int, asyncio.Queue] = {}
        self.processing: Dict[int, bool] = {}

        # Offline capability
        self.is_online = self._check_connectivity()

        # Execution Controller (Stop/Kill System)
        try:
            from src.core.execution_control import get_execution_controller

            self.execution_controller = get_execution_controller()
        except ImportError:
            self.execution_controller = None
            logger.warning("Execution control not available")

        # Relationship System
        try:
            from src.agent import get_relationship_system

            self.relationship_system = get_relationship_system(str(user_id))
        except ImportError:
            self.relationship_system = None
            logger.warning("Relationship system not available")

        # Personality System
        try:
            self.personality_system = get_personality_system()
        except Exception:
            self.personality_system = None
            logger.warning("Personality system not available")

    def _check_connectivity(self) -> bool:
        try:
            import socket

            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except Exception:
            return False

    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate mathematical expressions without using eval()"""
        import ast
        import operator

        valid_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def eval_node(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                return valid_ops[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                return valid_ops[type(node.op)](eval_node(node.operand))
            elif isinstance(node, ast.Num):
                return node.n
            else:
                raise ValueError(f"Unsupported operation: {ast.dump(node)}")

        tree = ast.parse(expression, mode="eval")
        return eval_node(tree.body)

    def _load_config(self) -> Dict:
        """Load configuration from YAML"""
        try:
            import yaml

            if os.path.exists(self.config_path):
                with open(self.config_path) as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
        return {}

    def _save_config(self):
        """Save configuration"""
        try:
            import yaml

            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                yaml.dump(self.config, f)
        except Exception as e:
            logger.error(f"Config save error: {e}")

    def _get_main_keyboard(self) -> ReplyKeyboardMarkup:
        """Get main menu keyboard"""
        keyboard = [
            [
                KeyboardButton("🎤 Voice"),
                KeyboardButton("📷 Photo"),
                KeyboardButton("📎 File"),
            ],
            [
                KeyboardButton("📋 Tasks"),
                KeyboardButton("📅 Calendar"),
                KeyboardButton("📝 Notes"),
            ],
            [
                KeyboardButton("📊 Reports"),
                KeyboardButton("⚙️ Settings"),
            ],
            [KeyboardButton("❓ Help")],
        ]
        return ReplyKeyboardMarkup(
            keyboard, resize_keyboard=True, one_time_keyboard=False
        )

    def _get_settings_keyboard(self) -> InlineKeyboardMarkup:
        """Get inline settings keyboard"""
        keyboard = [
            [InlineKeyboardButton("🔊 Voice On/Off", callback_data="toggle_voice")],
            [
                InlineKeyboardButton("📝 Style", callback_data="set_style"),
                InlineKeyboardButton("📏 Length", callback_data="set_length"),
            ],
            [InlineKeyboardButton("😊 Emoji On/Off", callback_data="toggle_emoji")],
            [InlineKeyboardButton("🔙 Back", callback_data="back_main")],
        ]
        return InlineKeyboardMarkup(keyboard)

    # ============== COMMAND HANDLERS ==============

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        profile = await self.state_manager.get_user_profile(user_id)

        welcome = f"""
🤖 *AURA v3 - Your Personal AI*

Hey{", " + profile.display_name if profile.display_name else ""}! I'm your personal AI assistant that feels like having FRIDAY at your fingertips.

*What I can do:*
• 💬 Chat naturally - just send a message
• 🎤 Voice messages - I'll transcribe & respond
• 📷 Photos - I'll analyze or extract text
• 📎 Files - Process documents instantly
• ⏰ Reminders - Set and track reminders
• 🧠 Remember things about you

        *Quick commands:*
 /status - Check system status
 /relationship - See our bond
 /logs - See what I'm doing
 /help - See all commands
 /settings - Customize your experience

 *Just start chatting!*
 Your data stays private - 100% on your device.
 """
        await update.message.reply_text(
            welcome, parse_mode="Markdown", reply_markup=self._get_main_keyboard()
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show all commands"""
        help_text = """
*📋 AURA Commands*

*🎛️ Control:*
/start - Start AURA
/stop - Graceful stop (finish action)
/force-stop - Force stop (finish tool)
/kill - Emergency kill (stop everything)
/execution-status - Show execution status
/restart - Restart AURA
/status - System status

*⚙️ Setup:*
/setup - Interactive setup wizard
/test - Test system functionality
/config - View configuration  

*🔒 Security:*
/security - Show security status
/lock - Lock AURA
/unlock <pin> - Unlock AURA
/setpin - Set/change PIN

*🔐 Privacy:*
/privacy - Show privacy settings
/privacy <tier> - Set default tier
/permissions - List all permissions

*💬 Communication:*
/setmodel <name> - Set LLM model
/setvoice <engine> - Set TTS engine
/style <formal/casual> - Adjust style
/voice - Toggle voice responses

        *🧠 Memory:*
/remember <fact> - Store information
/search <query> - Search history
/summarize - Summarize conversation
/clear - Clear conversation

*🔍 Transparency:*
/logs - Show recent background activity
/logs thought - Show only agent thoughts
/logs tools - Show only tool executions
/logs clear - Clear log history

*🎯 Mission Control:*
/dashboard - Main dashboard (alias: /mission)
/inner - See what AURA is thinking
/mood - AURA's current feelings
/personality - AURA's personality traits
/morals - AURA's moral framework
/tools - Tool execution status

 *📊 Info:*
 /models - List available models
 /voices - List TTS engines
 /memory - Memory statistics
 /interests - Show discovered interests
 /relationship - Your bond with AURA
 /stats - Your usage stats

 *📝 Reports:*
/report - Get your daily digest
/report today - Today's summary
/report weekly - Weekly review
/report settings - Report preferences

 *💬 Natural Language:*
You can also say things like:
"You can access my photos" - Grant permission
"Don't read my messages without asking" - Restrict

*Or use the menu below:*
"""
        await update.message.reply_text(
            help_text, parse_mode="Markdown", reply_markup=self._get_main_keyboard()
        )

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show system status"""
        status = []

        # Online status
        status.append(
            f"{'🟢' if self.is_online else '🔴'} Network: {'Online' if self.is_online else 'Offline (using local models)'}"
        )

        # LLM Status
        llm_loaded = self.aura.llm.is_loaded() if hasattr(self.aura, "llm") else False
        status.append(f"🤖 LLM: {'✅ Loaded' if llm_loaded else '❌ Not loaded'}")

        # Model info
        if hasattr(self.aura, "llm") and self.aura.llm.config:
            status.append(f"📦 Model: {self.aura.llm.config.model_type}")

        # Voice
        voice = self.config.get("voice", {}).get("engine", "espeak")
        status.append(f"🔊 Voice: {voice}")

        # Security
        level = self.config.get("security", {}).get("default_level", "L2")
        status.append(f"🛡️ Security: {level}")

        # User sessions
        session_count = len(self.user_sessions)
        status.append(f"💬 Sessions: {session_count}")

        # User profile
        user_id = update.effective_user.id
        profile = await self.state_manager.get_user_profile(user_id)
        status.append(f"📊 Interactions: {profile.interaction_count}")

        await update.message.reply_text("\n".join(status))

    async def config_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current configuration"""
        config_text = "*⚙️ Current Configuration*\n\n"

        # LLM
        llm = self.config.get("llm", {})
        config_text += f"*LLM:*\n"
        config_text += f"  Model: {llm.get('model_type', 'llama')}\n"
        config_text += f"  Path: {llm.get('model_path', 'Not set')}\n"
        config_text += f"  Context: {llm.get('max_context', 4096)}\n\n"

        # Voice
        voice = self.config.get("voice", {})
        config_text += f"*Voice:*\n"
        config_text += f"  Engine: {voice.get('engine', 'espeak')}\n"
        config_text += f"  Language: {voice.get('language', 'en')}\n\n"

        # Security
        security = self.config.get("security", {})
        config_text += f"*Security:*\n"
        config_text += f"  Level: {security.get('default_level', 'L2')}\n"
        config_text += (
            f"  Banking Block: {security.get('banking_protection', True)}\n\n"
        )

        # Personal
        user_id = update.effective_user.id
        profile = await self.state_manager.get_user_profile(user_id)
        config_text += f"*Your Preferences:*\n"
        config_text += f"  Name: {profile.display_name or 'Not set'}\n"
        config_text += f"  Style: {profile.formality_level.value}\n"
        config_text += f"  Response: {profile.response_length.value}\n"
        config_text += f"  Emoji: {'On' if profile.uses_emoji else 'Off'}\n"
        config_text += f"  Voice: {'On' if profile.voice_enabled else 'Off'}"

        await update.message.reply_text(config_text, parse_mode="Markdown")

    async def setmodel_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Set LLM model"""
        if not context.args:
            await update.message.reply_text(
                "Usage: /setmodel <model_name>\n\nAvailable:\n- llama\n- qwen\n- mistral\n- phi"
            )
            return

        model = context.args[0].lower()
        valid = ["llama", "qwen", "mistral", "phi"]

        if model not in valid:
            await update.message.reply_text(f"Invalid model. Use: {', '.join(valid)}")
            return

        if "llm" not in self.config:
            self.config["llm"] = {}
        self.config["llm"]["model_type"] = model
        self._save_config()

        await update.message.reply_text(
            f"✅ Model set to: {model}\nRestart with /restart to apply"
        )

    async def setvoice_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Set TTS engine"""
        if not context.args:
            await update.message.reply_text(
                "Usage: /setvoice <engine>\n\nAvailable:\n- espeak (free, fast)\n- pyttsx3 (free, natural)\n- google (needs internet)"
            )
            return

        engine = context.args[0].lower()
        valid = ["espeak", "pyttsx3", "google", "edge"]

        if engine not in valid:
            await update.message.reply_text(f"Invalid. Use: {', '.join(valid)}")
            return

        if "voice" not in self.config:
            self.config["voice"] = {}
        self.config["voice"]["engine"] = engine
        self._save_config()

        await update.message.reply_text(f"✅ Voice engine: {engine}")

    async def setlevel_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Set security level"""
        if not context.args:
            await update.message.reply_text(
                "Usage: /setlevel <L1|L2|L3|L4>\n\nLevels:\nL1 - Full trust\nL2 - Normal\nL3 - Restricted\nL4 - Maximum"
            )
            return

        level = context.args[0].upper()
        if level not in ["L1", "L2", "L3", "L4"]:
            await update.message.reply_text("Invalid. Use L1, L2, L3, or L4")
            return

        if "security" not in self.config:
            self.config["security"] = {}
        self.config["security"]["default_level"] = level
        self._save_config()

        await update.message.reply_text(f"✅ Security level: {level}")

    async def style_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set communication style"""
        user_id = update.effective_user.id

        if not context.args:
            profile = await self.state_manager.get_user_profile(user_id)
            await update.message.reply_text(
                f"Current style: {profile.formality_level.value}\n\n"
                "Usage: /style <formal|casual|friendly>"
            )
            return

        style = context.args[0].lower()
        valid_styles = {
            "formal": FormalityLevel.FORMAL,
            "casual": FormalityLevel.CASUAL,
            "friendly": FormalityLevel.FRIENDLY,
        }

        if style not in valid_styles:
            await update.message.reply_text("Use: formal, casual, or friendly")
            return

        profile = await self.state_manager.get_user_profile(user_id)
        profile.formality_level = valid_styles[style]
        await self.state_manager.save_user_profile(profile)

        await update.message.reply_text(f"✅ Style set to: {style}")

    async def voice_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Toggle voice responses"""
        user_id = update.effective_user.id
        profile = await self.state_manager.get_user_profile(user_id)

        profile.voice_enabled = not profile.voice_enabled
        await self.state_manager.save_user_profile(profile)

        status = "enabled" if profile.voice_enabled else "disabled"
        await update.message.reply_text(f"✅ Voice responses {status}")

    async def remember_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Store user information"""
        user_id = update.effective_user.id

        if not context.args:
            await update.message.reply_text(
                "Usage: /remember <information>\n\nExamples:\n"
                "/remember John's birthday is March 15\n"
                "/remember I prefer short responses\n"
                "/remember Call me by my nickname 'Ace'"
            )
            return

        fact = " ".join(context.args)
        profile = await self.state_manager.get_user_profile(user_id)

        # Parse the fact
        fact_lower = fact.lower()

        if "nickname" in fact_lower or "call me" in fact_lower:
            # Extract name
            match = re.search(r"['\"](.+?)['\"]", fact)
            if match:
                profile.display_name = match.group(1)
                await self.state_manager.save_user_profile(profile)
                await update.message.reply_text(
                    f"✓ Got it! I'll call you {profile.display_name}"
                )
                return

        if "prefer" in fact_lower or "like" in fact_lower:
            # It's a preference
            profile.preferences[fact] = True
            await self.state_manager.save_user_profile(profile)
            await update.message.reply_text(f"✓ Got it! I'll remember that preference.")
            return

        # Store as fact
        key = fact.split()[0] if fact.split() else "info"
        profile.facts[key] = fact
        await self.state_manager.save_user_profile(profile)

        await update.message.reply_text(f"✓ Got it! I'll remember: {fact}")

    async def search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Search conversation history"""
        user_id = update.effective_user.id

        if not context.args:
            await update.message.reply_text("Usage: /search <query>")
            return

        query = " ".join(context.args).lower()
        history = await self.state_manager.get_history(user_id, limit=50)

        results = [msg for msg in history if query in msg.get("content", "").lower()]

        if not results:
            await update.message.reply_text(f"No results found for '{query}'")
            return

        response = f"🔍 *Results for '{query}':*\n\n"
        for i, msg in enumerate(results[-5:], 1):
            content = msg.get("content", "")[:100]
            response += f"{i}. {msg.get('role', '')}: {content}...\n"

        await update.message.reply_text(response, parse_mode="Markdown")

    async def summarize_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Summarize conversation"""
        user_id = update.effective_user.id
        history = await self.state_manager.get_history(user_id, limit=20)

        if not history:
            await update.message.reply_text("No conversation history to summarize.")
            return

        # Simple summary - count messages
        user_msgs = sum(1 for msg in history if msg.get("role") == "user")
        aura_msgs = sum(1 for msg in history if msg.get("role") == "assistant")

        first_msg = history[0].get("content", "")[:50] if history else ""
        last_msg = history[-1].get("content", "")[:50] if history else ""

        summary = f"""
*📝 Conversation Summary*

*Messages:*
• You: {user_msgs}
• AURA: {aura_msgs}

*Started with:*
"{first_msg}..."

*Recent:*
"{last_msg}..."

*To clear history:* /clear
"""
        await update.message.reply_text(summary, parse_mode="Markdown")

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show user statistics"""
        user_id = update.effective_user.id
        profile = await self.state_manager.get_user_profile(user_id)

        stats = f"""
*📊 Your Stats*

*Activity:*
• Total interactions: {profile.interaction_count}
• Peak hours: {", ".join(map(str, profile.peak_hours[-5:])) or "N/A"}

*Preferences:*
• Style: {profile.formality_level.value}
• Response length: {profile.response_length.value}
• Emoji: {"Yes" if profile.uses_emoji else "No"}
• Voice: {"On" if profile.voice_enabled else "Off"}

*Memory:*
• Facts stored: {len(profile.facts)}
• Preferences: {len(profile.preferences)}
• Common intents: {", ".join(profile.common_intents[-3:]) or "N/A"}
"""
        await update.message.reply_text(stats, parse_mode="Markdown")

    async def models_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List available models"""
        models_text = """
📦 *Available LLM Models*

*For mobile (4GB RAM):*
• qwen2.5-1b - Best balance (625MB)
• phi-3-mini - Fast (2.5GB)
• mistral-7b - Powerful (4GB+)

*Quantization:*
• q4_k_m - 4-bit (smaller)
• q5_k_m - 5-bit (balanced)
• q8_0 - 8-bit (better quality)

*Download:*
Use /setmodel to select
Then download GGUF file to models/
"""
        await update.message.reply_text(models_text, parse_mode="Markdown")

    async def voices_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List TTS engines"""
        voices_text = """
🔊 *Available TTS Engines*

*Free (Offline):*
• espeak - Fast, works in Termux
• pyttsx3 - More natural

*Online (Need Internet):*
• google - Google TTS
• edge - Microsoft Edge TTS

*Setup:*
/setvoice <engine>

*Toggle voice:*
/voice
"""
        await update.message.reply_text(voices_text, parse_mode="Markdown")

    async def memory_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show memory stats"""
        if hasattr(self.aura, "memory"):
            stats = self.aura.memory.get_stats()
            await update.message.reply_text(f"📊 *Memory Stats*\n\n{stats}")
        else:
            await update.message.reply_text("Memory stats not available")

    async def interests_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Show or manage discovered interests"""
        args = context.args if hasattr(context, "args") else []

        try:
            from src.agent.interest_learner import (
                get_interest_detector,
                InterestCategory,
            )

            detector = get_interest_detector()

            if not args:
                await update.message.reply_text(
                    detector.format_interests_for_display(), parse_mode="Markdown"
                )
                return

            subcommand = args[0].lower()

            if subcommand == "analyze":
                await update.message.reply_text("🔍 Running interest analysis...")
                await detector.run_full_analysis()
                await update.message.reply_text(
                    detector.format_interests_for_display(), parse_mode="Markdown"
                )

            elif subcommand == "add" and len(args) > 1:
                category_name = args[1].lower()
                try:
                    category = InterestCategory(category_name)
                    detector.add_manual_interest(category)
                    await update.message.reply_text(
                        f"✅ Added {category.value} to your interests!"
                    )
                except ValueError:
                    valid = ", ".join([c.value for c in InterestCategory])
                    await update.message.reply_text(f"Invalid category. Valid: {valid}")

            elif subcommand == "remove" and len(args) > 1:
                category_name = args[1].lower()
                try:
                    category = InterestCategory(category_name)
                    detector.remove_interest(category)
                    await update.message.reply_text(
                        f"❌ Removed {category.value} from your interests!"
                    )
                except ValueError:
                    await update.message.reply_text("Invalid category")

            elif subcommand == "confirm" and len(args) > 1:
                category_name = args[1].lower()
                try:
                    category = InterestCategory(category_name)
                    detector.confirm_interest(category, True)
                    await update.message.reply_text(
                        f"✅ Confirmed {category.value} as an interest!"
                    )
                except ValueError:
                    await update.message.reply_text("Invalid category")

            elif subcommand == "deny" and len(args) > 1:
                category_name = args[1].lower()
                try:
                    category = InterestCategory(category_name)
                    detector.confirm_interest(category, False)
                    await update.message.reply_text(
                        f"❌ Denied {category.value} as an interest."
                    )
                except ValueError:
                    await update.message.reply_text("Invalid category")

            elif subcommand == "list":
                categories = [c.value for c in InterestCategory]
                await update.message.reply_text(
                    "📋 *Available Categories:*\n\n"
                    + "\n".join([f"- {c}" for c in categories]),
                    parse_mode="Markdown",
                )

            else:
                await update.message.reply_text(
                    "*📊 Interests Help:*\n\n"
                    "/interests - Show interests\n"
                    "/interests analyze - Run analysis\n"
                    "/interests add <cat> - Add interest\n"
                    "/interests remove <cat> - Remove\n"
                    "/interests confirm <cat> - Confirm\n"
                    "/interests deny <cat> - Deny\n"
                    "/interests list - List categories",
                    parse_mode="Markdown",
                )

        except ImportError:
            await update.message.reply_text("Interest learner not available")

    async def relationship_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /relationship command"""
        args = context.args if hasattr(context, "args") else []

        if not self.relationship_system:
            await update.message.reply_text("Relationship system not available")
            return

        try:
            from src.agent.relationship_system import RelationshipCommands

            result = await RelationshipCommands.handle_relationship_command(
                args, self.relationship_system
            )
            await update.message.reply_text(result, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"Error: {str(e)[:100]}")

    async def logs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show or manage transparent logs"""
        user_id = update.effective_user.id
        args = context.args if hasattr(context, "args") else []

        # Try to get transparent logger
        try:
            from src.core.transparent_logger import (
                get_transparent_logger,
                VerbosityLevel,
            )

            logger = get_transparent_logger()
        except ImportError:
            await update.message.reply_text("Transparent logger not available.")
            return

        # Handle subcommands
        if args:
            subcommand = args[0].lower()

            if subcommand == "clear":
                # Clear log history
                logger.clear_logs(user_id)
                await update.message.reply_text("✅ Log history cleared!")
                return

            elif subcommand == "thought":
                # Show only thoughts
                thoughts = logger.get_thoughts(user_id, limit=10)
                settings = logger._get_verbosity(user_id)
                formatted = logger.format_for_telegram(thoughts, settings)
                await update.message.reply_text(formatted, parse_mode="Markdown")
                return

            elif subcommand == "tools":
                # Show only tool executions
                actions = logger.get_actions(user_id, limit=10)
                settings = logger._get_verbosity(user_id)
                formatted = logger.format_for_telegram(actions, settings)
                await update.message.reply_text(formatted, parse_mode="Markdown")
                return

            elif subcommand == "minimal":
                # Set minimal verbosity
                logger.set_verbosity(user_id, level=VerbosityLevel.MINIMAL)
                await update.message.reply_text(
                    "✅ Log verbosity set to minimal (key highlights only)"
                )
                return

            elif subcommand == "normal":
                # Set normal verbosity
                logger.set_verbosity(user_id, level=VerbosityLevel.NORMAL)
                await update.message.reply_text("✅ Log verbosity set to normal")
                return

            elif subcommand == "detailed":
                # Set detailed verbosity
                logger.set_verbosity(user_id, level=VerbosityLevel.DETAILED)
                await update.message.reply_text(
                    "✅ Log verbosity set to detailed (full details)"
                )
                return

            elif subcommand == "help":
                # Show help
                help_text = """
📋 *Logs Command Help*

/logs - Show recent activity
/logs thought - Show only agent thoughts
/logs tools - Show only tool executions
/logs clear - Clear log history

*Verbosity Settings:*
/logs minimal - Key highlights only
/logs normal - Balanced view (default)
/logs detailed - Full technical details
"""
                await update.message.reply_text(help_text, parse_mode="Markdown")
                return

        # Default: show recent logs
        logs = logger.get_logs(user_id, limit=10)
        settings = logger._get_verbosity(user_id)

        # Check if processing
        status = logger.get_status()
        if status.is_processing:
            status_msg = (
                f"{status.get_progress_indicator()} {status.get_display_message()}\n\n"
            )
        else:
            status_msg = ""

        formatted = logger.format_for_telegram(logs, settings)
        await update.message.reply_text(status_msg + formatted, parse_mode="Markdown")

    async def privacy_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show or set privacy settings"""
        try:
            from src.core.privacy_tiers import get_permission_manager, PrivacyTier
        except ImportError:
            await update.message.reply_text("Privacy system not available.")
            return

        pm = get_permission_manager()

        if not context.args:
            status = pm.get_permission_status()
            default = status.get("default_tier", "sensitive")

            response = f"🔒 *Privacy Settings*\n\n"
            response += f"*Default Tier:* {default.upper()}\n\n"
            response += "*Category Tiers:*\n"

            for cat, info in status.get("categories", {}).items():
                effective = info.get("effective", "public")
                override = info.get("override")
                if override:
                    response += f"• {cat}: {effective.upper()} (override: {override})\n"
                else:
                    response += f"• {cat}: {effective.upper()}\n"

            response += "\n*Commands:*\n"
            response += "• /privacy public - Set default to public\n"
            response += "• /privacy sensitive - Set default to sensitive\n"
            response += "• /privacy private - Set default to private\n"
            response += "• /permissions - Detailed permissions\n"

            await update.message.reply_text(response, parse_mode="Markdown")
            return

        tier_arg = context.args[0].lower()
        valid_tiers = {
            "public": PrivacyTier.PUBLIC,
            "sensitive": PrivacyTier.SENSITIVE,
            "private": PrivacyTier.PRIVATE,
        }

        if tier_arg not in valid_tiers:
            await update.message.reply_text(
                "Usage: /privacy <public|sensitive|private>"
            )
            return

        pm.set_default_tier(valid_tiers[tier_arg])
        await update.message.reply_text(
            f"✅ Default privacy tier set to: {tier_arg.upper()}"
        )

    async def permissions_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """List detailed permissions"""
        try:
            from src.core.privacy_tiers import get_permission_manager
        except ImportError:
            await update.message.reply_text("Privacy system not available.")
            return

        pm = get_permission_manager()
        status = pm.get_permission_status()

        response = "🔐 *Detailed Permissions*\n\n"

        # Group by tier
        by_tier = {"private": [], "sensitive": [], "public": []}
        for cat, info in status.get("categories", {}).items():
            tier = info.get("effective", "public")
            by_tier[tier].append(cat)

        for tier, cats in by_tier.items():
            if cats:
                response += f"*{tier.upper()}:*\n"
                for cat in sorted(cats):
                    grant = status["categories"][cat].get("grant")
                    if grant:
                        response += f"  ✓ {cat} ({grant})\n"
                    else:
                        response += f"  • {cat}\n"
                response += "\n"

        await update.message.reply_text(response, parse_mode="Markdown")

    async def lock_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Lock AURA - require PIN to unlock"""
        try:
            from src.core.security_layers import get_security_layers

            security = get_security_layers()

            if not security.auth.has_pin_setup():
                await update.message.reply_text(
                    "🔐 *Lock Setup*\n\n"
                    "No PIN is set up yet. Let's create one!\n\n"
                    "Send your PIN (4-6 digits):"
                )
                self.user_sessions[update.effective_user.id] = "setup_lock_pin"
                return

            security.lock()
            await update.message.reply_text(
                "🔒 *AURA Locked*\n\n"
                "AURA has been locked. Use /unlock to regain access."
            )

        except ImportError:
            await update.message.reply_text("❌ Security module not available.")

    async def unlock_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Unlock AURA with PIN"""
        if not context.args:
            await update.message.reply_text(
                "🔓 *Unlock AURA*\n\nUsage: /unlock <PIN>\n\nExample: /unlock 1234"
            )
            return

        pin = context.args[0]

        try:
            from src.core.security_layers import get_security_layers

            security = get_security_layers()

            if not security.auth.has_pin_setup():
                await update.message.reply_text(
                    "⚠️ No PIN is set up. Use /lock to set one up."
                )
                return

            if security.auth.is_locked():
                remaining = security.auth.get_lock_time_remaining()
                if remaining:
                    mins = remaining // 60
                    secs = remaining % 60
                    await update.message.reply_text(
                        f"⏳ Too many failed attempts.\nTry again in {mins}m {secs}s"
                    )
                    return

            success = await security.auth.verify_pin(pin)

            if success:
                await update.message.reply_text(
                    "🔓 *AURA Unlocked*\n\nWelcome back! AURA is now unlocked."
                )
            else:
                await update.message.reply_text(
                    "❌ *Incorrect PIN*\n\n"
                    "The PIN you entered is incorrect. Please try again."
                )

        except ImportError:
            await update.message.reply_text("❌ Security module not available.")

    async def setpin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set or change PIN"""
        try:
            from src.core.security_layers import get_security_layers

            security = get_security_layers()

            if not context.args:
                if security.auth.has_pin_setup():
                    await update.message.reply_text(
                        "🔐 *Change PIN*\n\n"
                        "Current PIN is set.\n\n"
                        "To change: /setpin <old_pin> <new_pin>\n\n"
                        "Example: /setpin 1234 5678"
                    )
                else:
                    await update.message.reply_text(
                        "🔐 *Set Up PIN*\n\n"
                        "Usage: /setpin <pin>\n\n"
                        "Example: /setpin 1234\n\n"
                        "PIN must be 4-6 digits."
                    )
                return

            if security.auth.has_pin_setup():
                if len(context.args) < 2:
                    await update.message.reply_text(
                        "To change PIN: /setpin <old_pin> <new_pin>"
                    )
                    return

                old_pin = context.args[0]
                new_pin = context.args[1]

                success = security.auth.change_pin(old_pin, new_pin)
                if success:
                    await update.message.reply_text(
                        "✅ *PIN Changed*\n\nYour PIN has been updated successfully."
                    )
                else:
                    await update.message.reply_text(
                        "❌ *Failed*\n\n"
                        "Could not change PIN. Check your current PIN is correct."
                    )
            else:
                pin = context.args[0]
                if len(pin) < 4 or len(pin) > 6 or not pin.isdigit():
                    await update.message.reply_text("❌ PIN must be 4-6 digits.")
                    return

                success = await security.auth.setup_pin(pin)
                if success:
                    await update.message.reply_text(
                        "✅ *PIN Set*\n\n"
                        "Your PIN has been set up successfully!\n\n"
                        "Use /lock to lock AURA and /unlock <pin> to unlock."
                    )
                else:
                    await update.message.reply_text("❌ Failed to set PIN.")

        except ImportError:
            await update.message.reply_text("❌ Security module not available.")

    async def security_status_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Show security status"""
        try:
            from src.core.security_layers import get_security_layers

            security = get_security_layers()

            enc_status = (
                "✅ Enabled" if security.is_encryption_enabled() else "❌ Disabled"
            )
            auth_status = (
                "✅ Required"
                if security.requires_authentication()
                else "⚠️ Not required"
            )
            pin_status = "✅ Set" if security.auth.has_pin_setup() else "❌ Not set"
            lock_status = "🔒 Locked" if security.is_locked() else "🔓 Unlocked"

            response = f"""
🛡️ *Security Status*

*Encryption:* {enc_status}
*Authentication:* {auth_status}
*PIN:* {pin_status}
*Lock Status:* {lock_status}

*Session Timeout:* {security.session.config.session_timeout_minutes} minutes
*Auto-lock on background:* {security.session.config.auto_lock_on_background}

*Commands:*
/lock - Lock AURA
/unlock <pin> - Unlock AURA
/setpin - Set/change PIN
"""
            await update.message.reply_text(response, parse_mode="Markdown")

        except ImportError:
            await update.message.reply_text("❌ Security module not available.")

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Clear conversation"""
        user_id = update.effective_user.id
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        await update.message.reply_text(
            "✅ Conversation cleared! Starting fresh.",
            reply_markup=self._get_main_keyboard(),
        )

    async def setup_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Interactive setup wizard"""
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        welcome = """
🔧 *AURA Setup Wizard*

I'll guide you through setting up AURA.

*Step 1: Telegram Token*
"""

        token = self.config.get("telegram", {}).get("token", "")

        if not token:
            welcome += """
You haven't set up a Telegram token yet.

*How to get a token:*
1. Open Telegram → Search @BotFather
2. Send /newbot
3. Follow the prompts
4. Copy the token

Then send me your token to continue.
"""
            await update.message.reply_text(welcome, parse_mode="Markdown")
            self.user_sessions[user_id] = "setup_token"
            return

        welcome += f"✅ Token configured: `{token[:10]}...`\n\n"

        llm_loaded = (
            hasattr(self.aura, "llm") and self.aura.llm.is_loaded()
            if hasattr(self.aura, "llm")
            else False
        )

        welcome += f"*Step 2: LLM Model*\n"
        if llm_loaded:
            welcome += "✅ LLM loaded and ready!\n\n"
        else:
            welcome += "⚠️ LLM not loaded (optional)\n\n"

        level = self.config.get("security", {}).get("default_level", "L2")
        welcome += f"*Step 3: Security Level*\n"
        welcome += f"Current: {level}\n\n"

        welcome += "*Step 4: Personalize*\n"
        welcome += "Tell me about yourself!\n"
        welcome += "• /remember My name is [name]\n"
        welcome += "• /style casual\n"
        welcome += "• /voice to toggle voice\n\n"

        welcome += "*Setup Complete!*\n"
        welcome += "Run /test to verify everything works."

        await update.message.reply_text(welcome, parse_mode="Markdown")

    async def test_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Test if everything works"""
        tests = []

        test_name = "Python Imports"
        try:
            import aiofiles
            import yaml

            tests.append((test_name, True, "All core imports OK"))
        except ImportError as e:
            tests.append((test_name, False, str(e)))

        test_name = "Config File"
        if os.path.exists(self.config_path):
            tests.append((test_name, True, f"Loaded"))
        else:
            tests.append((test_name, False, "Not found"))

        test_name = "Telegram API"
        token = self.config.get("telegram", {}).get("token", "")
        if token:
            tests.append((test_name, True, f"Token set"))
        else:
            tests.append((test_name, False, "No token"))

        test_name = "LLM Module"
        if hasattr(self.aura, "llm"):
            llm_loaded = (
                self.aura.llm.is_loaded()
                if hasattr(self.aura.llm, "is_loaded")
                else False
            )
            tests.append((test_name, True, "Ready" if llm_loaded else "No model"))
        else:
            tests.append((test_name, False, "Not initialized"))

        test_name = "Memory"
        if hasattr(self.aura, "memory"):
            tests.append((test_name, True, "Available"))
        else:
            tests.append((test_name, False, "Not initialized"))

        test_name = "State Manager"
        try:
            profile = await self.state_manager.get_user_profile(
                update.effective_user.id
            )
            tests.append((test_name, True, f"Working"))
        except Exception as e:
            tests.append((test_name, False, str(e)))

        result = "🧪 *Test Results*\n\n"

        passed = 0
        for name, success, msg in tests:
            status = "✅" if success else "❌"
            result += f"{status} *{name}*: {msg}\n"
            if success:
                passed += 1

        result += f"\n*{passed}/{len(tests)} tests passed*"

        await update.message.reply_text(result, parse_mode="Markdown")

    async def settings_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Show settings menu"""
        await update.message.reply_text(
            "*⚙️ Settings*\n\nChoose an option:",
            parse_mode="Markdown",
            reply_markup=self._get_settings_keyboard(),
        )

    # ============== MEDIA HANDLERS ==============

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice messages with transcription"""
        user_id = update.effective_user.id

        # Show typing indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        # Acknowledge receipt
        await update.message.reply_text("🎤 Transcribing your voice message...")

        try:
            # Get voice file
            voice = update.message.voice
            file = await context.bot.get_file(voice.file_id)

            # Download to temp file
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                temp_path = f.name

            await file.download_to_drive(temp_path)

            # Transcribe (placeholder - would use actual STT)
            # In production, use whisper.cpp, Vosk, or API
            transcription = await self._transcribe_audio(temp_path)

            # Clean up
            os.unlink(temp_path)

            if transcription:
                await update.message.reply_text(
                    f"📝 *You said:*\n\n{transcription}", parse_mode="Markdown"
                )

                # Process as regular message
                response = await self._process_message_with_context(
                    transcription, user_id, "voice"
                )

                # Adapt response
                response = await self.personalization.adapt_response(response, user_id)

                await update.message.reply_text(response)

                # Save to history
                await self.state_manager.add_to_history(user_id, "user", transcription)
                await self.state_manager.add_to_history(user_id, "assistant", response)
            else:
                await update.message.reply_text(
                    "Sorry, I couldn't transcribe that. Try again or type your message."
                )

        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            await update.message.reply_text(f"Error processing voice: {str(e)[:100]}")

    async def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file - placeholder for actual STT"""
        # This would integrate with:
        # - whisper.cpp (offline)
        # - Vosk (offline)
        # - OpenAI Whisper API (online)
        # - Google Speech-to-Text (online)

        # For now, return a placeholder
        # In production, implement actual transcription
        try:
            # Try offline whisper.cpp if available
            # return await self._whisper_cpp_transcribe(audio_path)
            pass
        except:
            pass

        # Return placeholder for demo
        return "[Transcription would appear here - configure STT for voice input]"

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo messages with OCR/analysis"""
        user_id = update.effective_user.id

        # Show typing
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        photo = update.message.photo[-1]  # Highest resolution

        try:
            # Download photo
            file = await context.bot.get_file(photo.file_id)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                temp_path = f.name

            await file.download_to_drive(temp_path)

            # Analyze or extract text
            # In production, use OCR (Tesseract, EasyOCR) or vision model
            result = await self._analyze_image(temp_path)

            os.unlink(temp_path)

            if result:
                await update.message.reply_text(result)
            else:
                await update.message.reply_text(
                    "📷 I received your photo! Image analysis would appear here with vision model integration."
                )

        except Exception as e:
            logger.error(f"Photo processing error: {e}")
            await update.message.reply_text(f"Error processing photo: {str(e)[:100]}")

    async def _analyze_image(self, image_path: str) -> str:
        """Analyze image - placeholder for vision model"""
        # Would integrate with:
        # - LLaVA (offline)
        # - BakLlava (offline)
        # - GPT-4V (online)
        # - Google Vision API (online)

        # For OCR, would use:
        # - Tesseract
        # - EasyOCR
        # - PaddleOCR

        return None  # Placeholder

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document uploads"""
        user_id = update.effective_user.id

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        document = update.message.document
        file_name = document.file_name or "document"

        # Detect file type
        file_ext = os.path.splitext(file_name)[1].lower()

        await update.message.reply_text(f"📎 Processing {file_name}...")

        try:
            # Download file
            file = await context.bot.get_file(document.file_id)

            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as f:
                temp_path = f.name

            await file.download_to_drive(temp_path)

            # Process based on type
            result = await self._process_document(temp_path, file_ext)

            os.unlink(temp_path)

            if result:
                await update.message.reply_text(result)
            else:
                await update.message.reply_text(
                    f"📄 Received {file_name}! Document processing would appear here."
                )

        except Exception as e:
            logger.error(f"Document processing error: {e}")
            await update.message.reply_text(f"Error: {str(e)[:100]}")

    async def _process_document(self, file_path: str, file_ext: str) -> str:
        """Process document based on type"""
        # PDF - would extract text
        # Text files - read directly
        # Spreadsheets - parse data
        # Images - analyze

        return None  # Placeholder

    # ============== MESSAGE HANDLING ==============

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages - chat with AURA"""
        user_id = update.effective_user.id
        message_text = update.message.text

        # Handle keyboard button presses
        if message_text in [
            "🎤 Voice",
            "📷 Photo",
            "📎 File",
            "📋 Tasks",
            "📅 Calendar",
            "📝 Notes",
            "⚙️ Settings",
            "❓ Help",
        ]:
            await self._handle_menu_button(update, context, message_text)
            return

        # Check for natural language permission statements
        try:
            from src.core.privacy_tiers import get_permission_manager

            pm = get_permission_manager()
            permission_response = pm.handle_natural_permission(message_text)

            if permission_response:
                # Track permission grant for relationship (trust boost)
                if (
                    self.relationship_system
                    and "granted" in permission_response.lower()
                ):
                    self.relationship_system.record_permission_change(
                        permission_level=2
                    )
                await update.message.reply_text(permission_response)
                return
        except ImportError:
            pass  # Privacy system not available

        # Show typing indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        # Start transparent logging - thinking phase
        try:
            from src.core.transparent_logger import get_transparent_logger, LogPrivacy

            tlogger = get_transparent_logger()
            tlogger.start_processing(phase="thinking")
            tlogger.log_thought(
                content=f"Processing: {message_text[:50]}...",
                category="message",
                privacy=LogPrivacy.SENSITIVE,
            )
        except ImportError:
            tlogger = None

        # Process message with personalization
        response = await self._process_message_with_context(
            message_text, user_id, "text"
        )

        # Track interaction for relationship system (passive)
        if self.relationship_system:
            try:
                self.relationship_system.process_message(message_text)
            except Exception as e:
                logger.debug(f"Relationship tracking error: {e}")

        # Stop transparent logging
        if tlogger:
            tlogger.stop_processing()

        # Detect mood and adapt
        mood = self.mood_detector.detect(message_text)
        response = self.mood_detector.adapt_for_mood(response, mood)

        # Personalize response
        response = await self.personalization.adapt_response(response, user_id)

        # Learn from interaction
        intent = self.intent_classifier.classify(
            message_text, await self.state_manager.get_user_profile(user_id)
        )
        await self.personalization.learn_from_interaction(
            user_id, message_text, response, intent
        )

        # Save to history
        await self.state_manager.add_to_history(user_id, "user", message_text)
        await self.state_manager.add_to_history(user_id, "assistant", response)

        await update.message.reply_text(response)

        # Optionally send voice response
        profile = await self.state_manager.get_user_profile(user_id)
        if profile.voice_enabled and profile.prefers_voice:
            # Would convert to voice and send
            pass

    async def _handle_menu_button(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, button: str
    ):
        """Handle menu button presses"""
        if button == "❓ Help":
            await self.help_command(update, context)
        elif button == "⚙️ Settings":
            await self.settings_command(update, context)
        elif button == "🎤 Voice":
            await update.message.reply_text(
                "🎤 Send me a voice message and I'll transcribe it!",
                reply_markup=self._get_main_keyboard(),
            )
        elif button == "📷 Photo":
            await update.message.reply_text(
                "📷 Send me a photo and I'll analyze it!",
                reply_markup=self._get_main_keyboard(),
            )
        elif button == "📎 File":
            await update.message.reply_text(
                "📎 Send me a document and I'll process it!",
                reply_markup=self._get_main_keyboard(),
            )
        elif button == "📋 Tasks":
            await update.message.reply_text(
                "📋 Task management coming soon!",
                reply_markup=self._get_main_keyboard(),
            )
        elif button == "📅 Calendar":
            await update.message.reply_text(
                "📅 Calendar integration coming soon!",
                reply_markup=self._get_main_keyboard(),
            )
        elif button == "📝 Notes":
            await update.message.reply_text(
                "📝 Notes feature coming soon! Use /remember to store info.",
                reply_markup=self._get_main_keyboard(),
            )
        elif button == "📊 Reports":
            try:
                from src.agent.daily_reporter import get_daily_reporter

                reporter = get_daily_reporter(self.aura)
                response = await reporter.generate_report(user_id)
                await update.message.reply_text(
                    response,
                    parse_mode="Markdown",
                    reply_markup=self._get_main_keyboard(),
                )
            except Exception as e:
                await update.message.reply_text(
                    f"Reports feature: {str(e)[:100]}",
                    reply_markup=self._get_main_keyboard(),
                )

    async def _process_message_with_context(
        self, message: str, user_id: int, source: str
    ) -> str:
        """Process message with full context"""

        # Create/get session
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = f"telegram_{user_id}"

        session_id = self.user_sessions[user_id]

        # Get user profile for context
        profile = await self.state_manager.get_user_profile(user_id)

        # Build context
        context = {
            "source": source,
            "user_id": user_id,
            "formality": profile.formality_level.value,
            "display_name": profile.display_name,
        }

        # Process through AURA
        try:
            response = await self.aura.process(message, session_id, context=context)
            return response
        except Exception as e:
            logger.error(f"Process error: {e}")
            return f"I encountered an error: {str(e)[:100]}"

    # ============== INLINE QUERIES ==============

    async def handle_inline_query(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle inline queries"""
        query = update.inline_query.query
        user_id = update.inline_query.from_user.id

        if not query:
            return

        results = []

        # Parse query type
        if query.lower().startswith("search:"):
            search_term = query[7:].strip()
            history = await self.state_manager.get_history(user_id, limit=20)
            matches = [
                msg
                for msg in history
                if search_term.lower() in msg.get("content", "").lower()
            ]

            for i, msg in enumerate(matches[:5]):
                results.append(
                    {
                        "type": "article",
                        "id": str(i),
                        "title": msg.get("content", "")[:50],
                        "description": f"Found in {msg.get('role', 'chat')}",
                        "input_message_content": {
                            "message_text": f"📋 *Found:*\n\n{msg.get('content', '')}",
                            "parse_mode": "Markdown",
                        },
                    }
                )

        elif query.lower().startswith("calc:"):
            expression = query[5:].strip()
            try:
                result = self._safe_eval(expression)
                results.append(
                    {
                        "type": "article",
                        "id": "1",
                        "title": f"= {result}",
                        "description": expression,
                        "input_message_content": {
                            "message_text": f"🧮 *Calculation:*\n\n{expression} = {result}",
                            "parse_mode": "Markdown",
                        },
                    }
                )
            except Exception as e:
                logger.warning(f"Calculation error: {e}")

        elif query.lower().startswith("convert:"):
            # Unit conversion placeholder
            results.append(
                {
                    "type": "article",
                    "id": "1",
                    "title": "Unit Converter",
                    "description": "Usage: convert: 50 f to c",
                    "input_message_content": {
                        "message_text": "📐 Send me conversions like: convert: 50 f to c",
                    },
                }
            )

        else:
            # Default: search history
            history = await self.state_manager.get_history(user_id, limit=10)
            for i, msg in enumerate(history[-3:]):
                results.append(
                    {
                        "type": "article",
                        "id": str(i),
                        "title": msg.get("content", "")[:40],
                        "description": f"From {msg.get('role', 'chat')}",
                        "input_message_content": {
                            "message_text": msg.get("content", ""),
                        },
                    }
                )

        await update.inline_query.answer(results, cache_time=60)

    # ============== CALLBACK QUERIES ==============

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        query = update.callback_query
        user_id = query.from_user.id

        await query.answer()

        if query.data == "toggle_voice":
            profile = await self.state_manager.get_user_profile(user_id)
            profile.voice_enabled = not profile.voice_enabled
            await self.state_manager.save_user_profile(profile)
            status = "enabled" if profile.voice_enabled else "disabled"
            await query.edit_message_text(f"✅ Voice responses {status}")

        elif query.data == "toggle_emoji":
            profile = await self.state_manager.get_user_profile(user_id)
            profile.uses_emoji = not profile.uses_emoji
            await self.state_manager.save_user_profile(profile)
            status = "enabled" if profile.uses_emoji else "disabled"
            await query.edit_message_text(f"✅ Emoji {status}")

        elif query.data == "set_style":
            await query.edit_message_text(
                "*Choose style:*",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "😊 Friendly", callback_data="style_friendly"
                            )
                        ],
                        [
                            InlineKeyboardButton(
                                "👔 Formal", callback_data="style_formal"
                            )
                        ],
                        [
                            InlineKeyboardButton(
                                "😎 Casual", callback_data="style_casual"
                            )
                        ],
                        [
                            InlineKeyboardButton(
                                "🔙 Back", callback_data="back_settings"
                            )
                        ],
                    ]
                ),
            )

        elif query.data.startswith("style_"):
            style = query.data.split("_")[1]
            profile = await self.state_manager.get_user_profile(user_id)
            profile.formality_level = FormalityLevel(style.upper())
            await self.state_manager.save_user_profile(profile)
            await query.edit_message_text(f"✅ Style set to {style}")

        elif query.data == "set_length":
            await query.edit_message_text(
                "*Choose response length:*",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "📝 Short", callback_data="length_short"
                            )
                        ],
                        [
                            InlineKeyboardButton(
                                "📏 Balanced", callback_data="length_balanced"
                            )
                        ],
                        [InlineKeyboardButton("📚 Long", callback_data="length_long")],
                        [
                            InlineKeyboardButton(
                                "🔙 Back", callback_data="back_settings"
                            )
                        ],
                    ]
                ),
            )

        elif query.data.startswith("length_"):
            length = query.data.split("_")[1]
            profile = await self.state_manager.get_user_profile(user_id)
            profile.response_length = ResponseLength(length.upper())
            await self.state_manager.save_user_profile(profile)
            await query.edit_message_text(f"✅ Response length set to {length}")

        elif query.data == "back_main":
            await query.edit_message_text(
                "*Main Menu*",
                parse_mode="Markdown",
                reply_markup=self._get_main_keyboard(),
            )

        elif query.data == "back_settings":
            await query.edit_message_text(
                "*⚙️ Settings*",
                parse_mode="Markdown",
                reply_markup=self._get_settings_keyboard(),
            )

    # ============== DASHBOARD COMMANDS ==============

    async def dashboard_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Show main mission control dashboard"""
        if not hasattr(self, "_dashboard_renderer"):
            from src.channels.telegram_dashboard import DashboardRenderer

            self._dashboard_renderer = DashboardRenderer(self.aura)

        await update.message.reply_text(
            await self._dashboard_renderer.handle_dashboard(), parse_mode="Markdown"
        )

    async def mission_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Alias for /dashboard - Show mission control"""
        await self.dashboard_command(update, context)

    async def inner_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show AURA's inner voice/thoughts"""
        if not hasattr(self, "_dashboard_renderer"):
            from src.channels.telegram_dashboard import DashboardRenderer

            self._dashboard_renderer = DashboardRenderer(self.aura)

        await update.message.reply_text(
            await self._dashboard_renderer.handle_inner(), parse_mode="Markdown"
        )

    async def mood_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show AURA's mood/feelings - now uses personality system"""
        if not self.personality_system:
            await update.message.reply_text("Personality system not available.")
            return

        mood = self.personality_system.get_mood()
        response = self.personality_system.express_feelings()

        await update.message.reply_text(
            f"💭 *My Current Mood*\n\n"
            f"Feeling: {mood['description']} {mood['emoji']}\n"
            f"Intensity: {mood['intensity']:.0%}\n\n"
            f"{response}",
            parse_mode="Markdown",
        )

    async def personality_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /personality command"""
        if not self.personality_system:
            await update.message.reply_text("Personality system not available.")
            return

        args = context.args if hasattr(context, "args") and context.args else []
        response = PersonalityCommands.handle_personality_command(
            args, self.personality_system
        )

        await update.message.reply_text(response, parse_mode="Markdown")

    async def morals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /morals command"""
        if not self.personality_system:
            await update.message.reply_text("Personality system not available.")
            return

        args = context.args if hasattr(context, "args") and context.args else []
        response = PersonalityCommands.handle_morals_command(
            args, self.personality_system
        )

        await update.message.reply_text(response, parse_mode="Markdown")

    async def tools_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show tools status"""
        if not hasattr(self, "_dashboard_renderer"):
            from src.channels.telegram_dashboard import DashboardRenderer

            self._dashboard_renderer = DashboardRenderer(self.aura)

        await update.message.reply_text(
            await self._dashboard_renderer.handle_tools(), parse_mode="Markdown"
        )

    async def report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /report command for daily/weekly reports"""
        try:
            from src.agent.daily_reporter import get_daily_reporter

            reporter = get_daily_reporter(self.aura)
            response = await reporter.handle_report_command(update, context)
            await update.message.reply_text(response, parse_mode="Markdown")
        except ImportError:
            await update.message.reply_text("Daily reporter not available.")
        except Exception as e:
            logger.error(f"Report command error: {e}")
            await update.message.reply_text(f"Error generating report: {str(e)[:100]}")

    # ============== CONTROL COMMANDS ==============

    async def restart_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Restart AURA"""
        await update.message.reply_text("🔄 Restarting AURA...")
        try:
            await self.aura.shutdown()
            await self.aura.initialize()
            await update.message.reply_text("✅ AURA restarted successfully!")
        except Exception as e:
            await update.message.reply_text(f"❌ Restart failed: {str(e)[:100]}")

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop AURA"""
        await update.message.reply_text("🛑 Stopping AURA...")
        try:
            await self.aura.shutdown()
            await update.message.reply_text(
                "✅ AURA stopped. To restart, run the bot again."
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Stop failed: {str(e)[:100]}")

    # ============== EXECUTION CONTROL COMMANDS ==============

    async def execution_stop_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Graceful stop - finish current atomic action, then stop"""
        if not self.execution_controller:
            await update.message.reply_text("❌ Execution control not available")
            return

        await update.message.reply_text(
            "🛑 *Graceful Stop Requested*\n\n"
            "Finishing current atomic action, then stopping...",
            parse_mode="Markdown",
        )

        try:
            event = await self.execution_controller.stop(reason="telegram_stop")
            await update.message.reply_text(
                f"✅ Stop initiated\n"
                f"Event: `{event.event_id}`\n"
                f"Previous state: {event.state_before.value}",
                parse_mode="Markdown",
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Stop failed: {str(e)[:100]}")

    async def execution_force_stop_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Force stop - finish current tool, then stop agent loop"""
        if not self.execution_controller:
            await update.message.reply_text("❌ Execution control not available")
            return

        await update.message.reply_text(
            "⚠️ *Force Stop Requested*\n\n"
            "Finishing current tool, then stopping agent loop...",
            parse_mode="Markdown",
        )

        try:
            event = await self.execution_controller.force_stop(
                reason="telegram_force_stop"
            )
            status = self.execution_controller.get_status()
            await update.message.reply_text(
                f"✅ Force stop initiated\n"
                f"Event: `{event.event_id}`\n"
                f"Current action: {status.get('current_action', {}).get('description', 'None')}",
                parse_mode="Markdown",
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Force stop failed: {str(e)[:100]}")

    async def execution_kill_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Emergency kill - stop everything immediately"""
        if not self.execution_controller:
            await update.message.reply_text("❌ Execution control not available")
            return

        await update.message.reply_text(
            "🚨 *EMERGENCY KILL*\n\n"
            "Stopping everything immediately!\n"
            "Saving state and cleaning up...",
            parse_mode="Markdown",
        )

        try:
            event = await self.execution_controller.kill(reason="telegram_kill")
            await update.message.reply_text(
                f"🛑 Kill completed!\n\n"
                f"Event: `{event.event_id}`\n"
                f"State before: {event.state_before.value}\n"
                f"Cleaned up: {', '.join(event.cleanup_performed) if event.cleanup_performed else 'None'}",
                parse_mode="Markdown",
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Kill failed: {str(e)[:100]}")

    async def execution_status_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Show execution status"""
        if not self.execution_controller:
            await update.message.reply_text("❌ Execution control not available")
            return

        status = self.execution_controller.get_status()

        current = status.get("current_action")
        current_str = (
            f"{current['type']}: {current['description']}" if current else "None"
        )

        message = f"""
*⚙️ Execution Status*

*State:* `{status["state"]}`
*Stop Level:* `{status["stop_level"]}`
*Conversation:* `{status["conversation_id"]}`
*Loop Count:* {status["loop_count"]}

*Current Action:*
{current_str}

*Pending Actions:* {status["pending_actions"]}
*Active Timeouts:* {status["active_timeouts"]}
*Can Accept Input:* {status["can_accept_input"]}

*Commands:*
/stop - Graceful stop
/force-stop - Force stop
/kill - Emergency kill
"""
        await update.message.reply_text(message, parse_mode="Markdown")

    # ============== ERROR HANDLING ==============

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Update {update} caused error {context.error}")
        if update and update.message:
            await update.message.reply_text(
                "Sorry, something went wrong. Please try again."
            )

    # ============== RUN ==============

    def run(self, token: str):
        """Start the bot"""
        if not TELEGRAM_AVAILABLE:
            print("ERROR: python-telegram-bot not installed!")
            print("Install: pip install python-telegram-bot")
            return

        app = Application.builder().token(token).build()

        # Add error handler
        app.add_error_handler(self.error_handler)

        # Commands
        app.add_handler(CommandHandler("start", self.start_command))
        app.add_handler(CommandHandler("help", self.help_command))
        app.add_handler(CommandHandler("status", self.status_command))
        app.add_handler(CommandHandler("config", self.config_command))
        app.add_handler(CommandHandler("setmodel", self.setmodel_command))
        app.add_handler(CommandHandler("setvoice", self.setvoice_command))
        app.add_handler(CommandHandler("setlevel", self.setlevel_command))
        app.add_handler(CommandHandler("style", self.style_command))
        app.add_handler(CommandHandler("voice", self.voice_command))
        app.add_handler(CommandHandler("remember", self.remember_command))
        app.add_handler(CommandHandler("search", self.search_command))
        app.add_handler(CommandHandler("summarize", self.summarize_command))
        app.add_handler(CommandHandler("stats", self.stats_command))
        app.add_handler(CommandHandler("models", self.models_command))
        app.add_handler(CommandHandler("voices", self.voices_command))
        app.add_handler(CommandHandler("memory", self.memory_command))
        app.add_handler(CommandHandler("interests", self.interests_command))
        app.add_handler(CommandHandler("relationship", self.relationship_command))
        app.add_handler(CommandHandler("logs", self.logs_command))
        app.add_handler(CommandHandler("privacy", self.privacy_command))
        app.add_handler(CommandHandler("permissions", self.permissions_command))
        app.add_handler(CommandHandler("clear", self.clear_command))
        app.add_handler(CommandHandler("restart", self.restart_command))
        app.add_handler(CommandHandler("stop", self.stop_command))
        app.add_handler(CommandHandler("setup", self.setup_command))
        app.add_handler(CommandHandler("test", self.test_command))
        app.add_handler(CommandHandler("settings", self.settings_command))

        # Security commands
        app.add_handler(CommandHandler("lock", self.lock_command))
        app.add_handler(CommandHandler("unlock", self.unlock_command))
        app.add_handler(CommandHandler("setpin", self.setpin_command))
        app.add_handler(CommandHandler("security", self.security_status_command))

        # Dashboard commands
        app.add_handler(CommandHandler("dashboard", self.dashboard_command))
        app.add_handler(CommandHandler("mission", self.mission_command))
        app.add_handler(CommandHandler("inner", self.inner_command))
        app.add_handler(CommandHandler("mood", self.mood_command))
        app.add_handler(CommandHandler("personality", self.personality_command))
        app.add_handler(CommandHandler("morals", self.morals_command))
        app.add_handler(CommandHandler("tools", self.tools_command))

        # Report commands
        app.add_handler(CommandHandler("report", self.report_command))

        # Execution control commands
        app.add_handler(CommandHandler("stop", self.execution_stop_command))
        app.add_handler(CommandHandler("force-stop", self.execution_force_stop_command))
        app.add_handler(CommandHandler("kill", self.execution_kill_command))
        app.add_handler(
            CommandHandler("execution-status", self.execution_status_command)
        )

        # Media handlers
        app.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
        app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        app.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))

        # Inline queries
        app.add_handler(InlineQueryHandler(self.handle_inline_query))

        # Callback queries
        app.add_handler(CallbackQueryHandler(self.handle_callback))

        # Regular messages (must be last)
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

        print("🤖 AURA Telegram Bot starting... (Friday-like mode)")
        app.run_polling()


# For running directly
if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from main import AURA
    import yaml

    # Load config
    if os.path.exists("config/config.yaml"):
        with open("config/config.yaml") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Get token
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        print("ERROR: Set TELEGRAM_TOKEN environment variable")
        print("Get token from @BotFather on Telegram")
        exit(1)

    # Create and run
    aura = AURA(config)
    bot = AuraTelegramBot(aura)
    bot.run(token)
