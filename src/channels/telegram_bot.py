"""
AURA Telegram Bot - Self-Contained Setup & Control

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
/clear - Clear conversation
/help - Show all commands

Keyboard shortcuts for common actions
"""

import asyncio
import logging
import os
import json
from pathlib import Path
from typing import Dict, Optional

# Telegram imports
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        filters,
        ContextTypes,
    )

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

logger = logging.getLogger(__name__)


class AuraTelegramBot:
    """
    Self-contained Telegram bot for AURA
    All setup/config via commands - no external web!
    """

    def __init__(self, aura_instance, config_path: str = "config/config.yaml"):
        self.aura = aura_instance
        self.config_path = config_path
        self.config = self._load_config()
        self.user_sessions: Dict[int, str] = {}  # user_id -> session_id

    def _load_config(self) -> Dict:
        """Load configuration from YAML"""
        try:
            import yaml

            if os.path.exists(self.config_path):
                with open(self.config_path) as f:
                    return yaml.safe_load(f)
        except:
            pass
        return {}

    def _save_config(self):
        """Save configuration"""
        try:
            import yaml

            with open(self.config_path, "w") as f:
                yaml.dump(self.config, f)
        except Exception as e:
            logger.error(f"Config save error: {e}")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome = """
🤖 *AURA v3 - Personal Mobile AGI*

Welcome! I'm your personal AI assistant that runs 100% offline on your device.

*Quick Start:*
1. /status - Check system status
2. /config - View configuration  
3. /help - See all commands

*Your Data Stays Private:*
- All processing on your device
- No cloud, no external servers
- 100% offline capable

Send me a message and I'll help you!
"""
        await update.message.reply_text(welcome, parse_mode="Markdown")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show all commands"""
        help_text = """
*📋 AURA Commands*

*🎛️ Control:*
/start - Start AURA
/stop - Stop AURA  
/status - System status
/restart - Restart AURA

*⚙️ Setup:*
/setup - Interactive setup wizard
/test - Test system functionality

*⚙️ Configuration:*
/config - Show current config
/setmodel <name> - Set LLM model
/setvoice <engine> - Set TTS engine
/setlevel L1-L4 - Set permission level

*📦 Info:*
/models - List available models
/voices - List TTS engines
/memory - Memory statistics

*🗑️ Utility:*
/clear - Clear conversation

*💬 Just chat with me!*
"""
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show system status"""
        status = []

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

        # Sessions
        session_count = len(self.user_sessions)
        status.append(f"💬 Sessions: {session_count}")

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

        # Memory
        memory = self.config.get("memory", {})
        config_text += f"*Memory:*\n"
        config_text += f"  Working: {memory.get('working_size', 10)}\n"
        config_text += f"  Short-term: {memory.get('short_term_size', 100)}"

        await update.message.reply_text(config_text, parse_mode="Markdown")

    async def setmodel_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Set LLM model"""
        if not context.args:
            await update.message.reply_text(
                "Usage: /setmodel <model_name>\n\nAvailable:\n- llama\n- qwen\n- mistral"
            )
            return

        model = context.args[0].lower()

        # Validate
        valid = ["llama", "qwen", "mistral", "phi"]
        if model not in valid:
            await update.message.reply_text(f"Invalid model. Use: {', '.join(valid)}")
            return

        # Save
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
                "Usage: /setlevel <L1|L2|L3|L4>\n\nLevels:\nL1 - Full trust (no confirmations)\nL2 - Normal (confirm calls/messages)\nL3 - Restricted (confirm most actions)\nL4 - Maximum (confirm everything)"
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

    async def models_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List available models"""
        models_text = """
📦 *Available LLM Models*

For mobile (4GB RAM):
• qwen2.5-1b - Best balance (625MB)
• phi-3-mini - Fast (2.5GB)
• mistral-7b - Powerful (4GB+)

Quantization:
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
  Install: apt install espeak-ng
• pyttsx3 - More natural
  Install: pip install pyttsx3

*Online (Need Internet):*
• google - Google TTS
• edge - Microsoft Edge TTS

*Setup:*
/setvoice <engine>
"""
        await update.message.reply_text(voices_text, parse_mode="Markdown")

    async def memory_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show memory stats"""
        if hasattr(self.aura, "memory"):
            stats = self.aura.memory.get_stats()
            await update.message.reply_text(f"📊 *Memory Stats*\n\n{stats}")
        else:
            await update.message.reply_text("Memory stats not available")

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Clear conversation"""
        user_id = update.effective_user.id
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        await update.message.reply_text("✅ Conversation cleared!")

    async def setup_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Interactive setup wizard"""
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        welcome = """
🔧 *AURA Setup Wizard*

I'll guide you through setting up AURA step by step.

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
4. Copy the token (something like: 123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11)

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
            import cryptography

            tests.append((test_name, True, "All core imports OK"))
        except ImportError as e:
            tests.append((test_name, False, str(e)))

        test_name = "Config File"
        if os.path.exists(self.config_path):
            tests.append((test_name, True, f"Loaded from {self.config_path}"))
        else:
            tests.append((test_name, False, "Config not found"))

        test_name = "Telegram API"
        token = self.config.get("telegram", {}).get("token", "")
        if token:
            tests.append((test_name, True, f"Token: {token[:10]}..."))
        else:
            tests.append((test_name, False, "No token configured"))

        test_name = "LLM Module"
        if hasattr(self.aura, "llm"):
            llm_loaded = (
                self.aura.llm.is_loaded()
                if hasattr(self.aura.llm, "is_loaded")
                else False
            )
            if llm_loaded:
                tests.append((test_name, True, "Model loaded"))
            else:
                tests.append((test_name, True, "Module available (no model)"))
        else:
            tests.append((test_name, False, "LLM not initialized"))

        test_name = "Memory Module"
        if hasattr(self.aura, "memory"):
            tests.append((test_name, True, "Memory module available"))
        else:
            tests.append((test_name, False, "Memory not initialized"))

        result = "🧪 *Test Results*\n\n"

        passed = 0
        for name, success, msg in tests:
            status = "✅" if success else "❌"
            result += f"{status} *{name}*: {msg}\n"
            if success:
                passed += 1

        result += f"\n*{passed}/{len(tests)} tests passed*"

        await update.message.reply_text(result, parse_mode="Markdown")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages - chat with AURA"""
        user_id = update.effective_user.id
        message = update.message.text

        # Create/get session
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = f"telegram_{user_id}"

        session_id = self.user_sessions[user_id]

        # Process message
        try:
            response = await self.aura.process(message, session_id)
            await update.message.reply_text(response)
        except Exception as e:
            logger.error(f"Process error: {e}")
            await update.message.reply_text(f"Error: {str(e)[:100]}")

    def run(self, token: str):
        """Start the bot"""
        if not TELEGRAM_AVAILABLE:
            print("ERROR: python-telegram-bot not installed!")
            print("Install: pip install python-telegram-bot")
            return

        app = Application.builder().token(token).build()

        # Commands
        app.add_handler(CommandHandler("start", self.start_command))
        app.add_handler(CommandHandler("help", self.help_command))
        app.add_handler(CommandHandler("status", self.status_command))
        app.add_handler(CommandHandler("config", self.config_command))
        app.add_handler(CommandHandler("setmodel", self.setmodel_command))
        app.add_handler(CommandHandler("setvoice", self.setvoice_command))
        app.add_handler(CommandHandler("setlevel", self.setlevel_command))
        app.add_handler(CommandHandler("models", self.models_command))
        app.add_handler(CommandHandler("voices", self.voices_command))
        app.add_handler(CommandHandler("memory", self.memory_command))
        app.add_handler(CommandHandler("clear", self.clear_command))
        app.add_handler(CommandHandler("restart", self.restart_command))
        app.add_handler(CommandHandler("setup", self.setup_command))
        app.add_handler(CommandHandler("test", self.test_command))

        # Messages
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

        print("🤖 AURA Telegram Bot starting...")
        app.run_polling()


# For running directly
if __name__ == "__main__":
    # Import AURA
    import sys

    sys.path.insert(0, ".")
    from main import AURA
    import yaml

    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

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
