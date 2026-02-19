"""
AURA v3 - Next-Generation Personal Mobile AGI Assistant
Entry Point

Usage:
    python main.py --mode cli       # Command line interface
    python main.py --mode voice     # Voice interface
    python main.py --mode api       # REST API server
    python main.py --mode telegram  # Telegram bot
"""

import asyncio
import argparse
import logging
import signal
import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import yaml

from src.agent.loop import ReActAgent, AgentFactory
from src.tools.registry import ToolRegistry, ToolExecutor
from src.memory import HierarchicalMemory
from src.learning.engine import LearningEngine
from src.context.detector import ContextDetector
from src.security.permissions import SecurityLayer, PermissionLevel
from src.llm import LLMRunner, MockLLM
from src.channels.voice import VoiceChannel
from src.session.manager import SessionManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/aura.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("AURA")


class AURA:
    """Main AURA Application"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm: Optional[LLMRunner] = None
        self.agent: Optional[ReActAgent] = None
        self.memory: Optional[HierarchicalMemory] = None
        self.tools: Optional[ToolRegistry] = None
        self.tool_executor: Optional[ToolExecutor] = None
        self.security: Optional[SecurityLayer] = None
        self.learning: Optional[LearningEngine] = None
        self.context: Optional[ContextDetector] = None
        self.voice: Optional[VoiceChannel] = None
        self.sessions: Optional[SessionManager] = None
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize all components in order"""
        logger.info("Initializing AURA v3...")
        start_time = datetime.now()

        try:
            logger.info("[1/7] Loading LLM...")
            llm_config = self.config.get("llm", {})
            model_path = llm_config.get("model_path", "models/qwen2.5-1b-q5_k_m.gguf")

            if os.path.exists(model_path):
                self.llm = LLMRunner(
                    model_path=model_path,
                    max_context=llm_config.get("max_context", 4096),
                    n_gpu_layers=llm_config.get("n_gpu_layers", 0),
                    temperature=llm_config.get("temperature", 0.7),
                    max_tokens=llm_config.get("max_tokens", 512),
                )
                logger.info(f"LLM loaded from {model_path}")
            else:
                logger.warning(f"Model not found at {model_path}, using MockLLM")
                self.llm = MockLLM()

            logger.info("[2/7] Initializing memory system...")
            memory_config = self.config.get("memory", {})
            self.memory = HierarchicalMemory(
                working_size=memory_config.get("working_size", 10),
                short_term_size=memory_config.get("short_term_size", 100),
                db_path=memory_config.get("db_path", "data/memories/aura.db"),
                self_model_path=memory_config.get(
                    "self_model_path", "data/memories/self_model.db"
                ),
            )
            await self.memory.initialize()

            logger.info("[3/7] Loading tools...")
            self.tools = ToolRegistry()
            self.tool_executor = ToolExecutor(self.tools, self.security)
            self.tools.load_all_tools()
            logger.info(f"Loaded {len(self.tools.list_tools())} tools")

            logger.info("[4/7] Setting up security layer...")
            security_config = self.config.get("security", {})
            self.security = SecurityLayer(
                default_level=PermissionLevel[
                    security_config.get("default_level", "L2")
                ],
                audit_log_path=security_config.get("audit_log", "logs/security.log"),
                banking_protection=security_config.get("banking_protection", True),
            )

            logger.info("[5/7] Initializing learning engine...")
            learning_config = self.config.get("learning", {})
            self.learning = LearningEngine(
                patterns_path=learning_config.get("patterns_path", "data/patterns"),
                min_confidence=learning_config.get("min_confidence", 0.6),
                max_patterns=learning_config.get("max_patterns", 1000),
            )
            await self.learning.initialize()

            logger.info("[6/7] Setting up context detector...")
            context_config = self.config.get("context", {})
            self.context = ContextDetector(
                work_start_hour=context_config.get("work_start_hour", 9),
                work_end_hour=context_config.get("work_end_hour", 18),
                sleep_start_hour=context_config.get("sleep_start_hour", 23),
                sleep_end_hour=context_config.get("sleep_end_hour", 7),
            )

            logger.info("[7/7] Initializing session manager...")
            session_config = self.config.get("session", {})
            self.sessions = SessionManager(
                storage_path=session_config.get("storage_path", "data/sessions"),
                encryption=session_config.get("encryption", True),
                max_history=session_config.get("max_history", 100),
            )
            await self.sessions.initialize()

            voice_config = self.config.get("voice", {})
            self.voice = VoiceChannel(
                tts_engine=voice_config.get("tts_engine", "sarvam"),
                language=voice_config.get("language", "en"),
                speed=voice_config.get("speed", 1.0),
            )

            logger.info("[8/8] Creating ReAct agent...")
            perf_config = self.config.get("performance", {})
            self.agent = AgentFactory.create(
                llm=self.llm,
                memory=self.memory,
                tools=self.tools,
                security=self.security,
                learning=self.learning,
                context=self.context,
                max_iterations=perf_config.get("max_react_iterations", 10),
                tool_timeout=perf_config.get("tool_timeout", 30),
            )

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"AURA v3 initialized successfully in {elapsed:.2f}s")
            self._running = True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    async def process(
        self, user_input: str, session_id: str = None, context: Dict[str, Any] = None
    ) -> str:
        """Process user input through ReAct agent"""
        if not self._running:
            raise RuntimeError("AURA is not initialized")

        session_id = session_id or self.sessions.create_session()
        session = await self.sessions.get_session(session_id)

        current_context = await self.context.detect()
        if context:
            current_context.update(context)

        await self.memory.add_to_working(
            role="user",
            content=user_input,
            metadata={"session_id": session_id, "context": current_context},
        )

        try:
            response = await self.agent.run(
                query=user_input, session=session, context=current_context
            )

            await self.memory.add_to_working(
                role="assistant", content=response, metadata={"session_id": session_id}
            )

            await self.sessions.update_session(session_id, user_input, response)

            await self.learning.learn_from_interaction(
                query=user_input,
                response=response,
                context=current_context,
                tools_used=getattr(self.agent, "_last_tools_used", []),
            )

            return response

        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return f"I encountered an error: {str(e)}. Please try again."

    async def run_cli(self):
        """Run command-line interface"""
        logger.info("Starting CLI mode...")
        print("\n" + "=" * 60)
        print("   AURA v3 - Personal Mobile AGI Assistant")
        print("=" * 60)
        print("Type 'exit' or 'quit' to stop.")
        print("Type 'status' to see system status.")
        print("=" * 60 + "\n")

        session_id = self.sessions.create_session()

        while self._running and not self._shutdown_event.is_set():
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == "status":
                    await self._print_status()
                    continue

                print("\nAURA: ", end="", flush=True)
                response = await self.process(user_input, session_id)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n\nInterrupted. Shutting down...")
                break
            except EOFError:
                break

    async def run_voice(self):
        """Run voice interface"""
        logger.info("Starting Voice mode...")
        print("\n" + "=" * 60)
        print("   AURA v3 - Voice Interface")
        print("=" * 60)
        print("Listening... Say 'stop' to exit.")
        print("=" * 60 + "\n")

        session_id = self.sessions.create_session()

        while self._running and not self._shutdown_event.is_set():
            try:
                user_input = await self.voice.listen()

                if not user_input:
                    continue

                if user_input.lower().strip() in ["stop", "exit", "quit"]:
                    await self.voice.speak("Goodbye!")
                    break

                print(f"You: {user_input}")
                response = await self.process(user_input, session_id)
                print(f"AURA: {response}")

                await self.voice.speak(response)

            except Exception as e:
                logger.error(f"Voice error: {e}")
                await self.voice.speak("I encountered an error. Please try again.")

    async def run_api(self, port: int = 8080):
        """Run REST API server"""
        from aiohttp import web

        logger.info(f"Starting API server on port {port}...")

        async def handle_chat(request):
            try:
                data = await request.json()
                user_input = data.get("message", "")
                session_id = data.get("session_id")

                if not user_input:
                    return web.json_response(
                        {"error": "No message provided"}, status=400
                    )

                response = await self.process(user_input, session_id)

                return web.json_response(
                    {
                        "response": response,
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

        async def handle_status(request):
            return web.json_response(
                {
                    "status": "running" if self._running else "stopped",
                    "tools_loaded": len(self.tools.list_tools()) if self.tools else 0,
                    "sessions_active": len(self.sessions.list_sessions())
                    if self.sessions
                    else 0,
                }
            )

        async def handle_health(request):
            return web.json_response({"healthy": self._running})

        app = web.Application()
        app.router.add_post("/chat", handle_chat)
        app.router.add_get("/status", handle_status)
        app.router.add_get("/health", handle_health)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()

        logger.info(f"API server running at http://0.0.0.0:{port}")
        print(f"\nAURA API Server running on port {port}")
        print("Endpoints: POST /chat, GET /status, GET /health\n")

        await self._shutdown_event.wait()
        await runner.cleanup()

    async def run_telegram(self, token: str = None):
        """Run Telegram bot interface"""
        logger.info("Starting Telegram bot mode...")

        token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            logger.error("No Telegram bot token provided")
            return

        try:
            from telegram import Update
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler,
                filters,
            )

            async def handle_message(update, context):
                user_input = update.message.text
                user_id = str(update.effective_user.id)

                session_id = f"telegram_{user_id}"

                response = await self.process(user_input, session_id)
                await update.message.reply_text(response)

            async def handle_start(update, context):
                await update.message.reply_text(
                    "Hello! I'm AURA, your personal AI assistant. "
                    "How can I help you today?"
                )

            app = Application.builder().token(token).build()
            app.add_handler(CommandHandler("start", handle_start))
            app.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
            )

            await app.initialize()
            await app.start()
            await app.updater.start_polling()

            logger.info("Telegram bot started")
            await self._shutdown_event.wait()

            await app.updater.stop()
            await app.stop()

        except ImportError:
            logger.error("python-telegram-bot not installed")
            raise

    async def _print_status(self):
        """Print system status"""
        status = {
            "LLM": "Loaded" if self.llm else "Not loaded",
            "Memory": "Initialized" if self.memory else "Not initialized",
            "Tools": len(self.tools.list_tools()) if self.tools else 0,
            "Security Level": self.security.default_level.name
            if self.security
            else "None",
            "Sessions": len(self.sessions.list_sessions()) if self.sessions else 0,
            "Learning Patterns": await self.learning.get_pattern_count()
            if self.learning
            else 0,
            "Running": self._running,
        }

        print("\n--- System Status ---")
        for key, value in status.items():
            print(f"  {key}: {value}")
        print()

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down AURA...")
        self._running = False
        self._shutdown_event.set()

        if self.sessions:
            await self.sessions.save_all()

        if self.memory:
            await self.memory.persist()

        if self.llm:
            self.llm.unload()

        logger.info("AURA shutdown complete")


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(path)

    if not config_path.exists():
        logger.warning(f"Config file not found: {path}, using defaults")
        return get_default_config()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        "llm": {
            "model_type": "llama",
            "model_path": "models/qwen2.5-1b-q5_k_m.gguf",
            "quantization": "q5_k_m",
            "max_context": 4096,
            "n_gpu_layers": 0,
            "temperature": 0.7,
            "max_tokens": 512,
        },
        "memory": {
            "working_size": 10,
            "short_term_size": 100,
            "db_path": "data/memories/aura.db",
            "self_model_path": "data/memories/self_model.db",
        },
        "learning": {
            "patterns_path": "data/patterns",
            "min_confidence": 0.6,
            "max_patterns": 1000,
        },
        "security": {
            "default_level": "L2",
            "audit_log": "logs/security.log",
            "banking_protection": True,
        },
        "context": {
            "work_start_hour": 9,
            "work_end_hour": 18,
            "sleep_start_hour": 23,
            "sleep_end_hour": 7,
        },
        "voice": {"tts_engine": "sarvam", "language": "en", "speed": 1.0},
        "session": {
            "storage_path": "data/sessions",
            "encryption": True,
            "max_history": 100,
        },
        "performance": {
            "max_react_iterations": 10,
            "tool_timeout": 30,
            "thermal_monitoring": True,
        },
    }


async def main():
    parser = argparse.ArgumentParser(
        description="AURA v3 - Personal Mobile AGI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --mode cli
    python main.py --mode voice
    python main.py --mode api --port 8080
    python main.py --mode telegram
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "voice", "api", "telegram"],
        default="cli",
        help="Interface mode to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--model", type=str, default=None, help="Override model path")
    parser.add_argument("--port", type=int, default=8080, help="Port for API mode")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model:
        config["llm"]["model_path"] = args.model

    aura = AURA(config)

    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(aura.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass

    try:
        await aura.initialize()

        if args.mode == "cli":
            await aura.run_cli()
        elif args.mode == "voice":
            await aura.run_voice()
        elif args.mode == "api":
            await aura.run_api(args.port)
        elif args.mode == "telegram":
            await aura.run_telegram()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        await aura.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
