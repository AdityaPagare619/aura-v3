"""
AURA Voice System - Voice Pipeline (DEPRECATED)
Coordinates STT, TTS, and hot word detection

DEPRECATION NOTICE:
    This module is deprecated and will be removed in a future version.
    Please migrate to the real-time pipeline:

    BEFORE (deprecated):
        from aura.voice import VoicePipeline, TelegramVoiceHandler
        pipeline = VoicePipeline(config)
        handler = TelegramVoiceHandler(pipeline)

    AFTER (recommended):
        from aura.voice import RealtimeVoicePipeline, TelegramVoiceAdapter
        pipeline = RealtimeVoicePipeline(config)
        adapter = TelegramVoiceAdapter(pipeline)

    The new pipeline provides:
    - Streaming STT/TTS with lower latency
    - Voice Activity Detection (VAD)
    - Multiple operating modes (push-to-talk, wake-word, continuous)
    - Latency budget tracking
    - Emergency hotword detection

    See real_time_pipeline.py for documentation.
"""

import asyncio
from datetime import datetime
import json
import logging
import os
import tempfile
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .stt import STTConfig, STTProcessor
from .tts import TTSConfig, TTSEngine, TTSQueueManager
from .hotword import HotWordConfig, HotWordDetector

logger = logging.getLogger(__name__)

# Emit deprecation warning on import
warnings.warn(
    "The voice.pipeline module is deprecated. "
    "Use voice.real_time_pipeline (RealtimeVoicePipeline, TelegramVoiceAdapter) instead.",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class VoicePipelineConfig:
    """Configuration for voice pipeline"""

    stt_config: STTConfig
    tts_config: TTSConfig
    hotword_config: HotWordConfig
    enable_hotword: bool = True
    enable_continuous: bool = False
    cache_dir: str = "/data/data/com.termux/files/home/.cache/aura/voice"
    telegram_mode: bool = True


class VoicePipeline:
    """
    Complete voice processing pipeline

    Handles: Telegram voice → STT → Intent → TTS → Audio output
    """

    def __init__(self, config: VoicePipelineConfig):
        self.config = config

        self.stt = STTProcessor(config.stt_config)
        self.tts_engine = TTSEngine(config.tts_config)
        self.tts_queue = TTSQueueManager(self.tts_engine)

        if config.enable_hotword:
            self.hotword = HotWordDetector(config.hotword_config)
        else:
            self.hotword = None

        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

        self._queue_thread: Optional[threading.Thread] = None
        self._continuous_listening = False
        self._intent_handler: Optional[Callable] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the voice pipeline"""
        logger.info("Initializing voice pipeline...")

        try:
            if not self.stt.engine.load_model():
                logger.warning("STT model not loaded, will lazy-load")

            if self.hotword and self.config.enable_hotword:
                if not self.hotword.load_model():
                    logger.warning("Hot word model not loaded")

            self.tts_queue.start_processing()

            self._initialized = True
            logger.info("Voice pipeline initialized")
            return True

        except Exception as e:
            logger.error(f"Pipeline init error: {e}")
            return False

    def set_intent_handler(self, handler: Callable) -> None:
        """Set intent handler for processing transcribed text"""
        self._intent_handler = handler

    async def process_telegram_voice(
        self, voice_file_path: str, intent_handler: Optional[Callable] = None
    ) -> str:
        """
        Process a voice message from Telegram

        Args:
            voice_file_path: Path to voice message audio
            intent_handler: Optional override for intent processing

        Returns:
            Response text that was spoken
        """
        handler = intent_handler or self._intent_handler

        logger.info(f"Processing voice message: {voice_file_path}")

        try:
            result = self.stt.process_voice_message(voice_file_path)
            text = result.get("text", "")

            if not text:
                response = "I couldn't understand that. Please try again."
            else:
                logger.info(f"Transcribed: {text}")

                if handler:
                    response = await handler(text)
                else:
                    response = "Voice received but no intent handler configured."

                self._log_interaction(text, response)

        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            response = "Sorry, I had trouble processing that."

        self.speak(response)
        return response

    async def process_audio_file(
        self, audio_path: str, language: Optional[str] = None
    ) -> dict:
        """
        Process an audio file

        Args:
            audio_path: Path to audio file
            language: Optional language code

        Returns:
            Transcription result
        """
        return self.stt.process_voice_message(audio_path)

    def speak(self, text: str, priority: int = 0) -> None:
        """
        Add text to TTS queue

        Args:
            text: Text to speak
            priority: Priority level (0-10)
        """
        if not text.strip():
            return

        self.tts_queue.enqueue(text, priority)

        if self._queue_thread is None or not self._queue_thread.is_alive():
            self._queue_thread = threading.Thread(
                target=self.tts_queue.process_queue, daemon=True
            )
            self._queue_thread.start()

    def speak_immediate(self, text: str) -> None:
        """
        Interrupt current speech and speak immediately

        Args:
            text: Text to speak
        """
        if not text.strip():
            return

        self.tts_queue.interrupt()
        self.tts_engine.speak(text, blocking=True)

    def speak_async(self, text: str, callback: Optional[Callable] = None) -> None:
        """
        Speak text asynchronously

        Args:
            text: Text to speak
            callback: Optional callback when done
        """
        if text.strip():
            self.tts_engine.speak_async(text, callback)

    def start_continuous_listening(self) -> bool:
        """
        Start continuous ambient listening

        Returns:
            True if started successfully
        """
        if not self.config.enable_continuous:
            logger.warning("Continuous listening not enabled in config")
            return False

        if not self.hotword:
            logger.warning("Hot word detector not available")
            return False

        def on_hotword():
            logger.info("Hot word detected!")
            asyncio.create_task(self._handle_voice_command_mode())

        if self.hotword.start_listening(on_hotword):
            self._continuous_listening = True
            logger.info("Continuous listening started")
            return True

        return False

    def stop_continuous_listening(self) -> None:
        """Stop continuous listening"""
        if self.hotword:
            self.hotword.stop_listening()

        self._continuous_listening = False
        logger.info("Continuous listening stopped")

    async def _handle_voice_command_mode(self) -> None:
        """Handle active voice command mode after hot word"""
        try:
            self.speak_immediate("Yes?")

            await asyncio.sleep(0.5)

            user_input = await self._listen_for_command()

            if user_input:
                if self._intent_handler:
                    response = await self._intent_handler(user_input)
                else:
                    response = "Voice received but no intent handler."

                self.speak(response)

        except Exception as e:
            logger.error(f"Voice command mode error: {e}")
            self.speak("Sorry, I encountered an error.")

    async def _listen_for_command(self, timeout: float = 5.0) -> str:
        """Listen for voice command"""
        try:
            audio_path = await self._capture_audio(timeout)
            if not audio_path:
                logger.warning("No audio captured")
                return ""
            return await self.stt.transcribe(audio_path, language="en")
        except Exception as e:
            logger.error(f"Listen error: {e}")
            return ""

    async def _capture_audio(self, timeout: float) -> Optional[str]:
        """Capture audio from microphone"""
        import tempfile
        import subprocess

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        try:
            result = subprocess.run(
                [
                    "arecord",
                    "-f",
                    "cd",
                    "-t",
                    "wav",
                    "-d",
                    str(int(timeout)),
                    temp_path,
                ],
                capture_output=True,
                timeout=timeout + 5,
            )
            if result.returncode == 0 and Path(temp_path).stat().st_size > 0:
                return temp_path
            return None
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
            if Path(temp_path).exists():
                Path(temp_path).unlink(missing_ok=True)
            return None

    def _log_interaction(self, input_text: str, response: str) -> None:
        """Log voice interaction for learning"""
        log_file = Path(self.config.cache_dir) / "voice_interactions.jsonl"

        entry = {
            "timestamp": datetime.now().isoformat(),
            "input": input_text,
            "output": response,
        }

        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Logging error: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            "initialized": self._initialized,
            "stt_loaded": self.stt.engine.is_loaded,
            "continuous_listening": self._continuous_listening,
            "tts_queue_size": self.tts_queue.get_queue_size(),
            "hotword_active": self.hotword is not None and self._continuous_listening,
        }

    def clear_cache(self) -> None:
        """Clear all caches"""
        self.stt.clear_cache()

    async def shutdown(self) -> None:
        """Clean shutdown of voice pipeline"""
        logger.info("Shutting down voice pipeline...")

        if self._continuous_listening:
            self.stop_continuous_listening()

        self.tts_queue.stop()

        if self.stt.engine:
            self.stt.engine.unload_model()

        self._initialized = False
        logger.info("Voice pipeline shutdown complete")


class TelegramVoiceHandler:
    """Handle voice messages from Telegram"""

    def __init__(self, pipeline: VoicePipeline):
        self.pipeline = pipeline
        self._download_dir = Path("/data/data/com.termux/files/home/.cache/aura/voice")
        self._download_dir.mkdir(parents=True, exist_ok=True)

    async def handle_voice_message(self, message) -> str:
        """
        Process incoming voice message

        Args:
            message: Telegram message object with voice

        Returns:
            Response text
        """
        voice_file = await self._download_voice(message)

        if not voice_file:
            return "Failed to download voice message."

        response = await self.pipeline.process_telegram_voice(voice_file)

        Path(voice_file).unlink(missing_ok=True)

        return response

    async def _download_voice(self, message) -> Optional[str]:
        """Download voice message to local file"""
        try:
            bot = message.bot
            file = await bot.get_file(message.voice.file_id)

            temp_file = self._download_dir / f"voice_{message.message_id}.ogg"

            await file.download_to_drive(str(temp_file))

            return str(temp_file)

        except Exception as e:
            logger.error(f"Voice download error: {e}")
            return None


__all__ = ["VoicePipelineConfig", "VoicePipeline", "TelegramVoiceHandler"]
