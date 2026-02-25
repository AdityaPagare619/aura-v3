"""
AURA Voice Channel - Multiple TTS Engine Support
User chooses: eSpeak-ng, pyttsx3, SherpaTTS, or API-based (optional)
"""

import asyncio
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class TTSEngine(Enum):
    """Available TTS engines"""

    ESPEAK = "espeak"  # Free, in Termux
    PYTTSX3 = "pyttsx3"  # Free, offline
    COQUI = "coqui"  # Free, neural
    GOOGLE_TTS = "google"  # Requires internet
    EDGE_TTS = "edge"  # Requires internet
    SARVAM = "sarvam"  # Requires API key
    MOCK = "mock"  # For testing


@dataclass
class VoiceProfile:
    voice_id: str
    name: str
    language: str
    gender: str = "neutral"


class BaseTTS(ABC):
    """Base class for TTS engines"""

    @abstractmethod
    async def synthesize(self, text: str, voice_id: str = None) -> bytes:
        """Convert text to audio bytes"""
        pass

    @abstractmethod
    async def get_available_voices(self) -> List[VoiceProfile]:
        """List available voices"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this TTS engine is available"""
        pass


class EspeakTTS(BaseTTS):
    """
    eSpeak-ng TTS - Available in Termux
    Install: apt install espeak-ng
    """

    LANGUAGES = {
        "en": "english",
        "hi": "hindi",
        "bn": "bengali",
        "ta": "tamil",
        "te": "telugu",
        "mr": "marathi",
        "gu": "gujarati",
        "kn": "kannada",
        "ml": "malayalam",
        "pa": "punjabi",
    }

    async def synthesize(self, text: str, voice_id: str = None) -> bytes:
        """Generate speech using eSpeak-ng"""

        lang = "english"
        if voice_id and voice_id in self.LANGUAGES:
            lang = self.LANGUAGES[voice_id]

        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".wav", delete=False
            ) as f:
                temp_wav = f.name

            # Run espeak
            cmd = ["espeak-ng", "-w", temp_wav, "-v", lang, text]
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            # Read audio
            with open(temp_wav, "rb") as f:
                audio = f.read()

            os.unlink(temp_wav)
            return audio

        except FileNotFoundError:
            logger.warning("espeak-ng not installed")
            return b""
        except Exception as e:
            logger.error(f"espeak error: {e}")
            return b""

    async def get_available_voices(self) -> List[VoiceProfile]:
        return [
            VoiceProfile("en", "English", "en", "neutral"),
            VoiceProfile("hi", "Hindi", "hi", "neutral"),
        ]

    def is_available(self) -> bool:
        return os.path.exists("/usr/bin/espeak-ng") or os.path.exists(
            "/data/data/com.termux/files/usr/bin/espeak-ng"
        )


class Pyttsx3TTS(BaseTTS):
    """
    pyttsx3 - Offline TTS for Python
    Install: pip install pyttsx3
    Works on Android/Termux
    """

    def __init__(self):
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        try:
            import pyttsx3

            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)
            self.engine.setProperty("volume", 1.0)
        except Exception as e:
            logger.warning(f"pyttsx3 not available: {e}")

    async def synthesize(self, text: str, voice_id: str = None) -> bytes:
        if self.engine is None:
            return b""

        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".wav", delete=False
            ) as f:
                temp_wav = f.name

            self.engine.save_to_file(text, temp_wav)
            self.engine.runAndWait()

            with open(temp_wav, "rb") as f:
                audio = f.read()

            os.unlink(temp_wav)
            return audio
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
            return b""

    async def get_available_voices(self) -> List[VoiceProfile]:
        if self.engine is None:
            return []

        voices = []
        for voice in self.engine.getProperty("voices"):
            voices.append(
                VoiceProfile(
                    voice.id,
                    voice.name,
                    voice.languages[0] if voice.languages else "en",
                    "neutral",
                )
            )
        return voices

    def is_available(self) -> bool:
        return self.engine is not None


class VoiceChannel:
    """
    Main voice channel - user chooses TTS engine
    """

    def __init__(self, engine: str = "espeak", config: Dict = None):
        self.engine_name = engine
        self.config = config or {}
        self.tts = self._init_tts(engine)
        self.currently_speaking = False

    def _init_tts(self, engine: str) -> BaseTTS:
        """Initialize TTS engine based on user choice"""

        # Try each engine in order of preference
        if engine == "espeak":
            tts = EspeakTTS()
            if tts.is_available():
                return tts
            logger.info("eSpeak not available, trying pyttsx3")

        if engine in ["espeak", "pyttsx3", "auto"]:
            tts = Pyttsx3TTS()
            if tts.is_available():
                return tts
            logger.info("pyttsx3 not available")

        # Fallback to mock
        return MockTTS()

    async def speak(self, text: str, voice_id: str = None):
        """Convert text to speech and play"""

        if not text:
            return

        self.currently_speaking = True

        try:
            # Generate audio
            audio = await self.tts.synthesize(text, voice_id)

            if audio:
                # Play audio (platform-specific)
                await self._play_audio(audio)
            else:
                logger.warning("No audio generated")

        finally:
            self.currently_speaking = False

    async def _play_audio(self, audio: bytes):
        """Play audio bytes - platform specific"""

        # Try different players based on platform
        players = ["aplay", "paplay", "ffplay", "play"]

        for player in players:
            try:
                process = await asyncio.create_subprocess_exec(
                    player,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await process.communicate(input=audio)
                return
            except FileNotFoundError:
                continue

        logger.warning("No audio player found")

    async def listen(self, timeout: int = 10) -> Optional[str]:
        """Voice input - requires STT (Speech-to-Text)"""
        # Simplified: Return None (no voice input implemented)
        # Could integrate with: whisper.cpp, Vosk, or online APIs
        return None

    def is_speaking(self) -> bool:
        return self.currently_speaking

    async def stop_speaking(self):
        """Stop current speech"""
        self.currently_speaking = False


class MockTTS(BaseTTS):
    """Mock TTS for testing"""

    async def synthesize(self, text: str, voice_id: str = None) -> bytes:
        return b""

    async def get_available_voices(self) -> List[VoiceProfile]:
        return [VoiceProfile("mock", "Mock Voice", "en")]

    def is_available(self) -> bool:
        return True


# Export
__all__ = ["VoiceChannel", "TTSEngine", "VoiceProfile", "EspeakTTS", "Pyttsx3TTS"]
