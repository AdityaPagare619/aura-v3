"""
AURA Channels Module

Provides communication channels for AURA (Voice, Telegram, API).
"""

from .voice import VoiceChannel, TTSEngine, EspeakTTS, Pyttsx3TTS

__all__ = [
    "VoiceChannel",
    "TTSEngine",
    "EspeakTTS",
    "Pyttsx3TTS",
]
