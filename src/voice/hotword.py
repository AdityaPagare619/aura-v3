"""
AURA Voice System - Hotword Detection Module
Wake word detection and Voice Activity Detection (VAD)
"""

import asyncio
import logging
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class HotwordBackend(Enum):
    """Hotword detection backend types"""

    PORCUPINE = "porcupine"
    SNIPS = "snips"
    POCKETSPHINX = "pocketsphinx"
    WEbrtc_VAD = "webrtc_vad"
    SILERO = "silero_vad"
    CUSTOM = "custom"


@dataclass
class HotWordConfig:
    """Hotword Detection Configuration"""

    backend: HotwordBackend = HotwordBackend.PORCUPINE
    keywords: List[str] = field(default_factory=lambda: ["hey aura", "aura"])
    sensitivity: float = 0.5
    model_path: Optional[str] = None
    library_path: Optional[str] = None
    sample_rate: int = 16000
    buffer_duration: float = 0.5
    min_keyword_duration: float = 1.0
    enable_vad: bool = True
    vad_aggressiveness: int = 2  # 0-3
    vad_padding_duration: float = 0.5
    silence_threshold: float = 0.01
    energy_threshold: float = 0.02


@dataclass
class VADResult:
    """Voice Activity Detection Result"""

    is_speaking: bool
    confidence: float = 0.0
    audio_level: float = 0.0
    timestamp: float = 0.0
    speech_start: Optional[float] = None
    speech_end: Optional[float] = None


@dataclass
class HotwordResult:
    """Hotword Detection Result"""

    keyword: str
    confidence: float = 0.0
    timestamp: float = 0.0
    audio_start: float = 0.0
    audio_end: float = 0.0


class VADEngineBase(ABC):
    """Base class for VAD engines"""

    @abstractmethod
    async def load(self, config: HotWordConfig) -> bool:
        """Load VAD model"""
        pass

    @abstractmethod
    async def detect(self, audio_chunk: bytes) -> VADResult:
        """Detect voice activity in audio chunk"""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if loaded"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset VAD state"""
        pass


class WebRTCVADEngine(VADEngineBase):
    """WebRTC VAD - fast, lightweight voice activity detection"""

    def __init__(self):
        self.vad = None
        self.config = None
        self._is_speaking = False
        self._speech_start_time: Optional[float] = None
        self._speech_end_time: Optional[float] = None

    async def load(self, config: HotWordConfig) -> bool:
        """Load WebRTC VAD"""
        try:
            import webrtcvad

            self.vad = webrtcvad.Vad(config.vad_aggressiveness)
            self.config = config
            logger.info(
                f"WebRTC VAD loaded (aggressiveness: {config.vad_aggressiveness})"
            )
            return True

        except ImportError:
            logger.error("webrtcvad not installed: pip install webrtcvad")
            return False
        except Exception as e:
            logger.error(f"Failed to load WebRTC VAD: {e}")
            return False

    async def detect(self, audio_chunk: bytes) -> VADResult:
        """Detect voice activity"""
        if not self.vad:
            return VADResult(is_speaking=False)

        try:
            # WebRTC VAD expects 16-bit PCM
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)

            # Calculate energy
            energy = np.abs(audio_np).mean() / 32768.0

            # Check if is speech
            is_speech = self.vad.is_speech(
                audio_chunk, sample_rate=self.config.sample_rate
            )

            timestamp = time.time()

            # Track speech boundaries
            if is_speech and not self._is_speaking:
                self._is_speaking = True
                self._speech_start_time = timestamp
                self._speech_end_time = None
            elif not is_speech and self._is_speaking:
                # Check for end of speech
                if self._speech_end_time is None:
                    self._speech_end_time = timestamp

            return VADResult(
                is_speaking=is_speech,
                confidence=1.0 if is_speech else 0.0,
                audio_level=energy,
                timestamp=timestamp,
                speech_start=self._speech_start_time,
                speech_end=self._speech_end_time,
            )

        except Exception as e:
            logger.error(f"VAD detection error: {e}")
            return VADResult(is_speaking=False)

    def is_loaded(self) -> bool:
        return self.vad is not None

    def reset(self) -> None:
        self._is_speaking = False
        self._speech_start_time = None
        self._speech_end_time = None


class SileroVADEngine(VADEngineBase):
    """Silero VAD - high accuracy voice activity detection"""

    def __init__(self):
        self.model = None
        self.config = None
        self._is_speaking = False
        self._speech_start_time: Optional[float] = None
        self._speech_end_time: Optional[float] = None

    async def load(self, config: HotWordConfig) -> bool:
        """Load Silero VAD"""
        try:
            import torch
            from SileroVAD import load_silero_vad

            logger.info("Loading Silero VAD...")

            def _load():
                return load_silero_vad()

            self.model = await asyncio.to_thread(_load)
            self.config = config
            logger.info("Silero VAD loaded successfully")
            return True

        except ImportError:
            logger.warning("Silero VAD not available, falling back to WebRTC")
            return False
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            return False

    async def detect(self, audio_chunk: bytes) -> VADResult:
        """Detect voice activity using Silero"""
        if not self.model:
            return VADResult(is_speaking=False)

        try:
            import torch

            # Convert to tensor
            audio_np = np.frombuffer(audio_chunk, dtype=np.float32)
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

            # Get speech probability
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.config.sample_rate).item()

            is_speech = speech_prob > 0.5
            timestamp = time.time()

            # Track speech boundaries
            if is_speech and not self._is_speaking:
                self._is_speaking = True
                self._speech_start_time = timestamp
            elif not is_speech and self._is_speaking:
                self._speech_end_time = timestamp

            return VADResult(
                is_speaking=is_speech,
                confidence=speech_prob,
                audio_level=np.abs(audio_np).mean(),
                timestamp=timestamp,
                speech_start=self._speech_start_time,
                speech_end=self._speech_end_time,
            )

        except Exception as e:
            logger.error(f"Silero VAD error: {e}")
            return VADResult(is_speaking=False)

    def is_loaded(self) -> bool:
        return self.model is not None

    def reset(self) -> None:
        self._is_speaking = False
        self._speech_start_time = None
        self._speech_end_time = None


class EnergyBasedVAD(VADEngineBase):
    """Simple energy-based VAD - fallback option"""

    def __init__(self):
        self.config = None
        self._is_speaking = False
        self._speech_start_time: Optional[float] = None
        self._speech_end_time: Optional[float] = None
        self._silence_count = 0

    async def load(self, config: HotWordConfig) -> bool:
        """Initialize energy-based VAD"""
        self.config = config
        logger.info("Energy-based VAD initialized")
        return True

    async def detect(self, audio_chunk: bytes) -> VADResult:
        """Detect voice activity using energy threshold"""
        try:
            # Convert to float
            audio_np = (
                np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            )

            # Calculate RMS energy
            energy = np.sqrt(np.mean(audio_np**2))

            is_speech = energy > self.config.energy_threshold
            timestamp = time.time()

            # Track speech boundaries
            if is_speech:
                self._silence_count = 0
                if not self._is_speaking:
                    self._is_speaking = True
                    self._speech_start_time = timestamp
            else:
                self._silence_count += 1
                if self._is_speaking and self._silence_count > 3:
                    self._is_speaking = False
                    self._speech_end_time = timestamp

            return VADResult(
                is_speaking=is_speaking,
                confidence=min(energy / self.config.energy_threshold, 1.0),
                audio_level=energy,
                timestamp=timestamp,
                speech_start=self._speech_start_time,
                speech_end=self._speech_end_time,
            )

        except Exception as e:
            logger.error(f"Energy VAD error: {e}")
            return VADResult(is_speaking=False)

    def is_loaded(self) -> bool:
        return True

    def reset(self) -> None:
        self._is_speaking = False
        self._silence_count = 0
        self._speech_start_time = None
        self._speech_end_time = None


class HotWordDetectorBase(ABC):
    """Base class for hotword detection"""

    @abstractmethod
    async def load_model(self, config: HotWordConfig) -> bool:
        """Load hotword model"""
        pass

    @abstractmethod
    async def detect(self, audio_chunk: bytes) -> Optional[HotwordResult]:
        """Detect hotword in audio chunk"""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if loaded"""
        pass


class PorcupineHotwordEngine(HotWordDetectorBase):
    """Picovoice Porcupine - wake word detection"""

    def __init__(self):
        self.porcupine = None
        self.config = None
        self._keyword_index = {}

    async def load_model(self, config: HotWordConfig) -> bool:
        """Load Porcupine"""
        try:
            import pvporcupine

            # Create keyword indexes
            for i, keyword in enumerate(config.keywords):
                self._keyword_index[keyword] = i

            # Build keyword paths
            keyword_paths = []
            if config.model_path:
                keyword_paths.append(config.model_path)

            self.porcupine = pvporcupine.create(
                keywords=config.keywords,
                sensitivities=[config.sensitivity] * len(config.keywords),
                keyword_paths=keyword_paths if keyword_paths else None,
            )

            self.config = config
            logger.info(f"Porcupine loaded: {config.keywords}")
            return True

        except ImportError:
            logger.error("pvporcupine not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load Porcupine: {e}")
            return False

    async def detect(self, audio_chunk: bytes) -> Optional[HotwordResult]:
        """Detect hotword"""
        if not self.porcupine:
            return None

        try:
            # Convert to 16-bit PCM
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)

            # Detect
            keyword_index = self.porcupine.process(audio_np)

            if keyword_index >= 0:
                keyword = self.config.keywords[keyword_index]
                return HotwordResult(
                    keyword=keyword,
                    confidence=1.0,
                    timestamp=time.time(),
                )

        except Exception as e:
            logger.error(f"Hotword detection error: {e}")

        return None

    def is_loaded(self) -> bool:
        return self.porcupine is not None


class CustomHotwordEngine(HotWordDetectorBase):
    """Custom hotword detection using template matching"""

    def __init__(self):
        self.config = None
        self._templates: Dict[str, np.ndarray] = {}

    async def load_model(self, config: HotWordConfig) -> bool:
        """Initialize custom hotword detector"""
        self.config = config
        logger.info(f"Custom hotword detector initialized: {config.keywords}")
        return True

    async def detect(self, audio_chunk: bytes) -> Optional[HotwordResult]:
        """Detect hotword using simple energy pattern matching"""
        # Simple implementation - detect specific energy patterns
        # In production, would use proper keyword spotting model

        audio_np = (
            np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        )
        energy = np.sqrt(np.mean(audio_np**2))

        # Simple threshold-based detection (placeholder)
        if energy > 0.1:
            return HotwordResult(
                keyword=self.config.keywords[0] if self.config.keywords else "unknown",
                confidence=0.5,
                timestamp=time.time(),
            )

        return None

    def is_loaded(self) -> bool:
        return True


class HotWordDetector:
    """
    Complete hotword detection system
    Combines VAD and wake word detection
    """

    def __init__(self, config: HotWordConfig):
        self.config = config
        self.vad = self._create_vad()
        self.hotword = self._create_hotword()
        self._is_listening = False
        self._listening_thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._callbacks: List[Callable[[], None]] = []
        self._vad_callbacks: List[Callable[[VADResult], None]] = []

    def _create_vad(self) -> VADEngineBase:
        """Create VAD engine"""
        return WebRTCVADEngine()  # Default

    def _create_hotword(self) -> HotWordDetectorBase:
        """Create hotword engine"""
        if self.config.backend == HotwordBackend.PORCUPINE:
            return PorcupineHotwordEngine()
        else:
            return CustomHotwordEngine()

    async def load_model(self) -> bool:
        """Load models"""
        vad_loaded = await self.vad.load(self.config)

        # Try to load hotword, but continue without if fails
        hotword_loaded = await self.hotword.load_model(self.config)

        if not vad_loaded:
            logger.warning("VAD not loaded, using fallback")
            fallback_vad = EnergyBasedVAD()
            await fallback_vad.load(self.config)
            self.vad = fallback_vad

        return vad_loaded or True

    def add_hotword_callback(self, callback: Callable[[], None]) -> None:
        """Add callback for hotword detection"""
        self._callbacks.append(callback)

    def remove_hotword_callback(self, callback: Callable[[], None]) -> None:
        """Remove hotword callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def add_vad_callback(self, callback: Callable[[VADResult], None]) -> None:
        """Add callback for VAD results"""
        self._vad_callbacks.append(callback)

    def remove_vad_callback(self, callback: Callable[[VADResult], None]) -> None:
        """Remove VAD callback"""
        if callback in self._vad_callbacks:
            self._vad_callbacks.remove(callback)

    def start_listening(
        self,
        on_hotword: Optional[Callable[[], None]] = None,
        audio_source: Optional[Callable[[], bytes]] = None,
    ) -> bool:
        """Start continuous listening"""
        if self._is_listening:
            logger.warning("Already listening")
            return False

        if on_hotword:
            self._callbacks.append(on_hotword)

        self._is_listening = True

        # Start listening thread
        self._listening_thread = threading.Thread(
            target=self._listening_loop, args=(audio_source,), daemon=True
        )
        self._listening_thread.start()

        logger.info("Hotword listening started")
        return True

    def _listening_loop(self, audio_source: Optional[Callable[[], bytes]]) -> None:
        """Main listening loop"""
        import pyaudio

        audio = None
        stream = None

        try:
            audio = pyaudio.PyAudio()

            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=int(
                    self.config.sample_rate * self.config.buffer_duration
                ),
            )

            logger.info("Audio stream opened for listening")

            while self._is_listening:
                try:
                    # Read audio chunk
                    chunk = stream.read(
                        int(self.config.sample_rate * self.config.buffer_duration),
                        exception_on_overflow=False,
                    )

                    # Process through VAD
                    if self.config.enable_vad:
                        vad_result = asyncio.run(self.vad.detect(chunk))

                        # Notify VAD callbacks
                        for callback in self._vad_callbacks:
                            try:
                                callback(vad_result)
                            except Exception as e:
                                logger.error(f"VAD callback error: {e}")

                    # Process through hotword detector
                    result = asyncio.run(self.hotword.detect(chunk))

                    if result:
                        logger.info(f"Hotword detected: {result.keyword}")

                        # Notify callbacks
                        for callback in self._callbacks:
                            try:
                                callback()
                            except Exception as e:
                                logger.error(f"Hotword callback error: {e}")

                    # Also allow custom audio source
                    if audio_source:
                        chunk = audio_source()
                        if chunk:
                            self._audio_queue.put(chunk)

                except Exception as e:
                    logger.error(f"Listening loop error: {e}")
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Listening setup error: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            if audio:
                audio.terminate()

    def stop_listening(self) -> None:
        """Stop listening"""
        self._is_listening = False

        if self._listening_thread:
            self._listening_thread.join(timeout=2.0)

        logger.info("Hotword listening stopped")

    def is_listening(self) -> bool:
        """Check if listening"""
        return self._is_listening

    def get_audio_level(self, audio_chunk: bytes) -> float:
        """Get current audio level"""
        audio_np = (
            np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        )
        return np.sqrt(np.mean(audio_np**2))


class EmergencyHotwordDetector:
    """Special detector for emergency hotwords"""

    EMERGENCY_KEYWORDS = {
        "help",
        "emergency",
        "danger",
        "help me",
        "call 911",
        "call emergency",
        "stop",
        "don't touch",
        "leave me alone",
    }

    def __init__(self):
        self._is_enabled = False

    async def load(self) -> bool:
        """Load emergency detector"""
        self._is_enabled = True
        logger.info("Emergency hotword detector loaded")
        return True

    async def detect_emergency(self, text: str) -> bool:
        """
        Detect emergency keywords in transcribed text
        Returns True if emergency detected
        """
        if not self._is_enabled:
            return False

        text_lower = text.lower()

        for keyword in self.EMERGENCY_KEYWORDS:
            if keyword in text_lower:
                logger.warning(f"Emergency keyword detected: {keyword}")
                return True

        return False

    def add_emergency_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for emergency detection"""
        self._emergency_callback = callback

    async def check_and_notify(self, text: str) -> None:
        """Check for emergency and notify"""
        if await self.detect_emergency(text):
            if hasattr(self, "_emergency_callback"):
                self._emergency_callback(text)


def create_vad_engine(config: HotWordConfig) -> VADEngineBase:
    """Factory for VAD engine"""
    return WebRTCVADEngine()


def create_hotword_detector(config: HotWordConfig) -> HotWordDetector:
    """Factory for hotword detector"""
    return HotWordDetector(config)


__all__ = [
    "HotWordConfig",
    "HotwordResult",
    "VADResult",
    "VADEngineBase",
    "HotWordDetector",
    "EmergencyHotwordDetector",
    "HotwordBackend",
    "create_vad_engine",
    "create_hotword_detector",
]
