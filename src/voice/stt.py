"""
AURA Voice System - STT (Speech-to-Text) Module
Streaming-capable speech recognition with partial results
"""

import asyncio
import logging
import os
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, AsyncIterator
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class STTBackend(Enum):
    """STT backend types"""

    WHISPER = "whisper"
    WHISPER_CPP = "whisper_cpp"
    COQUI = "coqui"
    FASTER_WHISPER = "faster_whisper"
    AZURE = "azure"
    GOOGLE = "google"


@dataclass
class STTConfig:
    """STT Configuration"""

    backend: STTBackend = STTBackend.FASTER_WHISPER
    model_name: str = "base"
    model_path: Optional[str] = None
    language: str = "en"
    sample_rate: int = 16000
    buffer_duration: float = 0.5  # seconds
    min_speech_duration: float = 0.3
    max_speech_duration: float = 30.0
    use_gpu: bool = True
    beam_size: int = 5
    vad_filter: bool = True
    compute_type: str = "float16"
    streaming: bool = True
    enable_partial_results: bool = True
    device: str = "auto"


@dataclass
class STTResult:
    """STT Recognition Result"""

    text: str
    partial: bool = False
    confidence: float = 0.0
    language: str = "en"
    audio_duration: float = 0.0
    processing_time: float = 0.0
    timestamps: Optional[Dict[str, float]] = None
    backend: STTBackend = STTBackend.FASTER_WHISPER

    def __post_init__(self):
        if self.timestamps is None:
            self.timestamps = {}


class STTEngineBase(ABC):
    """Base class for STT engines"""

    @abstractmethod
    async def load_model(self, config: STTConfig) -> bool:
        """Load the STT model"""
        pass

    @abstractmethod
    async def transcribe(
        self, audio_data: bytes, language: Optional[str] = None, partial: bool = False
    ) -> STTResult:
        """Transcribe audio data"""
        pass

    @abstractmethod
    async def transcribe_stream(self, audio_chunk: bytes) -> AsyncIterator[STTResult]:
        """Stream transcribe audio chunks"""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model"""
        pass


class FasterWhisperEngine(STTEngineBase):
    """Faster Whisper STT engine - best for real-time"""

    def __init__(self):
        self.model = None
        self.config = None
        self._lock = threading.Lock()

    async def load_model(self, config: STTConfig) -> bool:
        """Load faster-whisper model"""
        self.config = config

        try:
            from faster_whisper import WhisperModel

            compute_type = config.compute_type
            if not config.use_gpu:
                compute_type = "int8"

            logger.info(f"Loading faster-whisper model: {config.model_name}")

            # Run in thread to avoid blocking
            def _load():
                return WhisperModel(
                    config.model_name,
                    device=config.device,
                    compute_type=compute_type,
                )

            self.model = await asyncio.to_thread(_load)
            logger.info("Faster-whisper model loaded successfully")
            return True

        except ImportError:
            logger.error("faster-whisper not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load faster-whisper: {e}")
            return False

    async def transcribe(
        self, audio_data: bytes, language: Optional[str] = None, partial: bool = False
    ) -> STTResult:
        """Transcribe audio data"""
        if not self.model:
            return STTResult(text="", partial=False, backend=STTBackend.FASTER_WHISPER)

        language = language or self.config.language

        try:
            start_time = time.perf_counter()

            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.float32)

            # Run transcription
            def _transcribe():
                segments, info = self.model.transcribe(
                    audio_np,
                    language=language,
                    beam_size=self.config.beam_size,
                    vad_filter=self.config.vad_filter,
                    word_timestamps=False,
                )

                text = " ".join([seg.text for seg in segments])
                return text, info.language, info.language_probability

            text, lang, confidence = await asyncio.to_thread(_transcribe)
            processing_time = time.perf_counter() - start_time

            return STTResult(
                text=text.strip(),
                partial=partial,
                confidence=confidence,
                language=lang,
                processing_time=processing_time,
                backend=STTBackend.FASTER_WHISPER,
            )

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return STTResult(
                text="", partial=partial, backend=STTBackend.FASTER_WHISPER
            )

    async def transcribe_stream(self, audio_chunk: bytes) -> AsyncIterator[STTResult]:
        """Stream transcribe - yields partial results"""
        if not self.model:
            return

        try:
            audio_np = np.frombuffer(audio_chunk, dtype=np.float32)

            # For streaming, use faster-whisper's segment-based approach
            segments_generator = self.model.streaming(
                language=self.config.language,
            )

            # This would be implemented based on the specific streaming API
            # For now, yield partial results as audio accumulates
            yield STTResult(
                text="",
                partial=True,
                backend=STTBackend.FASTER_WHISPER,
            )

        except Exception as e:
            logger.error(f"Streaming transcription error: {e}")

    def is_loaded(self) -> bool:
        return self.model is not None

    def unload(self) -> None:
        with self._lock:
            self.model = None
            import gc

            gc.collect()


class WhisperCppEngine(STTEngineBase):
    """Whisper.cpp STT engine - lightweight, fast"""

    def __init__(self):
        self.model = None
        self.config = None
        self._lock = threading.Lock()

    async def load_model(self, config: STTConfig) -> bool:
        """Load whisper.cpp model via python bindings"""
        self.config = config

        try:
            from whispercpp import Whisper

            model_path = config.model_path or f"models/ggml-{config.model_name}.bin"
            logger.info(f"Loading whisper.cpp: {model_path}")

            def _load():
                return Whisper.from_pretrained(model_path)

            self.model = await asyncio.to_thread(_load)
            logger.info("Whisper.cpp loaded successfully")
            return True

        except ImportError:
            logger.error("whispercpp not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load whisper.cpp: {e}")
            return False

    async def transcribe(
        self, audio_data: bytes, language: Optional[str] = None, partial: bool = False
    ) -> STTResult:
        """Transcribe audio data"""
        if not self.model:
            return STTResult(text="", partial=False, backend=STTBackend.WHISPER_CPP)

        language = language or self.config.language

        try:
            start_time = time.perf_counter()

            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.float32)

            # Run transcription
            def _transcribe():
                result = self.model.transcribe(
                    audio_np,
                    language=language,
                    reset=True,
                )
                return result["text"]

            text = await asyncio.to_thread(_transcribe)
            processing_time = time.perf_counter() - start_time

            return STTResult(
                text=text.strip(),
                partial=partial,
                processing_time=processing_time,
                backend=STTBackend.WHISPER_CPP,
            )

        except Exception as e:
            logger.error(f"Whisper.cpp transcription error: {e}")
            return STTResult(text="", partial=partial, backend=STTBackend.WHISPER_CPP)

    async def transcribe_stream(self, audio_chunk: bytes) -> AsyncIterator[STTResult]:
        """Stream transcribe"""
        # Similar to above - yields partial results
        if not self.model:
            return

        yield STTResult(
            text="",
            partial=True,
            backend=STTBackend.WHISPER_CPP,
        )

    def is_loaded(self) -> bool:
        return self.model is not None

    def unload(self) -> None:
        with self._lock:
            self.model = None
            import gc

            gc.collect()


class CoquiEngine(STTEngineBase):
    """Coqui STT engine - open source alternative"""

    def __init__(self):
        self.model = None
        self.config = None

    async def load_model(self, config: STTConfig) -> bool:
        """Load Coqui STT model"""
        self.config = config

        try:
            from TTS.tts.configs.bark_config import BarkConfig

            # Coqui STT loading would go here
            logger.info("Coqui STT loading not fully implemented")
            return False

        except ImportError:
            logger.error("Coqui TTS not installed")
            return False

    async def transcribe(
        self, audio_data: bytes, language: Optional[str] = None, partial: bool = False
    ) -> STTResult:
        return STTResult(text="", partial=partial, backend=STTBackend.COQUI)

    async def transcribe_stream(self, audio_chunk: bytes) -> AsyncIterator[STTResult]:
        return

    def is_loaded(self) -> bool:
        return self.model is not None

    def unload(self) -> None:
        self.model = None


class STTProcessor:
    """
    Main STT processor with streaming support
    Handles audio buffer management and provides partial results
    """

    def __init__(self, config: STTConfig):
        self.config = config
        self.engine = self._create_engine()
        self._audio_buffer: List[bytes] = []
        self._buffer_lock = threading.Lock()
        self._is_processing = False
        self._last_partial_text = ""
        self._callbacks: List[Callable[[STTResult], None]] = []

    def _create_engine(self) -> STTEngineBase:
        """Create STT engine based on config"""
        if self.config.backend == STTBackend.FASTER_WHISPER:
            return FasterWhisperEngine()
        elif self.config.backend == STTBackend.WHISPER_CPP:
            return WhisperCppEngine()
        elif self.config.backend == STTBackend.COQUI:
            return CoquiEngine()
        else:
            return FasterWhisperEngine()  # Default

    async def initialize(self) -> bool:
        """Initialize the STT engine"""
        logger.info("Initializing STT processor...")
        return await self.engine.load_model(self.config)

    def add_callback(self, callback: Callable[[STTResult], None]) -> None:
        """Add callback for transcription results"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[STTResult], None]) -> None:
        """Remove callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def process_voice_message(self, audio_path: str) -> Dict[str, Any]:
        """Process a voice message file (legacy API)"""
        try:
            import wave

            # Read audio file
            with wave.open(audio_path, "rb") as wf:
                # Convert to 16kHz mono if needed
                frames = wf.readframes(wf.getnframes())
                audio_data = self._convert_audio(frames, wf.getparams())

            result = await self.engine.transcribe(audio_data, self.config.language)

            return {
                "text": result.text,
                "confidence": result.confidence,
                "language": result.language,
                "processing_time": result.processing_time,
            }

        except Exception as e:
            logger.error(f"Voice message processing error: {e}")
            return {"text": "", "error": str(e)}

    async def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe audio file"""
        result = await self.process_voice_message(audio_path)
        return result.get("text", "")

    def _convert_audio(self, frames: bytes, params) -> bytes:
        """Convert audio to 16kHz mono float32"""
        # This would use scipy or soundfile for proper conversion
        # For now, pass through
        return frames

    async def stream_audio_chunk(self, audio_chunk: bytes) -> Optional[STTResult]:
        """
        Process an audio chunk in streaming mode
        Returns partial result if available
        """
        if not self.engine.is_loaded():
            return None

        with self._buffer_lock:
            self._audio_buffer.append(audio_chunk)

        # Accumulate audio and process periodically
        total_size = sum(len(chunk) for chunk in self._audio_buffer)
        expected_size = int(
            self.config.sample_rate * self.config.buffer_duration * 2
        )  # 16-bit

        if total_size >= expected_size:
            # Combine buffer
            combined_audio = b"".join(self._audio_buffer)

            # Clear buffer for next chunk
            with self._buffer_lock:
                self._audio_buffer = []

            # Transcribe
            if self.config.enable_partial_results:
                result = await self.engine.transcribe(combined_audio, partial=True)

                # Only notify on new text
                if result.text != self._last_partial_text:
                    self._last_partial_text = result.text
                    for callback in self._callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

                    return result

        return None

    async def finalize_stream(self) -> STTResult:
        """Finalize streaming and get final result"""
        if not self.engine.is_loaded():
            return STTResult(text="")

        # Process remaining buffer
        with self._buffer_lock:
            if not self._audio_buffer:
                return STTResult(text="")
            combined_audio = b"".join(self._audio_buffer)
            self._audio_buffer = []

        result = await self.engine.transcribe(combined_audio, partial=False)
        self._last_partial_text = ""

        return result

    def clear_buffer(self) -> None:
        """Clear the audio buffer"""
        with self._buffer_lock:
            self._audio_buffer = []
            self._last_partial_text = ""

    def is_processing(self) -> bool:
        """Check if currently processing"""
        return self._is_processing

    def clear_cache(self) -> None:
        """Clear any cached data"""
        self.clear_buffer()


def create_stt_processor(config: Optional[STTConfig] = None) -> STTProcessor:
    """Factory function to create STT processor"""
    config = config or STTConfig()
    return STTProcessor(config)


# Latency calculations for real-time optimization
def estimate_stt_latency(audio_duration: float, model_size: str = "base") -> float:
    """
    Estimate STT latency based on audio duration and model size
    Returns expected processing time in seconds
    """
    # Rough estimates based on faster-whisper benchmarks
    processing_rates = {
        "tiny": 0.1,  # 10x realtime
        "base": 0.2,  # 5x realtime
        "small": 0.4,  # 2.5x realtime
        "medium": 0.8,  # 1.25x realtime
        "large": 1.5,  # 0.67x realtime
    }

    rate = processing_rates.get(model_size, 0.2)
    return audio_duration * rate


__all__ = [
    "STTConfig",
    "STTResult",
    "STTProcessor",
    "STTEngineBase",
    "STTBackend",
    "create_stt_processor",
    "estimate_stt_latency",
]
