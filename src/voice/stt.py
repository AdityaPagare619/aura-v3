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

    VOSK = "vosk"  # Lightweight, mobile-friendly
    WHISPER = "whisper"
    WHISPER_CPP = "whisper_cpp"
    COQUI = "coqui"
    FASTER_WHISPER = "faster_whisper"
    AZURE = "azure"
    GOOGLE = "google"
    MOCK = "mock"  # Fallback when no backend available


@dataclass
class STTConfig:
    """STT Configuration"""

    backend: STTBackend = STTBackend.FASTER_WHISPER  # Changed by get_default_backend() at runtime
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

# Platform detection and backend selection
# =============================================================================

def detect_platform() -> str:
    """Detect the current platform"""
    import os
    import platform
    if "ANDROID_ROOT" in os.environ or os.path.exists("/data/data/com.termux"):
        return "termux"
    elif platform.system() == "Windows":
        return "windows"
    elif platform.system() == "Linux":
        return "linux"
    elif platform.system() == "Darwin":
        return "macos"
    return "unknown"


def get_default_backend() -> STTBackend:
    """Get the best available STT backend for current platform"""
    # Try Vosk first (lightweight, works everywhere)
    try:
        import vosk
        return STTBackend.VOSK
    except ImportError:
        pass
    
    # Try Faster Whisper
    try:
        import faster_whisper
        return STTBackend.FASTER_WHISPER
    except ImportError:
        pass
    
    # Try Whisper.cpp
    try:
        import whispercpp
        return STTBackend.WHISPER_CPP
    except ImportError:
        pass
    
    # Fallback to mock
    return STTBackend.MOCK





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


class VoskEngine(STTEngineBase):
    """Vosk STT engine - lightweight, mobile-friendly, offline"""
    
    VOSK_MODELS = {
        "tiny": "vosk-model-tiny-en-us-0.15",
        "small": "vosk-model-small-en-us-0.15",
        "medium": "vosk-model-en-us-0.15",
    }

    def __init__(self):
        self.model = None
        self.config = None
        self._lock = threading.Lock()
        
    async def load_model(self, config: STTConfig) -> bool:
        """Load Vosk model"""
        self.config = config
        
        try:
            import vosk
            
            model_name = config.model_name or "small"
            model_id = self.VOSK_MODELS.get(model_name, self.VOSK_MODELS["small"])
            
            model_path = config.model_path
            if not model_path:
                possible_paths = [
                    f"models/{model_id}",
                    f"models/vosk/{model_id}",
                    str(Path.home() / ".cache" / "aura" / "vosk" / model_id),
                    "/data/data/com.termux/files/home/.cache/aura/vosk/" + model_id,
                ]
                for path in possible_paths:
                    if Path(path).exists():
                        model_path = path
                        break
                        
            if not model_path or not Path(model_path).exists():
                logger.warning(f"Vosk model not found at: {model_path}")
                logger.info("Download models from: https://alphacephei.com/vosk/models")
                return False
                
            logger.info(f"Loading Vosk model from: {model_path}")
            
            def _load():
                return vosk.Model(model_path)
            
            self.model = await asyncio.to_thread(_load)
            logger.info("Vosk model loaded successfully")
            return True
            
        except ImportError:
            logger.error("vosk not installed: pip install vosk")
            return False
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            return False
    
    async def transcribe(
        self, audio_data: bytes, language: Optional[str] = None, partial: bool = False
    ) -> STTResult:
        """Transcribe audio data using Vosk"""
        if not self.model:
            return STTResult(text="", partial=False, backend=STTBackend.VOSK)
            
        language = language or self.config.language
        
        try:
            import vosk
            import json
            start_time = time.perf_counter()
            
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            def _transcribe():
                rec = vosk.KaldiRecognizer(self.model, self.config.sample_rate)
                chunk_size = 4000
                for i in range(0, len(audio_np), chunk_size):
                    chunk = audio_np[i:i+chunk_size]
                    rec.AcceptWaveform(chunk.tobytes())
                return rec.FinalResult()
            
            result_json = await asyncio.to_thread(_transcribe)
            result_dict = json.loads(result_json)
            
            text = result_dict.get("text", "")
            confidence = result_dict.get("confidence", 0.0)
            processing_time = time.perf_counter() - start_time
            
            return STTResult(
                text=text.strip(),
                partial=partial,
                confidence=confidence,
                language=language,
                processing_time=processing_time,
                backend=STTBackend.VOSK,
            )
            
        except Exception as e:
            logger.error(f"Vosk transcription error: {e}")
            return STTResult(text="", partial=partial, backend=STTBackend.VOSK)
    
    async def transcribe_stream(self, audio_chunk: bytes) -> AsyncIterator[STTResult]:
        if not self.model:
            return
        yield STTResult(text="", partial=True, backend=STTBackend.VOSK)
    
    def is_loaded(self) -> bool:
        return self.model is not None
    
    def unload(self) -> None:
        with self._lock:
            self.model = None
            import gc
            gc.collect()


class MockSTTEngine(STTEngineBase):
    """Mock STT engine for fallback"""
    
    def __init__(self):
        self._loaded = False
        
    async def load_model(self, config: STTConfig) -> bool:
        self._loaded = True
        logger.info("Mock STT engine loaded (no real backend available)")
        return True
        
    async def transcribe(
        self, audio_data: bytes, language: Optional[str] = None, partial: bool = False
    ) -> STTResult:
        return STTResult(
            text="[STT not available - install vosk or faster-whisper]",
            partial=partial,
            backend=STTBackend.MOCK
        )
        
    async def transcribe_stream(self, audio_chunk: bytes) -> AsyncIterator[STTResult]:
        yield STTResult(text="", partial=True, backend=STTBackend.MOCK)
        
    def is_loaded(self) -> bool:
        return self._loaded
        
    def unload(self) -> None:
        self._loaded = False


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
        # Apply default backend if not specified
        if config.backend == STTBackend.FASTER_WHISPER:
            config.backend = get_default_backend()
        self.config = config
        self.engine = self._create_engine()
        self._audio_buffer: List[bytes] = []
        self._buffer_lock = threading.Lock()
        self._is_processing = False
        self._last_partial_text = ""
        self._callbacks: List[Callable[[STTResult], None]] = []

    def _create_engine(self) -> STTEngineBase:
        """Create STT engine based on config"""
        if self.config.backend == STTBackend.VOSK:
            return VoskEngine()
        elif self.config.backend == STTBackend.FASTER_WHISPER:
            return FasterWhisperEngine()
        elif self.config.backend == STTBackend.WHISPER_CPP:
            return WhisperCppEngine()
        elif self.config.backend == STTBackend.COQUI:
            return CoquiEngine()
        elif self.config.backend == STTBackend.MOCK:
            return MockSTTEngine()
        else:
            # Auto-detect best available backend
            default = get_default_backend()
            logger.info(f"Backend {self.config.backend} not available, using {default}")
            self.config.backend = default
            return self._create_engine()

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
    "VoskEngine",
    "MockSTTEngine",
    "create_stt_processor",
    "estimate_stt_latency",
    "get_default_backend",
    "detect_platform",
]
