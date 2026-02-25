"""
AURA Voice System - STT (Speech-to-Text) Module
Streaming-capable speech recognition with partial results

Offline-capable backends:
- Vosk: Lightweight, mobile-friendly (recommended for Termux)
- Faster Whisper: High quality, GPU-accelerated
- Whisper.cpp: Medium weight, good quality

Installation:
    # For Vosk (lightweight, mobile-friendly)
    pip install vosk
    
    # For Faster Whisper (high quality)
    pip install faster-whisper
    
    # For Whisper.cpp bindings
    pip install whispercpp
"""

import asyncio
import logging
import os
import platform
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, AsyncIterator
from enum import Enum
import json

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


def detect_platform() -> str:
    """Detect the current platform"""
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
    logger.warning("No STT backend available, using mock")
    return STTBackend.MOCK


@dataclass
class STTConfig:
    """STT Configuration"""

    backend: STTBackend = field(default_factory=get_default_backend)
    model_name: str = "base"
    model_path: Optional[str] = None
    language: str = "en"
    sample_rate: int = 16000
    buffer_duration: float = 0.5  # seconds
    min_speech_duration: float = 0.3
    max_speech_duration: float = 30.0
    use_gpu: bool = False  # Default to CPU for mobile
    beam_size: int = 5
    vad_filter: bool = True
    compute_type: str = "int8"  # Default to int8 for mobile
    streaming: bool = True
    enable_partial_results: bool = True
    device: str = "auto"
    # Vosk-specific
    vocab_path: Optional[str] = None


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
    backend: STTBackend = STTBackend.VOSK

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


class MockSTTEngine(STTEngineBase):
    """Mock STT engine for fallback when no backend available"""
    
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


class VoskEngine(STTEngineBase):
    """Vosk STT engine - lightweight, mobile-friendly, offline"""
    
    # Vosk model sizes (smaller = faster, less accurate)
    VOSK_MODELS = {
        "tiny": "vosk-model-tiny-en-us-0.15",
        "small": "vosk-model-small-en-us-0.15",
        "medium": "vosk-model-en-us-0.15",
        "large": "vosk-model-en-us-lgraph-0.15",
    }

    def __init__(self):
        self.model = None
        self.config = None
        self._recognition_state = None
        self._lock = threading.Lock()
        
    async def load_model(self, config: STTConfig) -> bool:
        """Load Vosk model"""
        self.config = config
        
        try:
            import vosk
            
            # Determine model path
            model_name = config.model_name or "small"
            model_id = self.VOSK_MODELS.get(model_name, self.VOSK_MODELS["small"])
            
            # Check for custom model path
            model_path = config.model_path
            if not model_path:
                # Default model locations
                possible_paths = [
                    f"models/{model_id}",
                    f"models/vosk/{model_id}",
                    str(Path.home() / ".cache" / "aura" / "vosk" / model_id),
                    f"/data/data/com.termux/files/home/.cache/aura/vosk/{model_id}",
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
            start_time = time.perf_counter()
            
            # Convert bytes to proper format if needed
            # Vosk expects 16-bit mono PCM
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Create recognizer
          
