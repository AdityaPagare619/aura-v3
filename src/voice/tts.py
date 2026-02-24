"""
AURA Voice System - TTS (Text-to-Speech) Module
Streaming-capable speech synthesis with low latency
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
from typing import Any, Callable, Dict, List, Optional, AsyncIterator
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class TTSBackend(Enum):
    """TTS backend types"""

    COQUI = "coqui"
    PIPER = "piper"
    EDGE_TTS = "edge_tts"
    GOOGLE_TTS = "google_tts"
    PYTTSX3 = "pyttsx3"
    BARK = "bark"
    SPEECH_T5 = "speech_t5"


@dataclass
class TTSConfig:
    """TTS Configuration"""

    backend: TTSBackend = TTSBackend.COQUI
    model_name: str = "en_US-lessac-medium"
    model_path: Optional[str] = None
    voice_id: str = "en_US-JennyNeural"
    language: str = "en"
    sample_rate: int = 22050
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0
    cache_dir: str = "/data/data/com.termux/files/home/.cache/aura/tts"
    enable_streaming: bool = True
    chunk_size: int = 1024
    use_gpu: bool = False
    quality: str = "medium"  # low, medium, high
    streaming_latency_target: float = 0.3  # target latency for streaming


@dataclass
class TTSResult:
    """TTS Synthesis Result"""

    audio_data: Optional[bytes] = None
    audio_path: Optional[str] = None
    duration: float = 0.0
    text_length: int = 0
    processing_time: float = 0.0
    backend: TTSBackend = TTSBackend.COQUI
    error: Optional[str] = None


class TTSEngineBase(ABC):
    """Base class for TTS engines"""

    @abstractmethod
    async def load_model(self, config: TTSConfig) -> bool:
        """Load the TTS model"""
        pass

    @abstractmethod
    async def speak(
        self,
        text: str,
        blocking: bool = False,
        callback: Optional[Callable[[bytes], None]] = None,
    ) -> TTSResult:
        """Synthesize speech from text"""
        pass

    @abstractmethod
    async def stream_synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream synthesis - yields audio chunks"""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        pass

    @abstractmethod
    def interrupt(self) -> None:
        """Interrupt current speech"""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model"""
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Get sample rate"""
        pass


class CoquiTTSEngine(TTSEngineBase):
    """Coqui TTS engine - high quality, open source"""

    def __init__(self):
        self.model = None
        self.config = None
        self._current_task: Optional[asyncio.Task] = None
        self._is_speaking = False
        self._interrupt_event = asyncio.Event()
        self._lock = threading.Lock()

    async def load_model(self, config: TTSConfig) -> bool:
        """Load Coqui TTS model"""
        self.config = config

        try:
            from TTS.api import TTS

            logger.info(f"Loading Coqui TTS: {config.model_name}")

            def _load():
                return TTS(
                    model_name=config.model_name,
                    gpu=config.use_gpu,
                )

            self.model = await asyncio.to_thread(_load)
            logger.info("Coqui TTS loaded successfully")
            return True

        except ImportError:
            logger.error("TTS not installed: pip install TTS")
            return False
        except Exception as e:
            logger.error(f"Failed to load Coqui TTS: {e}")
            return False

    async def speak(
        self,
        text: str,
        blocking: bool = False,
        callback: Optional[Callable[[bytes], None]] = None,
    ) -> TTSResult:
        """Synthesize speech"""
        if not self.model:
            return TTSResult(error="Model not loaded")

        self._is_speaking = True
        self._interrupt_event.clear()

        try:
            start_time = time.perf_counter()

            # Run synthesis in thread to avoid blocking
            def _synthesize():
                wav = self.model.tts(text)
                audio_data = (wav * 32767).astype(np.int16).tobytes()
                return audio_data

            audio_data = await asyncio.to_thread(_synthesize)
            processing_time = time.perf_counter() - start_time

            # Calculate duration
            duration = len(audio_data) / (self.sample_rate * 2)

            # Play if callback provided
            if callback:
                callback(audio_data)

            return TTSResult(
                audio_data=audio_data,
                duration=duration,
                text_length=len(text),
                processing_time=processing_time,
                backend=TTSBackend.COQUI,
            )

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return TTSResult(error=str(e))
        finally:
            self._is_speaking = False

    async def stream_synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream synthesis - yields chunks as they're generated"""
        if not self.model:
            return

        self._is_speaking = True
        self._interrupt_event.clear()

        try:
            # Coqui supports streaming output
            # For each character or word, yield audio
            def _stream_generator():
                # This would use the streaming API
                # For now, generate full audio and chunk it
                wav = self.model.tts(text)
                audio_data = (wav * 32767).astype(np.int16).tobytes()

                # Yield in chunks
                chunk_size = self.config.chunk_size * 2  # 16-bit
                for i in range(0, len(audio_data), chunk_size):
                    if self._interrupt_event.is_set():
                        break
                    yield audio_data[i : i + chunk_size]

            for chunk in _stream_generator():
                if self._interrupt_event.is_set():
                    break
                yield chunk

        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")
        finally:
            self._is_speaking = False

    def is_loaded(self) -> bool:
        return self.model is not None

    def interrupt(self) -> None:
        """Interrupt current speech"""
        self._interrupt_event.set()
        self._is_speaking = False
        logger.debug("TTS interrupted")

    def unload(self) -> None:
        with self._lock:
            self.model = None
            import gc

            gc.collect()

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate if self.config else 22050


class PiperTTSEngine(TTSEngineBase):
    """Piper TTS engine - fast, low latency, high quality"""

    def __init__(self):
        self.model = None
        self.config = None
        self._process = None
        self._is_speaking = False
        self._interrupt_event = asyncio.Event()

    async def load_model(self, config: TTSConfig) -> bool:
        """Load Piper TTS"""
        self.config = config

        try:
            import subprocess

            model_path = config.model_path or f"models/{config.model_name}.onnx"

            # Check if piper is available
            result = subprocess.run(["which", "piper"], capture_output=True)

            if result.returncode != 0:
                logger.warning("Piper not found in PATH")
                return False

            logger.info(f"Piper TTS configured: {model_path}")
            self.model = True  # Placeholder - actual loading is per-request
            return True

        except Exception as e:
            logger.error(f"Failed to setup Piper: {e}")
            return False

    async def speak(
        self,
        text: str,
        blocking: bool = False,
        callback: Optional[Callable[[bytes], None]] = None,
    ) -> TTSResult:
        """Synthesize speech using Piper"""
        try:
            import subprocess
            import tempfile

            start_time = time.perf_counter()

            # Create temp file for output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name

            # Run piper
            model_path = (
                self.config.model_path or f"models/{self.config.model_name}.onnx"
            )

            process = await asyncio.create_subprocess_exec(
                "piper",
                "--model",
                model_path,
                "--output_file",
                output_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate(input=text.encode())

            processing_time = time.perf_counter() - start_time

            if process.returncode != 0:
                return TTSResult(error=stderr.decode())

            # Read output
            with open(output_path, "rb") as f:
                audio_data = f.read()

            # Clean up
            os.unlink(output_path)

            duration = len(audio_data) / (self.sample_rate * 2)

            if callback:
                callback(audio_data)

            return TTSResult(
                audio_data=audio_data,
                audio_path=output_path,
                duration=duration,
                text_length=len(text),
                processing_time=processing_time,
                backend=TTSBackend.PIPER,
            )

        except Exception as e:
            logger.error(f"Piper TTS error: {e}")
            return TTSResult(error=str(e))

    async def stream_synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream synthesis using piper's stdin/stdout"""
        # Piper supports --stdout for streaming
        try:
            import subprocess

            model_path = (
                self.config.model_path or f"models/{self.config.model_name}.onnx"
            )

            process = await asyncio.create_subprocess_exec(
                "piper",
                "--model",
                model_path,
                "--stdout",
                "--sentence_silence",
                "0.0",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await process.communicate(input=text.encode())

            # Yield chunks
            chunk_size = self.config.chunk_size * 2
            for i in range(0, len(stdout), chunk_size):
                yield stdout[i : i + chunk_size]

        except Exception as e:
            logger.error(f"Piper streaming error: {e}")

    def is_loaded(self) -> bool:
        return self.model is not None

    def interrupt(self) -> None:
        self._interrupt_event.set()
        if self._process:
            self._process.terminate()

    def unload(self) -> None:
        self.model = None
        import gc

        gc.collect()

    @property
    def sample_rate(self) -> int:
        return 22050  # Piper default


class EdgeTTSEngine(TTSEngineBase):
    """Edge TTS - Microsoft Azure Edge TTS, high quality, cloud-based"""

    def __init__(self):
        self.config = None
        self._is_speaking = False
        self._interrupt_event = asyncio.Event()

    async def load_model(self, config: TTSConfig) -> bool:
        """Edge TTS doesn't need model loading"""
        self.config = config
        logger.info("Edge TTS initialized")
        return True

    async def speak(
        self,
        text: str,
        blocking: bool = False,
        callback: Optional[Callable[[bytes], None]] = None,
    ) -> TTSResult:
        """Synthesize speech using Edge TTS"""
        try:
            import edge_tts
            import tempfile

            start_time = time.perf_counter()

            communicate = edge_tts.Communicate(text, self.config.voice_id)

            # Collect audio
            audio_chunks = []

            async def save_audio():
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_chunks.append(chunk["data"])

            await save_audio()

            audio_data = b"".join(audio_chunks)
            processing_time = time.perf_counter() - start_time

            duration = len(audio_data) / (self.sample_rate * 2)

            if callback:
                callback(audio_data)

            return TTSResult(
                audio_data=audio_data,
                duration=duration,
                text_length=len(text),
                processing_time=processing_time,
                backend=TTSBackend.EDGE_TTS,
            )

        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            return TTSResult(error=str(e))

    async def stream_synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream synthesis - yields chunks as they're generated"""
        try:
            import edge_tts

            communicate = edge_tts.Communicate(text, self.config.voice_id)

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        except Exception as e:
            logger.error(f"Edge TTS streaming error: {e}")

    def is_loaded(self) -> bool:
        return True

    def interrupt(self) -> None:
        self._interrupt_event.set()

    def unload(self) -> None:
        pass

    @property
    def sample_rate(self) -> int:
        return 24000  # Edge TTS default


class PyTTSx3Engine(TTSEngineBase):
    """PyTTSx3 - offline, cross-platform TTS"""

    def __init__(self):
        self.engine = None
        self.config = None

    async def load_model(self, config: TTSConfig) -> bool:
        """Initialize PyTTSx3"""
        try:
            import pyttsx3

            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 200 * config.speed)
            self.engine.setProperty("volume", config.volume)

            # Set voice
            voices = self.engine.getProperty("voices")
            if voices:
                self.engine.setProperty("voice", voices[0].id)

            self.config = config
            logger.info("PyTTSx3 initialized")
            return True

        except ImportError:
            logger.error("pyttsx3 not installed")
            return False
        except Exception as e:
            logger.error(f"PyTTSx3 init error: {e}")
            return False

    async def speak(
        self,
        text: str,
        blocking: bool = False,
        callback: Optional[Callable[[bytes], None]] = None,
    ) -> TTSResult:
        """Synthesize speech"""
        if not self.engine:
            return TTSResult(error="Engine not initialized")

        try:
            import tempfile

            start_time = time.perf_counter()

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name

            self.engine.save_to_file(text, output_path)
            self.engine.runAndWait()

            # Read audio
            with open(output_path, "rb") as f:
                audio_data = f.read()

            os.unlink(output_path)

            processing_time = time.perf_counter() - start_time
            duration = len(audio_data) / (self.sample_rate * 2)

            if callback:
                callback(audio_data)

            return TTSResult(
                audio_data=audio_data,
                duration=duration,
                text_length=len(text),
                processing_time=processing_time,
                backend=TTSBackend.PYTTSX3,
            )

        except Exception as e:
            logger.error(f"PyTTSx3 error: {e}")
            return TTSResult(error=str(e))

    async def stream_synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream synthesis - not supported in PyTTSx3"""
        # PyTTSx3 doesn't support streaming, so yield full result
        result = await self.speak(text)
        if result.audio_data:
            yield result.audio_data

    def is_loaded(self) -> bool:
        return self.engine is not None

    def interrupt(self) -> None:
        if self.engine:
            self.engine.stop()

    def unload(self) -> None:
        if self.engine:
            self.engine.stop()
            self.engine = None

    @property
    def sample_rate(self) -> int:
        return 22050  # PyTTSx3 default


class TTSQueueManager:
    """Manages TTS queue for sequential and priority speech"""

    def __init__(self, engine: TTSEngineBase):
        self.engine = engine
        self._queue: queue.Queue = queue.Queue()
        self._is_running = False
        self._current_text = ""
        self._processing_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def enqueue(self, text: str, priority: int = 0) -> None:
        """Add text to queue"""
        self._queue.put((priority, text, time.time()))

    def process_queue(self) -> None:
        """Process queue in order"""
        self._is_running = True

        while self._is_running:
            try:
                # Get next item with timeout
                priority, text, timestamp = self._queue.get(timeout=0.5)

                logger.debug(f"TTS: Speaking '{text[:50]}...'")

                # Run synthesis
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    result = loop.run_until_complete(
                        self.engine.speak(text, blocking=True)
                    )

                    if result.error:
                        logger.error(f"TTS error: {result.error}")

                finally:
                    loop.close()

                self._queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")

    def interrupt(self) -> None:
        """Interrupt current speech"""
        self.engine.interrupt()

        # Clear queue
        with self._queue.mutex:
            self._queue.queue.clear()

    def start_processing(self) -> None:
        """Start queue processing thread"""
        if not self._processing_thread or not self._processing_thread.is_alive():
            self._processing_thread = threading.Thread(
                target=self.process_queue, daemon=True
            )
            self._processing_thread.start()

    def stop(self) -> None:
        """Stop queue processing"""
        self._is_running = False
        self.engine.interrupt()

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self._queue.qsize()

    def get_current_text(self) -> str:
        """Get currently speaking text"""
        with self._lock:
            return self._current_text


class StreamingTTSPlayer:
    """Handles streaming audio playback with low latency"""

    def __init__(self, engine: TTSEngineBase):
        self.engine = engine
        self._audio_queue: queue.Queue = queue.Queue()
        self._is_playing = False
        self._playback_thread: Optional[threading.Thread] = None

    async def play_streaming(
        self, text: str, on_chunk: Optional[Callable[[bytes], None]] = None
    ) -> TTSResult:
        """Play TTS with streaming"""
        self._is_playing = True

        try:
            start_time = time.perf_counter()
            all_chunks = []

            async for chunk in self.engine.stream_synthesize(text):
                all_chunks.append(chunk)

                if on_chunk:
                    on_chunk(chunk)

            audio_data = b"".join(all_chunks)
            processing_time = time.perf_counter() - start_time

            return TTSResult(
                audio_data=audio_data,
                duration=len(audio_data) / (self.engine.sample_rate * 2),
                text_length=len(text),
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Streaming playback error: {e}")
            return TTSResult(error=str(e))
        finally:
            self._is_playing = False

    def interrupt(self) -> None:
        """Interrupt playback"""
        self._is_playing = False
        self.engine.interrupt()


class TTSEngine:
    """Main TTS engine wrapper with fallback support"""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.primary_engine = self._create_engine()
        self._is_loaded = False

    def _create_engine(self) -> TTSEngineBase:
        """Create TTS engine based on config"""
        if self.config.backend == TTSBackend.COQUI:
            return CoquiTTSEngine()
        elif self.config.backend == TTSBackend.PIPER:
            return PiperTTSEngine()
        elif self.config.backend == TTSBackend.EDGE_TTS:
            return EdgeTTSEngine()
        elif self.config.backend == TTSBackend.PYTTSX3:
            return PyTTSx3Engine()
        else:
            return CoquiTTSEngine()  # Default

    async def initialize(self) -> bool:
        """Initialize the TTS engine"""
        logger.info("Initializing TTS engine...")
        self._is_loaded = await self.primary_engine.load_model(self.config)
        return self._is_loaded

    async def speak(
        self,
        text: str,
        blocking: bool = False,
        callback: Optional[Callable[[bytes], None]] = None,
    ) -> TTSResult:
        """Speak text"""
        if not self._is_loaded:
            return TTSResult(error="TTS not initialized")
        return await self.primary_engine.speak(text, blocking, callback)

    async def stream_synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream synthesize"""
        if not self._is_loaded:
            return
        async for chunk in self.primary_engine.stream_synthesize(text):
            yield chunk

    def speak_async(self, text: str, callback: Optional[Callable] = None) -> None:
        """Speak asynchronously"""
        if text.strip():
            asyncio.create_task(self.speak(text, callback=callback))

    def interrupt(self) -> None:
        """Interrupt current speech"""
        self.primary_engine.interrupt()

    def is_loaded(self) -> bool:
        return self._is_loaded

    def unload(self) -> None:
        self.primary_engine.unload()
        self._is_loaded = False

    @property
    def sample_rate(self) -> int:
        return self.primary_engine.sample_rate


def create_tts_engine(config: Optional[TTSConfig] = None) -> TTSEngine:
    """Factory function to create TTS engine"""
    config = config or TTSConfig()
    return TTSEngine(config)


# Latency calculations for real-time optimization
def estimate_tts_latency(
    text_length: int, backend: TTSBackend = TTSBackend.COQUI
) -> float:
    """
    Estimate TTS latency based on text length and backend
    Returns expected processing time in seconds
    """
    # Rough estimates
    chars_per_second = {
        TTSBackend.COQUI: 100,  # ~100 chars/sec
        TTSBackend.PIPER: 80,  # ~80 chars/sec
        TTSBackend.EDGE_TTS: 150,  # ~150 chars/sec (cloud)
        TTSBackend.PYTTSX3: 60,  # ~60 chars/sec
    }

    rate = chars_per_second.get(backend, 50)
    return text_length / rate


__all__ = [
    "TTSConfig",
    "TTSResult",
    "TTSEngine",
    "TTSQueueManager",
    "StreamingTTSPlayer",
    "TTSBackend",
    "create_tts_engine",
    "estimate_tts_latency",
]
