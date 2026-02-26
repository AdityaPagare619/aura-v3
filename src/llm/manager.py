"""
AURA v3 LLM Integration
Manages local LLM models for offline operation
Supports: whisper.cpp (STT), piper (TTS), quantized LLMs
"""

import asyncio
import logging
import os
import subprocess
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Import ProductionLLM (canonical LLM backend)
try:
    from src.llm.production_llm import ProductionLLM, BackendType

    PRODUCTION_LLM_AVAILABLE = True
except ImportError:
    PRODUCTION_LLM_AVAILABLE = False
    logger.warning("ProductionLLM not available, using fallback mode")

# Legacy import for backward compatibility (deprecated)
try:
    from src.llm.real_llm import LLMConfig
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class LLMConfig:
        model_path: str = ""
        backend: str = "llama_cpp"
        n_ctx: int = 2048
        n_gpu_layers: int = 0
        n_threads: int = 4
        temperature: float = 0.7
        max_tokens: int = 512


class ModelType(Enum):
    """Types of local models"""

    STT = "speech_to_text"  # Whisper
    TTS = "text_to_speech"  # Piper
    LLM = "language_model"  # Quantized LLM
    VISION = "vision"  # Image understanding


class ModelStatus(Enum):
    """Status of a model"""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    NOT_FOUND = "not_found"


@dataclass
class ModelInfo:
    """Information about a loaded model"""

    name: str
    model_type: ModelType
    path: str
    status: ModelStatus
    size_mb: float = 0.0
    loaded_at: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class LLMResponse:
    """Response from LLM"""

    text: str
    model: str
    tokens_used: int = 0
    inference_time_ms: int = 0
    error: Optional[str] = None


@dataclass
class TTSResponse:
    """Response from TTS"""

    audio_path: str
    model: str
    duration_ms: int = 0
    error: Optional[str] = None


@dataclass
class STTResponse:
    """Response from STT"""

    text: str
    language: str = "en"
    confidence: float = 0.0
    model: str = ""
    error: Optional[str] = None


class LLMManager:
    """
    LLM Manager - handles local LLM models

    Designed for mobile/offline operation:
    - Whisper.cpp for speech-to-text
    - Piper for text-to-speech
    - Quantized LLMs (llama.cpp, etc.) for text generation

    All processing is local - no cloud APIs
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Loaded models
        self._loaded_models: Dict[str, ModelInfo] = {}

        # Production LLM integration (canonical backend)
        self._production_llm: Optional[ProductionLLM] = None
        if PRODUCTION_LLM_AVAILABLE:
            self._production_llm = ProductionLLM()

        # Settings
        self._default_stt_model = "base"
        self._default_tts_model = "en_US-lessac"
        self._default_llm_model = "llama-2-7b-chat.Q4_K_M.gguf"
        self._max_tokens = 512
        self._temperature = 0.7

    # =========================================================================
    # MODEL MANAGEMENT
    # =========================================================================

    async def load_model(self, model_name: str, model_type: ModelType) -> ModelInfo:
        """Load a model into memory"""
        logger.info(f"Loading model: {model_name} ({model_type.value})")

        model_info = ModelInfo(
            name=model_name,
            model_type=model_type,
            path=str(self.models_dir / model_name),
            status=ModelStatus.LOADING,
        )

        try:
            # Check if model file exists
            if not Path(model_info.path).exists():
                model_info.status = ModelStatus.NOT_FOUND
                model_info.error = f"Model file not found: {model_info.path}"
                return model_info

            # Get file size
            size_mb = Path(model_info.path).stat().st_size / (1024 * 1024)
            model_info.size_mb = size_mb

            # Load based on type (this is a placeholder - actual implementation
            # would load the model using appropriate library)
            if model_type == ModelType.STT:
                await self._load_whisper(model_name)
            elif model_type == ModelType.TTS:
                await self._load_piper(model_name)
            elif model_type == ModelType.LLM:
                await self._load_llm(model_name)

            model_info.status = ModelStatus.READY
            model_info.loaded_at = datetime.now()
            self._loaded_models[model_name] = model_info

            logger.info(f"Model loaded: {model_name} ({size_mb:.1f} MB)")

        except Exception as e:
            model_info.status = ModelStatus.ERROR
            model_info.error = str(e)
            logger.error(f"Failed to load model {model_name}: {e}")

        return model_info

    async def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            logger.info(f"Model unloaded: {model_name}")
            return True
        return False

    def get_model_status(self, model_name: str) -> Optional[ModelInfo]:
        """Get status of a model"""
        return self._loaded_models.get(model_name)

    def list_models(self) -> List[Dict]:
        """List all available models"""
        models = []
        for model_file in self.models_dir.glob("*"):
            if model_file.is_file():
                size_mb = model_file.stat().st_size / (1024 * 1024)
                loaded = model_file.name in self._loaded_models
                models.append(
                    {
                        "name": model_file.name,
                        "size_mb": size_mb,
                        "loaded": loaded,
                        "status": self._loaded_models.get(
                            model_file.name,
                            ModelInfo("", ModelType.LLM, "", ModelStatus.NOT_LOADED),
                        ).status.value,
                    }
                )
        return models

    # =========================================================================
    # STT (Speech to Text) - Whisper
    # =========================================================================

    async def _load_whisper(self, model_name: str):
        """Load Whisper model"""
        # Placeholder - actual implementation would use whisper.cpp or faster-whisper
        logger.info(f"Loading Whisper model: {model_name}")
        await asyncio.sleep(0.1)

    async def transcribe(
        self, audio_path: str, language: Optional[str] = None
    ) -> STTResponse:
        """
        Transcribe audio to text using Whisper

        Args:
            audio_path: Path to audio file
            language: Language code (auto-detect if None)
        """
        logger.info(f"Transcribing: {audio_path}")

        # Check if we have a loaded STT model
        stt_models = [
            m for m in self._loaded_models.values() if m.model_type == ModelType.STT
        ]
        if not stt_models:
            # Try to use default
            if not Path(
                self.models_dir / f"whisper-{self._default_stt_model}.bin"
            ).exists():
                return STTResponse(
                    text="",
                    error="No STT model loaded. Please load a Whisper model first.",
                )

        try:
            # This is a placeholder - actual implementation would use:
            # faster-whisper or whisper.cpp
            # Example: faster_whisper.Transcriber.transcribe(audio_path)

            # For now, return mock response
            return STTResponse(
                text="[Transcription would appear here]",
                language=language or "en",
                confidence=0.95,
                model=self._default_stt_model,
            )

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return STTResponse(text="", error=str(e))

    async def transcribe_microphone(
        self, duration_seconds: int = 5, language: Optional[str] = None
    ) -> STTResponse:
        """
        Transcribe from microphone in real-time
        Note: Requires audio capture capability
        """
        logger.info(f"Recording and transcribing {duration_seconds}s of audio")

        # This would capture audio from microphone
        # Then call transcribe() on the captured audio

        return STTResponse(
            text="[Microphone transcription would appear here]",
            language=language or "en",
            confidence=0.90,
            model=self._default_stt_model,
        )

    # =========================================================================
    # TTS (Text to Speech) - Piper
    # =========================================================================

    async def _load_piper(self, model_name: str):
        """Load Piper TTS model"""
        logger.info(f"Loading Piper model: {model_name}")
        await asyncio.sleep(0.1)

    async def speak(
        self, text: str, output_path: Optional[str] = None, voice: Optional[str] = None
    ) -> TTSResponse:
        """
        Convert text to speech using Piper

        Args:
            text: Text to speak
            output_path: Path to save audio (auto-generated if None)
            voice: Voice model to use
        """
        logger.info(f"Synthesizing speech: {text[:50]}...")

        # Check for loaded TTS model
        tts_models = [
            m for m in self._loaded_models.values() if m.model_type == ModelType.TTS
        ]
        if not tts_models:
            default_model = f"piper-{self._default_tts_model}"
            if not Path(self.models_dir / f"{default_model}.onnx").exists():
                return TTSResponse(
                    audio_path="",
                    error="No TTS model loaded. Please load a Piper model first.",
                )

        # Generate output path
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/tts/output_{timestamp}.wav"

        try:
            # This is a placeholder - actual implementation would use piper
            # Example: subprocess.run(['piper', '--model', model, '--output_file', output_path],
            #                     input=text, capture_output=True)

            # For now, return mock response
            return TTSResponse(
                audio_path=output_path,
                model=voice or self._default_tts_model,
                duration_ms=len(text) * 50,  # Rough estimate
            )

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return TTSResponse(audio_path="", error=str(e))

    async def speak_with_emotion(
        self, text: str, emotion: str, output_path: Optional[str] = None
    ) -> TTSResponse:
        """Speak with emotional tone"""
        # Map emotion to voice settings
        emotion_map = {
            "happy": {"pitch": 1.1, "speed": 1.0},
            "sad": {"pitch": 0.9, "speed": 0.9},
            "excited": {"pitch": 1.2, "speed": 1.1},
            "calm": {"pitch": 1.0, "speed": 0.95},
        }

        settings = emotion_map.get(emotion, {})
        logger.info(f"Speaking with emotion '{emotion}': {text[:50]}...")

        return await self.speak(text, output_path)

    # =========================================================================
    # LLM (Language Model)
    # =========================================================================

    async def _load_llm(self, model_name: str):
        """Load LLM model using ProductionLLM integration"""
        logger.info(f"Loading LLM model: {model_name}")

        if not PRODUCTION_LLM_AVAILABLE or self._production_llm is None:
            logger.warning("ProductionLLM not available, using fallback")
            await asyncio.sleep(0.1)
            return

        # Determine model path
        model_path = str(self.models_dir / model_name)

        # Load the model via ProductionLLM
        success = await self._production_llm.load_model(
            model_id=model_name,
            model_path=model_path,
        )
        if success:
            logger.info(f"ProductionLLM loaded successfully: {model_name}")
        else:
            logger.warning("Failed to load ProductionLLM, using fallback")

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Generate text using LLM

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 - 1.0)
            stop: Stop sequences
        """
        logger.info(f"Generating response for prompt: {prompt[:50]}...")

        # Check for loaded LLM
        llm_models = [
            m for m in self._loaded_models.values() if m.model_type == ModelType.LLM
        ]
        if not llm_models:
            return LLMResponse(
                text="",
                error="No LLM loaded. Please load a model first.",
            )

        try:
            # Try using ProductionLLM first
            if (
                PRODUCTION_LLM_AVAILABLE
                and self._production_llm is not None
                and self._production_llm.is_loaded
            ):
                result = await self._production_llm.generate(
                    prompt=prompt,
                    max_tokens=max_tokens or self._max_tokens,
                )

                return LLMResponse(
                    text=result.get("text", ""),
                    model=result.get("model", llm_models[0].name),
                    tokens_used=result.get(
                        "tokens_generated", len(result.get("text", "").split())
                    ),
                    inference_time_ms=int(result.get("latency_ms", 1000)),
                )

            # Fallback to mock response if ProductionLLM not available
            response_text = f"[LLM response would appear here for: {prompt[:50]}...]"

            return LLMResponse(
                text=response_text,
                model=llm_models[0].name,
                tokens_used=len(response_text.split()),
                inference_time_ms=1000,  # Rough estimate
            )

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return LLMResponse(text="", error=str(e))

    async def chat(
        self,
        message: str,
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Chat with LLM

        Args:
            message: User message
            conversation_history: Previous messages
            system_prompt: System prompt to set context
        """
        # Build prompt
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}")

        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}")

        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")

        prompt = "\n".join(prompt_parts)

        return await self.generate(
            prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

    # =========================================================================
    # CONTEXT MANAGEMENT
    # =========================================================================

    async def create_context(self, system_prompt: str) -> str:
        """Create a new chat context"""
        context_id = f"context_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Created context: {context_id}")
        return context_id

    # =========================================================================
    # SETTINGS
    # =========================================================================

    def set_default_stt_model(self, model: str):
        """Set default STT model"""
        self._default_stt_model = model
        logger.info(f"Default STT model set to: {model}")

    def set_default_tts_model(self, model: str):
        """Set default TTS model"""
        self._default_tts_model = model
        logger.info(f"Default TTS model set to: {model}")

    def set_default_llm_model(self, model: str):
        """Set default LLM model"""
        self._default_llm_model = model
        logger.info(f"Default LLM model set to: {model}")

    def set_generation_params(self, max_tokens: int = 512, temperature: float = 0.7):
        """Set default generation parameters"""
        self._max_tokens = max_tokens
        self._temperature = temperature
        logger.info(
            f"Generation params: max_tokens={max_tokens}, temperature={temperature}"
        )

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get LLM manager status"""
        return {
            "models_dir": str(self.models_dir),
            "loaded_models": [
                {
                    "name": m.name,
                    "type": m.model_type.value,
                    "status": m.status.value,
                    "size_mb": m.size_mb,
                    "loaded_at": m.loaded_at.isoformat() if m.loaded_at else None,
                }
                for m in self._loaded_models.values()
            ],
            "defaults": {
                "stt": self._default_stt_model,
                "tts": self._default_tts_model,
                "llm": self._default_llm_model,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            },
        }


# ==============================================================================
# FACTORY
# ==============================================================================

_llm_manager_instance: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get or create LLM manager instance"""
    global _llm_manager_instance
    if _llm_manager_instance is None:
        _llm_manager_instance = LLMManager()
    return _llm_manager_instance
