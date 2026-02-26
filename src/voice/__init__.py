"""
AURA v3 Voice Package
Voice input/output processing

Primary Pipeline (recommended):
- RealtimeVoicePipeline: Production-grade real-time voice pipeline
- TelegramVoiceAdapter: Telegram voice message processing

Modules:
- real_time_pipeline: Production-grade real-time voice pipeline (primary)
- pipeline: Legacy voice pipeline (deprecated, kept for backward compatibility)
- stt: Speech-to-text with streaming support
- tts: Text-to-speech with streaming support
- hotword: Wake word and VAD detection
"""

import warnings
from typing import TYPE_CHECKING

# Primary exports - real-time pipeline (recommended)
from .real_time_pipeline import (
    RealtimePipelineConfig,
    RealtimeVoicePipeline,
    RealtimePipelineFactory,
    PipelineMode,
    PipelineState,
    LatencyBudget,
    TelegramVoiceAdapter,
)

# STT exports
from .stt import (
    STTConfig,
    STTResult,
    STTProcessor,
    STTBackend,
    create_stt_processor,
    estimate_stt_latency,
)

# TTS exports
from .tts import (
    TTSConfig,
    TTSResult,
    TTSEngine,
    TTSBackend,
    create_tts_engine,
    estimate_tts_latency,
)

# Hotword exports
from .hotword import (
    HotWordConfig,
    HotwordResult,
    VADResult,
    VADEngineBase,
    HotWordDetector,
    EmergencyHotwordDetector,
    HotwordBackend,
    create_vad_engine,
    create_hotword_detector,
)


# Backward-compatible legacy exports with deprecation warnings
# These are kept to avoid breaking existing imports


def __getattr__(name: str):
    """Lazy import legacy pipeline classes with deprecation warnings."""

    if name == "VoicePipeline":
        warnings.warn(
            "VoicePipeline is deprecated. Use RealtimeVoicePipeline instead. "
            "See voice.real_time_pipeline for the modern implementation.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .pipeline import VoicePipeline

        return VoicePipeline

    elif name == "VoicePipelineConfig":
        warnings.warn(
            "VoicePipelineConfig is deprecated. Use RealtimePipelineConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .pipeline import VoicePipelineConfig

        return VoicePipelineConfig

    elif name == "TelegramVoiceHandler":
        warnings.warn(
            "TelegramVoiceHandler is deprecated. Use TelegramVoiceAdapter instead. "
            "The new adapter uses the real-time pipeline for better performance.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .pipeline import TelegramVoiceHandler

        return TelegramVoiceHandler

    raise AttributeError(f"module 'voice' has no attribute {name!r}")


# Convenience aliases
RealTimePipeline = RealtimeVoicePipeline  # Alternative naming


__all__ = [
    # Primary pipeline (recommended)
    "RealtimeVoicePipeline",
    "RealtimePipelineConfig",
    "RealtimePipelineFactory",
    "TelegramVoiceAdapter",
    "PipelineMode",
    "PipelineState",
    "LatencyBudget",
    # Convenience alias
    "RealTimePipeline",
    # Legacy pipeline (deprecated - backward compatibility)
    "VoicePipeline",
    "VoicePipelineConfig",
    "TelegramVoiceHandler",
    # STT
    "STTConfig",
    "STTResult",
    "STTProcessor",
    "STTBackend",
    "create_stt_processor",
    "estimate_stt_latency",
    # TTS
    "TTSConfig",
    "TTSResult",
    "TTSEngine",
    "TTSBackend",
    "create_tts_engine",
    "estimate_tts_latency",
    # Hotword
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
