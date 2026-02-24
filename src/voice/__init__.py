"""
AURA v3 Voice Package
Voice input/output processing

Modules:
- pipeline: Original voice pipeline (legacy)
- real_time_pipeline: Production-grade real-time voice pipeline
- stt: Speech-to-text with streaming support
- tts: Text-to-speech with streaming support
- hotword: Wake word and VAD detection
"""

from .pipeline import VoicePipeline, VoicePipelineConfig, TelegramVoiceHandler
from .real_time_pipeline import (
    RealtimePipelineConfig,
    RealtimeVoicePipeline,
    RealtimePipelineFactory,
    PipelineMode,
    PipelineState,
    LatencyBudget,
)
from .stt import (
    STTConfig,
    STTResult,
    STTProcessor,
    STTBackend,
    create_stt_processor,
    estimate_stt_latency,
)
from .tts import (
    TTSConfig,
    TTSResult,
    TTSEngine,
    TTSBackend,
    create_tts_engine,
    estimate_tts_latency,
)
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

__all__ = [
    # Legacy pipeline
    "VoicePipeline",
    "VoicePipelineConfig",
    "TelegramVoiceHandler",
    # Real-time pipeline
    "RealtimePipelineConfig",
    "RealtimeVoicePipeline",
    "RealtimePipelineFactory",
    "PipelineMode",
    "PipelineState",
    "LatencyBudget",
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
