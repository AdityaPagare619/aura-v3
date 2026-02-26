"""
AURA v3 Production LLM Integration
Production-grade LLM system with GROUND REALITY calculations for Android Termux

This module provides:
- Dynamic model loading with real memory calculations
- Streaming token generation for faster perceived response
- Model benchmark system with actual latency tracking
- Power consumption estimation
- Thermal throttling awareness
- Model recommendation engine based on device capabilities
- Fallback chain for model failures
- Performance monitoring with actual metrics
- User-definable models (not hardcoded)
"""

import asyncio
import logging
import os
import json
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """LLM backend types"""

    LLAMA_CPP = "llama_cpp"  # llama.cpp via python bindings
    GPT4ALL = "gpt4all"  # gpt4all
    OLLAMA = "ollama"  # Ollama server
    TRANSFORMERS = "transformers"  # HuggingFace transformers


class ModelCategory(Enum):
    """Model categories for different use cases"""

    SPEED = "speed"  # Maximum speed, basic tasks
    BALANCED = "balanced"  # Balanced performance
    REASONING = "reasoning"  # Mathematical/logical reasoning
    CODING = "coding"  # Code generation
    GENERAL = "general"  # General purpose


class QuantizationType(Enum):
    """Quantization levels"""

    Q2 = "q2"
    Q3 = "q3"
    Q4_0 = "q4_0"
    Q4_K_M = "q4_k_m"
    Q5_0 = "q5_0"
    Q5_K_M = "q5_k_m"
    Q6_K = "q6_k"
    Q8 = "q8"
    FP16 = "fp16"
    FP32 = "fp32"


@dataclass
class DeviceCapabilities:
    """Device hardware capabilities"""

    total_ram_mb: float = 0.0
    available_ram_mb: float = 0.0
    cpu_cores: int = 4
    cpu_freq_mhz: int = 2400
    has_npu: bool = False
    has_gpu: bool = False
    gpu_model: str = ""
    thermal_state: str = "nominal"  # nominal, fair, serious, critical

    def to_dict(self) -> Dict:
        return {
            "total_ram_mb": self.total_ram_mb,
            "available_ram_mb": self.available_ram_mb,
            "cpu_cores": self.cpu_cores,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "has_npu": self.has_npu,
            "has_gpu": self.has_gpu,
            "gpu_model": self.gpu_model,
            "thermal_state": self.thermal_state,
        }


@dataclass
class ModelSpec:
    """
    Complete model specification - user definable
    All models can be added by users, not hardcoded
    """

    id: str
    name: str
    model_type: str  # "llm", "stt", "tts"
    backend: BackendType
    model_path: str
    filename: str

    # Model parameters (for LLM)
    parameters_billion: float = 0.0  # e.g., 1.5 for 1.5B model
    quantization: QuantizationType = QuantizationType.Q4_K_M

    # Memory calculations
    file_size_mb: float = 0.0

    # Performance (from benchmarks or user input)
    tokens_per_second: float = 0.0
    load_time_seconds: float = 0.0
    time_to_first_token_ms: float = 0.0

    # Context
    context_length: int = 2048

    # Category
    category: ModelCategory = ModelCategory.BALANCED

    # Languages supported
    languages: List[str] = field(default_factory=lambda: ["en"])

    # Metadata
    description: str = ""
    url: str = ""
    added_by: str = "system"  # "system" or username
    added_at: datetime = field(default_factory=datetime.now)

    # Benchmarks (if available)
    mmlu_score: float = 0.0  # General knowledge
    code_score: float = 0.0  # Coding
    math_score: float = 0.0  # Math reasoning

    def calculate_ram_required(self, context_tokens: int = 0) -> float:
        """
        REAL CALCULATION: RAM needed = (model_size * quantization_factor) + (context_size * 4 bytes) + overhead

        Quantization factors (relative to FP16):
        - Q2: ~0.25x
        - Q3: ~0.375x
        - Q4: ~0.5x
        - Q5: ~0.625x
        - Q6: ~0.75x
        - Q8: ~1.0x
        - FP16: ~2.0x
        - FP32: ~4.0x
        """
        quant_factors = {
            QuantizationType.Q2: 0.25,
            QuantizationType.Q3: 0.375,
            QuantizationType.Q4_0: 0.5,
            QuantizationType.Q4_K_M: 0.5,
            QuantizationType.Q5_0: 0.625,
            QuantizationType.Q5_K_M: 0.625,
            QuantizationType.Q6_K: 0.75,
            QuantizationType.Q8: 1.0,
            QuantizationType.FP16: 2.0,
            QuantizationType.FP32: 4.0,
        }

        factor = quant_factors.get(self.quantization, 0.5)

        # Base model memory (FP16 would be ~2 bytes per parameter)
        base_memory_mb = (
            self.parameters_billion * 2000
        )  # 2GB per billion params in FP16
        model_memory_mb = base_memory_mb * factor

        # Context memory: 4 bytes per token (KV cache)
        context_memory_mb = (context_tokens * 4) / (1024 * 1024)

        # Overhead: ~10% for runtime buffers
        overhead_mb = model_memory_mb * 0.1

        total = model_memory_mb + context_memory_mb + overhead_mb

        logger.debug(
            f"RAM calculation for {self.name}: "
            f"model={model_memory_mb:.1f}MB + context={context_memory_mb:.1f}MB + "
            f"overhead={overhead_mb:.1f}MB = {total:.1f}MB"
        )

        return total

    def calculate_estimated_speed(self, device: DeviceCapabilities) -> float:
        """
        REAL CALCULATION: Expected tokens/sec based on model size and phone CPU

        Base speed per billion parameters (Q4, 4GB RAM class device):
        - Single core: ~3 tokens/sec per billion
        - Multi-core scales with ~0.6x efficiency

        Adjustments:
        - CPU frequency: linear with frequency
        - Thermal throttling: can reduce by 20-50%
        - GPU/NPU: can add 1.5-2x boost
        """
        if self.tokens_per_second > 0:
            return self.tokens_per_second

        # Base calculation
        base_tps = 4.0 * self.parameters_billion  # ~4 TPS per billion on 4GB RAM device

        # CPU frequency adjustment (baseline 2400MHz)
        freq_factor = device.cpu_freq_mhz / 2400
        base_tps *= freq_factor

        # Multi-threading efficiency
        if device.cpu_cores > 1:
            # Diminishing returns: 2 cores = 1.7x, 4 cores = 2.5x, 8 cores = 3.2x
            core_factor = 1 + 0.7 * (device.cpu_cores - 1) ** 0.5
            base_tps *= min(core_factor, 3.0)

        # Thermal throttling
        thermal_factors = {
            "nominal": 1.0,
            "fair": 0.85,
            "serious": 0.6,
            "critical": 0.3,
        }
        base_tps *= thermal_factors.get(device.thermal_state, 1.0)

        # GPU/NPU boost
        if device.has_gpu or device.has_npu:
            base_tps *= 1.5

        logger.debug(
            f"Estimated speed for {self.name} on {device.cpu_cores}c: {base_tps:.1f} TPS"
        )

        return max(base_tps, 1.0)  # Minimum 1 TPS

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "model_type": self.model_type,
            "backend": self.backend.value,
            "parameters_billion": self.parameters_billion,
            "quantization": self.quantization.value,
            "file_size_mb": self.file_size_mb,
            "tokens_per_second": self.tokens_per_second,
            "load_time_seconds": self.load_time_seconds,
            "context_length": self.context_length,
            "category": self.category.value,
            "languages": self.languages,
            "description": self.description,
            "mmlu_score": self.mmlu_score,
            "code_score": self.code_score,
            "math_score": self.math_score,
        }


@dataclass
class PipelineLatency:
    """
    GROUND REALITY latency tracking
    Total response time = STT_latency + LLM_latency + TTS_latency
    """

    stt_latency_ms: float = 0.0
    llm_time_to_first_token_ms: float = 0.0
    llm_token_generation_ms: float = 0.0
    llm_total_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0

    # Calculated
    total_latency_ms: float = 0.0
    estimated_audio_duration_ms: float = 0.0

    def calculate_total(self) -> float:
        """Calculate total pipeline latency"""
        self.total_latency_ms = (
            self.stt_latency_ms
            + self.llm_time_to_first_token_ms
            + self.llm_token_generation_ms
            + self.tts_latency_ms
        )
        return self.total_latency_ms

    def is_within_target(self, target_ms: float = 3000) -> bool:
        """Check if total latency is within target (default 3 seconds for voice)"""
        return self.total_latency_ms <= target_ms

    def to_dict(self) -> Dict:
        return {
            "stt_latency_ms": self.stt_latency_ms,
            "llm_time_to_first_token_ms": self.llm_time_to_first_token_ms,
            "llm_token_generation_ms": self.llm_token_generation_ms,
            "llm_total_latency_ms": self.llm_total_latency_ms,
            "tts_latency_ms": self.tts_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "estimated_audio_duration_ms": self.estimated_audio_duration_ms,
            "within_3s_target": self.is_within_target(),
        }


@dataclass
class ModelBenchmark:
    """Actual benchmark results"""

    model_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Load metrics
    load_time_ms: float = 0.0
    memory_used_mb: float = 0.0

    # Generation metrics
    tokens_generated: int = 0
    total_generation_time_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    tokens_per_second: float = 0.0

    # Context metrics
    context_length_used: int = 0
    peak_memory_mb: float = 0.0

    # Device state
    device_temp_celsius: float = 0.0
    cpu_usage_percent: float = 0.0

    def calculate_efficiency(self) -> float:
        """Tokens per second per MB of memory"""
        if self.memory_used_mb > 0:
            return self.tokens_per_second / self.memory_used_mb
        return 0.0

    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat(),
            "load_time_ms": self.load_time_ms,
            "memory_used_mb": self.memory_used_mb,
            "tokens_generated": self.tokens_generated,
            "total_generation_time_ms": self.total_generation_time_ms,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "tokens_per_second": self.tokens_per_second,
            "context_length_used": self.context_length_used,
            "peak_memory_mb": self.peak_memory_mb,
            "device_temp_celsius": self.device_temp_celsius,
            "cpu_usage_percent": self.cpu_usage_percent,
            "efficiency": self.calculate_efficiency(),
        }


@dataclass
class PowerConsumption:
    """Power consumption estimation"""

    cpu_power_watts: float = 0.0
    memory_power_watts: float = 0.0
    gpu_power_watts: float = 0.0
    total_power_watts: float = 0.0

    # Per-operation estimates
    per_token_power_mwh: float = 0.0  # milliwatt-hours per token

    def estimate_from_benchmark(
        self, benchmark: ModelBenchmark, duration_seconds: float
    ) -> float:
        """
        Estimate power consumption for a generation
        Based on CPU usage and duration
        """
        # CPU: ~2W per core at 100% usage on mobile
        cpu_watts = (benchmark.cpu_usage_percent / 100) * 2.0 * 4  # 4 cores baseline

        # Memory: ~0.5W per GB
        memory_watts = (benchmark.memory_used_mb / 1024) * 0.5

        self.cpu_power_watts = cpu_watts
        self.memory_power_watts = memory_watts
        self.total_power_watts = cpu_watts + memory_watts

        # Per-token estimate
        if benchmark.tokens_generated > 0:
            self.per_token_power_mwh = (
                self.total_power_watts * duration_seconds * 1000
            ) / benchmark.tokens_generated

        return self.total_power_watts

    def to_dict(self) -> Dict:
        return {
            "cpu_power_watts": self.cpu_power_watts,
            "memory_power_watts": self.memory_power_watts,
            "gpu_power_watts": self.gpu_power_watts,
            "total_power_watts": self.total_power_watts,
            "per_token_power_mwh": self.per_token_power_mwh,
        }


class ModelRegistry:
    """
    User-definable model registry
    Allows adding ANY model, not hardcoded
    """

    DEFAULT_MODELS: List[ModelSpec] = [
        # Tier 1: Speed-focused
        ModelSpec(
            id="llama-3.2-1b-q4",
            name="Llama 3.2 1B Q4",
            model_type="llm",
            backend=BackendType.LLAMA_CPP,
            model_path="models/llama-3.2-1b-instruct-q4_k_m.gguf",
            filename="llama-3.2-1b-instruct-q4_k_m.gguf",
            parameters_billion=1.0,
            quantization=QuantizationType.Q4_K_M,
            file_size_mb=700,
            tokens_per_second=13.0,
            load_time_seconds=4.0,
            time_to_first_token_ms=500,
            context_length=128000,
            category=ModelCategory.SPEED,
            languages=["en"],
            description="Meta's fastest 1B model, excellent for quick responses",
            mmlu_score=55.0,
        ),
        # Tier 2: Balanced (Recommended default)
        ModelSpec(
            id="qwen-2.5-1.5b-q4",
            name="Qwen 2.5 1.5B Q4",
            model_type="llm",
            backend=BackendType.LLAMA_CPP,
            model_path="models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
            parameters_billion=1.5,
            quantization=QuantizationType.Q4_K_M,
            file_size_mb=1000,
            tokens_per_second=11.0,
            load_time_seconds=5.0,
            time_to_first_token_ms=600,
            context_length=128000,
            category=ModelCategory.BALANCED,
            languages=["en", "zh", "es", "fr", "de", "ja", "ko", "ar", "ru", "pt"],
            description="Best overall: multilingual, coding, reasoning - 29+ languages",
            mmlu_score=62.0,
            code_score=45.0,
            math_score=40.0,
        ),
        # Tier 3: Balanced with more power
        ModelSpec(
            id="llama-3.2-3b-q4",
            name="Llama 3.2 3B Q4",
            model_type="llm",
            backend=BackendType.LLAMA_CPP,
            model_path="models/llama-3.2-3b-instruct-q4_k_m.gguf",
            filename="llama-3.2-3b-instruct-q4_k_m.gguf",
            parameters_billion=3.0,
            quantization=QuantizationType.Q4_K_M,
            file_size_mb=2000,
            tokens_per_second=8.5,
            load_time_seconds=10.0,
            time_to_first_token_ms=800,
            context_length=128000,
            category=ModelCategory.BALANCED,
            languages=["en"],
            description="Meta's balanced 3B model - better reasoning than 1B",
            mmlu_score=65.0,
        ),
        # Tier 4: Reasoning focus
        ModelSpec(
            id="deepseek-r1-1.5b-q4",
            name="DeepSeek-R1 Distill 1.5B Q4",
            model_type="llm",
            backend=BackendType.LLAMA_CPP,
            model_path="models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf",
            filename="deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf",
            parameters_billion=1.5,
            quantization=QuantizationType.Q4_K_M,
            file_size_mb=1000,
            tokens_per_second=9.5,
            load_time_seconds=6.0,
            time_to_first_token_ms=700,
            context_length=128000,
            category=ModelCategory.REASONING,
            languages=["en", "zh"],
            description="Best for math and reasoning - includes chain-of-thought",
            mmlu_score=58.0,
            math_score=70.0,
        ),
        # Tier 5: Coding focus
        ModelSpec(
            id="qwen-2.5-3b-q4",
            name="Qwen 2.5 3B Q4",
            model_type="llm",
            backend=BackendType.LLAMA_CPP,
            model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf",
            filename="qwen2.5-3b-instruct-q4_k_m.gguf",
            parameters_billion=3.0,
            quantization=QuantizationType.Q4_K_M,
            file_size_mb=1900,
            tokens_per_second=9.0,
            load_time_seconds=10.0,
            time_to_first_token_ms=750,
            context_length=128000,
            category=ModelCategory.CODING,
            languages=["en", "zh", "es", "fr", "de", "ja", "ko", "ar", "ru", "pt"],
            description="Excellent coding performance, good reasoning",
            mmlu_score=68.0,
            code_score=55.0,
            math_score=50.0,
        ),
        # STT Models
        ModelSpec(
            id="whisper-tiny",
            name="Whisper Tiny",
            model_type="stt",
            backend=BackendType.LLAMA_CPP,  # whisper.cpp
            model_path="models/whisper-tiny.bin",
            filename="whisper-tiny.bin",
            parameters_billion=0.039,
            file_size_mb=75,
            tokens_per_second=0,  # N/A for STT
            load_time_seconds=2.0,
            context_length=0,
            category=ModelCategory.SPEED,
            languages=["en", "multilingual"],
            description="Fastest STT, requires ~200MB RAM",
        ),
        ModelSpec(
            id="whisper-base",
            name="Whisper Base",
            model_type="stt",
            backend=BackendType.LLAMA_CPP,
            model_path="models/whisper-base.bin",
            filename="whisper-base.bin",
            parameters_billion=0.074,
            file_size_mb=150,
            load_time_seconds=3.0,
            context_length=0,
            category=ModelCategory.BALANCED,
            languages=["en", "multilingual"],
            description="Best accuracy/speed balance for STT",
        ),
        # TTS Models
        ModelSpec(
            id="piper-medium",
            name="Piper Medium",
            model_type="tts",
            backend=BackendType.LLAMA_CPP,
            model_path="models/piper/en_US-lessac-medium.onnx",
            filename="en_US-lessac-medium.onnx",
            file_size_mb=500,
            load_time_seconds=3.0,
            context_length=0,
            category=ModelCategory.BALANCED,
            languages=["en"],
            description="Good quality English TTS, ~500MB",
        ),
    ]

    def __init__(self, user_models_path: str = "config/user_models.json"):
        self.user_models_path = Path(user_models_path)
        self._models: Dict[str, ModelSpec] = {}
        self._load_models()

    def _load_models(self):
        """Load default and user models"""
        # Load defaults
        for model in self.DEFAULT_MODELS:
            self._models[model.id] = model

        # Load user models
        if self.user_models_path.exists():
            try:
                with open(self.user_models_path) as f:
                    user_data = json.load(f)
                    for model_data in user_data.get("models", []):
                        model = self._parse_model_data(model_data)
                        if model:
                            self._models[model.id] = model
                logger.info(f"Loaded {len(user_data.get('models', []))} user models")
            except Exception as e:
                logger.error(f"Failed to load user models: {e}")

    def _parse_model_data(self, data: Dict) -> Optional[ModelSpec]:
        """Parse model data from JSON"""
        try:
            return ModelSpec(
                id=data["id"],
                name=data["name"],
                model_type=data.get("model_type", "llm"),
                backend=BackendType(data.get("backend", "llama_cpp")),
                model_path=data["model_path"],
                filename=data.get("filename", ""),
                parameters_billion=data.get("parameters_billion", 0.0),
                quantization=QuantizationType(data.get("quantization", "q4_k_m")),
                file_size_mb=data.get("file_size_mb", 0.0),
                tokens_per_second=data.get("tokens_per_second", 0.0),
                load_time_seconds=data.get("load_time_seconds", 0.0),
                time_to_first_token_ms=data.get("time_to_first_token_ms", 0.0),
                context_length=data.get("context_length", 2048),
                category=ModelCategory(data.get("category", "balanced")),
                languages=data.get("languages", ["en"]),
                description=data.get("description", ""),
                url=data.get("url", ""),
                added_by=data.get("added_by", "user"),
                mmlu_score=data.get("mmlu_score", 0.0),
                code_score=data.get("code_score", 0.0),
                math_score=data.get("math_score", 0.0),
            )
        except Exception as e:
            logger.error(f"Failed to parse model data: {e}")
            return None

    def add_model(self, model: ModelSpec) -> bool:
        """Add a new model (user-definable)"""
        if model.id in self._models:
            logger.warning(f"Model {model.id} already exists, updating")

        self._models[model.id] = model
        self._save_user_model(model)
        logger.info(f"Added model: {model.name} ({model.id})")
        return True

    def _save_user_model(self, model: ModelSpec):
        """Save user model to file"""
        user_models = []

        if self.user_models_path.exists():
            try:
                with open(self.user_models_path) as f:
                    user_models = json.load(f).get("models", [])
            except:
                pass

        # Check if already exists
        existing_ids = [m["id"] for m in user_models]
        if model.id in existing_ids:
            user_models = [m for m in user_models if m["id"] != model.id]

        user_models.append(model.to_dict())

        self.user_models_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.user_models_path, "w") as f:
            json.dump({"models": user_models}, f, indent=2)

    def remove_model(self, model_id: str) -> bool:
        """Remove a user-added model"""
        if model_id not in self._models:
            return False

        model = self._models[model_id]
        if model.added_by == "system":
            logger.warning(f"Cannot remove system model: {model_id}")
            return False

        del self._models[model_id]

        # Also remove from user models file
        if self.user_models_path.exists():
            try:
                with open(self.user_models_path) as f:
                    user_models = json.load(f).get("models", [])
                user_models = [m for m in user_models if m["id"] != model_id]
                with open(self.user_models_path, "w") as f:
                    json.dump({"models": user_models}, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to update user models file: {e}")

        return True

    def get_model(self, model_id: str) -> Optional[ModelSpec]:
        """Get model by ID"""
        return self._models.get(model_id)

    def list_models(self, model_type: Optional[str] = None) -> List[ModelSpec]:
        """List all models, optionally filtered by type"""
        models = list(self._models.values())
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        return sorted(models, key=lambda m: m.name)

    def find_models_for_device(self, device: DeviceCapabilities) -> List[ModelSpec]:
        """
        Find models that fit in device memory
        Returns sorted by best match (performance vs capability)
        """
        available_ram = device.available_ram_mb

        suitable_models = []
        for model in self._models.values():
            if model.model_type != "llm":
                continue

            # Check if model fits with some context headroom
            ram_needed = model.calculate_ram_required(2048)  # 2K context
            if ram_needed <= available_ram * 0.7:  # Leave 30% headroom
                suitable_models.append((model, ram_needed))

        # Sort by: RAM efficiency within device constraints
        suitable_models.sort(key=lambda x: x[1])

        return [m[0] for m in suitable_models]


class ModelRecommendationEngine:
    """
    Model recommendation engine based on device capabilities
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def get_recommendations(
        self,
        device: DeviceCapabilities,
        use_case: str = "general",
        priority: str = "balanced",  # speed, balanced, quality
    ) -> List[Dict[str, Any]]:
        """
        Get model recommendations based on device and use case

        Args:
            device: Device capabilities
            use_case: "general", "coding", "reasoning", "speed"
            priority: "speed", "balanced", "quality"
        """
        recommendations = []

        # Get all LLM models that fit
        suitable_models = self.registry.find_models_for_device(device)

        for model in suitable_models:
            score = self._calculate_score(model, device, use_case, priority)

            # Calculate expected latency
            estimated_tps = model.calculate_estimated_speed(device)
            estimated_latency = self._estimate_latency(model, estimated_tps)

            recommendations.append(
                {
                    "model": model.to_dict(),
                    "score": score,
                    "estimated_tps": estimated_tps,
                    "estimated_latency_ms": estimated_latency,
                    "ram_required_mb": model.calculate_ram_required(2048),
                    "fits_in_memory": model.calculate_ram_required(2048)
                    <= device.available_ram_mb * 0.7,
                    "within_3s_target": estimated_latency <= 3000,
                }
            )

        # Sort by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        return recommendations[:5]  # Top 5

    def _calculate_score(
        self,
        model: ModelSpec,
        device: DeviceCapabilities,
        use_case: str,
        priority: str,
    ) -> float:
        """Calculate suitability score for a model"""
        score = 50.0  # Base score

        # Use case matching
        use_case_match = {
            "general": {
                "general": 20,
                "balanced": 15,
                "speed": 10,
                "reasoning": 5,
                "coding": 5,
            },
            "coding": {"coding": 25, "balanced": 10, "general": 5},
            "reasoning": {"reasoning": 25, "balanced": 10, "general": 5},
            "speed": {"speed": 25, "balanced": 10, "general": 5},
        }
        score += use_case_match.get(use_case, {}).get(model.category.value, 0)

        # Priority matching
        if priority == "speed":
            score += 20 if model.category == ModelCategory.SPEED else 0
            score -= model.parameters_billion * 5  # Penalize larger models
        elif priority == "quality":
            score += model.mmlu_score * 0.3
            score += model.code_score * 0.2
            score += model.math_score * 0.2
        else:  # balanced
            score -= abs(2.0 - model.parameters_billion) * 5  # Prefer ~2B models

        # Performance estimate
        estimated_tps = model.calculate_estimated_speed(device)
        score += estimated_tps * 0.5

        # Memory efficiency
        ram_per_param = model.file_size_mb / max(model.parameters_billion, 0.1)
        score += (2000 - ram_per_param) * 0.01  # Prefer smaller per-param

        # Language support (bonus for multilingual)
        score += len(model.languages) * 0.5

        return max(score, 0)

    def _estimate_latency(
        self,
        model: ModelSpec,
        estimated_tps: float,
        max_tokens: int = 50,
    ) -> float:
        """Estimate total response latency in ms"""
        # Time to first token (prefill)
        tfft_ms = model.time_to_first_token_ms

        # Token generation
        gen_time_ms = (max_tokens / max(estimated_tps, 1)) * 1000

        # Add STT and TTS estimates
        stt_estimate_ms = 1500  # Base STT latency
        tts_estimate_ms = 500  # Base TTS latency

        return stt_estimate_ms + tfft_ms + gen_time_ms + tts_estimate_ms


class FallbackChain:
    """
    Fallback chain: if model X fails, try Y
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._chains: Dict[str, List[str]] = {}
        self._init_default_chains()

    def _init_default_chains(self):
        """Initialize default fallback chains"""
        self._chains = {
            "llm": [
                "qwen-2.5-1.5b-q4",
                "llama-3.2-1b-q4",
                "deepseek-r1-1.5b-q4",
            ],
            "stt": [
                "whisper-base",
                "whisper-tiny",
            ],
            "tts": [
                "piper-medium",
            ],
        }

    def set_chain(self, model_type: str, chain: List[str]):
        """Set custom fallback chain"""
        self._chains[model_type] = chain

    def get_chain(self, model_type: str) -> List[str]:
        """Get fallback chain for model type"""
        return self._chains.get(model_type, [])

    def get_fallback(self, failed_model_id: str) -> Optional[str]:
        """Get fallback model for a failed model"""
        # Find which chain contains this model
        for model_type, chain in self._chains.items():
            if failed_model_id in chain:
                idx = chain.index(failed_model_id)
                if idx + 1 < len(chain):
                    return chain[idx + 1]
        return None

    def add_to_chain(self, model_type: str, model_id: str, position: int = -1):
        """Add model to fallback chain"""
        if model_type not in self._chains:
            self._chains[model_type] = []

        if model_id not in self._chains[model_type]:
            if position < 0:
                self._chains[model_type].append(model_id)
            else:
                self._chains[model_type].insert(position, model_id)


class PerformanceMonitor:
    """
    Performance monitoring with actual metrics
    """

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self._benchmarks: deque = deque(maxlen=max_history)
        self._latencies: deque = deque(maxlen=max_history)
        self._current_benchmark: Optional[ModelBenchmark] = None

    def start_benchmark(self, model_id: str) -> ModelBenchmark:
        """Start a new benchmark"""
        self._current_benchmark = ModelBenchmark(model_id=model_id)
        self._current_benchmark.timestamp = datetime.now()
        self._start_time = time.time()
        return self._current_benchmark

    def end_benchmark(
        self,
        tokens_generated: int,
        generation_time_ms: float,
        time_to_first_token_ms: float,
    ):
        """End benchmark and record results"""
        if not self._current_benchmark:
            return

        self._current_benchmark.tokens_generated = tokens_generated
        self._current_benchmark.total_generation_time_ms = generation_time_ms
        self._current_benchmark.time_to_first_token_ms = time_to_first_token_ms

        if tokens_generated > 0 and generation_time_ms > 0:
            self._current_benchmark.tokens_per_second = tokens_generated / (
                generation_time_ms / 1000
            )

        # Get system metrics
        try:
            process = psutil.Process()
            self._current_benchmark.memory_used_mb = process.memory_info().rss / (
                1024 * 1024
            )
            self._current_benchmark.cpu_usage_percent = process.cpu_percent()
        except:
            pass

        self._benchmarks.append(self._current_benchmark)
        self._current_benchmark = None

    def record_latency(self, latency: PipelineLatency):
        """Record pipeline latency"""
        self._latencies.append(latency)

    def get_stats(self) -> Dict:
        """Get performance statistics"""
        if not self._benchmarks:
            return {"error": "No benchmarks recorded"}

        recent = list(self._benchmarks)[-10:]  # Last 10

        return {
            "total_benchmarks": len(self._benchmarks),
            "avg_tps": sum(b.tokens_per_second for b in recent) / len(recent),
            "avg_load_time_ms": sum(b.load_time_ms for b in recent) / len(recent),
            "avg_tfft_ms": sum(b.time_to_first_token_ms for b in recent) / len(recent),
            "avg_memory_mb": sum(b.memory_used_mb for b in recent) / len(recent),
            "avg_cpu_percent": sum(b.cpu_usage_percent for b in recent) / len(recent),
            "avg_latency_ms": sum(l.total_latency_ms for l in self._latencies)
            / max(len(self._latencies), 1),
            "within_3s_rate": sum(1 for l in self._latencies if l.is_within_target())
            / max(len(self._latencies), 1),
        }

    def get_benchmarks(self, limit: int = 10) -> List[Dict]:
        """Get recent benchmarks"""
        return [b.to_dict() for b in list(self._benchmarks)[-limit:]]


class ThermalMonitor:
    """
    Thermal throttling awareness
    """

    def __init__(self):
        self._thermal_file = "/sys/class/thermal/thermal_zone0/temp"
        self._last_check = 0
        self._current_temp = 0.0
        self._check_interval = 5  # seconds

    def get_thermal_state(self) -> str:
        """Get current thermal state"""
        current_time = time.time()
        if current_time - self._last_check < self._check_interval:
            return self._get_state_from_temp(self._current_temp)

        try:
            if Path(self._thermal_file).exists():
                with open(self._thermal_file) as f:
                    temp_millidegrees = int(f.read().strip())
                    self._current_temp = temp_millidegrees / 1000.0
            else:
                # Fallback: estimate from CPU usage
                self._current_temp = self._estimate_temp()
        except:
            self._current_temp = self._estimate_temp()

        self._last_check = current_time
        return self._get_state_from_temp(self._current_temp)

    def _get_state_from_temp(self, temp_celsius: float) -> str:
        """Map temperature to thermal state"""
        if temp_celsius < 40:
            return "nominal"
        elif temp_celsius < 50:
            return "fair"
        elif temp_celsius < 65:
            return "serious"
        else:
            return "critical"

    def _estimate_temp(self) -> float:
        """Estimate temperature from CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            return 35 + (cpu_percent / 100) * 25  # 35-60C range
        except:
            return 40.0

    def should_throttle(self) -> bool:
        """Check if we should throttle due to heat"""
        return self.get_thermal_state() in ["serious", "critical"]

    def get_throttle_factor(self) -> float:
        """Get throttling factor based on thermal state"""
        state = self.get_thermal_state()
        factors = {
            "nominal": 1.0,
            "fair": 0.85,
            "serious": 0.6,
            "critical": 0.3,
        }
        return factors.get(state, 1.0)


class StreamingGenerator:
    """
    Streaming token generator for faster perceived response
    """

    def __init__(self, backend=None):
        self.backend = backend

    async def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 100,
        callback: Optional[Callable[[str], None]] = None,
    ) -> AsyncIterator[str]:
        """
        Generate tokens streaming for faster perceived response

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            callback: Optional callback for each token

        Yields:
            Generated tokens one at a time
        """
        if not self.backend:
            return

        try:
            # For llama.cpp, we can use streaming
            if hasattr(self.backend, "create_chat_completion"):
                # OpenAI-compatible interface
                response = await asyncio.to_thread(
                    self.backend.create_chat_completion,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    stream=True,
                )

                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        if callback:
                            callback(token)
                        yield token
            else:
                # Fallback: non-streaming
                result = await asyncio.to_thread(
                    self.backend,
                    prompt,
                    max_tokens=max_tokens,
                    stream=False,
                )
                text = result.get("choices", [{}])[0].get("text", "")
                if callback:
                    callback(text)
                yield text

        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield f"[Error: {str(e)}]"


class ProductionLLM:
    """
    Main production LLM integration class
    Combines all components for production use
    """

    def __init__(self, config_path: Optional[str] = None):
        # Components
        self.registry = ModelRegistry()
        self.recommender = ModelRecommendationEngine(self.registry)
        self.fallback_chain = FallbackChain(self.registry)
        self.performance_monitor = PerformanceMonitor()
        self.thermal_monitor = ThermalMonitor()
        self.streaming_generator = StreamingGenerator()

        # Current state
        self.current_model: Optional[ModelSpec] = None
        self.backend = None
        self.is_loaded = False

        # Device detection
        self.device = self._detect_device()

        # Config
        self.config_path = config_path
        self._load_config()

        logger.info(
            f"Production LLM initialized - Device: {self.device.cpu_cores}c, {self.device.total_ram_mb:.0f}MB RAM"
        )

    def _detect_device(self) -> DeviceCapabilities:
        """Detect device capabilities"""
        device = DeviceCapabilities()

        try:
            # RAM detection
            mem = psutil.virtual_memory()
            device.total_ram_mb = mem.total / (1024 * 1024)
            device.available_ram_mb = mem.available / (1024 * 1024)

            # CPU detection
            device.cpu_cores = psutil.cpu_count(logical=True)

            # Try to get CPU frequency
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    device.cpu_freq_mhz = cpu_freq.current
            except:
                device.cpu_freq_mhz = 2400  # Default assumption

            # Thermal state
            device.thermal_state = self.thermal_monitor.get_thermal_state()

        except Exception as e:
            logger.error(f"Failed to detect device: {e}")
            # Fallback defaults
            device.total_ram_mb = 4096
            device.available_ram_mb = 2048
            device.cpu_cores = 4
            device.cpu_freq_mhz = 2400

        return device

    def _load_config(self):
        """Load configuration"""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path) as f:
                    config = json.load(f)

                # Load custom fallback chains
                if "fallback_chains" in config:
                    for model_type, chain in config["fallback_chains"].items():
                        self.fallback_chain.set_chain(model_type, chain)

                logger.info("Configuration loaded")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")

    def _save_config(self):
        """Save configuration"""
        if not self.config_path:
            return

        config = {
            "fallback_chains": {k: v for k, v in self.fallback_chain._chains.items()},
            "current_model": self.current_model.id if self.current_model else None,
        }

        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    async def load_model(self, model_id: str) -> bool:
        """
        Load a model by ID with fallback support
        """
        model = self.registry.get_model(model_id)
        if not model:
            logger.error(f"Model not found: {model_id}")
            return False

        # Check memory
        ram_needed = model.calculate_ram_required(2048)
        if ram_needed > self.device.available_ram_mb * 0.8:
            logger.warning(
                f"Model {model.name} may not fit: "
                f"needs {ram_needed:.0f}MB, available {self.device.available_ram_mb:.0f}MB"
            )

        # Load model
        success = await self._do_load_model(model)

        if not success:
            # Try fallback
            fallback_id = self.fallback_chain.get_fallback(model_id)
            if fallback_id:
                logger.info(f"Primary model failed, trying fallback: {fallback_id}")
                return await self.load_model(fallback_id)
            return False

        self.current_model = model
        self.is_loaded = True

        # Save config
        self._save_config()

        return True

    async def _do_load_model(self, model: ModelSpec) -> bool:
        """Actually load the model"""
        start_time = time.time()

        try:
            if model.backend == BackendType.LLAMA_CPP:
                return await self._load_llama_cpp(model)
            elif model.backend == BackendType.GPT4ALL:
                return await self._load_gpt4all(model)
            elif model.backend == BackendType.OLLAMA:
                return await self._load_ollama(model)
            elif model.backend == BackendType.TRANSFORMERS:
                return await self._load_transformers(model)
            else:
                logger.error(f"Unsupported backend: {model.backend}")
                return False

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
        finally:
            load_time_ms = (time.time() - start_time) * 1000
            if self.performance_monitor._current_benchmark:
                self.performance_monitor._current_benchmark.load_time_ms = load_time_ms

    async def _load_llama_cpp(self, model: ModelSpec) -> bool:
        """Load using llama.cpp"""
        try:
            from llama_cpp import Llama

            # Check if model file exists
            model_path = Path(model.model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            # Detect thermal throttling
            throttle_factor = self.thermal_monitor.get_throttle_factor()
            n_threads = max(2, int(self.device.cpu_cores * throttle_factor))

            self.backend = Llama(
                model_path=str(model_path),
                n_ctx=min(model.context_length, 4096),  # Cap at 4K for mobile
                n_gpu_layers=0,  # CPU only on Termux
                n_threads=n_threads,
                verbose=False,
            )

            self.streaming_generator.backend = self.backend

            logger.info(f"Loaded {model.name} in {n_threads} threads")
            return True

        except ImportError:
            logger.error("llama-cpp-python not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load llama.cpp: {e}")
            return False

    async def _load_gpt4all(self, model: ModelSpec) -> bool:
        """Load using GPT4All"""
        try:
            from gpt4all import GPT4All

            model_path = Path(model.model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            self.backend = GPT4All(str(model_path))
            logger.info(f"Loaded {model.name} via GPT4All")
            return True

        except ImportError:
            logger.error("gpt4all not installed")
            return False

    async def _load_ollama(self, model: ModelSpec) -> bool:
        """Load using Ollama (server mode)"""
        try:
            import requests

            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if not response.ok:
                logger.error("Ollama server not running")
                return False

            # Just verify connection - Ollama loads models on demand
            self.backend = model.id  # Store model name
            logger.info(f"Connected to Ollama for model {model.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False

    async def _load_transformers(self, model: ModelSpec) -> bool:
        """Load using HuggingFace transformers"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info(f"Loading transformers model: {model.model_path}")

            # Check available memory and set device
            if torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.float16
            else:
                device_map = "cpu"
                torch_dtype = torch.float32
                logger.info("CUDA not available, using CPU")

            self._transformers_model = AutoModelForCausalLM.from_pretrained(
                model.model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,  # Mobile-first: reduce memory
            )
            self._transformers_tokenizer = AutoTokenizer.from_pretrained(
                model.model_path
            )

            self.backend = "transformers"
            logger.info(f"Transformers model loaded: {model.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load transformers model: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text with full latency tracking
        """
        if not self.is_loaded or not self.current_model:
            return {"error": "No model loaded", "text": ""}

        # Start benchmark
        benchmark = self.performance_monitor.start_benchmark(self.current_model.id)
        latency = PipelineLatency()

        try:
            # Thermal check
            if self.thermal_monitor.should_throttle():
                logger.warning("Thermal throttling active - expect slower response")

            start_time = time.time()

            # Generate
            if streaming_callback:
                # Streaming mode
                full_text = ""
                first_token_time = None

                async for token in self.streaming_generator.generate_streaming(
                    prompt, max_tokens, streaming_callback
                ):
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_text += token

                generation_time = (time.time() - start_time) * 1000
                tokens_generated = len(full_text.split())

                latency.llm_time_to_first_token_ms = (
                    (first_token_time - start_time) * 1000 if first_token_time else 0
                )
                latency.llm_token_generation_ms = (
                    generation_time - latency.llm_time_to_first_token_ms
                )

            else:
                # Non-streaming mode
                if self.current_model.backend == BackendType.LLAMA_CPP:
                    result = await asyncio.to_thread(
                        self.backend,
                        prompt,
                        max_tokens=max_tokens,
                        temperature=0.7,
                    )
                    full_text = result.get("choices", [{}])[0].get("text", "").strip()
                elif self.current_model.backend == BackendType.OLLAMA:
                    full_text = await self._generate_ollama(prompt, max_tokens)
                elif self.current_model.backend == BackendType.TRANSFORMERS:
                    full_text = await self._generate_transformers(prompt, max_tokens)
                else:
                    full_text = "[Error: Unsupported backend for generation]"

                generation_time = (time.time() - start_time) * 1000
                tokens_generated = len(full_text.split())

                # Estimate TFFT (Time to First Token)
                tfft_estimate = self.current_model.time_to_first_token_ms
                latency.llm_time_to_first_token_ms = tfft_estimate
                latency.llm_token_generation_ms = generation_time - tfft_estimate

            latency.llm_total_latency_ms = generation_time

            # End benchmark
            self.performance_monitor.end_benchmark(
                tokens_generated=tokens_generated,
                generation_time_ms=generation_time,
                time_to_first_token_ms=latency.llm_time_to_first_token_ms,
            )

            latency.calculate_total()
            self.performance_monitor.record_latency(latency)

            return {
                "text": full_text,
                "model": self.current_model.id,
                "tokens_generated": tokens_generated,
                "latency_ms": latency.total_latency_ms,
                "latency_details": latency.to_dict(),
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {"error": str(e), "text": ""}

    async def _generate_ollama(self, prompt: str, max_tokens: int) -> str:
        """Generate using Ollama API"""
        import requests

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.backend,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7,
                    },
                },
                timeout=120,
            )

            if response.ok:
                return response.json().get("response", "").strip()
            return f"[Error: {response.status_code}]"

        except Exception as e:
            return f"[Error: {str(e)}]"

    async def _generate_transformers(self, prompt: str, max_tokens: int) -> str:
        """Generate using HuggingFace transformers"""
        import torch

        try:

            def _run():
                inputs = self._transformers_tokenizer(prompt, return_tensors="pt").to(
                    self._transformers_model.device
                )

                with torch.no_grad():
                    outputs = self._transformers_model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self._transformers_tokenizer.eos_token_id,
                    )

                return self._transformers_tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )

            result = await asyncio.to_thread(_run)

            # Remove prompt from output (transformers includes it)
            if result.startswith(prompt):
                result = result[len(prompt) :]

            return result.strip()

        except Exception as e:
            return f"[Error: {str(e)}]"

    async def chat(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Chat with the model"""
        # Build prompt
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}")

        if history:
            for msg in history[-10:]:  # Last 10 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}")

        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")

        prompt = "\n".join(prompt_parts)

        return await self.generate(prompt)

    def get_recommendations(self, use_case: str = "general") -> List[Dict]:
        """Get model recommendations for current device"""
        # Update thermal state
        self.device.thermal_state = self.thermal_monitor.get_thermal_state()

        return self.recommender.get_recommendations(
            device=self.device,
            use_case=use_case,
        )

    def get_status(self) -> Dict:
        """Get system status"""
        return {
            "loaded": self.is_loaded,
            "current_model": self.current_model.to_dict()
            if self.current_model
            else None,
            "device": self.device.to_dict(),
            "thermal_state": self.thermal_monitor.get_thermal_state(),
            "performance": self.performance_monitor.get_stats(),
        }

    def add_custom_model(self, model: ModelSpec) -> bool:
        """Add a custom model (user-definable)"""
        return self.registry.add_model(model)

    def unload(self):
        """Unload model to free memory"""
        self.backend = None
        self.is_loaded = False
        self.current_model = None

        # Force garbage collection
        import gc

        gc.collect()

        logger.info("Model unloaded, memory freed")


# Factory function
_production_llm_instance: Optional[ProductionLLM] = None


def get_production_llm(config_path: Optional[str] = None) -> ProductionLLM:
    """Get or create production LLM instance"""
    global _production_llm_instance
    if _production_llm_instance is None:
        _production_llm_instance = ProductionLLM(config_path)
    return _production_llm_instance


# Example usage and testing
if __name__ == "__main__":

    async def test_production_llm():
        """Test the production LLM system"""
        logging.basicConfig(level=logging.INFO)

        # Create instance
        pllm = ProductionLLM()

        # Print device info
        print(f"\n=== Device Capabilities ===")
        print(
            f"RAM: {pllm.device.total_ram_mb:.0f}MB total, {pllm.device.available_ram_mb:.0f}MB available"
        )
        print(f"CPU: {pllm.device.cpu_cores} cores @ {pllm.device.cpu_freq_mhz}MHz")
        print(f"Thermal: {pllm.device.thermal_state}")

        # Get recommendations
        print(f"\n=== Model Recommendations ===")
        recommendations = pllm.get_recommendations(use_case="general")
        for i, rec in enumerate(recommendations):
            print(f"\n{i + 1}. {rec['model']['name']}")
            print(f"   Score: {rec['score']:.1f}")
            print(f"   Est. TPS: {rec['estimated_tps']:.1f}")
            print(f"   Est. Latency: {rec['estimated_latency_ms']:.0f}ms")
            print(f"   RAM Required: {rec['ram_required_mb']:.0f}MB")
            print(f"   Within 3s target: {rec['within_3s_target']}")

        # List available models
        print(f"\n=== Available Models ===")
        for model in pllm.registry.list_models("llm"):
            print(f"- {model.name} ({model.id})")
            print(
                f"  Params: {model.parameters_billion}B, Quant: {model.quantization.value}"
            )
            print(f"  Size: {model.file_size_mb}MB, TPS: {model.tokens_per_second}")

        # Performance stats
        print(f"\n=== Performance Stats ===")
        print(pllm.performance_monitor.get_stats())

    asyncio.run(test_production_llm())
