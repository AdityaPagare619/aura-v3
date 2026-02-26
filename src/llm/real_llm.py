"""
AURA v3 Real LLM Integration

DEPRECATED: This module is deprecated as of Wave 3 Task 3.1 (LLM Unification).
Use ProductionLLM from src.llm.production_llm instead.

The functionality has been absorbed into ProductionLLM, which now supports
all backends: LLAMA_CPP, GPT4ALL, OLLAMA, and TRANSFORMERS.

This file is kept for backward compatibility but will be removed in a future version.
"""

import warnings

warnings.warn(
    "real_llm.py is deprecated. Use ProductionLLM from src.llm.production_llm instead.",
    DeprecationWarning,
    stacklevel=2,
)

import asyncio
import logging
import os
import subprocess
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """LLM backend types"""

    LLAMA_CPP = "llama_cpp"  # llama.cpp via python bindings
    GPT4ALL = "gpt4all"  # gpt4all
    OLLAMA = "ollama"  # Ollama server
    TRANSFORMERS = "transformers"  # HuggingFace transformers


@dataclass
class LLMConfig:
    """Real LLM configuration"""

    model_path: str
    backend: BackendType
    n_ctx: int = 2048  # Context window
    n_gpu_layers: int = 0  # GPU layers (for llama.cpp)
    n_threads: int = 4  # CPU threads
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    max_tokens: int = 512


class RealLLMIntegration:
    """
    Real LLM integration that actually generates text.
    Supports multiple backends.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config
        self.backend = None
        self.model = None
        self.is_loaded = False

    async def load(self, config: LLMConfig) -> bool:
        """Load the model"""
        self.config = config

        try:
            if config.backend == BackendType.LLAMA_CPP:
                return await self._load_llama_cpp(config)
            elif config.backend == BackendType.GPT4ALL:
                return await self._load_gpt4all(config)
            elif config.backend == BackendType.OLLAMA:
                return await self._load_ollama(config)
            elif config.backend == BackendType.TRANSFORMERS:
                return await self._load_transformers(config)
            else:
                logger.error(f"Unknown backend: {config.backend}")
                return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    async def _load_llama_cpp(self, config: LLMConfig) -> bool:
        """Load using llama-cpp-python"""
        try:
            from llama_cpp import Llama

            logger.info(f"Loading llama.cpp model: {config.model_path}")

            self.backend = Llama(
                model_path=config.model_path,
                n_ctx=config.n_ctx,
                n_gpu_layers=config.n_gpu_layers,
                n_threads=config.n_threads,
                verbose=False,
            )

            self.is_loaded = True
            logger.info("Model loaded successfully")
            return True

        except ImportError:
            logger.error(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load llama.cpp: {e}")
            return False

    async def _load_gpt4all(self, config: LLMConfig) -> bool:
        """Load using gpt4all"""
        try:
            from gpt4all import GPT4All

            logger.info(f"Loading GPT4All model: {config.model_path}")

            self.backend = GPT4All(config.model_path)
            self.is_loaded = True
            logger.info("GPT4All model loaded")
            return True

        except ImportError:
            logger.error("gpt4all not installed. Run: pip install gpt4all")
            return False

    async def _load_ollama(self, config: LLMConfig) -> bool:
        """Connect to Ollama server"""
        # Ollama runs as a server, just verify connection
        try:
            import requests

            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.ok:
                self.is_loaded = True
                logger.info("Connected to Ollama server")
                return True
        except:
            pass
        logger.warning("Ollama not running. Start with: ollama serve")
        return False

    async def _load_transformers(self, config: LLMConfig) -> bool:
        """Load using HuggingFace transformers"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading transformers model: {config.model_path}")

            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path, torch_dtype="auto", device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

            self.is_loaded = True
            logger.info("Transformers model loaded")
            return True

        except Exception as e:
            logger.error(f"Failed to load transformers: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate text"""

        if not self.is_loaded:
            return "[Error: No model loaded]"

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        try:
            if self.config.backend == BackendType.LLAMA_CPP:
                return await self._generate_llama_cpp(
                    prompt, max_tokens, temperature, stop
                )
            elif self.config.backend == BackendType.GPT4ALL:
                return await self._generate_gpt4all(prompt, max_tokens, temperature)
            elif self.config.backend == BackendType.OLLAMA:
                return await self._generate_ollama(prompt, max_tokens, temperature)
            elif self.config.backend == BackendType.TRANSFORMERS:
                return await self._generate_transformers(
                    prompt, max_tokens, temperature
                )
            else:
                return "[Error: Unknown backend]"
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"[Error: {str(e)}]"

    async def _generate_llama_cpp(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]],
    ) -> str:
        """Generate using llama.cpp"""

        # Run in thread to avoid blocking
        def _run():
            return self.backend(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or [],
                echo=False,
            )

        result = await asyncio.to_thread(_run)
        return result["choices"][0]["text"].strip()

    async def _generate_gpt4all(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """Generate using GPT4All"""

        def _run():
            return self.backend.generate(
                prompt, max_tokens=max_tokens, temp=temperature
            )

        result = await asyncio.to_thread(_run)
        return result.strip()

    async def _generate_ollama(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """Generate using Ollama"""
        import requests

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2",  # Default model
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            },
            timeout=120,
        )

        if response.ok:
            return response.json().get("response", "").strip()
        return f"[Error: {response.status_code}]"

    async def _generate_transformers(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """Generate using transformers"""

        def _run():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        result = await asyncio.to_thread(_run)
        # Remove prompt from output
        if result.startswith(prompt):
            result = result[len(prompt) :]
        return result.strip()

    async def chat(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Chat with the model"""

        # Build prompt from history
        prompt = ""

        if system_prompt:
            prompt += f"System: {system_prompt}\n"

        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"{role.capitalize()}: {content}\n"

        prompt += f"User: {message}\nAssistant:"

        return await self.generate(prompt)

    def unload(self):
        """Unload model to free memory"""
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        import gc

        gc.collect()


# Default configuration for different device tiers
def get_default_config(device_tier: str = "medium") -> LLMConfig:
    """Get default config for device tier"""

    configs = {
        "low": LLMConfig(
            model_path="models/phi-2.Q4_K_M.gguf",
            backend=BackendType.LLAMA_CPP,
            n_ctx=1024,
            n_gpu_layers=0,
            n_threads=2,
            max_tokens=256,
        ),
        "medium": LLMConfig(
            model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
            backend=BackendType.LLAMA_CPP,
            n_ctx=2048,
            n_gpu_layers=24,
            n_threads=4,
            max_tokens=512,
        ),
        "high": LLMConfig(
            model_path="models/codellama-13b-instruct.Q4_K_M.gguf",
            backend=BackendType.LLAMA_CPP,
            n_ctx=4096,
            n_gpu_layers=32,
            n_threads=6,
            max_tokens=1024,
        ),
    }

    return configs.get(device_tier, configs["medium"])


# Factory
def create_llm(config: Optional[LLMConfig] = None) -> RealLLMIntegration:
    """Create LLM integration"""
    return RealLLMIntegration(config)
