"""
LLM Module - Unified LLM interface for AURA v3

This module provides a unified interface to various LLM backends.
The canonical implementation is ProductionLLM (production_llm.py).

LLMRunner and LLMConfig are preserved for backward compatibility.
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM - preserved for backward compatibility"""

    model_path: str = None
    model_type: str = "llama"
    quantization: str = "q5_k_m"
    max_context: int = 4096
    n_gpu_layers: int = 0
    temperature: float = 0.7
    max_tokens: int = 512


class LLMRunner:
    """
    Persistent LLM - stays loaded in memory between requests.

    This is a backward-compatible wrapper around ProductionLLM.
    Interface preserved: generate(messages) -> {"content": str, "usage": dict, "model": str}
    """

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self._production_llm = None
        self._lock = asyncio.Lock()
        self.loaded = False

    async def _ensure_production_llm(self):
        """Lazy-load ProductionLLM"""
        if self._production_llm is None:
            from .production_llm import ProductionLLM

            self._production_llm = ProductionLLM()

    async def load_model(self):
        """Load model ONCE at startup - CRITICAL for mobile performance"""
        async with self._lock:
            if self.loaded:
                return

            await self._ensure_production_llm()

            # If config has a model path, try to load it
            if self.config.model_path and os.path.exists(self.config.model_path):
                try:
                    # Register model with ProductionLLM
                    from .production_llm import BackendType

                    success = await self._production_llm.load_model(
                        model_id=self.config.model_type,
                        model_path=self.config.model_path,
                    )

                    if success:
                        self.loaded = True
                        logger.info("LLM model loaded via ProductionLLM")
                        return
                except Exception as e:
                    logger.error(f"Failed to load model via ProductionLLM: {e}")

            # Fallback: mark as loaded but with no model (mock mode)
            logger.warning(f"Model not found: {self.config.model_path}, using mock")
            self.loaded = True

    async def generate(
        self, messages: List[Dict], max_tokens: int = None, temperature: float = None
    ) -> Dict:
        """Generate response from conversation"""

        if not self.loaded:
            await self.load_model()

        await self._ensure_production_llm()

        # Check if ProductionLLM has a model loaded
        if self._production_llm.is_loaded:
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)

            try:
                result = await self._production_llm.generate(
                    prompt=prompt,
                    max_tokens=max_tokens or self.config.max_tokens,
                )

                return {
                    "content": result.get("text", ""),
                    "usage": {
                        "tokens_generated": result.get("tokens_generated", 0),
                        "latency_ms": result.get("latency_ms", 0),
                    },
                    "model": result.get("model", self.config.model_type),
                }
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return await self._mock_generate(messages)
        else:
            # Use mock
            return await self._mock_generate(messages)

    async def generate_with_tools(
        self, messages: List[Dict], tool_schemas: str = None
    ) -> Dict:
        """Generate with tool awareness - returns tool call or response"""

        if not self.loaded:
            await self.load_model()

        # Add tool schemas to system message if provided
        if tool_schemas:
            # Find system message and enhance it
            for msg in messages:
                if msg.get("role") == "system":
                    msg["content"] += f"\n\n{tool_schemas}"
                    break

        # Generate
        result = await self.generate(messages)

        # Try to parse as tool call
        content = result.get("content", "")

        if '"type": "tool_call"' in content or '"tool":' in content:
            # Return as tool call format
            return {"type": "tool_call", "content": content}
        else:
            return {"type": "response", "content": content}

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages to prompt format"""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

        prompt += "Assistant: "
        return prompt

    async def _mock_generate(self, messages: List[Dict]) -> Dict:
        """Mock generation for testing without model"""
        await asyncio.sleep(0.1)  # Simulate processing

        # Simple response based on last message
        last_msg = messages[-1].get("content", "") if messages else ""

        responses = [
            "I understand. Let me help you with that.",
            "I'll take care of it right away.",
            "Got it! I'll execute that action for you.",
            "I'm analyzing your request and will respond shortly.",
        ]

        # Check for tool call simulation
        if "tool" in last_msg.lower() or "call" in last_msg.lower():
            return {
                "type": "tool_call",
                "content": '{"type": "tool_call", "tool": "mock_tool", "parameters": {}}',
            }

        import random

        return {
            "content": random.choice(responses),
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "model": "mock",
        }

    def get_memory_usage(self) -> Dict:
        """Return current memory stats"""
        if self._production_llm and self._production_llm.is_loaded:
            return {
                "loaded": True,
                "model_path": self.config.model_path,
                "context_size": self.config.max_context,
                "quantization": self.config.quantization,
            }

        return {"loaded": False, "memory_mb": 0}

    def is_loaded(self) -> bool:
        return self.loaded

    async def unload_model(self):
        """Unload model from memory"""
        if self._production_llm:
            await self._production_llm.unload_model()
        self.loaded = False
        logger.info("Model unloaded")

    def unload(self):
        """Sync wrapper for unload_model"""
        self.loaded = False
        logger.info("Model unload requested")


class MockLLM(LLMRunner):
    """Mock LLM for testing - always uses mock generation"""

    def __init__(self):
        # Initialize with default config but no real model
        super().__init__(LLMConfig())
        self.loaded = True  # Pretend loaded for mock

    async def load_model(self):
        """Mock doesn't need to load"""
        self.loaded = True

    async def generate(self, messages, max_tokens=None, temperature=None):
        """Always use mock generation"""
        return await self._mock_generate(messages)

    async def unload_model(self):
        """Mock unload - nothing to do"""
        self.loaded = False
        logger.info("Mock model unloaded")

    def unload(self):
        """Sync wrapper for unload"""
        self.loaded = False


# Re-export get_llm_manager from manager.py for backward compatibility
def get_llm_manager():
    """Get the global LLM manager instance"""
    from .manager import get_llm_manager as _get_llm_manager

    return _get_llm_manager()


# Export for easy use
__all__ = ["LLMRunner", "LLMConfig", "MockLLM", "get_llm_manager"]
