import os
import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    model_path: str = None
    model_type: str = "llama"
    quantization: str = "q5_k_m"
    max_context: int = 4096
    n_gpu_layers: int = 0
    temperature: float = 0.7
    max_tokens: int = 512


class LLMRunner:
    """Persistent LLM - stays loaded in memory between requests"""

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.model = None
        self.loaded = False
        self._lock = asyncio.Lock()

    async def load_model(self):
        """Load model ONCE at startup - CRITICAL for mobile performance"""
        async with self._lock:
            if self.loaded:
                return

            # Try to import llama_cpp
            try:
                from llama_cpp import Llama

                # Check if model exists
                if not self.config.model_path or not os.path.exists(
                    self.config.model_path
                ):
                    logger.warning(
                        f"Model not found: {self.config.model_path}, using mock"
                    )
                    self.model = None
                    self.loaded = True
                    return

                # Load model with mobile-optimized settings
                self.model = Llama(
                    model_path=self.config.model_path,
                    n_ctx=self.config.max_context,
                    n_gpu_layers=self.config.n_gpu_layers,  # 0 for CPU
                    n_threads=4,  # Mobile CPU optimization
                    n_threads_batch=4,
                    verbose=False,
                    use_mmap=True,  # Memory mapping for mobile
                    use_mlock=False,  # Don't lock in memory on mobile
                )
                self.loaded = True
                logger.info("LLM model loaded successfully")

            except ImportError:
                logger.warning("llama-cpp-python not installed, using mock")
                self.model = None
                self.loaded = True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
                self.loaded = True

    async def generate(
        self, messages: List[Dict], max_tokens: int = None, temperature: float = None
    ) -> Dict:
        """Generate response from conversation"""

        if not self.loaded:
            await self.load_model()

        if self.model is None:
            # Use mock
            return await self._mock_generate(messages)

        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages)

        try:
            # Run in executor to not block
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.model(
                    prompt,
                    max_tokens=max_tokens or self.config.max_tokens,
                    temperature=temperature or self.config.temperature,
                    stop=["</s>", "<|end|>"],
                ),
            )

            return {
                "content": result["choices"][0]["text"],
                "usage": result.get("usage", {}),
                "model": self.config.model_type,
            }
        except Exception as e:
            logger.error(f"Generation error: {e}")
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
        if self.model is None:
            return {"loaded": False, "memory_mb": 0}

        return {
            "loaded": True,
            "model_path": self.config.model_path,
            "context_size": self.config.max_context,
            "quantization": self.config.quantization,
        }

    def is_loaded(self) -> bool:
        return self.loaded

    async def unload_model(self):
        """Unload model from memory"""
        self.model = None
        self.loaded = False
        logger.info("Model unloaded")

    def unload(self):
        """Sync wrapper for unload_model"""
        # For sync context, just set loaded to False
        self.loaded = False
        logger.info("Model unload requested")


# Export for easy use
__all__ = ["LLMRunner", "LLMConfig", "MockLLM"]


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
