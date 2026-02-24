"""
Tests for AURA v3 LLM Manager Integration

Tests the LLM manager with real_llm integration
"""

import unittest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm.manager import LLMManager, ModelType, ModelStatus, LLMResponse


class TestLLMManager(unittest.TestCase):
    """Test LLMManager class"""

    def setUp(self):
        """Set up test"""
        self.manager = LLMManager(models_dir="test_models")

    def test_default_values(self):
        """Test default manager values"""
        self.assertEqual(self.manager._default_stt_model, "base")
        self.assertEqual(self.manager._default_tts_model, "en_US-lessac")
        self.assertEqual(self.manager._default_llm_model, "llama-2-7b-chat.Q4_K_M.gguf")
        self.assertEqual(self.manager._max_tokens, 512)
        self.assertEqual(self.manager._temperature, 0.7)

    def test_models_directory_created(self):
        """Test models directory is created"""
        self.assertTrue(self.manager.models_dir.exists())

    def test_get_status(self):
        """Test getting status"""
        status = self.manager.get_status()
        self.assertIn("models_dir", status)
        self.assertIn("loaded_models", status)
        self.assertIn("defaults", status)

    def test_set_generation_params(self):
        """Test setting generation parameters"""
        self.manager.set_generation_params(max_tokens=256, temperature=0.5)
        self.assertEqual(self.manager._max_tokens, 256)
        self.assertEqual(self.manager._temperature, 0.5)

    def test_set_default_models(self):
        """Test setting default models"""
        self.manager.set_default_stt_model("tiny")
        self.manager.set_default_tts_model("en_US-lessac")
        self.manager.set_default_llm_model("test-model.gguf")

        self.assertEqual(self.manager._default_stt_model, "tiny")
        self.assertEqual(self.manager._default_tts_model, "en_US-lessac")
        self.assertEqual(self.manager._default_llm_model, "test-model.gguf")


class TestLLMManagerAsync(unittest.TestCase):
    """Async tests for LLMManager"""

    def setUp(self):
        """Set up test"""
        self.manager = LLMManager(models_dir="test_models")

    def test_load_model_not_found(self):
        """Test loading non-existent model"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            self.manager.load_model("nonexistent.gguf", ModelType.LLM)
        )

        self.assertEqual(result.status, ModelStatus.NOT_FOUND)
        loop.close()

    def test_unload_model(self):
        """Test unloading model"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Should return False for non-loaded model
        result = loop.run_until_complete(self.manager.unload_model("test.gguf"))
        self.assertFalse(result)

        loop.close()

    def test_get_model_status_nonexistent(self):
        """Test getting status of non-existent model"""
        status = self.manager.get_model_status("nonexistent.gguf")
        self.assertIsNone(status)

    def test_list_models_empty(self):
        """Test listing models when directory is empty"""
        models = self.manager.list_models()
        self.assertIsInstance(models, list)


class TestLLMResponse(unittest.TestCase):
    """Test LLMResponse class"""

    def test_create_response(self):
        """Test creating LLM response"""
        response = LLMResponse(
            text="Hello, world!",
            model="test-model",
            tokens_used=10,
            inference_time_ms=500,
        )
        self.assertEqual(response.text, "Hello, world!")
        self.assertEqual(response.model, "test-model")
        self.assertEqual(response.tokens_used, 10)
        self.assertEqual(response.inference_time_ms, 500)
        self.assertIsNone(response.error)

    def test_create_error_response(self):
        """Test creating error response"""
        response = LLMResponse(
            text="",
            model="test-model",
            error="Model not found",
        )
        self.assertEqual(response.text, "")
        self.assertEqual(response.error, "Model not found")


class TestModelType(unittest.TestCase):
    """Test ModelType enum"""

    def test_model_types(self):
        """Test all model types"""
        self.assertEqual(ModelType.STT.value, "speech_to_text")
        self.assertEqual(ModelType.TTS.value, "text_to_speech")
        self.assertEqual(ModelType.LLM.value, "language_model")
        self.assertEqual(ModelType.VISION.value, "vision")


class TestModelStatus(unittest.TestCase):
    """Test ModelStatus enum"""

    def test_model_statuses(self):
        """Test all model statuses"""
        self.assertEqual(ModelStatus.NOT_LOADED.value, "not_loaded")
        self.assertEqual(ModelStatus.LOADING.value, "loading")
        self.assertEqual(ModelStatus.READY.value, "ready")
        self.assertEqual(ModelStatus.ERROR.value, "error")
        self.assertEqual(ModelStatus.NOT_FOUND.value, "not_found")


if __name__ == "__main__":
    unittest.main()
