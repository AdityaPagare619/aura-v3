"""
Tests for AURA v3 Inner Voice Stream UI System
"""

import unittest
import asyncio
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ui.inner_voice import (
    InnerVoiceStream,
    InnerVoiceSettings,
    ThoughtCategory,
    ThoughtTone,
    ReasoningChain,
    get_inner_voice_stream,
)


class TestInnerVoiceSettings(unittest.TestCase):
    """Test InnerVoiceSettings"""

    def test_default_values(self):
        """Test default settings"""
        settings = InnerVoiceSettings()
        self.assertTrue(settings.is_visible)
        self.assertFalse(settings.always_show_why)
        self.assertTrue(settings.show_doubts)
        self.assertEqual(settings.max_visible_thoughts, 15)
        self.assertTrue(settings.show_confidence)

    def test_custom_values(self):
        """Test custom settings"""
        settings = InnerVoiceSettings(
            is_visible=False, always_show_why=True, max_visible_thoughts=20
        )
        self.assertFalse(settings.is_visible)
        self.assertTrue(settings.always_show_why)
        self.assertEqual(settings.max_visible_thoughts, 20)


class TestReasoningChain(unittest.TestCase):
    """Test ReasoningChain"""

    def test_create_chain(self):
        """Test creating reasoning chain"""
        chain = ReasoningChain(
            steps=["Step 1", "Step 2"],
            evidence=["Evidence 1"],
            confidence=0.7,
            sources=["source1"],
        )
        self.assertEqual(len(chain.steps), 2)
        self.assertEqual(chain.confidence, 0.7)


class TestThoughtCategory(unittest.TestCase):
    """Test ThoughtCategory enum"""

    def test_categories(self):
        """Test all categories exist"""
        categories = [
            ThoughtCategory.OBSERVATION,
            ThoughtCategory.REASONING,
            ThoughtCategory.DOUBT,
            ThoughtCategory.GOAL,
            ThoughtCategory.CONCERN,
            ThoughtCategory.REFLECTION,
            ThoughtCategory.DECISION,
            ThoughtCategory.LEARNING,
        ]
        self.assertEqual(len(categories), 8)


class TestThoughtTone(unittest.TestCase):
    """Test ThoughtTone enum"""

    def test_tones(self):
        """Test all tones exist"""
        tones = [
            ThoughtTone.CURIOUS,
            ThoughtTone.CONFIDENT,
            ThoughtTone.UNCERTAIN,
            ThoughtTone.HESITANT,
            ThoughtTone.EXCITED,
            ThoughtTone.CONCERNED,
            ThoughtTone.HOPEFUL,
        ]
        self.assertEqual(len(tones), 7)


# Synchronous tests for InnerVoiceStream (using unittest.TestCase)
class TestInnerVoiceStreamSync(unittest.TestCase):
    """Synchronous tests for InnerVoiceStream class"""

    def setUp(self):
        """Set up test with temp storage"""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()
        self.stream = InnerVoiceStream(storage_path=self.temp_file.name)

    def tearDown(self):
        """Clean up temp file"""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass

    def test_default_state(self):
        """Test default state"""
        self.assertEqual(len(self.stream.thoughts), 0)
        self.assertTrue(self.stream.settings.is_visible)


# Pytest fixture for async tests
@pytest.fixture
def inner_voice_stream():
    """Create InnerVoiceStream instance for async tests"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    temp_file.close()
    stream = InnerVoiceStream(storage_path=temp_file.name)
    yield stream
    try:
        os.unlink(temp_file.name)
    except:
        pass


# Async tests using pytest-native class (NOT unittest.TestCase)
class TestInnerVoiceStreamAsync:
    """Async tests for InnerVoiceStream - pytest-native class"""

    @pytest.mark.asyncio
    async def test_initialize(self, inner_voice_stream):
        """Test initialization"""
        await inner_voice_stream.initialize()
        assert inner_voice_stream.store is not None

    @pytest.mark.asyncio
    async def test_generate_thought(self, inner_voice_stream):
        """Test generating a thought"""
        thought = await inner_voice_stream.generate_thought(
            content="User seems happy today",
            category=ThoughtCategory.OBSERVATION,
            reasoning_steps=["User's tone was positive", "They laughed"],
            evidence=["Tone analysis: positive"],
            confidence=0.8,
            tone=ThoughtTone.CURIOUS,
        )
        assert thought is not None
        assert thought.content == "User seems happy today"
        assert thought.category == ThoughtCategory.OBSERVATION

    @pytest.mark.asyncio
    async def test_explain_action(self, inner_voice_stream):
        """Test action explanation"""
        explanation = await inner_voice_stream.explain_action(
            action="set_reminder",
            human_reasoning="because you always forget this kind of thing",
            reasoning_steps=[
                "User mentioned deadline",
                "Past pattern shows forgetfulness",
            ],
            evidence=["User said 'don't forget'"],
        )
        assert explanation is not None
        assert explanation.action == "set_reminder"
        assert "because" in explanation.human_reasoning

    @pytest.mark.asyncio
    async def test_get_visible_thoughts(self, inner_voice_stream):
        """Test getting visible thoughts"""
        # Add some thoughts
        for i in range(5):
            await inner_voice_stream.generate_thought(
                content=f"Thought {i}",
                category=ThoughtCategory.REASONING,
                reasoning_steps=["Test"],
                evidence=["Test"],
            )

        thoughts = inner_voice_stream.get_visible_thoughts(count=3)
        assert len(thoughts) == 3

    @pytest.mark.asyncio
    async def test_correct_thought(self, inner_voice_stream):
        """Test correcting a thought"""
        thought = await inner_voice_stream.generate_thought(
            content="Test thought",
            category=ThoughtCategory.OBSERVATION,
            reasoning_steps=["Test"],
            evidence=["Test"],
        )

        corrected = await inner_voice_stream.correct_thought(
            thought_id=thought.id, correction="You're wrong about this"
        )

        assert corrected is not None
        assert corrected.user_corrected is True
        assert corrected.correction_count == 1

    @pytest.mark.asyncio
    async def test_confirm_thought(self, inner_voice_stream):
        """Test confirming a thought"""
        thought = await inner_voice_stream.generate_thought(
            content="Test thought",
            category=ThoughtCategory.OBSERVATION,
            reasoning_steps=["Test"],
            evidence=["Test"],
        )

        confirmed = await inner_voice_stream.confirm_thought(thought.id)

        assert confirmed is not None
        assert confirmed.confirmation_count == 1

    @pytest.mark.asyncio
    async def test_update_settings(self, inner_voice_stream):
        """Test updating settings"""
        settings = await inner_voice_stream.update_settings(
            always_show_why=True, max_visible_thoughts=10
        )

        assert settings.always_show_why is True
        assert settings.max_visible_thoughts == 10

    @pytest.mark.asyncio
    async def test_context_thought_generation(self, inner_voice_stream):
        """Test context-aware thought generation"""
        thought = await inner_voice_stream.generate_context_thought(
            context_type="reminder",
            observations=["you mentioned this before"],
            user_patterns={"forget_rate": 0.7},
        )

        assert thought is not None
        assert "reminder" in (thought.related_action or "")

    @pytest.mark.asyncio
    async def test_get_thought_snippets(self, inner_voice_stream):
        """Test getting formatted snippets"""
        # Add thoughts
        for i in range(3):
            await inner_voice_stream.generate_thought(
                content=f"Thought {i}",
                category=ThoughtCategory.OBSERVATION,
                reasoning_steps=["Step 1"],
                evidence=["Evidence 1"],
                confidence=0.6,
            )

        snippets = inner_voice_stream.get_thought_snippets(count=2)

        assert len(snippets) == 2
        assert "content" in snippets[0]
        assert "category" in snippets[0]

    @pytest.mark.asyncio
    async def test_statistics(self, inner_voice_stream):
        """Test getting statistics"""
        # Add some thoughts
        for i in range(3):
            await inner_voice_stream.generate_thought(
                content=f"Thought {i}",
                category=ThoughtCategory.REASONING,
                reasoning_steps=["Test"],
                evidence=["Test"],
            )

        stats = inner_voice_stream.get_statistics()

        assert stats["total_thoughts"] == 3
        assert "settings" in stats

    @pytest.mark.asyncio
    async def test_hidden_mode(self, inner_voice_stream):
        """Test hidden mode returns no thoughts"""
        await inner_voice_stream.update_settings(is_visible=False)

        # Add thoughts
        for i in range(3):
            await inner_voice_stream.generate_thought(
                content=f"Thought {i}",
                category=ThoughtCategory.REASONING,
                reasoning_steps=["Test"],
                evidence=["Test"],
            )

        thoughts = inner_voice_stream.get_visible_thoughts()
        assert len(thoughts) == 0


class TestInnerVoiceIntegration(unittest.TestCase):
    """Integration tests"""

    def setUp(self):
        """Set up test with temp storage"""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()

    def tearDown(self):
        """Clean up"""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass

    def test_get_inner_voice_stream_singleton(self):
        """Test singleton pattern"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Test with direct instantiation since singleton doesn't accept params
        from src.ui.inner_voice import _inner_voice_stream
        import src.ui.inner_voice as inner_voice_module

        # Reset global for test
        inner_voice_module._inner_voice_stream = None

        stream1 = loop.run_until_complete(get_inner_voice_stream())
        stream2 = loop.run_until_complete(get_inner_voice_stream())

        # Should be same instance
        self.assertIs(stream1, stream2)

        loop.close()


if __name__ == "__main__":
    unittest.main()
