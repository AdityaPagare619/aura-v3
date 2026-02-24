"""
Tests for AURA v3 Inner Voice Stream UI System
"""

import unittest
import asyncio
import os
import sys
import tempfile

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


class TestInnerVoiceStream(unittest.TestCase):
    """Test InnerVoiceStream class"""

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

    @asyncio.coroutine
    def test_initialize(self):
        """Test initialization"""
        yield from self.stream.initialize()
        self.assertIsNotNone(self.stream.store)

    @asyncio.coroutine
    def test_generate_thought(self):
        """Test generating a thought"""
        thought = yield from self.stream.generate_thought(
            content="User seems happy today",
            category=ThoughtCategory.OBSERVATION,
            reasoning_steps=["User's tone was positive", "They laughed"],
            evidence=["Tone analysis: positive"],
            confidence=0.8,
            tone=ThoughtTone.CURIOUS,
        )
        self.assertIsNotNone(thought)
        self.assertEqual(thought.content, "User seems happy today")
        self.assertEqual(thought.category, ThoughtCategory.OBSERVATION)

    @asyncio.coroutine
    def test_explain_action(self):
        """Test action explanation"""
        explanation = yield from self.stream.explain_action(
            action="set_reminder",
            human_reasoning="you always forget this kind of thing",
            reasoning_steps=[
                "User mentioned deadline",
                "Past pattern shows forgetfulness",
            ],
            evidence=["User said 'don't forget'"],
        )
        self.assertIsNotNone(explanation)
        self.assertEqual(explanation.action, "set_reminder")
        self.assertIn("because", explanation.human_reasoning)

    @asyncio.coroutine
    def test_get_visible_thoughts(self):
        """Test getting visible thoughts"""
        # Add some thoughts
        for i in range(5):
            yield from self.stream.generate_thought(
                content=f"Thought {i}",
                category=ThoughtCategory.REASONING,
                reasoning_steps=["Test"],
                evidence=["Test"],
            )

        thoughts = self.stream.get_visible_thoughts(count=3)
        self.assertEqual(len(thoughts), 3)

    @asyncio.coroutine
    def test_correct_thought(self):
        """Test correcting a thought"""
        thought = yield from self.stream.generate_thought(
            content="Test thought",
            category=ThoughtCategory.OBSERVATION,
            reasoning_steps=["Test"],
            evidence=["Test"],
        )

        corrected = yield from self.stream.correct_thought(
            thought_id=thought.id, correction="You're wrong about this"
        )

        self.assertIsNotNone(corrected)
        self.assertTrue(corrected.user_corrected)
        self.assertEqual(corrected.correction_count, 1)

    @asyncio.coroutine
    def test_confirm_thought(self):
        """Test confirming a thought"""
        thought = yield from self.stream.generate_thought(
            content="Test thought",
            category=ThoughtCategory.OBSERVATION,
            reasoning_steps=["Test"],
            evidence=["Test"],
        )

        confirmed = yield from self.stream.confirm_thought(thought.id)

        self.assertIsNotNone(confirmed)
        self.assertEqual(confirmed.confirmation_count, 1)

    @asyncio.coroutine
    def test_update_settings(self):
        """Test updating settings"""
        settings = yield from self.stream.update_settings(
            always_show_why=True, max_visible_thoughts=10
        )

        self.assertTrue(settings.always_show_why)
        self.assertEqual(settings.max_visible_thoughts, 10)

    @asyncio.coroutine
    def test_context_thought_generation(self):
        """Test context-aware thought generation"""
        thought = yield from self.stream.generate_context_thought(
            context_type="reminder",
            observations=["you mentioned this before"],
            user_patterns={"forget_rate": 0.7},
        )

        self.assertIsNotNone(thought)
        self.assertIn("reminder", thought.related_action or "")

    @asyncio.coroutine
    def test_get_thought_snippets(self):
        """Test getting formatted snippets"""
        # Add thoughts
        for i in range(3):
            yield from self.stream.generate_thought(
                content=f"Thought {i}",
                category=ThoughtCategory.OBSERVATION,
                reasoning_steps=["Step 1"],
                evidence=["Evidence 1"],
                confidence=0.6,
            )

        snippets = self.stream.get_thought_snippets(count=2)

        self.assertEqual(len(snippets), 2)
        self.assertIn("content", snippets[0])
        self.assertIn("category", snippets[0])

    @asyncio.coroutine
    def test_statistics(self):
        """Test getting statistics"""
        # Add some thoughts
        for i in range(3):
            yield from self.stream.generate_thought(
                content=f"Thought {i}",
                category=ThoughtCategory.REASONING,
                reasoning_steps=["Test"],
                evidence=["Test"],
            )

        stats = self.stream.get_statistics()

        self.assertEqual(stats["total_thoughts"], 3)
        self.assertIn("settings", stats)

    @asyncio.coroutine
    def test_hidden_mode(self):
        """Test hidden mode returns no thoughts"""
        yield from self.stream.update_settings(is_visible=False)

        # Add thoughts
        for i in range(3):
            yield from self.stream.generate_thought(
                content=f"Thought {i}",
                category=ThoughtCategory.REASONING,
                reasoning_steps=["Test"],
                evidence=["Test"],
            )

        thoughts = self.stream.get_visible_thoughts()
        self.assertEqual(len(thoughts), 0)


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
