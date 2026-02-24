"""
Tests for AURA v3 Inner Voice System

Tests the UX features that make AURA feel like a living companion:
- Inner voice stream
- Feelings & Trust Meter
- Character sheets
- Thought bubbles
"""

import unittest
from datetime import datetime
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.services.inner_voice import (
    InnerVoiceSystem,
    AuraFeeling,
    TrustLevel,
    InnerThought,
    ThoughtBubble,
    TrustMeter,
    CharacterSheet,
    get_inner_voice,
)


class TestAuraFeeling(unittest.TestCase):
    """Test AuraFeeling enum"""

    def test_feeling_values(self):
        """Test all feeling values exist"""
        feelings = [
            AuraFeeling.CURIOUS,
            AuraFeeling.FOCUSED,
            AuraFeeling.CONFIDENT,
            AuraFeeling.UNCERTAIN,
            AuraFeeling.CONCERNED,
            AuraFeeling.HAPPY,
            AuraFeeling.EXCITED,
            AuraFeeling.CALM,
            AuraFeeling.TIRED,
            AuraFeeling.HOPEFUL,
        ]
        self.assertGreater(len(feelings), 5)


class TestTrustLevel(unittest.TestCase):
    """Test TrustLevel enum"""

    def test_trust_levels(self):
        """Test all trust levels"""
        levels = [
            TrustLevel.LEARNING,
            TrustLevel.GETTING_KNOW,
            TrustLevel.COMFORTABLE,
            TrustLevel.DEEP_BOND,
            TrustLevel.PARTNER,
        ]
        self.assertEqual(len(levels), 5)


class TestTrustMeter(unittest.TestCase):
    """Test TrustMeter class"""

    def test_default_values(self):
        """Test default trust meter values"""
        tm = TrustMeter()
        self.assertEqual(tm.understanding_work, 0.0)
        self.assertEqual(tm.understanding_mood, 0.0)
        self.assertEqual(tm.understanding_goals, 0.0)
        self.assertEqual(tm.trust_level, TrustLevel.LEARNING)

    def test_get_overall(self):
        """Test overall calculation"""
        tm = TrustMeter()
        tm.understanding_work = 8.0
        tm.understanding_mood = 6.0
        tm.understanding_goals = 7.0
        tm.understanding_relationships = 5.0

        overall = tm.get_overall()
        expected = 8.0 * 0.3 + 6.0 * 0.25 + 7.0 * 0.25 + 5.0 * 0.2
        self.assertAlmostEqual(overall, expected, places=1)

    def test_get_trust_description_learning(self):
        """Test trust description for learning level"""
        tm = TrustMeter()
        tm.understanding_work = 1.0
        tm.understanding_mood = 1.0
        tm.understanding_goals = 1.0
        tm.understanding_relationships = 1.0

        desc = tm.get_trust_description()
        self.assertIn("learning", desc.lower())

    def test_get_trust_description_deep_bond(self):
        """Test trust description for deep bond"""
        tm = TrustMeter()
        tm.understanding_work = 9.0
        tm.understanding_mood = 9.0
        tm.understanding_goals = 9.0
        tm.understanding_relationships = 9.0

        desc = tm.get_trust_description()
        self.assertIn("really well", desc.lower())


class TestCharacterSheet(unittest.TestCase):
    """Test CharacterSheet class"""

    def test_default_values(self):
        """Test default character sheet"""
        cs = CharacterSheet()
        self.assertEqual(cs.user_traits, [])
        self.assertEqual(cs.user_goals, [])
        self.assertEqual(cs.days_known, 0)
        self.assertEqual(cs.total_interactions, 0)
        self.assertEqual(cs.successful_tasks, 0)
        self.assertEqual(cs.failed_tasks, 0)

    def test_add_user_goal(self):
        """Test adding user goal"""
        cs = CharacterSheet()
        cs.user_goals.append(
            {
                "goal": "Learn Python",
                "progress": 0.5,
                "added_at": "2026-02-22",
            }
        )
        self.assertEqual(len(cs.user_goals), 1)
        self.assertEqual(cs.user_goals[0]["goal"], "Learn Python")


class TestInnerThought(unittest.TestCase):
    """Test InnerThought class"""

    def test_create_thought(self):
        """Test creating an inner thought"""
        thought = InnerThought(
            id="test_1",
            content="User seems happy",
            reasoning="Positive tone detected",
            confidence=0.8,
            timestamp=datetime.now(),
        )
        self.assertEqual(thought.content, "User seems happy")
        self.assertEqual(thought.confidence, 0.8)


class TestThoughtBubble(unittest.TestCase):
    """Test ThoughtBubble class"""

    def test_create_bubble(self):
        """Test creating a thought bubble"""
        bubble = ThoughtBubble(
            id="bubble_1",
            content="I think user is stressed",
            category="observation",
            is_sensitive=False,
            timestamp=datetime.now(),
        )
        self.assertEqual(bubble.content, "I think user is stressed")
        self.assertFalse(bubble.is_revealed)
        self.assertIsNone(bubble.user_response)

    def test_sensitive_bubble(self):
        """Test sensitive bubble"""
        bubble = ThoughtBubble(
            id="bubble_2",
            content="Secret observation",
            category="concern",
            is_sensitive=True,
            timestamp=datetime.now(),
        )
        self.assertTrue(bubble.is_sensitive)


class TestInnerVoiceSystem(unittest.TestCase):
    """Test InnerVoiceSystem class"""

    def setUp(self):
        """Set up test"""
        self.system = InnerVoiceSystem()

    def test_default_state(self):
        """Test default system state"""
        self.assertEqual(len(self.system.thought_stream), 0)
        self.assertEqual(len(self.system.thought_bubbles), 0)
        self.assertEqual(self.system.feeling_state.primary, AuraFeeling.CALM)

    def test_add_thought(self):
        """Test adding a thought"""
        thought = asyncio.run(
            self.system.add_thought(
                content="Testing thought",
                reasoning="Unit test",
                confidence=0.9,
            )
        )
        self.assertIsNotNone(thought)
        self.assertEqual(thought.content, "Testing thought")
        self.assertEqual(len(self.system.thought_stream), 1)

    def test_get_recent_thoughts(self):
        """Test getting recent thoughts"""
        # Add multiple thoughts
        for i in range(15):
            asyncio.run(
                self.system.add_thought(
                    content=f"Thought {i}",
                    reasoning="Test",
                )
            )

        recent = self.system.get_recent_thoughts(10)
        self.assertEqual(len(recent), 10)

    def test_create_thought_bubble(self):
        """Test creating thought bubble"""
        bubble = asyncio.run(
            self.system.create_thought_bubble(
                content="Test bubble",
                category="observation",
            )
        )
        self.assertIsNotNone(bubble)
        self.assertEqual(len(self.system.thought_bubbles), 1)

    def test_update_feeling(self):
        """Test updating feeling"""
        asyncio.run(
            self.system.update_feeling(
                primary=AuraFeeling.HAPPY,
                intensity=0.8,
                cause="User completed a task",
            )
        )
        self.assertEqual(self.system.feeling_state.primary, AuraFeeling.HAPPY)
        self.assertEqual(self.system.feeling_state.intensity, 0.8)
        self.assertEqual(self.system.feeling_state.cause, "User completed a task")

    def test_update_trust_from_interaction(self):
        """Test updating trust from interaction"""
        asyncio.run(
            self.system.update_trust_from_interaction(
                interaction_type="task_completion",
                success=True,
            )
        )
        self.assertGreater(self.system.trust_meter.understanding_work, 0)

    def test_trust_level_progression(self):
        """Test trust level progression"""
        # Simulate multiple successful interactions
        for _ in range(25):
            asyncio.run(
                self.system.update_trust_from_interaction(
                    interaction_type="task_completion",
                    success=True,
                )
            )

        # Trust should have increased
        self.assertGreater(self.system.trust_meter.understanding_work, 4.0)

    def test_format_feeling_for_user(self):
        """Test formatting feeling for user"""
        asyncio.run(
            self.system.update_feeling(
                primary=AuraFeeling.EXCITED,
                intensity=0.7,
            )
        )
        msg = self.system.format_feeling_for_user()
        self.assertIn("EXCITED", msg.upper())

    def test_format_trust_for_user(self):
        """Test formatting trust for user"""
        msg = self.system.format_trust_for_user()
        self.assertIn("Connection", msg)
        self.assertIn("trust", msg.lower())

    def test_format_character_sheet(self):
        """Test formatting character sheet"""
        msg = self.system.format_character_sheet()
        self.assertIn("YOUR SHEET", msg)
        self.assertIn("AURA'S SHEET", msg)

    def test_max_thoughts_limit(self):
        """Test that thought stream has max limit"""
        # Add more than max thoughts
        for i in range(60):
            asyncio.run(
                self.system.add_thought(
                    content=f"Thought {i}",
                    reasoning="Test",
                )
            )

        # Should be limited to max_thoughts
        self.assertEqual(len(self.system.thought_stream), self.system.max_thoughts)


class TestInnerVoiceIntegration(unittest.TestCase):
    """Integration tests for InnerVoiceSystem"""

    def test_get_inner_voice_singleton(self):
        """Test singleton pattern"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        voice1 = loop.run_until_complete(get_inner_voice())
        voice2 = loop.run_until_complete(get_inner_voice())

        # Should be same instance
        self.assertIs(voice1, voice2)

        loop.close()


if __name__ == "__main__":
    unittest.main()
