"""
Tests for AURA v3 Feelings & Trust Meter UI System
"""

import unittest
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ui.feelings_meter import (
    FeelingsMeter,
    AuraEmotion,
    UnderstandingDomain,
    TrustPhase,
    UnderstandingMetric,
    TrustState,
    get_feelings_meter,
)


class TestAuraEmotion(unittest.TestCase):
    """Test AuraEmotion enum"""

    def test_emotions(self):
        """Test all emotions exist"""
        emotions = [
            AuraEmotion.CURIOUS,
            AuraEmotion.FOCUSED,
            AuraEmotion.CONFIDENT,
            AuraEmotion.UNCERTAIN,
            AuraEmotion.CONCERNED,
            AuraEmotion.HAPPY,
            AuraEmotion.WORRIED,
            AuraEmotion.EXCITED,
            AuraEmotion.CALM,
            AuraEmotion.TIRED,
            AuraEmotion.FRUSTRATED,
            AuraEmotion.HOPEFUL,
            AuraEmotion.GRATEFUL,
            AuraEmotion.CONFUSED,
        ]
        self.assertGreaterEqual(len(emotions), 10)


class TestUnderstandingDomain(unittest.TestCase):
    """Test UnderstandingDomain enum"""

    def test_domains(self):
        """Test all domains exist"""
        domains = [
            UnderstandingDomain.WORK_STYLE,
            UnderstandingDomain.SLEEP_PATTERNS,
            UnderstandingDomain.MOOD_PATTERNS,
            UnderstandingDomain.GOALS,
            UnderstandingDomain.RELATIONSHIPS,
            UnderstandingDomain.PREFERENCES,
            UnderstandingDomain.PRODUCTIVITY,
            UnderstandingDomain.ENERGY_LEVELS,
        ]
        self.assertEqual(len(domains), 8)


class TestTrustPhase(unittest.TestCase):
    """Test TrustPhase enum"""

    def test_phases(self):
        """Test all phases"""
        phases = [
            TrustPhase.INTRODUCTION,
            TrustPhase.LEARNING,
            TrustPhase.UNDERSTANDING,
            TrustPhase.COMFORTABLE,
            TrustPhase.PARTNERSHIP,
        ]
        self.assertEqual(len(phases), 5)


class TestUnderstandingMetric(unittest.TestCase):
    """Test UnderstandingMetric class"""

    def test_create_metric(self):
        """Test creating metric"""
        metric = UnderstandingMetric(
            domain=UnderstandingDomain.WORK_STYLE,
            score=7.5,
            evidence=["Observation 1"],
            observations=["Observed work pattern"],
        )
        self.assertEqual(metric.score, 7.5)
        self.assertEqual(metric.domain, UnderstandingDomain.WORK_STYLE)

    def test_adjust_score(self):
        """Test adjusting score"""
        metric = UnderstandingMetric(domain=UnderstandingDomain.WORK_STYLE, score=5.0)

        metric.adjust_score(2.0)
        self.assertEqual(metric.score, 7.0)

        metric.adjust_score(-3.0)
        self.assertEqual(metric.score, 4.0)

        # Bounds
        metric.adjust_score(20.0)
        self.assertEqual(metric.score, 10.0)

        metric.adjust_score(-20.0)
        self.assertEqual(metric.score, 0.0)


class TestTrustState(unittest.TestCase):
    """Test TrustState class"""

    def test_default_state(self):
        """Test default trust state"""
        state = TrustState()
        self.assertEqual(state.phase, TrustPhase.INTRODUCTION)
        self.assertEqual(
            len(state.metrics), 0
        )  # Raw TrustState has no metrics; FeelingsMeter adds them
        self.assertEqual(state.total_interactions, 0)

    def test_get_overall_score(self):
        """Test overall score calculation"""
        state = TrustState()

        # Empty metrics
        self.assertEqual(state.get_overall_score(), 0.0)

        # Add some metrics
        state.metrics[UnderstandingDomain.WORK_STYLE.value] = UnderstandingMetric(
            domain=UnderstandingDomain.WORK_STYLE, score=8.0
        )
        state.metrics[UnderstandingDomain.GOALS.value] = UnderstandingMetric(
            domain=UnderstandingDomain.GOALS, score=6.0
        )

        overall = state.get_overall_score()
        self.assertGreater(overall, 0.0)

    def test_get_phase_description(self):
        """Test phase description"""
        state = TrustState()

        # Low score
        state.metrics[UnderstandingDomain.WORK_STYLE.value] = UnderstandingMetric(
            domain=UnderstandingDomain.WORK_STYLE, score=1.0
        )
        desc = state.get_phase_description()
        self.assertIn("getting started", desc)

        # High score - use 9.5 to ensure above threshold
        for domain in UnderstandingDomain:
            state.metrics[domain.value] = UnderstandingMetric(domain=domain, score=9.5)
        desc = state.get_phase_description()
        self.assertIn("deeply", desc)


class TestFeelingsMeter(unittest.TestCase):
    """Test FeelingsMeter class"""

    def setUp(self):
        """Set up test with temp storage"""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()
        self.meter = FeelingsMeter(storage_path=self.temp_file.name)

    def tearDown(self):
        """Clean up temp file"""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass

    def test_default_state(self):
        """Test default state"""
        self.assertEqual(self.meter.feeling_state.primary, AuraEmotion.CALM)
        self.assertEqual(self.meter.feeling_state.secondary, AuraEmotion.CURIOUS)
        self.assertEqual(len(self.meter.trust_state.metrics), 8)

    @asyncio.coroutine
    def test_initialize(self):
        """Test initialization"""
        yield from self.meter.initialize()
        self.assertIsNotNone(self.meter.store)

    @asyncio.coroutine
    def test_update_feeling(self):
        """Test updating feeling"""
        state = yield from self.meter.update_feeling(
            emotion=AuraEmotion.HAPPY,
            intensity=0.8,
            cause="User completed a task",
            evidence=["Task completed successfully"],
        )

        self.assertEqual(state.primary, AuraEmotion.HAPPY)
        self.assertEqual(state.intensity, 0.8)
        self.assertEqual(state.cause, "User completed a task")

    @asyncio.coroutine
    def test_format_feeling_message(self):
        """Test formatting feeling message"""
        yield from self.meter.update_feeling(
            emotion=AuraEmotion.CONFIDENT, intensity=0.7, cause="Good interaction"
        )

        message = self.meter.format_feeling_message()
        self.assertIn("confident", message)

    @asyncio.coroutine
    def test_update_understanding(self):
        """Test updating understanding"""
        metric = yield from self.meter.update_understanding(
            domain=UnderstandingDomain.WORK_STYLE,
            observation="User works late often",
            evidence="Logged 5 late nights this week",
            confidence=0.7,
        )

        self.assertGreater(metric.score, 0.0)

    @asyncio.coroutine
    def test_record_interaction_success(self):
        """Test recording successful interaction"""
        yield from self.meter.record_interaction(
            domain=UnderstandingDomain.WORK_STYLE, success=True
        )

        self.assertEqual(self.meter.trust_state.total_interactions, 1)
        self.assertEqual(self.meter.trust_state.successful_interactions, 1)

    @asyncio.coroutine
    def test_record_interaction_failure(self):
        """Test recording failed interaction"""
        yield from self.meter.record_interaction(
            domain=UnderstandingDomain.WORK_STYLE, success=False
        )

        self.assertEqual(self.meter.trust_state.total_interactions, 1)
        self.assertEqual(self.meter.trust_state.failed_interactions, 1)

    @asyncio.coroutine
    def test_correct_understanding(self):
        """Test correcting understanding"""
        # First update some understanding
        yield from self.meter.update_understanding(
            domain=UnderstandingDomain.SLEEP_PATTERNS,
            observation="User sleeps late",
            evidence="Late activity",
            confidence=0.6,
        )

        # Then correct it
        record = yield from self.meter.correct_understanding(
            domain=UnderstandingDomain.SLEEP_PATTERNS,
            correction="No, you're wrong about my sleep",
            explanation="I actually sleep early",
        )

        self.assertIsNotNone(record)
        self.assertEqual(record.domain, "sleep_patterns")

    @asyncio.coroutine
    def test_confirm_understanding(self):
        """Test confirming understanding"""
        yield from self.meter.confirm_understanding(domain=UnderstandingDomain.GOALS)

        metric = self.meter.trust_state.metrics[UnderstandingDomain.GOALS.value]
        self.assertGreater(metric.score, 0.0)
        self.assertEqual(metric.confirmations, 1)

    @asyncio.coroutine
    def test_get_understanding_meter(self):
        """Test getting understanding meter"""
        yield from self.meter.update_understanding(
            domain=UnderstandingDomain.WORK_STYLE,
            observation="Test",
            evidence="Test",
            confidence=0.5,
        )

        meter_data = self.meter.get_understanding_meter(UnderstandingDomain.WORK_STYLE)

        self.assertIn("score", meter_data)
        self.assertIn("domain", meter_data)

    @asyncio.coroutine
    def test_format_trust_message(self):
        """Test formatting trust message"""
        # Add some understanding
        yield from self.meter.update_understanding(
            domain=UnderstandingDomain.WORK_STYLE,
            observation="Test",
            evidence="Test",
            confidence=0.8,
        )

        message = self.meter.format_trust_message()
        self.assertIn("understand", message.lower())

    @asyncio.coroutine
    def test_generate_sleep_feeling(self):
        """Test generating sleep-related feeling"""
        state = yield from self.meter.generate_contextual_feeling(
            context="sleep", observations=["late night", "tired"]
        )

        # Should generate concerned or uncertain feeling
        self.assertIn(
            state.primary,
            [AuraEmotion.CONCERNED, AuraEmotion.UNCERTAIN, AuraEmotion.TIRED],
        )

    @asyncio.coroutine
    def test_generate_work_feeling(self):
        """Test generating work-related feeling"""
        state = yield from self.meter.generate_contextual_feeling(
            context="work", observations=["deadline approaching", "busy"]
        )

        # Should be focused
        self.assertEqual(state.primary, AuraEmotion.FOCUSED)

    @asyncio.coroutine
    def test_get_trends(self):
        """Test getting trends"""
        # Add some feelings
        for _ in range(3):
            yield from self.meter.update_feeling(
                emotion=AuraEmotion.HAPPY, intensity=0.7, cause="Test"
            )

        trends = self.meter.get_trends(days=7)

        self.assertIn("feeling_count", trends)
        self.assertIn("dominant_emotion", trends)

    @asyncio.coroutine
    def test_get_statistics(self):
        """Test getting statistics"""
        yield from self.meter.update_understanding(
            domain=UnderstandingDomain.WORK_STYLE,
            observation="Test",
            evidence="Test",
            confidence=0.5,
        )

        stats = self.meter.get_statistics()

        self.assertIn("total_interactions", stats)
        self.assertIn("trust_overall", stats)
        self.assertIn("current_feeling", stats)

    @asyncio.coroutine
    def test_feeling_state_transitions(self):
        """Test feeling changes after interactions"""
        # Success
        yield from self.meter.record_interaction(
            domain=UnderstandingDomain.WORK_STYLE, success=True
        )

        # Should be confident now
        self.assertIn(
            self.meter.feeling_state.primary, [AuraEmotion.CONFIDENT, AuraEmotion.HAPPY]
        )

        # Multiple failures
        for _ in range(5):
            yield from self.meter.record_interaction(
                domain=UnderstandingDomain.WORK_STYLE, success=False
            )

        # Should be uncertain
        self.assertEqual(self.meter.feeling_state.primary, AuraEmotion.UNCERTAIN)


class TestFeelingsMeterIntegration(unittest.TestCase):
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

    def test_get_feelings_meter_singleton(self):
        """Test singleton pattern"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Reset global for test
        import src.ui.feelings_meter as feelings_module

        feelings_module._feelings_meter = None

        meter1 = loop.run_until_complete(get_feelings_meter())
        meter2 = loop.run_until_complete(get_feelings_meter())

        # Should be same instance
        self.assertIs(meter1, meter2)

        loop.close()


if __name__ == "__main__":
    unittest.main()
