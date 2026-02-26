"""
Tests for AURA v3 Feelings & Trust Meter UI System
"""

import unittest
import asyncio
import os
import sys
import tempfile
import pytest

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
        assert len(emotions) >= 10


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
        assert len(domains) == 8


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
        assert len(phases) == 5


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
        assert metric.score == 7.5
        assert metric.domain == UnderstandingDomain.WORK_STYLE

    def test_adjust_score(self):
        """Test adjusting score"""
        metric = UnderstandingMetric(domain=UnderstandingDomain.WORK_STYLE, score=5.0)

        metric.adjust_score(2.0)
        assert metric.score == 7.0

        metric.adjust_score(-3.0)
        assert metric.score == 4.0

        # Bounds
        metric.adjust_score(20.0)
        assert metric.score == 10.0

        metric.adjust_score(-20.0)
        assert metric.score == 0.0


class TestTrustState(unittest.TestCase):
    """Test TrustState class"""

    def test_default_state(self):
        """Test default trust state"""
        state = TrustState()
        assert state.phase == TrustPhase.INTRODUCTION
        assert (
            len(state.metrics) == 0
        )  # Raw TrustState has no metrics; FeelingsMeter adds them
        assert state.total_interactions == 0

    def test_get_overall_score(self):
        """Test overall score calculation"""
        state = TrustState()

        # Empty metrics
        assert state.get_overall_score() == 0.0

        # Add some metrics
        state.metrics[UnderstandingDomain.WORK_STYLE.value] = UnderstandingMetric(
            domain=UnderstandingDomain.WORK_STYLE, score=8.0
        )
        state.metrics[UnderstandingDomain.GOALS.value] = UnderstandingMetric(
            domain=UnderstandingDomain.GOALS, score=6.0
        )

        overall = state.get_overall_score()
        assert overall > 0.0

    def test_get_phase_description(self):
        """Test phase description"""
        state = TrustState()

        # Low score
        state.metrics[UnderstandingDomain.WORK_STYLE.value] = UnderstandingMetric(
            domain=UnderstandingDomain.WORK_STYLE, score=1.0
        )
        desc = state.get_phase_description()
        assert "getting started" in desc

        # High score - use 9.5 to ensure above threshold
        for domain in UnderstandingDomain:
            state.metrics[domain.value] = UnderstandingMetric(domain=domain, score=9.5)
        desc = state.get_phase_description()
        assert "deeply" in desc


class TestFeelingsMeterSync(unittest.TestCase):
    """Test FeelingsMeter class - synchronous tests"""

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
        assert self.meter.feeling_state.primary == AuraEmotion.CALM
        assert self.meter.feeling_state.secondary == AuraEmotion.CURIOUS
        assert len(self.meter.trust_state.metrics) == 8


# Async tests for FeelingsMeter using pytest-native fixtures
@pytest.fixture
def feelings_meter():
    """Create a FeelingsMeter with temp storage"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    temp_file.close()
    meter = FeelingsMeter(storage_path=temp_file.name)
    yield meter
    try:
        os.unlink(temp_file.name)
    except:
        pass


class TestFeelingsMeterAsync:
    """Test FeelingsMeter class - async tests"""

    async def test_initialize(self, feelings_meter):
        """Test initialization"""
        await feelings_meter.initialize()
        assert feelings_meter.store is not None

    async def test_update_feeling(self, feelings_meter):
        """Test updating feeling"""
        state = await feelings_meter.update_feeling(
            emotion=AuraEmotion.HAPPY,
            intensity=0.8,
            cause="User completed a task",
            evidence=["Task completed successfully"],
        )

        assert state.primary == AuraEmotion.HAPPY
        assert state.intensity == 0.8
        assert state.cause == "User completed a task"

    async def test_format_feeling_message(self, feelings_meter):
        """Test formatting feeling message"""
        await feelings_meter.update_feeling(
            emotion=AuraEmotion.CONFIDENT, intensity=0.7, cause="Good interaction"
        )

        message = feelings_meter.format_feeling_message()
        assert "confident" in message

    async def test_update_understanding(self, feelings_meter):
        """Test updating understanding"""
        metric = await feelings_meter.update_understanding(
            domain=UnderstandingDomain.WORK_STYLE,
            observation="User works late often",
            evidence="Logged 5 late nights this week",
            confidence=0.7,
        )

        assert metric.score > 0.0

    async def test_record_interaction_success(self, feelings_meter):
        """Test recording successful interaction"""
        await feelings_meter.record_interaction(
            domain=UnderstandingDomain.WORK_STYLE, success=True
        )

        assert feelings_meter.trust_state.total_interactions == 1
        assert feelings_meter.trust_state.successful_interactions == 1

    async def test_record_interaction_failure(self, feelings_meter):
        """Test recording failed interaction"""
        await feelings_meter.record_interaction(
            domain=UnderstandingDomain.WORK_STYLE, success=False
        )

        assert feelings_meter.trust_state.total_interactions == 1
        assert feelings_meter.trust_state.failed_interactions == 1

    async def test_correct_understanding(self, feelings_meter):
        """Test correcting understanding"""
        # First update some understanding
        await feelings_meter.update_understanding(
            domain=UnderstandingDomain.SLEEP_PATTERNS,
            observation="User sleeps late",
            evidence="Late activity",
            confidence=0.6,
        )

        # Then correct it
        record = await feelings_meter.correct_understanding(
            domain=UnderstandingDomain.SLEEP_PATTERNS,
            correction="No, you're wrong about my sleep",
            explanation="I actually sleep early",
        )

        assert record is not None
        assert record.domain == "sleep_patterns"

    async def test_confirm_understanding(self, feelings_meter):
        """Test confirming understanding"""
        await feelings_meter.confirm_understanding(domain=UnderstandingDomain.GOALS)

        metric = feelings_meter.trust_state.metrics[UnderstandingDomain.GOALS.value]
        assert metric.score > 0.0
        assert metric.confirmations == 1

    async def test_get_understanding_meter(self, feelings_meter):
        """Test getting understanding meter"""
        await feelings_meter.update_understanding(
            domain=UnderstandingDomain.WORK_STYLE,
            observation="Test",
            evidence="Test",
            confidence=0.5,
        )

        meter_data = feelings_meter.get_understanding_meter(
            UnderstandingDomain.WORK_STYLE
        )

        assert "score" in meter_data
        assert "domain" in meter_data

    async def test_format_trust_message(self, feelings_meter):
        """Test formatting trust message"""
        # Add some understanding
        await feelings_meter.update_understanding(
            domain=UnderstandingDomain.WORK_STYLE,
            observation="Test",
            evidence="Test",
            confidence=0.8,
        )

        message = feelings_meter.format_trust_message()
        assert "understand" in message.lower()

    async def test_generate_sleep_feeling(self, feelings_meter):
        """Test generating sleep-related feeling"""
        state = await feelings_meter.generate_contextual_feeling(
            context="sleep", observations=["late night", "tired"]
        )

        # Should generate concerned or uncertain feeling
        assert state.primary in [
            AuraEmotion.CONCERNED,
            AuraEmotion.UNCERTAIN,
            AuraEmotion.TIRED,
        ]

    async def test_generate_work_feeling(self, feelings_meter):
        """Test generating work-related feeling"""
        state = await feelings_meter.generate_contextual_feeling(
            context="work", observations=["deadline approaching", "busy"]
        )

        # Should be focused
        assert state.primary == AuraEmotion.FOCUSED

    async def test_get_trends(self, feelings_meter):
        """Test getting trends"""
        # Add some feelings
        for _ in range(3):
            await feelings_meter.update_feeling(
                emotion=AuraEmotion.HAPPY, intensity=0.7, cause="Test"
            )

        trends = feelings_meter.get_trends(days=7)

        assert "feeling_count" in trends
        assert "dominant_emotion" in trends

    async def test_get_statistics(self, feelings_meter):
        """Test getting statistics"""
        await feelings_meter.update_understanding(
            domain=UnderstandingDomain.WORK_STYLE,
            observation="Test",
            evidence="Test",
            confidence=0.5,
        )

        stats = feelings_meter.get_statistics()

        assert "total_interactions" in stats
        assert "trust_overall" in stats
        assert "current_feeling" in stats

    async def test_feeling_state_transitions(self, feelings_meter):
        """Test feeling changes after interactions"""
        # Success
        await feelings_meter.record_interaction(
            domain=UnderstandingDomain.WORK_STYLE, success=True
        )

        # Should be confident now
        assert feelings_meter.feeling_state.primary in [
            AuraEmotion.CONFIDENT,
            AuraEmotion.HAPPY,
        ]

        # Multiple failures
        for _ in range(5):
            await feelings_meter.record_interaction(
                domain=UnderstandingDomain.WORK_STYLE, success=False
            )

        # Should be uncertain
        assert feelings_meter.feeling_state.primary == AuraEmotion.UNCERTAIN


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
        assert meter1 is meter2

        loop.close()


if __name__ == "__main__":
    unittest.main()
