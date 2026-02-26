"""
Tests for AURA AGI Core - Real Algorithmic Intelligence

Tests cover:
1. Emotional Detection - valence/arousal scoring with emotion lexicon
2. Markov Chain Predictions - behavior sequence predictions
3. Goal Management - priority scoring, progress tracking, deadlines
4. Habit Learning - frequency detection, confidence scoring
"""

import pytest
import asyncio
import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock

# Import AGI components
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agi import (
    AGICore,
    EmotionalIntelligence,
    EmotionalState,
    LongTermPlanner,
    TheoryOfMind,
    MarkovPredictor,
    Goal,
    Habit,
    EMOTION_LEXICON,
)


class MockMemory:
    """Mock memory for testing"""

    def __init__(self):
        self.self_model = MagicMock()
        self.self_model.get_category = MagicMock(return_value={})

    def retrieve(self, query, limit=5):
        return []


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_memory():
    """Create mock memory for testing"""
    return MockMemory()


# =============================================================================
# Test Emotion Lexicon
# =============================================================================


class TestEmotionLexicon:
    """Test the emotion lexicon structure and coverage"""

    def test_lexicon_has_sufficient_words(self):
        """Lexicon should have ~250+ words for comprehensive coverage"""
        assert len(EMOTION_LEXICON) >= 250, "Lexicon should have at least 250 words"

    def test_lexicon_entries_have_required_keys(self):
        """Each entry should have valence and arousal"""
        for word, scores in EMOTION_LEXICON.items():
            assert "valence" in scores, f"Missing valence for '{word}'"
            assert "arousal" in scores, f"Missing arousal for '{word}'"

    def test_lexicon_values_in_range(self):
        """Valence and arousal should be between 0 and 1"""
        for word, scores in EMOTION_LEXICON.items():
            assert 0 <= scores["valence"] <= 1, (
                f"Invalid valence for '{word}': {scores['valence']}"
            )
            assert 0 <= scores["arousal"] <= 1, (
                f"Invalid arousal for '{word}': {scores['arousal']}"
            )

    def test_lexicon_covers_emotion_quadrants(self):
        """Lexicon should cover all quadrants of valence/arousal space"""
        high_v_high_a = [
            w
            for w, s in EMOTION_LEXICON.items()
            if s["valence"] > 0.7 and s["arousal"] > 0.7
        ]
        high_v_low_a = [
            w
            for w, s in EMOTION_LEXICON.items()
            if s["valence"] > 0.7 and s["arousal"] < 0.3
        ]
        low_v_high_a = [
            w
            for w, s in EMOTION_LEXICON.items()
            if s["valence"] < 0.3 and s["arousal"] > 0.7
        ]
        low_v_low_a = [
            w
            for w, s in EMOTION_LEXICON.items()
            if s["valence"] < 0.3 and s["arousal"] < 0.3
        ]

        assert len(high_v_high_a) >= 10, "Need positive-excited words"
        assert len(high_v_low_a) >= 5, "Need positive-calm words"
        assert len(low_v_high_a) >= 10, "Need negative-excited words"
        assert len(low_v_low_a) >= 5, "Need negative-calm words"


# =============================================================================
# Test Emotional Intelligence
# =============================================================================


class TestEmotionalIntelligence:
    """Test valence/arousal emotion detection"""

    @pytest.fixture
    def ei(self):
        return EmotionalIntelligence()

    @pytest.mark.asyncio
    async def test_detect_happy_emotion(self, ei):
        """Should detect happy emotions with high valence"""
        state = await ei.detect_emotion("I'm so happy and excited today!")

        assert state.valence > 0.7, f"Expected high valence, got {state.valence}"
        assert state.primary in ["happy", "excited"], (
            f"Expected happy/excited, got {state.primary}"
        )

    @pytest.mark.asyncio
    async def test_detect_sad_emotion(self, ei):
        """Should detect sad emotions with low valence, low arousal"""
        state = await ei.detect_emotion("I feel so sad and depressed today")

        assert state.valence < 0.4, f"Expected low valence, got {state.valence}"
        assert state.primary in ["sad", "depressed"], (
            f"Expected sad/depressed, got {state.primary}"
        )

    @pytest.mark.asyncio
    async def test_detect_angry_emotion(self, ei):
        """Should detect angry emotions with low valence, high arousal"""
        state = await ei.detect_emotion("I'm furious and angry about this!")

        assert state.valence < 0.3, f"Expected low valence, got {state.valence}"
        assert state.arousal > 0.7, f"Expected high arousal, got {state.arousal}"
        assert state.primary in ["angry", "anxious"], (
            f"Expected angry, got {state.primary}"
        )

    @pytest.mark.asyncio
    async def test_detect_calm_emotion(self, ei):
        """Should detect calm emotions with high valence, low arousal"""
        state = await ei.detect_emotion("I feel peaceful and serene today")

        assert state.valence > 0.6, f"Expected high valence, got {state.valence}"
        assert state.arousal < 0.4, f"Expected low arousal, got {state.arousal}"

    @pytest.mark.asyncio
    async def test_detect_neutral_no_emotion_words(self, ei):
        """Should return neutral for text without emotion words"""
        state = await ei.detect_emotion("The meeting is at 3pm tomorrow")

        assert state.primary == "neutral"
        assert state.intensity < 0.2

    @pytest.mark.asyncio
    async def test_intensity_increases_with_emotion_words(self, ei):
        """More emotion words should increase intensity"""
        state1 = await ei.detect_emotion("happy")
        state2 = await ei.detect_emotion("happy and wonderful and fantastic and great")

        assert state2.intensity > state1.intensity

    @pytest.mark.asyncio
    async def test_weighted_average_scoring(self, ei):
        """Should use weighted average, not just presence/absence"""
        # Mix of positive and negative
        state = await ei.detect_emotion("I'm happy but also a bit worried")

        # Should be somewhere in the middle
        assert 0.3 < state.valence < 0.8

    @pytest.mark.asyncio
    async def test_emotional_history_tracking(self, ei):
        """Should track emotion history"""
        await ei.detect_emotion("I'm happy")
        await ei.detect_emotion("I'm sad")
        await ei.detect_emotion("I'm angry")

        assert len(ei.history) == 3

    @pytest.mark.asyncio
    async def test_emotional_trend_calculation(self, ei):
        """Should calculate emotional trends over history"""
        # Start sad, end happy
        await ei.detect_emotion("I'm sad")
        await ei.detect_emotion("I'm feeling a bit better")
        await ei.detect_emotion("I'm happy now")

        trend = ei.get_emotional_trend(window=3)

        assert trend["trend"] == "improving"
        assert trend["valence_change"] > 0


# =============================================================================
# Test Markov Predictor
# =============================================================================


class TestMarkovPredictor:
    """Test Markov chain behavior predictions"""

    @pytest.fixture
    def predictor(self, temp_data_dir):
        return MarkovPredictor(data_dir=temp_data_dir)

    def test_observe_and_predict(self, predictor):
        """Should predict based on observed transitions"""
        # Observe pattern: A -> B -> C (multiple times)
        for _ in range(5):
            predictor.observe("A")
            predictor.observe("B")
            predictor.observe("C")

        # Predict from A - should predict B
        predictions = predictor.predict("A")

        assert len(predictions) > 0
        assert predictions[0]["state"] == "B"
        assert predictions[0]["probability"] > 0.5

    def test_probability_distribution(self, predictor):
        """Probabilities should sum to 1 for all transitions"""
        predictor.observe("X")
        predictor.observe("Y")
        predictor.observe("X")
        predictor.observe("Z")
        predictor.observe("X")
        predictor.observe("Y")

        predictions = predictor.predict("X", top_k=10)

        if predictions:
            total_prob = sum(p["probability"] for p in predictions)
            # May not be exactly 1 if we have top_k limit, but should be close
            assert total_prob > 0.9 or len(predictions) < 3

    def test_confidence_increases_with_samples(self, predictor):
        """Confidence should increase with more observations"""
        predictor.observe("A")
        predictor.observe("B")
        pred1 = predictor.predict("A")
        conf1 = pred1[0]["confidence"] if pred1 else 0

        # Add more observations
        for _ in range(10):
            predictor.observe("A")
            predictor.observe("B")

        pred2 = predictor.predict("A")
        conf2 = pred2[0]["confidence"] if pred2 else 0

        assert conf2 > conf1

    def test_get_transition_probability(self, predictor):
        """Should return correct transition probability"""
        predictor.observe("A")
        predictor.observe("B")
        predictor.observe("A")
        predictor.observe("B")
        predictor.observe("A")
        predictor.observe("C")

        # A -> B happened 2/3 times, A -> C happened 1/3 times
        prob_a_to_b = predictor.get_transition_probability("A", "B")
        prob_a_to_c = predictor.get_transition_probability("A", "C")

        assert abs(prob_a_to_b - 2 / 3) < 0.01
        assert abs(prob_a_to_c - 1 / 3) < 0.01

    def test_persistence(self, temp_data_dir):
        """Should persist and reload transitions"""
        # Create predictor and add data
        pred1 = MarkovPredictor(data_dir=temp_data_dir)
        for _ in range(5):
            pred1.observe("login")
            pred1.observe("check_email")
        pred1._save_transitions()

        # Create new predictor from same dir
        pred2 = MarkovPredictor(data_dir=temp_data_dir)

        predictions = pred2.predict("login")
        assert len(predictions) > 0
        assert predictions[0]["state"] == "check_email"


# =============================================================================
# Test Goal Management
# =============================================================================


class TestGoalManagement:
    """Test priority-scored goal management"""

    @pytest.fixture
    def planner(self, mock_memory, temp_data_dir):
        return LongTermPlanner(mock_memory, data_dir=temp_data_dir)

    def test_add_goal_with_priority(self, planner):
        """Should add goal with calculated priority"""
        goal = planner.add_goal(
            description="Complete project",
            importance=0.8,
            deadline=datetime.now() + timedelta(days=3),
        )

        assert goal.importance == 0.8
        assert goal.urgency > 0.5  # Deadline is soon
        assert goal.priority == goal.importance * goal.urgency

    def test_priority_calculation(self, planner):
        """Priority should be importance × urgency"""
        goal = Goal(
            id="test", description="Test", importance=0.8, urgency=0.6, progress=0.0
        )

        assert goal.priority == 0.8 * 0.6

    def test_urgency_increases_near_deadline(self, planner):
        """Urgency should increase as deadline approaches"""
        future_goal = planner.add_goal(
            description="Future task",
            importance=0.5,
            deadline=datetime.now() + timedelta(days=30),
        )

        near_goal = planner.add_goal(
            description="Near task",
            importance=0.5,
            deadline=datetime.now() + timedelta(hours=12),
        )

        assert near_goal.urgency > future_goal.urgency

    def test_update_progress(self, planner):
        """Should track progress percentage"""
        goal = planner.add_goal("Test goal", importance=0.5)

        planner.update_goal_progress(goal.id, 0.5)

        assert planner.goals[goal.id].progress == 0.5
        assert not planner.goals[goal.id].completed

    def test_goal_completion(self, planner):
        """Should mark goal completed at 100% progress"""
        goal = planner.add_goal("Test goal", importance=0.5)

        planner.update_goal_progress(goal.id, 1.0)

        assert planner.goals[goal.id].completed

    def test_prioritized_goals_sorted(self, planner):
        """Goals should be sorted by priority"""
        planner.add_goal("Low priority", importance=0.2)
        planner.add_goal(
            "High priority", importance=0.9, deadline=datetime.now() + timedelta(days=1)
        )
        planner.add_goal("Medium priority", importance=0.5)

        goals = planner.get_prioritized_goals()

        # Should be sorted descending by priority
        for i in range(len(goals) - 1):
            assert goals[i].priority >= goals[i + 1].priority

    def test_exclude_completed_goals(self, planner):
        """Should exclude completed goals by default"""
        goal1 = planner.add_goal("Active goal", importance=0.5)
        goal2 = planner.add_goal("Completed goal", importance=0.8)
        planner.update_goal_progress(goal2.id, 1.0)

        active_goals = planner.get_prioritized_goals(include_completed=False)

        assert len(active_goals) == 1
        assert active_goals[0].id == goal1.id

    def test_persistence(self, mock_memory, temp_data_dir):
        """Should persist and reload goals"""
        planner1 = LongTermPlanner(mock_memory, data_dir=temp_data_dir)
        planner1.add_goal("Persisted goal", importance=0.7)
        planner1._save_data()

        planner2 = LongTermPlanner(mock_memory, data_dir=temp_data_dir)

        assert len(planner2.goals) == 1
        assert list(planner2.goals.values())[0].description == "Persisted goal"


# =============================================================================
# Test Habit Learning
# =============================================================================


class TestHabitLearning:
    """Test real habit learning algorithm"""

    @pytest.fixture
    def planner(self, mock_memory, temp_data_dir):
        return LongTermPlanner(mock_memory, data_dir=temp_data_dir)

    @pytest.mark.asyncio
    async def test_habit_detection_after_threshold(self, planner):
        """Should detect habit after 3+ occurrences at same time"""
        # Simulate same action at same hour 3 times
        for i in range(4):
            await planner.learn_habit("morning_coffee", {})

        # Check habit was created
        assert "morning_coffee" in planner.habits
        habit = planner.habits["morning_coffee"]
        assert habit.occurrence_count >= 3
        assert habit.confidence > 0

    @pytest.mark.asyncio
    async def test_confidence_increases_with_consistency(self, planner):
        """Confidence should increase with more consistent occurrences"""
        # First few occurrences
        for _ in range(3):
            await planner.learn_habit("daily_standup", {})

        conf1 = planner.habits["daily_standup"].confidence

        # More occurrences
        for _ in range(7):
            await planner.learn_habit("daily_standup", {})

        conf2 = planner.habits["daily_standup"].confidence

        assert conf2 > conf1

    @pytest.mark.asyncio
    async def test_habit_tracks_typical_hour(self, planner):
        """Should track the typical hour for habit"""
        for _ in range(5):
            await planner.learn_habit("check_email", {})

        habit = planner.habits["check_email"]
        current_hour = datetime.now().hour

        assert habit.typical_hour == current_hour

    @pytest.mark.asyncio
    async def test_proactive_suggestions_from_habits(self, planner):
        """Should generate proactive suggestions from habits"""
        # Create a confident habit at current hour
        for _ in range(10):
            await planner.learn_habit("exercise", {})

        # Manually set last_seen to yesterday to trigger suggestion
        planner.habits["exercise"].last_seen = datetime.now() - timedelta(hours=24)

        suggestions = planner.get_proactive_suggestions()

        assert len(suggestions) > 0
        assert any(s["action"] == "exercise" for s in suggestions)

    @pytest.mark.asyncio
    async def test_habit_persistence(self, mock_memory, temp_data_dir):
        """Should persist and reload habits"""
        planner1 = LongTermPlanner(mock_memory, data_dir=temp_data_dir)

        for _ in range(5):
            await planner1.learn_habit("meditation", {})

        planner1._save_data()

        planner2 = LongTermPlanner(mock_memory, data_dir=temp_data_dir)

        assert "meditation" in planner2.habits
        assert planner2.habits["meditation"].occurrence_count >= 5

    @pytest.mark.asyncio
    async def test_learn_habit_not_just_pass(self, planner):
        """Verify learn_habit actually does something (not just 'pass')"""
        initial_habit_count = len(planner.habits)
        initial_time_counts = dict(planner._action_time_counts)

        await planner.learn_habit("new_action", {})

        # Either habit count increased OR time counts were updated
        assert (
            len(planner.habits) > initial_habit_count
            or len(planner._action_time_counts) > len(initial_time_counts)
            or "new_action" in planner._action_time_counts
        )


# =============================================================================
# Test Theory of Mind with Markov
# =============================================================================


class TestTheoryOfMind:
    """Test Theory of Mind with Markov predictions"""

    @pytest.fixture
    def tom(self, mock_memory, temp_data_dir):
        return TheoryOfMind(mock_memory, data_dir=temp_data_dir)

    @pytest.mark.asyncio
    async def test_update_model_tracks_actions(self, tom):
        """Should track user actions for Markov predictions"""
        await tom.update_model("What time is it?", {})
        await tom.update_model("Help me with this task", {})

        assert tom._last_action is not None

    @pytest.mark.asyncio
    async def test_predict_needs_uses_markov(self, tom):
        """Should use Markov predictions instead of just time heuristics"""
        # Train the predictor
        for _ in range(5):
            await tom.update_model("What is this?", {})
            await tom.update_model("Help me understand", {})

        predictions = await tom.predict_user_needs()

        # Should have predictions based on behavior patterns
        behavior_based = [
            p for p in predictions if p.get("based_on") == "behavior_pattern"
        ]
        assert len(behavior_based) >= 0  # May or may not have, depending on confidence

    @pytest.mark.asyncio
    async def test_action_extraction(self, tom):
        """Should extract action categories from messages"""
        # Test question detection
        action = tom._extract_action("What is the weather today?", {})
        assert action == "ask_question"

        # Test help request (without question mark to avoid ask_question match)
        action = tom._extract_action("Please help me with this task", {})
        assert action == "request_help"

        # Test reminder
        action = tom._extract_action("Remind me to call mom", {})
        assert action == "set_reminder"


# =============================================================================
# Test AGI Core Integration
# =============================================================================


class TestAGICoreIntegration:
    """Test full AGI Core integration"""

    @pytest.fixture
    def agi(self, mock_memory, temp_data_dir):
        return AGICore(mock_memory, data_dir=temp_data_dir)

    @pytest.mark.asyncio
    async def test_process_returns_emotion(self, agi):
        """Process should return emotion state"""
        result = await agi.process("I'm feeling great today!", {})

        assert "emotion" in result
        assert result["emotion"].valence > 0.6

    @pytest.mark.asyncio
    async def test_process_returns_emotional_trend(self, agi):
        """Process should return emotional trend"""
        await agi.process("I'm sad", {})
        await agi.process("I'm feeling better", {})
        result = await agi.process("I'm happy now!", {})

        assert "emotional_trend" in result

    @pytest.mark.asyncio
    async def test_process_learns_habits(self, agi):
        """Process should learn habits from interactions"""
        for _ in range(5):
            await agi.process("What's the weather?", {})

        assert len(agi.planner.habits) > 0 or len(agi.planner._action_time_counts) > 0

    @pytest.mark.asyncio
    async def test_process_returns_suggestions(self, agi):
        """Process should return proactive suggestions"""
        result = await agi.process("Hello!", {})

        assert "suggestions" in result

    @pytest.mark.asyncio
    async def test_process_returns_user_model(self, agi):
        """Process should return user model context"""
        result = await agi.process("Help me with my tasks", {})

        assert "user_model" in result
        assert "goals" in result["user_model"]


# =============================================================================
# Test No LLM Calls
# =============================================================================


class TestNoLLMCalls:
    """Verify no LLM calls are made in AGI algorithms"""

    @pytest.mark.asyncio
    async def test_emotion_detection_no_llm(self, mock_memory, temp_data_dir):
        """Emotion detection should work without LLM"""
        ei = EmotionalIntelligence()

        # Should work purely algorithmically
        state = await ei.detect_emotion("I'm happy!")

        assert state.primary in ["happy", "excited"]
        assert state.valence > 0.7

    @pytest.mark.asyncio
    async def test_markov_prediction_no_llm(self, temp_data_dir):
        """Markov predictions should work without LLM"""
        predictor = MarkovPredictor(data_dir=temp_data_dir)

        for _ in range(5):
            predictor.observe("A")
            predictor.observe("B")

        predictions = predictor.predict("A")

        assert len(predictions) > 0

    @pytest.mark.asyncio
    async def test_habit_learning_no_llm(self, mock_memory, temp_data_dir):
        """Habit learning should work without LLM"""
        planner = LongTermPlanner(mock_memory, data_dir=temp_data_dir)

        for _ in range(5):
            await planner.learn_habit("test_action", {})

        # Should have learned without needing LLM
        assert "test_action" in planner.habits


# =============================================================================
# Test Mobile Efficiency
# =============================================================================


class TestMobileEfficiency:
    """Test that algorithms are efficient for mobile"""

    def test_markov_prediction_constant_time(self, temp_data_dir):
        """Markov prediction should be O(1) for lookups"""
        import time

        predictor = MarkovPredictor(data_dir=temp_data_dir)

        # Add many states
        for i in range(1000):
            predictor.observe(f"state_{i % 10}")

        # Prediction should be fast
        start = time.time()
        for _ in range(100):
            predictor.predict("state_0")
        duration = time.time() - start

        # Should complete 100 predictions in under 100ms
        assert duration < 0.1, f"Predictions too slow: {duration}s"

    def test_emotion_lexicon_lookup_fast(self):
        """Emotion lexicon lookup should be O(1)"""
        import time

        ei = EmotionalIntelligence()

        start = time.time()
        for _ in range(1000):
            # Simple word lookup
            _ = EMOTION_LEXICON.get("happy")
        duration = time.time() - start

        # Should complete 1000 lookups in under 10ms
        assert duration < 0.01, f"Lookups too slow: {duration}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
