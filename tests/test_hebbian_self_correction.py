"""
Tests for AURA v3 Hebbian Self-Correction

Tests both standalone mode (LocalActivationMap) and integrated mode (with neural_memory).
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock

from src.core.hebbian_self_correction import (
    LocalActivationMap,
    HebbianSelfCorrector,
    HebbianCorrection,
    ActionOutcome,
    create_hebbian_corrector,
    get_hebbian_corrector,
)


class TestLocalActivationMap:
    """Tests for LocalActivationMap (standalone Hebbian learning)"""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def activation_map(self, temp_storage):
        """Create LocalActivationMap with temp storage"""
        return LocalActivationMap(storage_path=temp_storage)

    def test_init_creates_storage(self, temp_storage):
        """Test that initialization creates storage directory and DB"""
        lam = LocalActivationMap(storage_path=temp_storage)

        assert os.path.exists(temp_storage)
        assert os.path.exists(lam.db_path)

    def test_strengthen_creates_bidirectional_connection(self, activation_map):
        """Test that strengthen creates bidirectional connections"""
        lam = activation_map

        strength = lam.strengthen("concept_a", "concept_b", 0.2)

        assert strength > 0
        assert "concept_b" in lam.connections["concept_a"]
        assert "concept_a" in lam.connections["concept_b"]
        assert (
            lam.connections["concept_a"]["concept_b"]
            == lam.connections["concept_b"]["concept_a"]
        )

    def test_strengthen_accumulates(self, activation_map):
        """Test that multiple strengthen calls accumulate strength"""
        lam = activation_map

        s1 = lam.strengthen("a", "b", 0.1)
        s2 = lam.strengthen("a", "b", 0.1)
        s3 = lam.strengthen("a", "b", 0.1)

        assert s3 > s1
        assert lam.connections["a"]["b"] >= 0.3

    def test_strengthen_capped_at_max(self, activation_map):
        """Test that strength doesn't exceed max"""
        lam = activation_map

        for _ in range(20):
            lam.strengthen("a", "b", 0.2)

        assert lam.connections["a"]["b"] <= lam.max_strength

    def test_weaken_reduces_strength(self, activation_map):
        """Test that weaken reduces connection strength"""
        lam = activation_map

        lam.strengthen("a", "b", 0.5)
        initial = lam.connections["a"]["b"]

        lam.weaken("a", "b", 0.2)

        assert lam.connections["a"]["b"] < initial

    def test_weaken_floored_at_min(self, activation_map):
        """Test that strength doesn't go below minimum"""
        lam = activation_map

        lam.strengthen("a", "b", 0.1)

        for _ in range(20):
            lam.weaken("a", "b", 0.2)

        assert lam.connections["a"]["b"] >= lam.min_strength

    def test_decay_reduces_all_connections(self, activation_map):
        """Test that decay reduces all connections"""
        lam = activation_map

        lam.strengthen("a", "b", 0.5)
        lam.strengthen("c", "d", 0.5)

        initial_ab = lam.connections["a"]["b"]
        initial_cd = lam.connections["c"]["d"]

        lam.decay(rate=0.1)

        assert lam.connections["a"]["b"] < initial_ab
        assert lam.connections["c"]["d"] < initial_cd

    def test_get_associated_returns_sorted(self, activation_map):
        """Test that get_associated returns sorted by strength"""
        lam = activation_map

        lam.strengthen("main", "weak", 0.1)
        lam.strengthen("main", "medium", 0.3)
        lam.strengthen("main", "strong", 0.5)

        associated = lam.get_associated("main", top_k=3)

        assert len(associated) == 3
        assert associated[0][0] == "strong"
        assert associated[1][0] == "medium"
        assert associated[2][0] == "weak"

    def test_get_associated_respects_top_k(self, activation_map):
        """Test that get_associated returns only top_k results"""
        lam = activation_map

        for i in range(10):
            lam.strengthen("main", f"concept_{i}", 0.1 * (i + 1))

        associated = lam.get_associated("main", top_k=3)

        assert len(associated) == 3

    def test_co_activate_strengthens_all_pairs(self, activation_map):
        """Test that co_activate strengthens all concept pairs"""
        lam = activation_map

        concepts = ["a", "b", "c"]
        lam.co_activate(concepts, amount=0.2)

        # All pairs should be connected
        assert "b" in lam.connections["a"]
        assert "c" in lam.connections["a"]
        assert "c" in lam.connections["b"]

    def test_record_outcome_tracks_success(self, activation_map):
        """Test that record_outcome tracks success"""
        lam = activation_map

        lam.record_outcome("action_1", success=True)
        lam.record_outcome("action_1", success=True)
        lam.record_outcome("action_1", success=False)

        meta = lam.concept_metadata["action_1"]
        assert meta["success_count"] == 2
        assert meta["failure_count"] == 1
        assert meta["access_count"] == 3

    def test_record_outcome_updates_emotional_valence(self, activation_map):
        """Test that success increases and failure decreases emotional valence"""
        lam = activation_map

        lam.record_outcome("happy", success=True)
        lam.record_outcome("happy", success=True)

        lam.record_outcome("sad", success=False)
        lam.record_outcome("sad", success=False)

        assert lam.concept_metadata["happy"]["emotional_valence"] > 0
        assert lam.concept_metadata["sad"]["emotional_valence"] < 0

    def test_get_concept_score(self, activation_map):
        """Test concept scoring based on success rate"""
        lam = activation_map

        # Create a successful concept
        for _ in range(5):
            lam.record_outcome("good", success=True)

        # Create a failing concept
        for _ in range(5):
            lam.record_outcome("bad", success=False)

        good_score = lam.get_concept_score("good")
        bad_score = lam.get_concept_score("bad")

        assert good_score > bad_score

    def test_persistence(self, temp_storage):
        """Test that connections persist across instances"""
        # Create and populate
        lam1 = LocalActivationMap(storage_path=temp_storage)
        lam1.strengthen("persistent_a", "persistent_b", 0.5)
        lam1.record_outcome("persistent_action", success=True)

        # Create new instance (should load from DB)
        lam2 = LocalActivationMap(storage_path=temp_storage)

        assert "persistent_b" in lam2.connections.get("persistent_a", {})
        assert "persistent_action" in lam2.concept_metadata

    def test_self_connections_ignored(self, activation_map):
        """Test that self-connections return 0"""
        lam = activation_map

        result = lam.strengthen("same", "same", 0.5)

        assert result == 0.0
        assert "same" not in lam.connections.get("same", {})

    def test_concepts_normalized(self, activation_map):
        """Test that concepts are normalized (lowercase, stripped)"""
        lam = activation_map

        lam.strengthen("  UPPER  ", "  LOWER  ", 0.3)

        assert "lower" in lam.connections["upper"]
        assert "upper" in lam.connections["lower"]


class TestHebbianSelfCorrector:
    """Tests for HebbianSelfCorrector"""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def standalone_corrector(self, temp_storage):
        """Create standalone corrector (no neural memory)"""
        return HebbianSelfCorrector(neural_memory=None, storage_path=temp_storage)

    @pytest.fixture
    def mock_neural_memory(self):
        """Create mock neural memory for integrated mode testing"""
        memory = MagicMock()
        memory.recall = AsyncMock(return_value=[])
        memory.get_neuron = Mock(return_value=None)
        return memory

    @pytest.fixture
    def integrated_corrector(self, temp_storage, mock_neural_memory):
        """Create corrector with neural memory"""
        return HebbianSelfCorrector(
            neural_memory=mock_neural_memory, storage_path=temp_storage
        )

    def test_standalone_mode_works(self, standalone_corrector):
        """Test that standalone mode initializes correctly"""
        assert standalone_corrector.is_standalone
        assert standalone_corrector.local_activation_map is not None

    def test_integrated_mode_works(self, integrated_corrector):
        """Test that integrated mode initializes correctly"""
        assert not integrated_corrector.is_standalone
        assert integrated_corrector.neural_memory is not None

    def test_set_neural_memory(self, standalone_corrector, mock_neural_memory):
        """Test dynamic neural memory injection"""
        assert standalone_corrector.is_standalone

        standalone_corrector.set_neural_memory(mock_neural_memory)

        assert not standalone_corrector.is_standalone
        assert standalone_corrector.neural_memory == mock_neural_memory

    @pytest.mark.asyncio
    async def test_record_success_outcome_standalone(self, standalone_corrector):
        """Test recording success outcome in standalone mode"""
        result = await standalone_corrector.record_outcome(
            action="send_message",
            params={"to": "user", "content": "hello"},
            outcome=ActionOutcome.SUCCESS,
            context={"activity": "chatting"},
        )

        # Success should return None (no alternatives needed)
        assert result is None

        # Should have strengthened local activation map
        score = standalone_corrector.local_activation_map.get_concept_score(
            "send_message"
        )
        assert score > 0.5  # Should be positive

    @pytest.mark.asyncio
    async def test_record_failure_outcome_returns_alternatives(
        self, standalone_corrector
    ):
        """Test that failure outcome returns alternatives"""
        # First, create some associations
        standalone_corrector.local_activation_map.strengthen(
            "failed_action", "alternative_1", 0.5
        )
        standalone_corrector.local_activation_map.strengthen(
            "failed_action", "alternative_2", 0.4
        )

        result = await standalone_corrector.record_outcome(
            action="failed_action", params={}, outcome=ActionOutcome.FAILURE, context={}
        )

        # Should return alternatives
        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_record_partial_outcome(self, standalone_corrector):
        """Test recording partial outcome"""
        result = await standalone_corrector.record_outcome(
            action="partial_action",
            params={},
            outcome=ActionOutcome.PARTIAL,
            context={},
        )

        # Partial can still return alternatives
        # (or None if none found)
        assert result is None or isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_action_recommendation_standalone(self, standalone_corrector):
        """Test getting action recommendation in standalone mode"""
        # Create history
        standalone_corrector.local_activation_map.record_outcome(
            "good_action", success=True
        )
        standalone_corrector.local_activation_map.record_outcome(
            "good_action", success=True
        )

        recommendation = await standalone_corrector.get_action_recommendation(
            desired_action="good_action", context={}
        )

        assert "recommended_action" in recommendation
        assert "confidence" in recommendation
        assert "reason" in recommendation
        assert "mode" in recommendation
        assert recommendation["mode"] == "standalone"

    @pytest.mark.asyncio
    async def test_get_action_recommendation_integrated(
        self, integrated_corrector, mock_neural_memory
    ):
        """Test getting action recommendation in integrated mode"""
        # Mock neuron
        mock_neuron = Mock()
        mock_neuron.id = "n1"
        mock_neuron.importance = 0.8
        mock_neuron.emotional_valence = 0.5
        mock_neuron.access_count = 10
        mock_neuron.content = "test_action:params"

        mock_neural_memory.recall = AsyncMock(return_value=[mock_neuron])

        recommendation = await integrated_corrector.get_action_recommendation(
            desired_action="test_action", context={}
        )

        assert "recommended_action" in recommendation
        assert recommendation["mode"] == "integrated"

    def test_record_user_correction(self, standalone_corrector):
        """Test recording user corrections"""
        standalone_corrector.record_user_correction(
            incorrect_action="wrong_action",
            correct_action="right_action",
            context={"topic": "testing"},
        )

        # Check that history was recorded
        assert len(standalone_corrector.correction_history) == 1

        # Check that correct action was strengthened
        correct_score = standalone_corrector.local_activation_map.get_concept_score(
            "right_action"
        )
        incorrect_score = standalone_corrector.local_activation_map.get_concept_score(
            "wrong_action"
        )

        assert correct_score > incorrect_score

    def test_apply_decay(self, standalone_corrector):
        """Test applying decay"""
        lam = standalone_corrector.local_activation_map

        lam.strengthen("a", "b", 0.5)
        initial = lam.connections["a"]["b"]

        standalone_corrector.apply_decay(rate=0.1)

        assert lam.connections["a"]["b"] < initial

    def test_get_correction_stats_empty(self, standalone_corrector):
        """Test stats with no corrections"""
        stats = standalone_corrector.get_correction_stats()

        assert stats["total_corrections"] == 0
        assert stats["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_get_correction_stats_with_history(self, standalone_corrector):
        """Test stats with some corrections"""
        # Record some outcomes
        await standalone_corrector.record_outcome(
            action="action1", params={}, outcome=ActionOutcome.FAILURE, context={}
        )
        await standalone_corrector.record_outcome(
            action="action2", params={}, outcome=ActionOutcome.FAILURE, context={}
        )

        stats = standalone_corrector.get_correction_stats()

        assert stats["total_corrections"] >= 2
        assert "failures" in stats
        assert "mode" in stats

    def test_context_concept_extraction(self, standalone_corrector):
        """Test that context concepts are properly extracted"""
        concepts = standalone_corrector._extract_context_concepts(
            action="test_action",
            params={"param1": "value1", "param2": 123},
            context={"current_activity": "testing", "location": "office"},
        )

        assert "test_action" in concepts
        assert "value1" in concepts
        assert "testing" in concepts
        assert "office" in concepts


class TestFactoryFunctions:
    """Tests for factory functions"""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_create_hebbian_corrector(self, temp_storage):
        """Test create_hebbian_corrector factory"""
        corrector = create_hebbian_corrector(storage_path=temp_storage)

        assert isinstance(corrector, HebbianSelfCorrector)
        assert corrector.is_standalone

    def test_create_hebbian_corrector_with_neural_memory(self, temp_storage):
        """Test create_hebbian_corrector with neural memory"""
        mock_memory = MagicMock()
        corrector = create_hebbian_corrector(
            neural_memory=mock_memory, storage_path=temp_storage
        )

        assert not corrector.is_standalone


class TestHebbianIntegration:
    """Integration tests for Hebbian self-correction"""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_full_learning_cycle(self, temp_storage):
        """Test complete learning cycle: action -> outcome -> recommendation"""
        corrector = create_hebbian_corrector(storage_path=temp_storage)

        # Simulate successful action multiple times
        for _ in range(5):
            await corrector.record_outcome(
                action="send_email",
                params={"to": "user@example.com"},
                outcome=ActionOutcome.SUCCESS,
                context={"task": "communication"},
            )

        # Simulate failed alternative
        await corrector.record_outcome(
            action="send_sms",
            params={"to": "1234567890"},
            outcome=ActionOutcome.FAILURE,
            context={"task": "communication"},
        )

        # Get recommendation for communication task
        recommendation = await corrector.get_action_recommendation(
            desired_action="send_email", context={"task": "communication"}
        )

        # Should recommend email (higher success rate)
        assert recommendation["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_user_correction_improves_recommendations(self, temp_storage):
        """Test that user corrections improve future recommendations"""
        corrector = create_hebbian_corrector(storage_path=temp_storage)

        # Initial wrong associations
        corrector.local_activation_map.strengthen("task", "wrong_action", 0.6)
        corrector.local_activation_map.strengthen("task", "right_action", 0.3)

        # User corrects
        corrector.record_user_correction(
            incorrect_action="wrong_action",
            correct_action="right_action",
            context={"task": "important"},
        )

        # Check that associations changed
        wrong_assoc = corrector.local_activation_map.connections.get("task", {}).get(
            "wrong_action", 0
        )
        right_assoc = corrector.local_activation_map.connections.get("task", {}).get(
            "right_action", 0
        )

        # Right action should be strengthened more than wrong
        right_score = corrector.local_activation_map.get_concept_score("right_action")
        wrong_score = corrector.local_activation_map.get_concept_score("wrong_action")

        assert right_score > wrong_score

    @pytest.mark.asyncio
    async def test_seamless_mode_switching(self, temp_storage):
        """Test seamless switching between standalone and integrated modes"""
        corrector = create_hebbian_corrector(storage_path=temp_storage)

        # Start standalone
        assert corrector.is_standalone

        await corrector.record_outcome(
            action="standalone_action",
            params={},
            outcome=ActionOutcome.SUCCESS,
            context={},
        )

        # Switch to integrated
        mock_memory = MagicMock()
        mock_memory.recall = AsyncMock(return_value=[])
        corrector.set_neural_memory(mock_memory)

        assert not corrector.is_standalone

        # Record in integrated mode
        await corrector.record_outcome(
            action="integrated_action",
            params={},
            outcome=ActionOutcome.SUCCESS,
            context={},
        )

        # Both actions should be recorded locally
        assert "standalone_action" in corrector.local_activation_map.concept_metadata
        assert "integrated_action" in corrector.local_activation_map.concept_metadata

    @pytest.mark.asyncio
    async def test_persistence_across_sessions(self, temp_storage):
        """Test that learning persists across corrector instances"""
        # Session 1: Learn something
        corrector1 = create_hebbian_corrector(storage_path=temp_storage)

        for _ in range(5):
            await corrector1.record_outcome(
                action="persistent_action",
                params={},
                outcome=ActionOutcome.SUCCESS,
                context={},
            )

        corrector1.record_user_correction("bad", "good", {})

        # Session 2: New instance should have learned
        corrector2 = create_hebbian_corrector(storage_path=temp_storage)

        # Check persistence
        assert "persistent_action" in corrector2.local_activation_map.concept_metadata
        meta = corrector2.local_activation_map.concept_metadata["persistent_action"]
        assert meta["success_count"] == 5
