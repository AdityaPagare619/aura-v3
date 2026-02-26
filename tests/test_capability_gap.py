"""
Capability Gap Handler Tests
Tests for capability gap detection, strategy execution, and gap history tracking
"""

import pytest
import asyncio
import os
import sys
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.addons.capability_gap import (
    CapabilityType,
    Strategy,
    PrerequisiteResult,
    GapHistoryEntry,
    CapabilityGap,
    GapResolution,
    CapabilityGapHandler,
    get_capability_gap_handler,
    REPEATED_GAP_THRESHOLD,
    GAP_HISTORY_RETENTION_DAYS,
    MAX_GAP_HISTORY_ENTRIES,
)


class TestCapabilityType:
    def test_enum_values_exist(self):
        assert CapabilityType.IMAGE_RECOGNITION.value == "image_recognition"
        assert CapabilityType.SPEECH_RECOGNITION.value == "speech_recognition"
        assert CapabilityType.WEB_ACCESS.value == "web_access"
        assert CapabilityType.LOCATION_ACCESS.value == "location_access"
        assert CapabilityType.CAMERA_ACCESS.value == "camera_access"

    def test_mobile_specific_capabilities(self):
        assert CapabilityType.CONTACTS_ACCESS.value == "contacts_access"
        assert CapabilityType.SMS_ACCESS.value == "sms_access"
        assert CapabilityType.NOTIFICATION_ACCESS.value == "notification_access"


class TestStrategy:
    def test_basic_strategy_creation(self):
        strategy = Strategy(
            name="test_strategy",
            description="A test strategy",
        )
        assert strategy.name == "test_strategy"
        assert strategy.description == "A test strategy"
        assert strategy.estimated_success == 0.5
        assert strategy.cost == 0.5
        assert strategy.is_last_resort is False

    def test_strategy_with_requirements(self):
        strategy = Strategy(
            name="file_search",
            description="Search files",
            required_tool="find_files",
            required_permission="storage",
            required_service="termux",
            prerequisites=["gallery_access"],
        )
        assert strategy.required_tool == "find_files"
        assert strategy.required_permission == "storage"
        assert strategy.required_service == "termux"
        assert "gallery_access" in strategy.prerequisites

    def test_strategy_with_custom_function(self):
        async def custom_fn(goal, context):
            return {"success": True, "data": "custom result"}

        strategy = Strategy(
            name="custom",
            description="Custom strategy",
            attempt_fn=custom_fn,
        )
        assert strategy.attempt_fn is not None


class TestPrerequisiteResult:
    def test_satisfied_result(self):
        result = PrerequisiteResult(
            satisfied=True,
            tool_available=True,
            permission_granted=True,
            service_running=True,
        )
        assert result.satisfied is True
        assert len(result.missing) == 0

    def test_unsatisfied_result(self):
        result = PrerequisiteResult(
            satisfied=False,
            tool_available=False,
            missing=["tool:find_files"],
            suggestions=["Install find_files tool"],
        )
        assert result.satisfied is False
        assert "tool:find_files" in result.missing
        assert len(result.suggestions) == 1


class TestGapHistoryEntry:
    def test_entry_creation(self):
        entry = GapHistoryEntry(
            capability="image_recognition",
            user_goal="Find my photo",
            timestamp=datetime.now().isoformat(),
            resolved=True,
            strategy_used="file_search",
            attempts=2,
        )
        assert entry.capability == "image_recognition"
        assert entry.resolved is True
        assert entry.strategy_used == "file_search"

    def test_unresolved_entry(self):
        entry = GapHistoryEntry(
            capability="web_access",
            user_goal="Search the web",
            timestamp=datetime.now().isoformat(),
            resolved=False,
            error="Network unavailable",
        )
        assert entry.resolved is False
        assert entry.error == "Network unavailable"


class TestCapabilityGap:
    def test_gap_creation(self):
        gap = CapabilityGap(
            capability=CapabilityType.IMAGE_RECOGNITION,
            user_goal="Find my cat photo",
        )
        assert gap.capability == CapabilityType.IMAGE_RECOGNITION
        assert gap.user_goal == "Find my cat photo"
        assert gap.resolved is False
        assert len(gap.attempted_strategies) == 0

    def test_gap_with_results(self):
        gap = CapabilityGap(
            capability=CapabilityType.WEB_ACCESS,
            user_goal="Search something",
            attempted_strategies=["cached_data", "local_knowledge"],
            results=[{"strategy": "cached_data", "result": {"success": False}}],
            resolved=False,
            partial_result="Some cached data found",
        )
        assert len(gap.attempted_strategies) == 2
        assert gap.partial_result is not None


class TestGapResolution:
    def test_successful_resolution(self):
        resolution = GapResolution(
            success=True,
            result={"files": ["photo1.jpg", "photo2.jpg"]},
            strategy_used="file_search",
            message="Found 2 files",
        )
        assert resolution.success is True
        assert resolution.strategy_used == "file_search"

    def test_partial_resolution(self):
        resolution = GapResolution(
            success=False,
            result={"cached": True},
            strategy_used="cached_data",
            message="Partial result from cache",
            partial=True,
        )
        assert resolution.success is False
        assert resolution.partial is True


class TestCapabilityGapHandler:
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def handler(self, temp_storage):
        """Create handler with temp storage"""
        return CapabilityGapHandler(storage_path=temp_storage)

    def test_handler_initialization(self, handler):
        assert handler._strategies is not None
        assert handler._gap_history is not None
        assert handler._service_cache is not None

    def test_default_strategies_registered(self, handler):
        # Check IMAGE_RECOGNITION strategies
        img_strategies = handler._strategies.get(CapabilityType.IMAGE_RECOGNITION, [])
        strategy_names = [s.name for s in img_strategies]
        assert "file_search" in strategy_names
        assert "chat_history_search" in strategy_names
        assert "metadata_analysis" in strategy_names

        # Check LOCATION_ACCESS strategies
        loc_strategies = handler._strategies.get(CapabilityType.LOCATION_ACCESS, [])
        loc_names = [s.name for s in loc_strategies]
        assert "last_known" in loc_names
        assert "manual_confirmation" in loc_names

    def test_register_custom_strategy(self, handler):
        custom_strategy = Strategy(
            name="custom_image_strategy",
            description="Custom image handling",
            estimated_success=0.9,
        )
        handler.register_strategy(CapabilityType.IMAGE_RECOGNITION, custom_strategy)

        strategies = handler._strategies[CapabilityType.IMAGE_RECOGNITION]
        names = [s.name for s in strategies]
        assert "custom_image_strategy" in names

    def test_strategies_sorted_by_success_cost_ratio(self, handler):
        # Add strategies with different ratios
        handler.register_strategy(
            CapabilityType.VIDEO_ANALYSIS,
            Strategy(
                name="low_ratio", description="Low", estimated_success=0.1, cost=1.0
            ),
        )
        handler.register_strategy(
            CapabilityType.VIDEO_ANALYSIS,
            Strategy(
                name="high_ratio", description="High", estimated_success=0.9, cost=0.1
            ),
        )

        strategies = handler._strategies[CapabilityType.VIDEO_ANALYSIS]
        # high_ratio should come first (higher success/cost ratio)
        assert strategies[0].name == "high_ratio"

    @pytest.mark.asyncio
    async def test_check_capability_available(self, handler):
        # FILE_ANALYSIS is available by default
        result = await handler.check_capability(CapabilityType.FILE_ANALYSIS)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_capability_unavailable(self, handler):
        # IMAGE_RECOGNITION is not in default available set
        result = await handler.check_capability(CapabilityType.IMAGE_RECOGNITION)
        assert result is False

    def test_extract_search_terms(self, handler):
        terms = handler._extract_search_terms("find my cat photo from vacation")
        assert "cat" in terms
        assert "photo" in terms
        assert "vacation" in terms
        # Stop words should be removed
        assert "find" not in terms
        assert "my" not in terms
        assert "from" not in terms

    def test_extract_search_terms_max_5(self, handler):
        terms = handler._extract_search_terms(
            "one two three four five six seven eight nine ten"
        )
        assert len(terms) <= 5

    def test_suggest_alternatives(self, handler):
        alternatives = handler.suggest_alternatives(CapabilityType.IMAGE_RECOGNITION)
        assert len(alternatives) <= 3
        assert all(isinstance(a, str) for a in alternatives)


class TestPrerequisiteChecking:
    @pytest.fixture
    def temp_storage(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def handler(self, temp_storage):
        return CapabilityGapHandler(storage_path=temp_storage)

    @pytest.mark.asyncio
    async def test_check_prerequisites_no_requirements(self, handler):
        strategy = Strategy(
            name="no_reqs",
            description="No requirements",
        )
        result = await handler._check_prerequisites(strategy)
        assert result.satisfied is True

    @pytest.mark.asyncio
    async def test_check_prerequisites_with_missing_tool(self, handler):
        # Mock registry to return None for tool
        mock_registry = MagicMock()
        mock_registry.get_tool_definition.return_value = None
        handler._tool_registry = mock_registry

        strategy = Strategy(
            name="needs_tool",
            description="Needs a tool",
            required_tool="nonexistent_tool",
        )
        result = await handler._check_prerequisites(strategy)
        assert result.tool_available is False
        assert "tool:nonexistent_tool" in result.missing

    @pytest.mark.asyncio
    async def test_check_prerequisites_with_available_tool(self, handler):
        # Mock registry to return tool definition
        mock_registry = MagicMock()
        mock_tool = MagicMock()
        mock_tool.handler = MagicMock()
        mock_registry.get_tool_definition.return_value = mock_tool
        handler._tool_registry = mock_registry

        strategy = Strategy(
            name="has_tool",
            description="Has tool",
            required_tool="find_files",
        )
        result = await handler._check_prerequisites(strategy)
        assert result.tool_available is True

    @pytest.mark.asyncio
    async def test_check_permission_granted(self, handler):
        # Mock permission manager
        mock_perm = AsyncMock()
        mock_perm.check_permission.return_value = True
        handler._permission_manager = mock_perm

        result = await handler._check_permission_granted("storage")
        assert result is True
        mock_perm.check_permission.assert_called_once_with("storage")

    @pytest.mark.asyncio
    async def test_check_permission_denied(self, handler):
        mock_perm = AsyncMock()
        mock_perm.check_permission.return_value = False
        handler._permission_manager = mock_perm

        result = await handler._check_permission_granted("camera")
        assert result is False

    @pytest.mark.asyncio
    async def test_service_cache_works(self, handler):
        # First check should cache
        handler._service_cache["test_service"] = (True, datetime.now())

        result = await handler._check_service_running("test_service")
        assert result is True

    @pytest.mark.asyncio
    async def test_service_cache_expires(self, handler):
        # Set expired cache entry
        old_time = datetime.now() - timedelta(seconds=120)
        handler._service_cache["test_service"] = (True, old_time)
        handler._service_cache_ttl = 60

        # Should trigger new check since cache expired
        with patch.object(
            handler, "_check_service_running_impl", new_callable=AsyncMock
        ) as mock:
            mock.return_value = False
            result = await handler._check_service_running("test_service")
            assert result is False
            mock.assert_called_once()


class TestGapHistory:
    @pytest.fixture
    def temp_storage(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def handler(self, temp_storage):
        return CapabilityGapHandler(storage_path=temp_storage)

    def test_record_gap_history(self, handler):
        handler._record_gap_history(
            CapabilityType.IMAGE_RECOGNITION,
            "Find photo",
            True,
            "file_search",
            2,
        )
        assert len(handler._persistent_gap_history) == 1
        entry = handler._persistent_gap_history[0]
        assert entry.capability == "image_recognition"
        assert entry.resolved is True

    def test_save_and_load_gap_history(self, temp_storage):
        handler1 = CapabilityGapHandler(storage_path=temp_storage)
        handler1._record_gap_history(
            CapabilityType.WEB_ACCESS,
            "Search web",
            False,
            None,
            3,
            "Network error",
        )
        handler1._save_gap_history()

        # Create new handler - should load saved history
        handler2 = CapabilityGapHandler(storage_path=temp_storage)
        assert len(handler2._persistent_gap_history) == 1
        entry = handler2._persistent_gap_history[0]
        assert entry.capability == "web_access"
        assert entry.error == "Network error"

    def test_gap_frequency_tracking(self, handler):
        for _ in range(5):
            handler._record_gap_history(
                CapabilityType.LOCATION_ACCESS,
                "Get location",
                False,
                None,
                1,
            )

        freq = handler._gap_frequency.get("location_access", [])
        assert len(freq) == 5

    def test_check_repeated_gap_below_threshold(self, handler):
        # Add fewer than threshold
        for _ in range(REPEATED_GAP_THRESHOLD - 1):
            handler._record_gap_history(
                CapabilityType.CAMERA_ACCESS,
                "Take photo",
                False,
                None,
                1,
            )

        suggestions = handler._check_repeated_gap(CapabilityType.CAMERA_ACCESS)
        assert len(suggestions) == 0

    def test_check_repeated_gap_above_threshold(self, handler):
        # Add more than threshold
        for _ in range(REPEATED_GAP_THRESHOLD + 1):
            handler._record_gap_history(
                CapabilityType.CAMERA_ACCESS,
                "Take photo",
                False,
                None,
                1,
            )

        suggestions = handler._check_repeated_gap(CapabilityType.CAMERA_ACCESS)
        assert len(suggestions) > 0

    def test_get_proactive_suggestions(self, handler):
        suggestions = handler._get_proactive_suggestions(
            CapabilityType.IMAGE_RECOGNITION
        )
        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)

    def test_get_gap_statistics(self, handler):
        handler._record_gap_history(
            CapabilityType.IMAGE_RECOGNITION, "Find photo", True, "file_search", 1
        )
        handler._record_gap_history(
            CapabilityType.IMAGE_RECOGNITION, "Find video", False, None, 2
        )
        handler._record_gap_history(
            CapabilityType.WEB_ACCESS, "Search", True, "cached_data", 1
        )

        stats = handler.get_gap_statistics()
        assert stats["total_gaps"] == 3
        assert stats["resolved_count"] == 2
        assert stats["unresolved_count"] == 1
        assert "image_recognition" in stats["by_capability"]

    def test_history_pruning(self, temp_storage):
        handler = CapabilityGapHandler(storage_path=temp_storage)

        # Add old entry (beyond retention period)
        old_timestamp = (
            datetime.now() - timedelta(days=GAP_HISTORY_RETENTION_DAYS + 1)
        ).isoformat()
        handler._persistent_gap_history.append(
            GapHistoryEntry(
                capability="old_cap",
                user_goal="old goal",
                timestamp=old_timestamp,
                resolved=True,
            )
        )

        # Add recent entry
        handler._record_gap_history(
            CapabilityType.WEB_ACCESS, "Recent", True, "cached", 1
        )

        handler._save_gap_history()

        # Reload and check old entry was pruned
        handler2 = CapabilityGapHandler(storage_path=temp_storage)
        caps = [e.capability for e in handler2._persistent_gap_history]
        assert "old_cap" not in caps
        assert "web_access" in caps


class TestHandleGap:
    @pytest.fixture
    def temp_storage(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def handler(self, temp_storage):
        return CapabilityGapHandler(storage_path=temp_storage)

    @pytest.mark.asyncio
    async def test_handle_gap_no_strategies(self, handler):
        # VIDEO_ANALYSIS has no default strategies
        result = await handler.handle_gap(
            "Analyze this video",
            CapabilityType.VIDEO_ANALYSIS,
        )
        assert result.success is False
        assert "No strategies available" in result.message

    @pytest.mark.asyncio
    async def test_handle_gap_text_fallback_success(self, handler):
        # SPEECH_RECOGNITION has text_fallback which always succeeds
        result = await handler.handle_gap(
            "Transcribe audio",
            CapabilityType.SPEECH_RECOGNITION,
        )
        assert result.success is True
        assert result.strategy_used == "text_fallback"

    @pytest.mark.asyncio
    async def test_handle_gap_records_history(self, handler):
        await handler.handle_gap(
            "Find location",
            CapabilityType.LOCATION_ACCESS,
        )
        assert len(handler._persistent_gap_history) > 0

    @pytest.mark.asyncio
    async def test_handle_gap_with_custom_strategy(self, handler):
        async def always_succeed(goal, ctx):
            return {"success": True, "data": "Custom success!"}

        custom = Strategy(
            name="always_works",
            description="Always succeeds",
            attempt_fn=always_succeed,
            estimated_success=1.0,
        )
        handler.register_strategy(CapabilityType.TRANSLATION, custom)

        result = await handler.handle_gap(
            "Translate this",
            CapabilityType.TRANSLATION,
        )
        assert result.success is True
        assert result.strategy_used == "always_works"

    @pytest.mark.asyncio
    async def test_handle_gap_tries_multiple_strategies(self, handler):
        call_count = {"count": 0}

        async def fail_twice(goal, ctx):
            call_count["count"] += 1
            if call_count["count"] < 3:
                return {"success": False, "error": "Not yet"}
            return {"success": True, "data": "Finally!"}

        # Clear existing strategies and add custom ones
        handler._strategies[CapabilityType.TEXT_ANALYSIS] = []

        for i in range(3):
            handler.register_strategy(
                CapabilityType.TEXT_ANALYSIS,
                Strategy(
                    name=f"strategy_{i}",
                    description=f"Strategy {i}",
                    attempt_fn=fail_twice,
                ),
            )

        result = await handler.handle_gap(
            "Analyze text",
            CapabilityType.TEXT_ANALYSIS,
        )
        # Should have tried multiple strategies
        assert call_count["count"] >= 1


class TestStrategyExecution:
    @pytest.fixture
    def temp_storage(self):
        temp_dir = tempfile.mkdtemp()
        # Create cache directory for cached strategy test
        cache_dir = Path(temp_dir).parent / "cache" / "web"
        cache_dir.mkdir(parents=True, exist_ok=True)
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(str(cache_dir.parent), ignore_errors=True)

    @pytest.fixture
    def handler(self, temp_storage):
        return CapabilityGapHandler(storage_path=temp_storage)

    @pytest.mark.asyncio
    async def test_strategy_local_knowledge(self, handler):
        result = await handler._strategy_local_knowledge("What is Python?", {})
        assert result["success"] is True
        assert result["data"]["type"] == "knowledge_query"

    @pytest.mark.asyncio
    async def test_strategy_manual(self, handler):
        result = await handler._strategy_manual("Enter your location", {})
        assert result["success"] is True
        assert result["data"]["type"] == "user_input_required"

    @pytest.mark.asyncio
    async def test_strategy_text_fallback(self, handler):
        result = await handler._strategy_text_fallback("Voice command", {})
        assert result["success"] is True
        assert result["data"]["type"] == "fallback_mode"
        assert result["data"]["fallback_mode"] == "text"

    @pytest.mark.asyncio
    async def test_strategy_last_location_no_history(self, handler):
        result = await handler._strategy_last_location("Where am I?", {})
        assert result["success"] is False
        assert "No location history" in result["error"]

    @pytest.mark.asyncio
    async def test_strategy_last_location_with_history(self, handler):
        # Create location history file
        history_path = Path(handler._storage_path) / "location_history.json"
        history_data = {
            "last_location": {"lat": 40.7128, "lon": -74.0060},
            "timestamp": datetime.now().isoformat(),
        }
        history_path.write_text(json.dumps(history_data))

        result = await handler._strategy_last_location("Where am I?", {})
        assert result["success"] is True
        assert result["partial"] is True
        assert result["data"]["lat"] == 40.7128

    @pytest.mark.asyncio
    async def test_strategy_metadata_no_file(self, handler):
        result = await handler._strategy_metadata("Analyze file", {})
        assert result["success"] is False
        assert "No file path" in result["error"]

    @pytest.mark.asyncio
    async def test_strategy_file_search_no_bridge(self, handler):
        handler._termux_bridge = None
        with patch.object(
            handler, "_get_termux_bridge", new_callable=AsyncMock
        ) as mock:
            mock.return_value = None
            result = await handler._strategy_file_search("Find photos", {})
            assert result["success"] is False
            assert "not available" in result["error"]


class TestSuccessPatterns:
    @pytest.fixture
    def temp_storage(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def handler(self, temp_storage):
        return CapabilityGapHandler(storage_path=temp_storage)

    def test_record_success(self, handler):
        handler._record_success("file_search", CapabilityType.IMAGE_RECOGNITION)
        patterns = handler.get_successful_patterns()
        assert "image_recognition:file_search" in patterns
        assert patterns["image_recognition:file_search"] == 1

    def test_record_success_increments(self, handler):
        handler._record_success("cached_data", CapabilityType.WEB_ACCESS)
        handler._record_success("cached_data", CapabilityType.WEB_ACCESS)
        handler._record_success("cached_data", CapabilityType.WEB_ACCESS)

        patterns = handler.get_successful_patterns()
        assert patterns["web_access:cached_data"] == 3


class TestGlobalInstance:
    def test_get_capability_gap_handler(self):
        # Reset global instance
        import src.addons.capability_gap as module

        module._gap_handler = None

        handler1 = get_capability_gap_handler()
        handler2 = get_capability_gap_handler()

        assert handler1 is handler2  # Same instance


class TestConstants:
    def test_threshold_constants(self):
        assert REPEATED_GAP_THRESHOLD == 3
        assert GAP_HISTORY_RETENTION_DAYS == 30
        assert MAX_GAP_HISTORY_ENTRIES == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
