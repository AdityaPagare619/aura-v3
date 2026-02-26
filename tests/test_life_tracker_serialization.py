"""
Tests for life_tracker.py serialization
Verifies that save/load cycles preserve all data types correctly
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

from src.services.life_tracker import (
    LifeTracker,
    LifeEvent,
    UserPattern,
    SocialInsight,
    EventCategory,
    EventPriority,
    EventStatus,
    _datetime_to_iso,
    _iso_to_datetime,
)


class TestDatetimeHelpers:
    """Test datetime serialization helpers"""

    def test_datetime_to_iso_with_datetime(self):
        dt = datetime(2026, 2, 26, 14, 30, 0)
        result = _datetime_to_iso(dt)
        assert result == "2026-02-26T14:30:00"

    def test_datetime_to_iso_with_none(self):
        result = _datetime_to_iso(None)
        assert result is None

    def test_iso_to_datetime_with_string(self):
        iso_str = "2026-02-26T14:30:00"
        result = _iso_to_datetime(iso_str)
        assert result == datetime(2026, 2, 26, 14, 30, 0)

    def test_iso_to_datetime_with_none(self):
        result = _iso_to_datetime(None)
        assert result is None

    def test_iso_to_datetime_with_datetime_passthrough(self):
        """If already datetime, should return as-is"""
        dt = datetime(2026, 2, 26, 14, 30, 0)
        result = _iso_to_datetime(dt)
        assert result == dt

    def test_iso_to_datetime_with_invalid_string(self):
        result = _iso_to_datetime("not-a-date")
        assert result is None


class TestLifeEventSerialization:
    """Test LifeEvent to_dict/from_dict"""

    def test_to_dict_basic(self):
        event = LifeEvent(
            id="test123",
            title="Test Event",
            description="A test event",
            category=EventCategory.WORK,
            priority=EventPriority.HIGH,
            status=EventStatus.PENDING,
        )
        data = event.to_dict()

        assert data["id"] == "test123"
        assert data["title"] == "Test Event"
        assert data["category"] == "work"  # Enum converted to value
        assert data["priority"] == 2  # Enum converted to int value
        assert data["status"] == "pending"  # Enum converted to value

    def test_to_dict_with_datetimes(self):
        event_date = datetime(2026, 3, 15, 10, 0)
        deadline = datetime(2026, 3, 14, 18, 0)

        event = LifeEvent(
            title="Meeting",
            event_date=event_date,
            deadline=deadline,
        )
        data = event.to_dict()

        assert data["event_date"] == "2026-03-15T10:00:00"
        assert data["deadline"] == "2026-03-14T18:00:00"

    def test_to_dict_with_lists(self):
        event = LifeEvent(
            title="Team Meeting",
            related_people=["Alice", "Bob"],
            tags=["work", "important"],
            notes=["Note 1", "Note 2"],
        )
        data = event.to_dict()

        assert data["related_people"] == ["Alice", "Bob"]
        assert data["tags"] == ["work", "important"]
        assert data["notes"] == ["Note 1", "Note 2"]

    def test_from_dict_basic(self):
        data = {
            "id": "abc123",
            "title": "Loaded Event",
            "category": "family",
            "priority": 3,  # CRITICAL
            "status": "completed",
        }
        event = LifeEvent.from_dict(data)

        assert event.id == "abc123"
        assert event.title == "Loaded Event"
        assert event.category == EventCategory.FAMILY
        assert event.priority == EventPriority.CRITICAL
        assert event.status == EventStatus.COMPLETED

    def test_from_dict_with_datetimes(self):
        data = {
            "title": "Event with dates",
            "created_at": "2026-02-26T10:00:00",
            "event_date": "2026-03-15T14:00:00",
            "deadline": "2026-03-14T23:59:59",
        }
        event = LifeEvent.from_dict(data)

        assert isinstance(event.created_at, datetime)
        assert event.created_at == datetime(2026, 2, 26, 10, 0, 0)
        assert isinstance(event.event_date, datetime)
        assert event.event_date == datetime(2026, 3, 15, 14, 0, 0)

    def test_roundtrip_preserves_data(self):
        """Save to dict, load back - all data should be identical"""
        original = LifeEvent(
            id="round123",
            title="Roundtrip Test",
            description="Testing full roundtrip",
            category=EventCategory.HEALTH,
            priority=EventPriority.MEDIUM,
            status=EventStatus.IN_PROGRESS,
            event_date=datetime(2026, 4, 1, 9, 0),
            deadline=datetime(2026, 3, 31, 17, 0),
            source="whatsapp",
            source_app="WhatsApp",
            related_people=["Mom", "Dad"],
            related_topics=["health", "appointment"],
            location="Hospital",
            actions_taken=["Called to confirm"],
            notes=["Bring insurance card"],
            insights={"importance": "high"},
            prep_status="in_progress",
            is_recurring=True,
            recurring_pattern="monthly",
            tags=["medical", "family"],
        )

        # Roundtrip
        data = original.to_dict()
        restored = LifeEvent.from_dict(data)

        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.description == original.description
        assert restored.category == original.category
        assert restored.priority == original.priority
        assert restored.status == original.status
        assert restored.event_date == original.event_date
        assert restored.deadline == original.deadline
        assert restored.source == original.source
        assert restored.source_app == original.source_app
        assert restored.related_people == original.related_people
        assert restored.related_topics == original.related_topics
        assert restored.location == original.location
        assert restored.actions_taken == original.actions_taken
        assert restored.notes == original.notes
        assert restored.insights == original.insights
        assert restored.prep_status == original.prep_status
        assert restored.is_recurring == original.is_recurring
        assert restored.recurring_pattern == original.recurring_pattern
        assert restored.tags == original.tags


class TestUserPatternSerialization:
    """Test UserPattern to_dict/from_dict"""

    def test_roundtrip_preserves_data(self):
        original = UserPattern(
            id="pattern1",
            pattern_type="timing",
            description="Morning meetings",
            confidence=0.85,
            first_seen=datetime(2026, 1, 1, 8, 0),
            last_seen=datetime(2026, 2, 26, 9, 0),
            frequency=15,
            context={"category": "work", "hour": 9},
        )

        data = original.to_dict()
        restored = UserPattern.from_dict(data)

        assert restored.id == original.id
        assert restored.pattern_type == original.pattern_type
        assert restored.description == original.description
        assert restored.confidence == original.confidence
        assert restored.first_seen == original.first_seen
        assert restored.last_seen == original.last_seen
        assert restored.frequency == original.frequency
        assert restored.context == original.context


class TestSocialInsightSerialization:
    """Test SocialInsight to_dict/from_dict"""

    def test_roundtrip_preserves_data(self):
        original = SocialInsight(
            id="insight1",
            platform="twitter",
            insight_type="interest",
            content="AI and machine learning",
            confidence=0.92,
            created_at=datetime(2026, 2, 20, 15, 30),
            actionable=True,
            suggested_action="Share relevant AI news",
        )

        data = original.to_dict()
        restored = SocialInsight.from_dict(data)

        assert restored.id == original.id
        assert restored.platform == original.platform
        assert restored.insight_type == original.insight_type
        assert restored.content == original.content
        assert restored.confidence == original.confidence
        assert restored.created_at == original.created_at
        assert restored.actionable == original.actionable
        assert restored.suggested_action == original.suggested_action


class TestLifeTrackerPersistence:
    """Test LifeTracker save/load functionality"""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary file for storage"""
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_save_and_load_events(self, temp_storage):
        """Test that events survive save/load cycle"""
        tracker = LifeTracker(storage_path=temp_storage)

        # Add an event directly to internal storage (bypassing async start)
        event = LifeEvent(
            id="persist1",
            title="Persistence Test",
            category=EventCategory.WORK,
            priority=EventPriority.HIGH,
            event_date=datetime(2026, 3, 1, 10, 0),
        )
        tracker._events[event.id] = event

        # Save
        await tracker._save_data()

        # Verify JSON is valid and readable
        with open(temp_storage, "r") as f:
            saved_data = json.load(f)

        assert len(saved_data["events"]) == 1
        assert saved_data["events"][0]["id"] == "persist1"
        assert saved_data["events"][0]["category"] == "work"

        # Create new tracker and load
        tracker2 = LifeTracker(storage_path=temp_storage)
        await tracker2._load_data()

        assert len(tracker2._events) == 1
        loaded_event = tracker2._events["persist1"]
        assert loaded_event.title == "Persistence Test"
        assert loaded_event.category == EventCategory.WORK
        assert loaded_event.priority == EventPriority.HIGH
        assert loaded_event.event_date == datetime(2026, 3, 1, 10, 0)

    @pytest.mark.asyncio
    async def test_save_and_load_all_data_types(self, temp_storage):
        """Test that events, patterns, and insights all survive"""
        tracker = LifeTracker(storage_path=temp_storage)

        # Add event
        tracker._events["e1"] = LifeEvent(
            id="e1", title="Event 1", category=EventCategory.FAMILY
        )

        # Add pattern
        tracker._patterns["p1"] = UserPattern(
            id="p1", pattern_type="timing", confidence=0.75
        )

        # Add insight
        tracker._social_insights["i1"] = SocialInsight(
            id="i1", platform="instagram", content="Loves photography"
        )

        # Save
        await tracker._save_data()

        # Load into fresh tracker
        tracker2 = LifeTracker(storage_path=temp_storage)
        await tracker2._load_data()

        assert len(tracker2._events) == 1
        assert len(tracker2._patterns) == 1
        assert len(tracker2._social_insights) == 1

        assert tracker2._events["e1"].category == EventCategory.FAMILY
        assert tracker2._patterns["p1"].confidence == 0.75
        assert tracker2._social_insights["i1"].platform == "instagram"

    @pytest.mark.asyncio
    async def test_json_is_human_readable(self, temp_storage):
        """Verify saved JSON is formatted and human-readable"""
        tracker = LifeTracker(storage_path=temp_storage)
        tracker._events["e1"] = LifeEvent(id="e1", title="Test")
        await tracker._save_data()

        with open(temp_storage, "r") as f:
            content = f.read()

        # Should be indented (not a single line)
        assert "\n" in content
        # Should be valid JSON
        data = json.loads(content)
        assert "events" in data

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self, temp_storage):
        """Loading from nonexistent file should not crash"""
        os.unlink(temp_storage)  # Remove the temp file
        tracker = LifeTracker(storage_path=temp_storage)
        await tracker._load_data()  # Should not raise
        assert len(tracker._events) == 0

    @pytest.mark.asyncio
    async def test_multiple_save_load_cycles(self, temp_storage):
        """Data should remain intact through multiple save/load cycles"""
        tracker = LifeTracker(storage_path=temp_storage)

        event = LifeEvent(
            id="cycle1",
            title="Multi-cycle Test",
            category=EventCategory.EDUCATION,
            priority=EventPriority.CRITICAL,
            event_date=datetime(2026, 5, 15, 12, 30),
            related_people=["Teacher", "Student"],
        )
        tracker._events[event.id] = event

        # Cycle 1
        await tracker._save_data()
        tracker2 = LifeTracker(storage_path=temp_storage)
        await tracker2._load_data()

        # Cycle 2
        await tracker2._save_data()
        tracker3 = LifeTracker(storage_path=temp_storage)
        await tracker3._load_data()

        # Cycle 3
        await tracker3._save_data()
        tracker4 = LifeTracker(storage_path=temp_storage)
        await tracker4._load_data()

        # Verify data integrity after 3 cycles
        final_event = tracker4._events["cycle1"]
        assert final_event.title == "Multi-cycle Test"
        assert final_event.category == EventCategory.EDUCATION
        assert final_event.priority == EventPriority.CRITICAL
        assert final_event.event_date == datetime(2026, 5, 15, 12, 30)
        assert final_event.related_people == ["Teacher", "Student"]
