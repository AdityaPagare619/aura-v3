"""
Tests for User Profile SQLite Persistence

Tests cover:
- Database creation and table setup
- Profile save and load
- Debounced persistence (threshold-based)
- Migration from in-memory to SQLite
- Concurrent access safety
- Error handling
"""

import os
import sqlite3
import tempfile
import threading
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.user_profile import (
    ProfilePersistence,
    DeepUserProfiler,
    BigFiveTrait,
    get_user_profiler,
    get_persistence,
    flush_all_profiles,
    clear_profile_cache,
    delete_user_profile,
    DATABASE_PATH,
    DATABASE_DIR,
    TRAIT_CHANGE_THRESHOLD,
    INTERACTION_SAVE_INTERVAL,
)


class TestProfilePersistence:
    """Tests for the ProfilePersistence class"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test"""
        # Reset singleton
        ProfilePersistence._instance = None
        ProfilePersistence._connections = {}

        # Use a temp database for testing
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = Path(self.temp_dir) / "test_profile.db"

        # Start patches - keep them active for the entire test
        self.patches = [
            patch("core.user_profile.DATABASE_PATH", self.temp_db),
            patch("core.user_profile.DATABASE_DIR", Path(self.temp_dir)),
        ]
        for p in self.patches:
            p.start()

        yield

        # Stop patches
        for p in self.patches:
            p.stop()

        # Close all connections before cleanup
        if ProfilePersistence._instance:
            ProfilePersistence._instance.close_all()
        ProfilePersistence._instance = None
        ProfilePersistence._connections = {}

        # Cleanup files
        import time

        time.sleep(0.1)  # Brief pause for Windows file handles
        try:
            for f in Path(self.temp_dir).glob("test_profile.db*"):
                try:
                    f.unlink()
                except:
                    pass
            os.rmdir(self.temp_dir)
        except:
            pass

    def test_database_creation(self):
        """Test that database and tables are created correctly"""
        persistence = ProfilePersistence()

        # Trigger connection/table creation by doing a simple operation
        persistence.list_profiles()

        # Verify database exists
        assert self.temp_db.exists()

        # Verify table structure
        conn = sqlite3.connect(str(self.temp_db))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(user_profiles)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            "id",
            "user_id",
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
            "interaction_count",
            "last_updated",
            "metadata",
        }
        assert columns == expected_columns
        conn.close()

    def test_save_and_load_profile(self):
        """Test saving and loading a profile"""
        persistence = ProfilePersistence()

        # Save a profile
        traits = {
            "openness": 0.7,
            "conscientiousness": 0.5,
            "extraversion": -0.3,
            "agreeableness": 0.2,
            "neuroticism": -0.1,
        }
        metadata = {"interests": ["coding", "music"]}

        result = persistence.save_profile(
            user_id="test_user_123",
            traits=traits,
            interaction_count=42,
            metadata=metadata,
        )
        assert result is True

        # Load the profile
        loaded = persistence.load_profile("test_user_123")
        assert loaded is not None
        assert loaded["user_id"] == "test_user_123"
        assert loaded["traits"]["openness"] == 0.7
        assert loaded["traits"]["extraversion"] == -0.3
        assert loaded["interaction_count"] == 42
        assert loaded["metadata"]["interests"] == ["coding", "music"]

    def test_load_nonexistent_profile(self):
        """Test loading a profile that doesn't exist"""
        persistence = ProfilePersistence()

        result = persistence.load_profile("nonexistent_user")
        assert result is None

    def test_update_existing_profile(self):
        """Test updating an existing profile (UPSERT)"""
        persistence = ProfilePersistence()

        # Initial save
        persistence.save_profile(
            user_id="update_test", traits={"openness": 0.5}, interaction_count=10
        )

        # Update
        persistence.save_profile(
            user_id="update_test",
            traits={"openness": 0.8, "extraversion": 0.3},
            interaction_count=25,
        )

        # Verify update
        loaded = persistence.load_profile("update_test")
        assert loaded["traits"]["openness"] == 0.8
        assert loaded["traits"]["extraversion"] == 0.3
        assert loaded["interaction_count"] == 25

    def test_delete_profile(self):
        """Test deleting a profile"""
        persistence = ProfilePersistence()

        # Create then delete
        persistence.save_profile(user_id="delete_me", traits={}, interaction_count=0)

        result = persistence.delete_profile("delete_me")
        assert result is True

        # Verify deleted
        assert persistence.load_profile("delete_me") is None

    def test_list_profiles(self):
        """Test listing all user IDs"""
        persistence = ProfilePersistence()

        # Create multiple profiles
        for i in range(3):
            persistence.save_profile(
                user_id=f"user_{i}", traits={}, interaction_count=i
            )

        user_ids = persistence.list_profiles()
        assert set(user_ids) == {"user_0", "user_1", "user_2"}


class TestDeepUserProfilerPersistence:
    """Tests for DeepUserProfiler with SQLite persistence"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test"""
        # Reset singletons and caches
        ProfilePersistence._instance = None
        ProfilePersistence._connections = {}

        # Clear profiler cache
        import core.user_profile as up

        up._profilers = {}
        up._persistence = None

        # Use temp database
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = Path(self.temp_dir) / "test_profile.db"

        self.patches = [
            patch("core.user_profile.DATABASE_PATH", self.temp_db),
            patch("core.user_profile.DATABASE_DIR", Path(self.temp_dir)),
        ]
        for p in self.patches:
            p.start()

        yield

        for p in self.patches:
            p.stop()

        # Close all connections before cleanup
        if ProfilePersistence._instance:
            ProfilePersistence._instance.close_all()
        ProfilePersistence._instance = None
        ProfilePersistence._connections = {}

        import core.user_profile as up

        up._profilers = {}
        up._persistence = None

        # Cleanup files
        import time

        time.sleep(0.1)  # Brief pause for Windows file handles
        try:
            for f in Path(self.temp_dir).glob("test_profile.db*"):
                try:
                    f.unlink()
                except:
                    pass
            os.rmdir(self.temp_dir)
        except:
            pass
        try:
            os.rmdir(self.temp_dir)
        except:
            pass

    def test_profiler_creates_empty_profile(self):
        """Test that a new profiler starts with default values"""
        profiler = DeepUserProfiler("new_user")

        # Check defaults
        for trait in BigFiveTrait:
            assert profiler.psychology.traits[trait] == 0.0

        assert profiler._interaction_count == 0

    def test_profiler_loads_from_persistence(self):
        """Test that profiler loads existing data from SQLite"""
        # First, save directly to persistence
        persistence = get_persistence()
        persistence.save_profile(
            user_id="persisted_user",
            traits={
                "openness": 0.6,
                "conscientiousness": 0.4,
                "extraversion": -0.2,
                "agreeableness": 0.3,
                "neuroticism": -0.1,
            },
            interaction_count=100,
            metadata={
                "communication": {
                    "formality_level": 0.5,
                    "prefers_concise": True,
                },
                "interests": ["python", "ai"],
            },
        )

        # Create profiler - should load from persistence
        profiler = DeepUserProfiler("persisted_user")

        assert profiler.psychology.traits[BigFiveTrait.OPENNESS] == 0.6
        assert profiler.psychology.traits[BigFiveTrait.EXTRAVERSION] == -0.2
        assert profiler._interaction_count == 100
        assert profiler.communication.formality_level == 0.5
        assert profiler.communication.prefers_concise is True
        assert "python" in profiler.context.topics_interested_in

    def test_observe_increments_interaction_count(self):
        """Test that observe() increments the interaction count"""
        profiler = DeepUserProfiler("interaction_test")

        assert profiler._interaction_count == 0

        profiler.observe("message", {"message": "Hello world"})
        assert profiler._interaction_count == 1

        profiler.observe("task", {"completed": True})
        assert profiler._interaction_count == 2

    def test_debounced_persistence_by_interaction_count(self):
        """Test that persistence happens after N interactions"""
        profiler = DeepUserProfiler("debounce_test")

        # Observe less than threshold
        for i in range(INTERACTION_SAVE_INTERVAL - 1):
            profiler.observe("message", {"message": f"msg {i}"})

        # Should not have saved yet
        persistence = get_persistence()
        loaded = persistence.load_profile("debounce_test")
        # May be None or have old count
        if loaded:
            assert loaded["interaction_count"] < INTERACTION_SAVE_INTERVAL

        # One more observation should trigger save
        profiler.observe("message", {"message": "trigger save"})

        loaded = persistence.load_profile("debounce_test")
        assert loaded is not None
        assert loaded["interaction_count"] == INTERACTION_SAVE_INTERVAL

    def test_debounced_persistence_by_trait_change(self):
        """Test that significant trait change triggers persistence"""
        profiler = DeepUserProfiler("trait_change_test")

        # Force save initial state
        profiler.save_to_persistence()
        initial_traits = profiler._last_saved_traits.copy()

        # Manually change a trait significantly
        profiler.psychology.traits[BigFiveTrait.OPENNESS] = (
            initial_traits.get("openness", 0.0) + TRAIT_CHANGE_THRESHOLD + 0.01
        )

        # Trigger persistence check
        profiler._persist_if_needed()

        # Verify saved
        persistence = get_persistence()
        loaded = persistence.load_profile("trait_change_test")
        assert loaded is not None
        assert (
            abs(
                loaded["traits"]["openness"]
                - profiler.psychology.traits[BigFiveTrait.OPENNESS]
            )
            < 0.001
        )

    def test_force_save(self):
        """Test force save regardless of thresholds"""
        profiler = DeepUserProfiler("force_save_test")

        # Modify slightly (below threshold)
        profiler.psychology.traits[BigFiveTrait.OPENNESS] = 0.01
        profiler._interaction_count = 3

        # Force save
        result = profiler.save_to_persistence()
        assert result is True

        # Verify
        persistence = get_persistence()
        loaded = persistence.load_profile("force_save_test")
        assert loaded["interaction_count"] == 3

    def test_get_user_profiler_caching(self):
        """Test that get_user_profiler returns cached instance"""
        profiler1 = get_user_profiler("cache_test")
        profiler2 = get_user_profiler("cache_test")

        assert profiler1 is profiler2

    def test_flush_all_profiles(self):
        """Test flushing all profiles to persistence"""
        # Create multiple profilers
        p1 = get_user_profiler("flush_1")
        p2 = get_user_profiler("flush_2")

        p1._interaction_count = 50
        p2._interaction_count = 75

        saved = flush_all_profiles()
        assert saved == 2

        # Verify both saved
        persistence = get_persistence()
        assert persistence.load_profile("flush_1")["interaction_count"] == 50
        assert persistence.load_profile("flush_2")["interaction_count"] == 75

    def test_delete_user_profile(self):
        """Test deleting a user profile from memory and SQLite"""
        profiler = get_user_profiler("delete_test")
        profiler.save_to_persistence()

        result = delete_user_profile("delete_test")
        assert result is True

        # Verify removed from memory
        import core.user_profile as up

        assert "delete_test" not in up._profilers

        # Verify removed from SQLite
        persistence = get_persistence()
        assert persistence.load_profile("delete_test") is None

    def test_auto_persist_disabled(self):
        """Test profiler with auto_persist=False"""
        profiler = DeepUserProfiler("no_persist", auto_persist=False)

        # Observe many times
        for i in range(INTERACTION_SAVE_INTERVAL + 5):
            profiler.observe("message", {"message": f"msg {i}"})

        # Should not have saved
        persistence = get_persistence()
        loaded = persistence.load_profile("no_persist")
        assert loaded is None

    def test_concurrent_access(self):
        """Test thread-safe concurrent access"""
        errors = []

        def worker(user_id: str, n_operations: int):
            try:
                profiler = DeepUserProfiler(f"concurrent_{user_id}")
                for i in range(n_operations):
                    profiler.observe("message", {"message": f"msg {i}"})
                profiler.save_to_persistence()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(str(i), 20)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all profiles saved
        persistence = get_persistence()
        for i in range(5):
            loaded = persistence.load_profile(f"concurrent_{i}")
            assert loaded is not None
            assert loaded["interaction_count"] == 20


class TestMigrationSupport:
    """Tests for migrating existing in-memory profiles to SQLite"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown"""
        ProfilePersistence._instance = None
        ProfilePersistence._connections = {}

        import core.user_profile as up

        up._profilers = {}
        up._persistence = None

        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = Path(self.temp_dir) / "test_profile.db"

        self.patches = [
            patch("core.user_profile.DATABASE_PATH", self.temp_db),
            patch("core.user_profile.DATABASE_DIR", Path(self.temp_dir)),
        ]
        for p in self.patches:
            p.start()

        yield

        for p in self.patches:
            p.stop()

        # Close all connections before cleanup
        if ProfilePersistence._instance:
            ProfilePersistence._instance.close_all()
        ProfilePersistence._instance = None
        ProfilePersistence._connections = {}

        import core.user_profile as up

        up._profilers = {}
        up._persistence = None

        # Cleanup files
        import time

        time.sleep(0.1)
        try:
            for f in Path(self.temp_dir).glob("test_profile.db*"):
                try:
                    f.unlink()
                except:
                    pass
            os.rmdir(self.temp_dir)
        except:
            pass
        try:
            os.rmdir(self.temp_dir)
        except:
            pass

    def test_existing_in_memory_profile_persists(self):
        """Test that an existing in-memory profile is saved to SQLite"""
        # Create profiler and modify it
        profiler = DeepUserProfiler("migrate_test")
        profiler.psychology.traits[BigFiveTrait.OPENNESS] = 0.8
        profiler.psychology.traits[BigFiveTrait.CONSCIENTIOUSNESS] = 0.6
        profiler.communication.prefers_concise = True
        profiler.context.topics_interested_in = ["testing", "python"]
        profiler._interaction_count = 50

        # Save to persistence
        profiler.save_to_persistence()

        # Reset singleton and load fresh
        ProfilePersistence._instance = None
        import core.user_profile as up

        up._persistence = None
        up._profilers = {}

        # Load fresh
        new_profiler = DeepUserProfiler("migrate_test")

        # Verify all data preserved
        assert new_profiler.psychology.traits[BigFiveTrait.OPENNESS] == 0.8
        assert new_profiler.psychology.traits[BigFiveTrait.CONSCIENTIOUSNESS] == 0.6
        assert new_profiler.communication.prefers_concise is True
        assert "testing" in new_profiler.context.topics_interested_in
        assert new_profiler._interaction_count == 50


class TestErrorHandling:
    """Tests for error handling and edge cases"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup"""
        ProfilePersistence._instance = None
        ProfilePersistence._connections = {}

        import core.user_profile as up

        up._profilers = {}
        up._persistence = None

        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = Path(self.temp_dir) / "test_profile.db"

        # Start patches - keep them active for the entire test
        self.patches = [
            patch("core.user_profile.DATABASE_PATH", self.temp_db),
            patch("core.user_profile.DATABASE_DIR", Path(self.temp_dir)),
        ]
        for p in self.patches:
            p.start()

        yield

        # Stop patches
        for p in self.patches:
            p.stop()

        # Close all connections before cleanup
        if ProfilePersistence._instance:
            ProfilePersistence._instance.close_all()
        ProfilePersistence._instance = None
        ProfilePersistence._connections = {}

        # Cleanup files
        import time

        time.sleep(0.1)  # Brief pause for Windows file handles
        try:
            for f in Path(self.temp_dir).glob("*"):
                try:
                    if f.is_file():
                        f.unlink()
                    elif f.is_dir():
                        import shutil

                        shutil.rmtree(f, ignore_errors=True)
                except:
                    pass
            os.rmdir(self.temp_dir)
        except:
            pass

    def test_corrupted_metadata_handling(self):
        """Test handling of corrupted JSON in metadata"""
        persistence = ProfilePersistence()

        # Trigger table creation
        persistence.list_profiles()

        # Insert corrupt metadata directly
        conn = sqlite3.connect(str(self.temp_db))
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_profiles (user_id, metadata)
            VALUES (?, ?)
        """,
            ("corrupt_user", "not valid json {{{"),
        )
        conn.commit()
        conn.close()

        # Should handle gracefully
        loaded = persistence.load_profile("corrupt_user")
        assert loaded is not None
        assert loaded["metadata"] == {}  # Defaults to empty dict

    def test_missing_database_directory(self):
        """Test that missing directory is created"""
        # Stop current patches to test with new paths
        for p in self.patches:
            p.stop()

        new_dir = Path(self.temp_dir) / "nested" / "dir"
        new_db = new_dir / "test.db"

        # Reset singleton to test new path creation
        ProfilePersistence._instance = None
        ProfilePersistence._connections = {}

        with patch("core.user_profile.DATABASE_PATH", new_db):
            with patch("core.user_profile.DATABASE_DIR", new_dir):
                persistence = ProfilePersistence()
                # Trigger connection to create directory
                persistence.list_profiles()

                assert new_dir.exists()
                assert new_db.exists()

                # Cleanup
                persistence.close_all()

        # Restart patches for teardown
        for p in self.patches:
            p.start()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
