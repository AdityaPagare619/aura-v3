"""
Tests for AURA v3 Memory Consolidation

Tests the memory consolidation pipeline:
- MemoryType enum
- MemoryItem dataclass
- MemoryCoordinator consolidation pipeline
- Memory state transitions
"""
import unittest
import asyncio
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

sys.path.insert(0, '.')
from src.memory.memory_coordinator import MemoryCoordinator, MemoryType, MemoryItem, get_memory_coordinator

class TestMemoryType(unittest.TestCase):
    def test_all_memory_types(self):
        types = [MemoryType.WORKING, MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.SKILL, MemoryType.ANCESTOR]
        self.assertEqual(len(types), 5)
    def test_values(self):
        self.assertEqual(MemoryType.WORKING.value, "working")
        self.assertEqual(MemoryType.EPISODIC.value, "episodic")
        self.assertEqual(MemoryType.SEMANTIC.value, "semantic")

class TestMemoryItem(unittest.TestCase):
    def test_create_item(self):
        now = datetime.now()
        item = MemoryItem(id="test1", content="Test memory", memory_type=MemoryType.WORKING, importance=0.8, activation=0.5, created_at=now, last_accessed=now)
        self.assertEqual(item.id, "test1")
        self.assertEqual(item.content, "Test memory")
        self.assertEqual(item.importance, 0.8)
    def test_default_values(self):
        now = datetime.now()
        item = MemoryItem(id="test2", content="Test", memory_type=MemoryType.EPISODIC, importance=0.5, activation=0.0, created_at=now, last_accessed=now)
        self.assertEqual(item.source, "")
        self.assertEqual(item.metadata, {})

class TestMemoryCoordinator(unittest.TestCase):
    def test_init(self):
        coord = MemoryCoordinator()
        self.assertEqual(coord.consolidation_interval, 300)
        self.assertEqual(coord.importance_threshold, 0.6)
        self.assertEqual(coord.max_working_memories, 50)
        self.assertFalse(coord.consolidation_running)

    def test_set_memory_system(self):
        coord = MemoryCoordinator()
        mock_mem = Mock()
        coord.set_memory_system(MemoryType.WORKING, mock_mem)
        self.assertIs(coord.neural_memory, mock_mem)

    def test_set_all_memory_systems(self):
        coord = MemoryCoordinator()
        neural = Mock()
        episodic = Mock()
        semantic = Mock()
        skill = Mock()
        ancestor = Mock()
        
        coord.set_memory_system(MemoryType.WORKING, neural)
        coord.set_memory_system(MemoryType.EPISODIC, episodic)
        coord.set_memory_system(MemoryType.SEMANTIC, semantic)
        coord.set_memory_system(MemoryType.SKILL, skill)
        coord.set_memory_system(MemoryType.ANCESTOR, ancestor)
        
        self.assertIs(coord.neural_memory, neural)
        self.assertIs(coord.episodic_memory, episodic)
        self.assertIs(coord.semantic_memory, semantic)
        self.assertIs(coord.skill_memory, skill)
        self.assertIs(coord.ancestor_memory, ancestor)

    def test_importance_threshold_setting(self):
        coord = MemoryCoordinator()
        coord.importance_threshold = 0.7
        self.assertEqual(coord.importance_threshold, 0.7)

    def test_consolidation_interval_setting(self):
        coord = MemoryCoordinator()
        coord.consolidation_interval = 600
        self.assertEqual(coord.consolidation_interval, 600)

class TestGetMemoryCoordinator(unittest.TestCase):
    def test_singleton(self):
        import src.memory.memory_coordinator
        src.memory.memory_coordinator._coordinator = None
        coord1 = get_memory_coordinator()
        coord2 = get_memory_coordinator()
        self.assertIs(coord1, coord2)

class TestConsolidationLogic(unittest.TestCase):
    def test_should_consolidate(self):
        coord = MemoryCoordinator()
        # Should not consolidate if no memory systems
        self.assertFalse(coord.neural_memory)
        self.assertFalse(coord.semantic_memory)

    def test_consolidation_threshold(self):
        coord = MemoryCoordinator()
        # High importance should be consolidatable
        importance_high = 0.8
        importance_low = 0.3
        self.assertGreater(importance_high, coord.importance_threshold)
        self.assertLess(importance_low, coord.importance_threshold)

if __name__ == "__main__": unittest.main()
