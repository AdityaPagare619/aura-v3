"""
Test Suite for Advanced Memory Systems
Verifies basic functionality of new memory modules
"""

import sys
import os
import tempfile
import time
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def test_sqlite_state_machine():
    """Test SQLite state machine"""
    print("\n=== Testing SQLiteStateMachine ===")

    from memory.sqlite_state_machine import SQLiteStateMachine, State, TransitionType

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        sm = SQLiteStateMachine(db_path)

        state = State(name="test_state", data={"key": "value"})
        sm.set_state(state)

        retrieved = sm.get_state("test_state")
        assert retrieved is not None
        assert retrieved.data["key"] == "value"

        transition = sm.transition(
            from_state="test_state",
            to_state="active",
            trigger="activate",
            transition_type=TransitionType.EXPLICIT,
        )
        assert transition is not None

        stats = sm.get_stats()
        print(f"  Stats: {stats}")
        print("  SQLiteStateMachine: PASSED")

    finally:
        if os.path.exists(db_path):
            os.remove(db_path.replace(".db", ".db"))


def test_local_vector_store():
    """Test local vector store"""
    print("\n=== Testing LocalVectorStore ===")

    from memory.local_vector_store import LocalVectorStore

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = LocalVectorStore(db_path, embedding_dim=64)

        entry1 = store.add(
            content="The quick brown fox jumps over the lazy dog",
            metadata={"category": "test"},
            importance=0.8,
        )
        assert entry1 is not None

        entry2 = store.add(
            content="Machine learning is a subset of artificial intelligence",
            metadata={"category": "tech"},
            importance=0.9,
        )

        results = store.search("artificial intelligence", top_k=5)
        assert len(results) > 0
        print(f"  Search results: {len(results)} found")

        stats = store.get_stats()
        print(f"  Stats: {stats}")
        print("  LocalVectorStore: PASSED")

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_temporal_knowledge_graph():
    """Test temporal knowledge graph"""
    print("\n=== Testing TemporalKnowledgeGraph ===")

    from memory.temporal_knowledge_graph import (
        TemporalKnowledgeGraph,
        FactStatus,
        RelationType,
    )

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        graph = TemporalKnowledgeGraph(db_path)

        entity = graph.add_entity("John Doe", "person", {"age": 30})
        assert entity is not None

        fact = graph.add_fact(
            subject="John Doe",
            predicate="works_at",
            object="TechCorp",
            confidence=0.9,
            source="user_input",
        )
        assert fact is not None

        entity2 = graph.add_entity("TechCorp", "company")

        relation_id = graph.add_relation(
            from_entity="John Doe",
            to_entity="TechCorp",
            relation_type=RelationType.RELATED_TO,
        )
        assert relation_id is not None

        facts = graph.what_was_true_at("John Doe")
        print(f"  Facts at time: {len(facts)} found")

        stats = graph.get_stats()
        print(f"  Stats: {stats}")
        print("  TemporalKnowledgeGraph: PASSED")

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_memory_optimizer():
    """Test memory optimizer"""
    print("\n=== Testing MemoryOptimizer ===")

    from memory.memory_optimizer import (
        MemoryOptimizer,
        MemoryArchiver,
        MemoryCompressor,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                created_at REAL
            )
        """)

        for i in range(10):
            conn.execute(
                "INSERT INTO memories (id, content, importance, access_count, created_at) VALUES (?, ?, ?, ?, ?)",
                (f"mem_{i}", f"Test memory {i}", 0.3, 0, time.time() - 86400 * 60),
            )

        conn.commit()
        conn.close()

        optimizer = MemoryOptimizer(
            db_paths={"test": db_path},
            config={"min_importance": 0.2, "inactivity_days": 30},
        )

        result = optimizer.optimize()
        print(
            f"  Optimization result: {result.items_deleted} deleted, {result.items_archived} archived"
        )

        compressor = MemoryCompressor()
        test_memory = {
            "content": "This is a test memory with some content",
            "importance": 0.5,
        }
        compressed = compressor.compress_memory(test_memory)
        print(f"  Compressed: {compressed.get('is_compressed', False)}")

        print("  MemoryOptimizer: PASSED")


def test_embedding_generator():
    """Test local embedding generator"""
    print("\n=== Testing LocalEmbeddingGenerator ===")

    from memory.local_vector_store import LocalEmbeddingGenerator

    gen = LocalEmbeddingGenerator(embedding_dim=64)

    emb1 = gen.generate("Hello world")
    assert len(emb1) == 64

    emb2 = gen.generate("Hello world")
    assert emb1 == emb2

    emb3 = gen.generate("Different text")
    assert emb1 != emb3

    print("  Embedding dimension: 64")
    print("  Same text produces same embedding: True")
    print("  LocalEmbeddingGenerator: PASSED")


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("Advanced Memory Systems Test Suite")
    print("=" * 50)

    tests = [
        test_sqlite_state_machine,
        test_local_vector_store,
        test_temporal_knowledge_graph,
        test_memory_optimizer,
        test_embedding_generator,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
