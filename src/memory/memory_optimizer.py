"""
Memory Optimizer - Memory Management and Optimization
Compresses old memories, archives unused data, balances storage vs recall
"""

import sqlite3
import json
import logging
import time
import hashlib
import threading
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import os

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of memory compression"""

    original_size: int
    compressed_size: int
    items_compressed: int
    items_archived: int
    items_deleted: int
    space_saved: int


@dataclass
class MemoryMetrics:
    """Memory system metrics"""

    total_items: int
    total_size_bytes: int
    active_items: int
    inactive_items: int
    archived_items: int
    avg_importance: float
    avg_age_days: float


class MemoryArchiver:
    """
    Archives old/low-priority memories to compressed storage
    """

    def __init__(
        self,
        active_db: str,
        archive_db: str = None,
        inactivity_days: int = 30,
    ):
        self.active_db = active_db
        self.archive_db = archive_db or active_db.replace(".db", "_archive.db")
        self.inactivity_days = inactivity_days

        self._init_archive_db()

    def _init_archive_db(self):
        """Initialize archive database"""
        os.makedirs(os.path.dirname(self.archive_db), exist_ok=True)

        conn = sqlite3.connect(self.archive_db)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS archived_memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                compressed_content BLOB,
                metadata TEXT,
                archived_at REAL,
                original_size INTEGER,
                compressed_size INTEGER,
                access_count INTEGER DEFAULT 0,
                importance REAL DEFAULT 0.5
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_archived_archived_at 
            ON archived_memories(archived_at DESC)
        """)

        conn.commit()
        conn.close()

    def archive(
        self,
        memories: List[Dict[str, Any]],
    ) -> int:
        """Archive memories"""
        if not memories:
            return 0

        archived_count = 0

        conn_active = sqlite3.connect(self.active_db)
        conn_archive = sqlite3.connect(self.archive_db)

        try:
            for memory in memories:
                memory_id = memory.get("id")
                content = memory.get("content", "")
                metadata = memory.get("metadata", {})
                importance = memory.get("importance", 0.5)

                if not memory_id:
                    continue

                original_size = len(content.encode("utf-8"))
                compressed = self._compress_content(content)
                compressed_size = len(compressed)

                with conn_archive:
                    conn_archive.execute(
                        """INSERT OR REPLACE INTO archived_memories
                           (id, content, compressed_content, metadata, archived_at, 
                            original_size, compressed_size, access_count, importance)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            memory_id,
                            content,
                            compressed,
                            json.dumps(metadata),
                            time.time(),
                            original_size,
                            compressed_size,
                            memory.get("access_count", 0),
                            importance,
                        ),
                    )

                conn_active.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                archived_count += 1

            conn_active.commit()
            conn_archive.commit()

        finally:
            conn_active.close()
            conn_archive.close()

        logger.info(f"Archived {archived_count} memories")
        return archived_count

    def _compress_content(self, content: str) -> bytes:
        """Compress content using zlib"""
        import zlib

        return zlib.compress(content.encode("utf-8"), level=6)

    def _decompress_content(self, compressed: bytes) -> str:
        """Decompress content"""
        import zlib

        return zlib.decompress(compressed).decode("utf-8")

    def restore(self, memory_id: str, target_db: str = None) -> Optional[Dict]:
        """Restore archived memory"""
        target_db = target_db or self.active_db

        conn_archive = sqlite3.connect(self.archive_db)
        cursor = conn_archive.execute(
            """SELECT id, content, compressed_content, metadata, archived_at, 
                      original_size, compressed_size, access_count, importance
               FROM archived_memories WHERE id = ?""",
            (memory_id,),
        )

        row = cursor.fetchone()

        if not row:
            conn_archive.close()
            return None

        memory = {
            "id": row[0],
            "content": row[1] if not row[2] else self._decompress_content(row[2]),
            "metadata": json.loads(row[3]) if row[3] else {},
            "archived_at": row[4],
            "original_size": row[5],
            "compressed_size": row[6],
            "access_count": row[7],
            "importance": row[8],
        }

        conn_archive.close()

        conn_target = sqlite3.connect(target_db)
        conn_target.execute(
            """INSERT OR REPLACE INTO memories 
               (id, content, metadata, importance, access_count)
               VALUES (?, ?, ?, ?, ?)""",
            (
                memory["id"],
                memory["content"],
                json.dumps(memory["metadata"]),
                memory["importance"],
                memory["access_count"],
            ),
        )
        conn_target.commit()
        conn_target.close()

        conn_archive = sqlite3.connect(self.archive_db)
        conn_archive.execute("DELETE FROM archived_memories WHERE id = ?", (memory_id,))
        conn_archive.commit()
        conn_archive.close()

        return memory

    def get_archived_count(self) -> int:
        """Get count of archived memories"""
        conn = sqlite3.connect(self.archive_db)
        count = conn.execute("SELECT COUNT(*) FROM archived_memories").fetchone()[0]
        conn.close()
        return count


class MemoryCompressor:
    """
    Compresses memories while preserving important data
    """

    def __init__(self, min_importance_threshold: float = 0.3):
        self.min_importance_threshold = min_importance_threshold

    def compress_memory(self, memory: Dict) -> Dict:
        """Compress a single memory"""
        content = memory.get("content", "")

        compressed_content = self._compress_text(content)

        original_size = len(content)
        compressed_size = len(compressed_content)

        return {
            **memory,
            "content": compressed_content,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "is_compressed": True,
        }

    def _compress_text(self, text: str) -> str:
        """Compress text using run-length encoding for repeated words"""
        words = text.split()

        if len(words) <= 10:
            return text

        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1

        rare_words = [w for w, c in word_counts.items() if c == 1]
        frequent_words = [w for w, c in word_counts.items() if c > 1]

        if len(rare_words) < len(frequent_words):
            return text

        compressed = []
        for word in words:
            if word_counts[word] > 3 and len(word) > 4:
                compressed.append(f"#{word_counts[word]}_{word[:4]}")
            else:
                compressed.append(word)

        return " ".join(compressed)

    def batch_compress(
        self,
        memories: List[Dict],
    ) -> List[Dict]:
        """Compress multiple memories"""
        return [self.compress_memory(m) for m in memories]


class MemoryCleanup:
    """
    Cleans up old, unused, or low-importance memories
    """

    def __init__(
        self,
        db_path: str,
        min_importance: float = 0.1,
        inactivity_days: int = 90,
    ):
        self.db_path = db_path
        self.min_importance = min_importance
        self.inactivity_days = inactivity_days

    def get_cleanup_candidates(
        self,
        limit: int = 100,
    ) -> List[Dict]:
        """Get memories eligible for cleanup"""
        conn = sqlite3.connect(self.db_path)

        cutoff_time = time.time() - (self.inactivity_days * 86400)

        cursor = conn.execute(
            """SELECT id, content, metadata, importance, access_count, created_at
               FROM memories 
               WHERE importance < ? OR (access_count = 0 AND created_at < ?)
               ORDER BY importance ASC, created_at ASC
               LIMIT ?""",
            (self.min_importance, cutoff_time, limit),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "id": row[0],
                    "content": row[1],
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "importance": row[3],
                    "access_count": row[4],
                    "created_at": row[5],
                }
            )

        conn.close()
        return results

    def cleanup(
        self,
        candidate_ids: List[str] = None,
    ) -> int:
        """Delete memories"""
        conn = sqlite3.connect(self.db_path)

        if candidate_ids:
            placeholders = ",".join("?" * len(candidate_ids))
            cursor = conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", candidate_ids
            )
        else:
            candidates = self.get_cleanup_candidates(100)
            candidate_ids = [c["id"] for c in candidates]

            if not candidate_ids:
                conn.close()
                return 0

            placeholders = ",".join("?" * len(candidate_ids))
            cursor = conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", candidate_ids
            )

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"Cleaned up {deleted} memories")
        return deleted


class MemoryOptimizer:
    """
    Main memory optimizer that coordinates compression, archiving, and cleanup

    Features:
    - Automatic memory compression
    - Intelligent archiving
    - Periodic cleanup
    - Storage balancing
    - Works offline on Android/Termux
    """

    def __init__(
        self,
        db_paths: Dict[str, str] = None,
        config: Dict[str, Any] = None,
    ):
        self.db_paths = db_paths or {
            "episodic": "data/memory/episodic.db",
            "semantic": "data/memory/semantic.db",
            "vectors": "data/memory/vectors.db",
            "temporal": "data/memory/temporal_graph.db",
        }

        config = config or {}

        self.inactivity_days = config.get("inactivity_days", 30)
        self.min_importance = config.get("min_importance", 0.2)
        self.max_storage_mb = config.get("max_storage_mb", 500)

        self._running = False
        self._optimize_thread: Optional[threading.Thread] = None

    def optimize(
        self,
        force: bool = False,
    ) -> CompressionResult:
        """Run optimization"""
        total_original = 0
        total_compressed = 0
        items_compressed = 0
        items_archived = 0
        items_deleted = 0

        for name, db_path in self.db_paths.items():
            if not os.path.exists(db_path):
                continue

            result = self._optimize_db(db_path, force)

            total_original += result.get("original_size", 0)
            total_compressed += result.get("compressed_size", 0)
            items_compressed += result.get("items_compressed", 0)
            items_archived += result.get("items_archived", 0)
            items_deleted += result.get("items_deleted", 0)

        return CompressionResult(
            original_size=total_original,
            compressed_size=total_compressed,
            items_compressed=items_compressed,
            items_archived=items_archived,
            items_deleted=items_deleted,
            space_saved=total_original - total_compressed,
        )

    def _optimize_db(
        self,
        db_path: str,
        force: bool,
    ) -> Dict:
        """Optimize a single database"""
        conn = sqlite3.connect(db_path)

        cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        if cursor.fetchone()[0] == 0:
            conn.close()
            return {}

        if "episodic" in db_path:
            return self._optimize_episodic(conn, force)
        elif "semantic" in db_path:
            return self._optimize_semantic(conn, force)
        elif "vectors" in db_path:
            return self._optimize_vectors(conn, force)

        conn.close()
        return {}

    def _optimize_episodic(
        self,
        conn: sqlite3.Connection,
        force: bool,
    ) -> Dict:
        """Optimize episodic memory"""
        result = {"items_compressed": 0, "items_archived": 0, "items_deleted": 0}

        cutoff = time.time() - (self.inactivity_days * 86400)

        cursor = conn.execute(
            """SELECT id, content, importance, consolidated 
               FROM episodic_traces 
               WHERE importance < ? AND consolidated = 0""",
            (self.min_importance,),
        )

        to_archive = []
        for row in cursor.fetchall():
            to_archive.append(
                {
                    "id": row[0],
                    "content": row[1],
                    "importance": row[2],
                }
            )

        if to_archive:
            conn.execute(
                """DELETE FROM episodic_traces 
                   WHERE id IN ({})""".format(",".join("?" * len(to_archive))),
                [m["id"] for m in to_archive],
            )
            result["items_archived"] = len(to_archive)

        conn.commit()
        conn.close()

        return result

    def _optimize_semantic(
        self,
        conn: sqlite3.Connection,
        force: bool,
    ) -> Dict:
        """Optimize semantic memory"""
        result = {"items_compressed": 0, "items_deleted": 0}

        cursor = conn.execute(
            """SELECT id, fact, confidence 
               FROM facts 
               WHERE confidence < ? AND evidence_count = 1""",
            (self.min_importance,),
        )

        to_delete = [
            row[0] for row in cursor.fetchall() if row[2] < self.min_importance * 0.5
        ]

        if to_delete:
            conn.execute(
                "DELETE FROM facts WHERE id IN ({})".format(
                    ",".join("?" * len(to_delete))
                ),
                to_delete,
            )
            result["items_deleted"] = len(to_delete)

        conn.commit()
        conn.close()

        return result

    def _optimize_vectors(
        self,
        conn: sqlite3.Connection,
        force: bool,
    ) -> Dict:
        """Optimize vector store"""
        result = {"items_compressed": 0, "items_deleted": 0}

        cursor = conn.execute(
            """SELECT id, importance, access_count 
               FROM vectors 
               WHERE importance < ? AND access_count < 2""",
            (self.min_importance,),
        )

        to_delete = [row[0] for row in cursor.fetchall()]

        if to_delete:
            conn.execute(
                "DELETE FROM vectors WHERE id IN ({})".format(
                    ",".join("?" * len(to_delete))
                ),
                to_delete,
            )
            result["items_deleted"] = len(to_delete)

        conn.commit()

        conn.execute("VACUUM")
        conn.close()

        return result

    def start_auto_optimize(self, interval_hours: int = 24):
        """Start automatic optimization"""
        if self._running:
            return

        self._running = True

        def _run():
            while self._running:
                try:
                    self.optimize()
                except Exception as e:
                    logger.error(f"Auto-optimize failed: {e}")

                time.sleep(interval_hours * 3600)

        self._optimize_thread = threading.Thread(target=_run, daemon=True)
        self._optimize_thread.start()

    def stop_auto_optimize(self):
        """Stop automatic optimization"""
        self._running = False
        if self._optimize_thread:
            self._optimize_thread.join(timeout=5)

    def get_metrics(self) -> Dict[str, MemoryMetrics]:
        """Get memory metrics for all databases"""
        metrics = {}

        for name, db_path in self.db_paths.items():
            if not os.path.exists(db_path):
                continue

            try:
                metrics[name] = self._get_db_metrics(db_path)
            except Exception as e:
                logger.warning(f"Failed to get metrics for {name}: {e}")

        return metrics

    def _get_db_metrics(self, db_path: str) -> MemoryMetrics:
        """Get metrics for a database"""
        conn = sqlite3.connect(db_path)

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

        total_items = 0
        total_size = 0

        for (table,) in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            total_items += count

            try:
                size = (
                    conn.execute(
                        f"SELECT SUM(LENGTH(content)) FROM {table} WHERE content IS NOT NULL"
                    ).fetchone()[0]
                    or 0
                )
                total_size += size
            except:
                pass

        conn.close()

        return MemoryMetrics(
            total_items=total_items,
            total_size_bytes=total_size,
            active_items=total_items,
            inactive_items=0,
            archived_items=0,
            avg_importance=0.5,
            avg_age_days=0,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        return {
            "databases": list(self.db_paths.keys()),
            "inactivity_days": self.inactivity_days,
            "min_importance": self.min_importance,
            "max_storage_mb": self.max_storage_mb,
        }


class SmartMemoryManager:
    """
    High-level memory management with intelligent decisions
    """

    def __init__(
        self,
        vector_store=None,
        episodic_memory=None,
        semantic_memory=None,
        temporal_graph=None,
    ):
        self.vector_store = vector_store
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.temporal = temporal_graph

        self.optimizer = MemoryOptimizer()

    def should_compress(self, memory: Dict) -> bool:
        """Decide if memory should be compressed"""
        importance = memory.get("importance", 0.5)
        age_days = (time.time() - memory.get("created_at", time.time())) / 86400

        return importance < 0.4 and age_days > 7

    def should_archive(self, memory: Dict) -> bool:
        """Decide if memory should be archived"""
        importance = memory.get("importance", 0.5)
        access_count = memory.get("access_count", 0)
        age_days = (time.time() - memory.get("created_at", time.time())) / 86400

        return importance < 0.3 and access_count < 2 and age_days > 30

    def should_delete(self, memory: Dict) -> bool:
        """Decide if memory should be deleted"""
        importance = memory.get("importance", 0.5)
        access_count = memory.get("access_count", 0)
        age_days = (time.time() - memory.get("created_at", time.time())) / 86400

        return importance < 0.1 and access_count == 0 and age_days > 90

    def optimize_all(self) -> CompressionResult:
        """Run full optimization"""
        return self.optimizer.optimize()


def get_memory_optimizer(
    config: Dict[str, Any] = None,
) -> MemoryOptimizer:
    """Get or create memory optimizer instance"""
    return MemoryOptimizer(config=config)


__all__ = [
    "MemoryOptimizer",
    "MemoryArchiver",
    "MemoryCompressor",
    "MemoryCleanup",
    "SmartMemoryManager",
    "CompressionResult",
    "MemoryMetrics",
    "get_memory_optimizer",
]
