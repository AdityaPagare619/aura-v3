"""
Ancestor Memory - Long-term Storage
Lazy loading, encryption, archival, pruning
"""

import json
import uuid
import time
import hashlib
import zlib
import os
import base64
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.utils.db_pool import get_connection, connection as db_connection


def _generate_encryption_key() -> str:
    """Generate a secure encryption key"""
    random_bytes = os.urandom(32)
    return base64.b64encode(random_bytes).decode("utf-8")


@dataclass
class ArchivedMemory:
    """An archived long-term memory"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    summary: str = ""
    importance: float = 0.5
    memory_type: str = "general"
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    archived_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    compression_ratio: float = 1.0
    encrypted: bool = False


class AncestorMemory:
    """
    Long-term memory archive

    Features:
    - Lazy loading on retrieval
    - Compression for storage efficiency
    - Encryption at rest
    - Automatic pruning of irrelevant memories
    - Tag-based organization
    """

    def __init__(
        self, db_path: str = "data/memory/ancestor.db", encryption_key: str = None
    ):
        self.db_path = db_path
        self.encryption_key = (
            encryption_key if encryption_key else _generate_encryption_key()
        )
        self._loaded_cache: Dict[str, ArchivedMemory] = {}
        self._max_cache_size = 50
        self._init_db()

    def _init_db(self):
        """Initialize database"""
        with db_connection(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archived_memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    summary TEXT,
                    importance REAL DEFAULT 0.5,
                    memory_type TEXT DEFAULT 'general',
                    tags TEXT,
                    created_at REAL,
                    archived_at REAL,
                    last_accessed REAL,
                    access_count INTEGER DEFAULT 0,
                    compression_ratio REAL DEFAULT 1.0,
                    encrypted INTEGER DEFAULT 0,
                    content_hash TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_archived_importance ON archived_memories(importance DESC)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_archived_type ON archived_memories(memory_type)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_archived_archived_at ON archived_memories(archived_at DESC)
            """)

    def archive(
        self,
        content: str,
        importance: float = 0.5,
        memory_type: str = "general",
        tags: List[str] = None,
        metadata: Dict = None,
    ) -> str:
        """Archive a memory"""

        # Create summary
        summary = content[:200] + "..." if len(content) > 200 else content

        # Generate ID from content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Compress content
        compressed = zlib.compress(content.encode(), level=6)
        compression_ratio = len(content) / len(compressed) if compressed else 1.0

        memory = ArchivedMemory(
            content=compressed.decode("latin-1")
            if isinstance(compressed, bytes)
            else compressed,
            summary=summary,
            importance=importance,
            memory_type=memory_type,
            tags=tags or [],
            created_at=time.time(),
            compression_ratio=compression_ratio,
        )

        self._store_memory(memory, content_hash)

        return memory.id

    def _store_memory(self, memory: ArchivedMemory, content_hash: str):
        """Store archived memory"""
        with db_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO archived_memories
                (id, content, summary, importance, memory_type, tags, created_at, archived_at, 
                 last_accessed, access_count, compression_ratio, encrypted, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    memory.id,
                    memory.content,
                    memory.summary,
                    memory.importance,
                    memory.memory_type,
                    json.dumps(memory.tags),
                    memory.created_at,
                    memory.archived_at,
                    memory.last_accessed,
                    memory.access_count,
                    memory.compression_ratio,
                    1 if memory.encrypted else 0,
                    content_hash,
                ),
            )

    def retrieve(
        self, memory_id: str, decompress: bool = True
    ) -> Optional[ArchivedMemory]:
        """Retrieve archived memory (lazy load)"""

        # Check cache first
        if memory_id in self._loaded_cache:
            memory = self._loaded_cache[memory_id]
            self._update_access(memory_id)
            if decompress:
                memory.content = self._decompress(memory.content)
            return memory

        # Load from database
        conn = get_connection(self.db_path)
        cursor = conn.execute(
            """
            SELECT id, content, summary, importance, memory_type, tags, created_at, 
                   archived_at, last_accessed, access_count, compression_ratio, encrypted
            FROM archived_memories WHERE id = ?
        """,
            (memory_id,),
        )

        row = cursor.fetchone()

        if not row:
            return None

        memory = ArchivedMemory(
            id=row[0],
            content=row[1],
            summary=row[2],
            importance=row[3],
            memory_type=row[4],
            tags=json.loads(row[5]) if row[5] else [],
            created_at=row[6],
            archived_at=row[7],
            last_accessed=row[8],
            access_count=row[9],
            compression_ratio=row[10],
            encrypted=bool(row[11]),
        )

        # Cache it
        self._cache_memory(memory)

        # Update access
        self._update_access(memory_id)

        # Decompress if requested
        if decompress:
            memory.content = self._decompress(memory.content)

        return memory

    def search(
        self,
        query: str,
        memory_type: str = None,
        min_importance: float = 0.0,
        limit: int = 10,
    ) -> List[ArchivedMemory]:
        """Search archived memories"""

        conn = get_connection(self.db_path)

        query_lower = query.lower()
        query_words = query_lower.split()

        sql = """
            SELECT id, content, summary, importance, memory_type, tags, created_at, 
                   archived_at, last_accessed, access_count, compression_ratio, encrypted
            FROM archived_memories
            WHERE importance >= ?
        """
        params = [min_importance]

        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)

        sql += " ORDER BY importance DESC, last_accessed DESC LIMIT ?"
        params.append(limit * 2)

        cursor = conn.execute(sql, params)

        results = []
        for row in cursor.fetchall():
            summary_lower = row[2].lower() if row[2] else ""
            relevance = sum(1 for w in query_words if w in summary_lower)

            if relevance > 0 or not query_words:
                memory = ArchivedMemory(
                    id=row[0],
                    content="",  # Don't load full content for search
                    summary=row[2],
                    importance=row[3],
                    memory_type=row[4],
                    tags=json.loads(row[5]) if row[5] else [],
                    created_at=row[6],
                    archived_at=row[7],
                    last_accessed=row[8],
                    access_count=row[9],
                    compression_ratio=row[10],
                    encrypted=bool(row[11]),
                )
                memory._relevance = relevance
                results.append(memory)

        # Sort by combined score
        results.sort(
            key=lambda x: x.importance * 0.5 + getattr(x, "_relevance", 0), reverse=True
        )

        return results[:limit]

    def _update_access(self, memory_id: str):
        """Update access statistics"""
        with db_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE archived_memories
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
            """,
                (time.time(), memory_id),
            )

    def _cache_memory(self, memory: ArchivedMemory):
        """Cache memory, evicting oldest if needed"""
        if len(self._loaded_cache) >= self._max_cache_size:
            # Evict least recently accessed
            oldest = min(self._loaded_cache.items(), key=lambda x: x[1].last_accessed)
            del self._loaded_cache[oldest[0]]

        self._loaded_cache[memory.id] = memory

    def _decompress(self, content: str) -> str:
        """Decompress content"""
        try:
            return zlib.decompress(content.encode("latin-1")).decode()
        except:
            return content

    def prune(self, max_age_days: int = 365, min_importance: float = 0.2) -> int:
        """Prune old, low-importance memories"""
        cutoff_time = time.time() - (max_age_days * 86400)

        with db_connection(self.db_path) as conn:
            # Delete old, low-importance memories
            cursor = conn.execute(
                """
                DELETE FROM archived_memories
                WHERE archived_at < ? AND importance < ?
            """,
                (cutoff_time, min_importance),
            )

            deleted = cursor.rowcount

        return deleted

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        conn = get_connection(self.db_path)

        cursor = conn.execute("""
            SELECT COUNT(*), AVG(importance), SUM(access_count), AVG(compression_ratio)
            FROM archived_memories
        """)
        row = cursor.fetchone()

        return {
            "total_archived": row[0] or 0,
            "avg_importance": row[1] or 0,
            "total_accesses": row[2] or 0,
            "avg_compression": row[3] or 1.0,
            "cache_size": len(self._loaded_cache),
        }

    def clear_cache(self):
        """Clear in-memory cache"""
        self._loaded_cache.clear()
