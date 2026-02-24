"""
Episodic Memory - Hippocampus CA3 Analogue
Fast encoding, pattern separation/completion, emotional tagging
"""

import sqlite3
import json
import uuid
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib


@dataclass
class Experience:
    """A single experience to encode"""

    content: str
    embedding: Optional[List[float]] = None
    context: Dict = field(default_factory=dict)
    outcome: Optional[str] = None
    emotional_valence: float = 0.0  # -1 to 1
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


@dataclass
class EpisodicTrace:
    """Encoded episodic memory trace"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sensory: Optional[List[float]] = None
    spatial: Optional[List[float]] = None
    temporal: Optional[List[float]] = None
    emotional: float = 0.0
    outcome: Optional[str] = None
    context: Dict = field(default_factory=dict)
    content: str = ""
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    consolidated: bool = False


class PatternSeparator:
    """Pattern separation to prevent interference"""

    def separate(
        self, embedding: List[float], existing_memories: List[List[float]] = None
    ) -> Dict:
        """
        Separate new encoding from existing memories
        Uses random projection for pattern separation
        """
        if not embedding:
            return {"sensory": None, "spatial": None, "temporal": None}

        emb = np.array(embedding)

        # Simple pattern separation: add noise proportional to similarity
        if existing_memories and len(existing_memories) > 0:
            existing = np.array(existing_memories)
            similarities = np.dot(existing, emb) / (
                np.linalg.norm(existing, axis=1) + 1e-8
            )
            noise_scale = np.clip(np.max(similarities), 0, 0.5)
            noise = np.random.randn(len(emb)) * noise_scale * 0.1
            emb = emb + noise

        # Split into components
        n = len(emb) // 3
        return {
            "sensory": emb[:n].tolist() if n > 0 else emb.tolist(),
            "spatial": emb[n : 2 * n].tolist() if n > 0 else emb.tolist(),
            "temporal": emb[2 * n :].tolist() if n > 0 else emb.tolist(),
        }


class PatternCompleter:
    """Pattern completion for fuzzy retrieval"""

    def complete(
        self, candidates: List[EpisodicTrace], query_embedding: List[float]
    ) -> List[EpisodicTrace]:
        """Complete missing details from partial cues"""
        if not candidates or not query_embedding:
            return candidates

        # Simple completion: boost items with similar embeddings
        query = np.array(query_embedding)

        completed = []
        for trace in candidates:
            if trace.sensory:
                trace_emb = np.array(trace.sensory + trace.spatial + trace.temporal)
                similarity = np.dot(query, trace_emb) / (
                    np.linalg.norm(query) * np.linalg.norm(trace_emb) + 1e-8
                )
                trace.importance = max(trace.importance, similarity)
            completed.append(trace)

        return sorted(completed, key=lambda x: x.importance, reverse=True)


class EpisodicMemory:
    """
    Fast-learning episodic memory with pattern completion

    Features:
    - Pattern separation (prevents interference)
    - Pattern completion (fuzzy retrieval)
    - Rapid encoding (< 100ms target)
    - Emotional valence tagging
    - Context preservation
    """

    def __init__(self, db_path: str = "data/memory/episodic.db"):
        self.db_path = db_path
        self.pattern_separator = PatternSeparator()
        self.pattern_completer = PatternCompleter()
        self._embedding_cache: Dict[str, List[float]] = {}
        self._max_memory_embeddings = 1000
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database"""
        import os

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic_traces (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                sensory BLOB,
                spatial BLOB,
                temporal BLOB,
                emotional REAL DEFAULT 0.0,
                outcome TEXT,
                context TEXT,
                importance REAL DEFAULT 0.5,
                created_at REAL,
                access_count INTEGER DEFAULT 0,
                consolidated INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodic_created ON episodic_traces(created_at DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodic_importance ON episodic_traces(importance DESC)
        """)
        conn.commit()
        conn.close()

    def encode(self, experience: Experience) -> EpisodicTrace:
        """Pattern-separate before storing"""
        existing = self.get_all_embeddings()
        separated = self.pattern_separator.separate(
            experience.embedding or [], existing_memories=existing
        )

        trace = EpisodicTrace(
            sensory=separated.get("sensory"),
            spatial=separated.get("spatial"),
            temporal=separated.get("temporal"),
            emotional=experience.emotional_valence,
            outcome=experience.outcome,
            context=experience.context,
            content=experience.content,
            importance=self._calculate_importance(experience),
        )

        self._store_trace(trace)
        return trace

    def _calculate_importance(self, experience: Experience) -> float:
        """Calculate initial importance from experience"""
        importance = 0.5

        # Emotional impact
        importance += abs(experience.emotional_valence) * 0.3

        # Outcome significance
        if experience.outcome:
            importance += 0.2

        # Content length (longer = potentially more important)
        if len(experience.content) > 200:
            importance += 0.1

        return min(1.0, importance)

    def _store_trace(self, trace: EpisodicTrace):
        """Store trace in database with transaction"""
        import os

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)

        def to_blob(arr):
            return json.dumps(arr) if arr else None

        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO episodic_traces
                    (id, content, sensory, spatial, temporal, emotional, outcome, context, importance, created_at, access_count, consolidated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        trace.id,
                        trace.content,
                        to_blob(trace.sensory),
                        to_blob(trace.spatial),
                        to_blob(trace.temporal),
                        trace.emotional,
                        trace.outcome,
                        json.dumps(trace.context),
                        trace.importance,
                        trace.created_at,
                        trace.access_count,
                        1 if trace.consolidated else 0,
                    ),
                )
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def retrieve(
        self, query_embedding: List[float], top_k: int = 10
    ) -> List[EpisodicTrace]:
        """Retrieve similar traces with pattern completion"""
        if not query_embedding:
            return self._get_recent(top_k)

        # Search for similar traces
        candidates = self._search_similar(query_embedding, top_k * 2)

        # Pattern complete
        completed = self.pattern_completer.complete(candidates, query_embedding)

        # Update access counts
        for trace in completed[:top_k]:
            self._update_access(trace.id)

        return completed[:top_k]

    def _search_similar(
        self, query_embedding: List[float], limit: int
    ) -> List[EpisodicTrace]:
        """Search for similar traces using simple cosine similarity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """
            SELECT id, content, sensory, spatial, temporal, emotional, outcome, context, importance, created_at, access_count, consolidated
            FROM episodic_traces
            ORDER BY importance DESC
            LIMIT ?
        """,
            (limit * 2,),
        )

        query = np.array(query_embedding) if query_embedding else None
        results = []

        for row in cursor.fetchall():
            sensory = json.loads(row[2]) if row[2] else []
            spatial = json.loads(row[3]) if row[3] else []
            temporal = json.loads(row[4]) if row[4] else []

            if query is not None and (sensory or spatial or temporal):
                trace_emb = np.array(
                    sensory
                    + spatial
                    + temporal[: len(query) - len(sensory) - len(spatial)]
                )
                if len(trace_emb) > 0:
                    sim = np.dot(query, trace_emb) / (
                        np.linalg.norm(query) * np.linalg.norm(trace_emb) + 1e-8
                    )
                    if sim < 0.3:  # Skip too dissimilar
                        continue

            results.append(
                EpisodicTrace(
                    id=row[0],
                    content=row[1],
                    sensory=sensory,
                    spatial=spatial,
                    temporal=temporal,
                    emotional=row[5],
                    outcome=row[6],
                    context=json.loads(row[7]) if row[7] else {},
                    importance=row[8],
                    created_at=row[9],
                    access_count=row[10],
                    consolidated=bool(row[11]),
                )
            )

        conn.close()
        return sorted(results, key=lambda x: x.importance, reverse=True)[:limit]

    def _get_recent(self, limit: int) -> List[EpisodicTrace]:
        """Get recent traces"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """
            SELECT id, content, sensory, spatial, temporal, emotional, outcome, context, importance, created_at, access_count, consolidated
            FROM episodic_traces
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (limit,),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                EpisodicTrace(
                    id=row[0],
                    content=row[1],
                    sensory=json.loads(row[2]) if row[2] else [],
                    spatial=json.loads(row[3]) if row[3] else [],
                    temporal=json.loads(row[4]) if row[4] else [],
                    emotional=row[5],
                    outcome=row[6],
                    context=json.loads(row[7]) if row[7] else {},
                    importance=row[8],
                    created_at=row[9],
                    access_count=row[10],
                    consolidated=bool(row[11]),
                )
            )

        conn.close()
        return results

    def _get_all_embeddings(self, max_sample: int = None) -> List[List[float]]:
        """Get embeddings for pattern separation (lazy with memory check)"""
        import sys

        conn = sqlite3.connect(self.db_path)

        # Count total first
        count_cursor = conn.execute("SELECT COUNT(*) FROM episodic_traces")
        total_count = count_cursor.fetchone()[0]
        conn.close()

        # Return empty if too many embeddings (memory protection)
        if total_count > self._max_memory_embeddings:
            return []

        # Limit sample size if specified
        limit = max_sample if max_sample else min(total_count, 500)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT sensory, spatial, temporal FROM episodic_traces LIMIT ?", (limit,)
        )

        embeddings = []
        for row in cursor.fetchall():
            emb = []
            if row[0]:
                emb.extend(json.loads(row[0]))
            if row[1]:
                emb.extend(json.loads(row[1]))
            if row[2]:
                emb.extend(json.loads(row[2]))
            if emb:
                embeddings.append(emb)

        conn.close()
        return embeddings

    def get_all_embeddings(self, max_sample: int = None) -> List[List[float]]:
        """Public wrapper with lazy loading and memory protection"""
        return self._get_all_embeddings(max_sample)

    def _update_access(self, trace_id: str):
        """Update access count with transaction"""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE episodic_traces 
                    SET access_count = access_count + 1 
                    WHERE id = ?
                """,
                    (trace_id,),
                )
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_unconsolidated(
        self, min_importance: float = 0.7, limit: int = 10
    ) -> List[EpisodicTrace]:
        """Get unconsolidated traces ready for consolidation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """
            SELECT id, content, sensory, spatial, temporal, emotional, outcome, context, importance, created_at, access_count, consolidated
            FROM episodic_traces
            WHERE importance >= ? AND consolidated = 0
            ORDER BY importance DESC
            LIMIT ?
        """,
            (min_importance, limit),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                EpisodicTrace(
                    id=row[0],
                    content=row[1],
                    sensory=json.loads(row[2]) if row[2] else [],
                    spatial=json.loads(row[3]) if row[3] else [],
                    temporal=json.loads(row[4]) if row[4] else [],
                    emotional=row[5],
                    outcome=row[6],
                    context=json.loads(row[7]) if row[7] else {},
                    importance=row[8],
                    created_at=row[9],
                    access_count=row[10],
                    consolidated=False,
                )
            )

        conn.close()
        return results

    def mark_consolidated(self, trace_id: str):
        """Mark trace as consolidated with transaction"""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE episodic_traces SET consolidated = 1 WHERE id = ?",
                    (trace_id,),
                )
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT COUNT(*), SUM(consolidated), AVG(importance), MAX(created_at)
            FROM episodic_traces
        """)
        row = cursor.fetchone()
        conn.close()

        return {
            "total_traces": row[0] or 0,
            "consolidated": row[1] or 0,
            "avg_importance": row[2] or 0,
            "oldest": row[3],
        }
