"""
Local Vector Store - Zvec-style Lightweight Vector Storage
Memory-efficient vector storage for RAG with local embedding generation
Optimized for Android/Termux (4GB RAM constraint)
"""

import sqlite3
import json
import logging
import hashlib
import time
import struct
import math
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorEntry:
    """A vector entry with metadata"""

    id: str
    vector: List[float]
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 0.5


@dataclass
class SearchResult:
    """Search result with score"""

    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class LocalEmbeddingGenerator:
    """
    Local embedding generation without external APIs
    Uses hash-based + statistical features for lightweight embeddings
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self._vocab: Dict[str, np.ndarray] = {}
        self._embedding_cache: Dict[str, List[float]] = {}
        self._max_cache_size = 1000

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().split()

    def _get_vocab_embedding(self, token: str) -> np.ndarray:
        """Get or create vocabulary embedding for token"""
        if token in self._vocab:
            return self._vocab[token]

        token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
        np.random.seed(token_hash % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        if len(self._vocab) < 10000:
            self._vocab[token] = embedding

        return embedding

    def generate(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for text
        Uses bag-of-words with vocabulary embeddings
        """
        if not text:
            return [0.0] * self.embedding_dim

        cache_key = hashlib.md5(text.encode()).hexdigest()
        if use_cache and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        tokens = self._tokenize(text)

        if not tokens:
            return [0.0] * self.embedding_dim

        embeddings = []
        for token in tokens:
            emb = self._get_vocab_embedding(token)
            embeddings.append(emb)

        combined = np.mean(embeddings, axis=0)

        text_hash = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        np.random.seed(text_hash % (2**32))
        noise = np.random.randn(self.embedding_dim) * 0.1
        combined = combined + noise

        combined = combined / (np.linalg.norm(combined) + 1e-8)

        result = combined.tolist()

        if use_cache:
            if len(self._embedding_cache) >= self._max_cache_size:
                self._embedding_cache.pop(next(iter(self._embedding_cache)))
            self._embedding_cache[cache_key] = result

        return result

    def batch_generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return [self.generate(text) for text in texts]

    def clear_cache(self):
        """Clear embedding cache"""
        self._embedding_cache.clear()


class QuantizedVectorStore:
    """
    Quantized vector storage for memory efficiency
    Uses product quantization for compression
    """

    def __init__(self, num_clusters: int = 16, sub_dim: int = 8):
        self.num_clusters = num_clusters
        self.sub_dim = sub_dim
        self.codebooks: List[np.ndarray] = []
        self._initialized = False

    def fit(self, vectors: List[np.ndarray]):
        """Build codebooks from vectors"""
        if not vectors:
            return

        vectors = [v for v in vectors if len(v) >= self.sub_dim]
        if not vectors:
            return

        total_dim = len(vectors[0])
        num_subvecs = total_dim // self.sub_dim

        self.codebooks = []

        for i in range(num_subvecs):
            subvecs = np.array(
                [v[i * self.sub_dim : (i + 1) * self.sub_dim] for v in vectors]
            )

            from sklearn.cluster import MiniBatchKMeans

            kmeans = MiniBatchKMeans(
                n_clusters=min(self.num_clusters, len(subvecs)),
                random_state=42,
                n_init=3,
            )
            kmeans.fit(subvecs)
            self.codebooks.append(kmeans.cluster_centers_)

        self._initialized = True

    def quantize(self, vector: np.ndarray) -> np.ndarray:
        """Quantize a vector"""
        if not self._initialized:
            return vector

        total_dim = len(vector)
        num_subvecs = total_dim // self.sub_dim

        codes = []
        for i in range(num_subvecs):
            subvec = vector[i * self.sub_dim : (i + 1) * self.sub_dim]

            codebook = self.codebooks[i]
            distances = np.linalg.norm(codebook - subvec, axis=1)
            codes.append(np.argmin(distances))

        return np.array(codes)

    def dequantize(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct vector from codes"""
        if not self._initialized:
            return np.array([])

        total_dim = len(codes) * self.sub_dim
        reconstructed = np.zeros(total_dim)

        for i, code in enumerate(codes):
            reconstructed[i * self.sub_dim : (i + 1) * self.sub_dim] = self.codebooks[
                i
            ][code]

        return reconstructed


class LocalVectorStore:
    """
    Lightweight vector store for RAG (Zvec-style)

    Features:
    - Local embedding generation (no external APIs)
    - Efficient similarity search
    - Memory-efficient quantized storage
    - Works offline on Android/Termux
    - SQLite-backed persistence
    """

    def __init__(
        self,
        db_path: str = "data/memory/vectors.db",
        embedding_dim: int = 128,
        max_vectors: int = 10000,
        use_quantization: bool = True,
    ):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.max_vectors = max_vectors
        self.use_quantization = use_quantization

        self.embedding_generator = LocalEmbeddingGenerator(embedding_dim)
        self.quantizer = QuantizedVectorStore()

        self._init_db()
        self._init_quantizer()

    def _init_db(self):
        """Initialize SQLite database"""
        import os

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                vector BLOB,
                metadata TEXT,
                created_at REAL,
                access_count INTEGER DEFAULT 0,
                importance REAL DEFAULT 0.5
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vectors_created 
            ON vectors(created_at DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vectors_importance 
            ON vectors(importance DESC)
        """)

        conn.commit()
        conn.close()

    def _init_quantizer(self):
        """Initialize quantizer with existing vectors"""
        if not self.use_quantization:
            return

        vectors = self._load_all_vectors()
        if len(vectors) > 100:
            try:
                import numpy as np

                vector_arrays = [np.array(v.vector) for v in vectors]
                self.quantizer.fit(vector_arrays)
                logger.info(f"Quantizer initialized with {len(vectors)} vectors")
            except ImportError:
                logger.warning("sklearn not available, skipping quantization")
                self.use_quantization = False

    def _load_all_vectors(self) -> List[VectorEntry]:
        """Load all vectors for quantizer training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT id, vector, content, metadata, created_at, access_count, importance FROM vectors LIMIT 1000"
        )

        results = []
        for row in cursor.fetchall():
            vector_data = row[1]
            if vector_data:
                import numpy as np

                vector = np.frombuffer(vector_data, dtype=np.float32).tolist()
            else:
                vector = []

            results.append(
                VectorEntry(
                    id=row[0],
                    vector=vector,
                    content=row[2],
                    metadata=json.loads(row[3]) if row[3] else {},
                    created_at=row[4],
                    access_count=row[5],
                    importance=row[6],
                )
            )

        conn.close()
        return results

    def add(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        importance: float = 0.5,
        vector: List[float] = None,
    ) -> VectorEntry:
        """Add a vector entry"""
        import os

        vector_id = f"vec_{hashlib.md5(content.encode()).hexdigest()[:12]}"

        if vector is None:
            vector = self.embedding_generator.generate(content)

        vector_bytes = np.array(vector, dtype=np.float32).tobytes()

        metadata = metadata or {}
        entry = VectorEntry(
            id=vector_id,
            vector=vector,
            content=content,
            metadata=metadata,
            importance=importance,
        )

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """INSERT OR REPLACE INTO vectors 
                       (id, content, vector, metadata, created_at, access_count, importance)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry.id,
                        entry.content,
                        vector_bytes,
                        json.dumps(entry.metadata),
                        entry.created_at,
                        entry.access_count,
                        entry.importance,
                    ),
                )

                count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
                if count > self.max_vectors:
                    conn.execute(
                        """DELETE FROM vectors WHERE id IN (
                           SELECT id FROM vectors 
                           ORDER BY importance ASC, access_count ASC, created_at ASC
                           LIMIT ?
                        )""",
                        (count - self.max_vectors,),
                    )
        finally:
            conn.close()

        return entry

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        filter_metadata: Dict[str, Any] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        query_vector = self.embedding_generator.generate(query)
        return self.search_by_vector(
            query_vector,
            top_k=top_k,
            min_score=min_score,
            filter_metadata=filter_metadata,
        )

    def search_by_vector(
        self,
        query_vector: List[float],
        top_k: int = 10,
        min_score: float = 0.0,
        filter_metadata: Dict[str, Any] = None,
    ) -> List[SearchResult]:
        """Search by vector"""
        import numpy as np

        if not query_vector:
            return []

        query = np.array(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)

        if query_norm > 0:
            query = query / query_norm

        conn = sqlite3.connect(self.db_path)

        filter_clause = ""
        params = []
        if filter_metadata:
            for key in filter_metadata:
                filter_clause += f" AND metadata LIKE ?"
                params.append(f'%"{key}":%')

        cursor = conn.execute(
            f"""SELECT id, content, vector, metadata, importance, access_count
                FROM vectors 
                WHERE 1=1 {filter_clause}
                ORDER BY importance DESC
                LIMIT ?""",
            params + [top_k * 3],
        )

        results = []
        for row in cursor.fetchall():
            vector_data = row[2]
            if not vector_data:
                continue

            vector = np.frombuffer(vector_data, dtype=np.float32)

            if self.use_quantization and self.quantizer._initialized:
                codes = self.quantizer.quantize(vector)
                vector = self.quantizer.dequantize(codes)

            vector_norm = np.linalg.norm(vector)
            if vector_norm == 0:
                continue

            vector = vector / vector_norm

            similarity = np.dot(query, vector)

            if similarity >= min_score:
                results.append(
                    SearchResult(
                        id=row[0],
                        content=row[1],
                        score=float(similarity),
                        metadata=json.loads(row[3]) if row[3] else {},
                    )
                )

                conn.execute(
                    "UPDATE vectors SET access_count = access_count + 1 WHERE id = ?",
                    (row[0],),
                )

        conn.commit()
        conn.close()

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def get(self, vector_id: str) -> Optional[VectorEntry]:
        """Get a vector by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """SELECT id, vector, content, metadata, created_at, access_count, importance
               FROM vectors WHERE id = ?""",
            (vector_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            vector_data = row[1]
            vector = (
                np.frombuffer(vector_data, dtype=np.float32).tolist()
                if vector_data
                else []
            )

            return VectorEntry(
                id=row[0],
                vector=vector,
                content=row[2],
                metadata=json.loads(row[3]) if row[3] else {},
                created_at=row[4],
                access_count=row[5],
                importance=row[6],
            )

        return None

    def delete(self, vector_id: str) -> bool:
        """Delete a vector"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("DELETE FROM vectors WHERE id = ?", (vector_id,))
        conn.commit()
        conn.close()

        return cursor.rowcount > 0

    def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """Update vector metadata"""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE vectors SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata), vector_id),
                )
            return True
        except Exception:
            return False
        finally:
            conn.close()

    def get_similar_by_id(
        self,
        vector_id: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Find similar vectors to existing one"""
        entry = self.get(vector_id)
        if not entry:
            return []

        return self.search_by_vector(entry.vector, top_k=top_k)

    def batch_add(
        self,
        entries: List[Tuple[str, Dict[str, Any], float]],
    ) -> List[VectorEntry]:
        """Batch add entries"""
        results = []

        for content, metadata, importance in entries:
            entry = self.add(content, metadata, importance)
            results.append(entry)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        conn = sqlite3.connect(self.db_path)

        cursor = conn.execute("""
            SELECT COUNT(*), AVG(importance), SUM(access_count), MAX(created_at)
            FROM vectors
        """)
        row = cursor.fetchone()

        conn.close()

        return {
            "total_vectors": row[0] or 0,
            "avg_importance": row[1] or 0,
            "total_accesses": row[2] or 0,
            "newest": row[3],
            "embedding_dim": self.embedding_dim,
            "quantization_enabled": self.use_quantization,
        }

    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_generator.clear_cache()

    def vacuum(self):
        """Vacuum database to reclaim space"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("VACUUM")
        conn.close()


def get_vector_store(
    db_path: str = "data/memory/vectors.db",
    embedding_dim: int = 128,
) -> LocalVectorStore:
    """Get or create vector store instance"""
    return LocalVectorStore(db_path, embedding_dim)


__all__ = [
    "LocalVectorStore",
    "LocalEmbeddingGenerator",
    "QuantizedVectorStore",
    "VectorEntry",
    "SearchResult",
    "get_vector_store",
]
