"""
Hierarchical Memory System for AURA v3
Implements multiple memory layers from immediate to long-term

Memory Hierarchy:
1. Immediate: Current thought processing
2. Working: Current task context (last few actions)
3. Short-term: Recent conversation (last hour)
4. Long-term: Persistent memories across sessions
5. Self-model: AURA's understanding of itself and user
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import hashlib

from src.utils.db_pool import get_connection, connection as db_connection

# Import and re-export from neural_memory
from src.memory.neural_memory import (
    NeuralMemory,
    get_neural_memory,
    MemoryType as NeuralMemoryType,
)

# Import and re-export from episodic_memory
from src.memory.episodic_memory import EpisodicMemory

# Import and re-export from semantic_memory
from src.memory.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """A single memory item"""

    id: str
    content: str
    importance: float  # 0-1
    timestamp: datetime
    memory_type: str  # interaction, fact, preference, self_model
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class ImmediateMemory:
    """
    Immediate Memory - Current processing context
    Holds the current thought/action/observation being processed
    """

    def __init__(self):
        self.current_thought = None
        self.current_action = None
        self.current_observation = None

    def set_thought(self, thought: str):
        self.current_thought = thought

    def set_action(self, action: str, params: Dict):
        self.current_action = {"tool": action, "params": params}

    def set_observation(self, observation: str):
        self.current_observation = observation

    def clear(self):
        """Clear after each iteration"""
        self.current_thought = None
        self.current_action = None
        self.current_observation = None

    def get_context(self) -> str:
        """Get current context as string"""
        parts = []
        if self.current_thought:
            parts.append(f"Thinking: {self.current_thought}")
        if self.current_action:
            parts.append(f"Acting: {self.current_action}")
        if self.current_observation:
            parts.append(f"Observing: {self.current_observation}")
        return " | ".join(parts) if parts else ""


class WorkingMemory:
    """
    Working Memory - Current task context
    Holds last N items relevant to current task
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.items: deque = deque(maxlen=max_size)
        self.current_task: Optional[str] = None

    def add(self, item: Dict):
        """Add item to working memory"""
        item["timestamp"] = datetime.now().isoformat()
        self.items.append(item)

    def get_recent(self, n: int = 5) -> List[Dict]:
        """Get last N items"""
        return list(self.items)[-n:]

    def clear(self):
        """Clear working memory"""
        self.items.clear()
        self.current_task = None

    def set_task(self, task: str):
        """Set current task"""
        self.current_task = task
        self.clear()

    def get_context(self) -> str:
        """Get context for LLM"""
        if not self.items:
            return ""

        lines = ["Recent context:"]
        for item in self.items:
            if item.get("type") == "action":
                lines.append(
                    f"- Did: {item.get('tool')} -> {item.get('result', {}).get('success', '?')}"
                )
            elif item.get("type") == "message":
                lines.append(f"- {item.get('role')}: {item.get('content', '')[:50]}")

        return "\n".join(lines)


class ShortTermMemory:
    """
    Short-term Memory - Recent conversation
    Stores interactions from last hour
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.items: deque = deque(maxlen=max_size)

    def add(self, role: str, content: str, metadata: Dict = None):
        """Add conversation turn"""
        self.items.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }
        )

    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return list(self.items)[-limit:]

    def get_context_for_llm(self, max_tokens: int = 2000) -> str:
        """Get context formatted for LLM"""
        lines = ["Conversation history:"]

        for item in self.items:
            role = item.get("role", "user")
            content = item.get("content", "")[:100]
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def prune_old(self, max_age_minutes: int = 60):
        """Remove items older than max_age_minutes"""
        cutoff = datetime.now() - timedelta(minutes=max_age_minutes)

        pruned = deque(maxlen=self.max_size)
        for item in self.items:
            item_time = datetime.fromisoformat(item["timestamp"])
            if item_time > cutoff:
                pruned.append(item)

        self.items = pruned


class LongTermMemory:
    """
    Long-term Memory - Persistent storage
    SQLite-backed memory with importance scoring
    """

    def __init__(self, db_path: str = "data/memories/aura_memories.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database"""
        with db_connection(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    memory_type TEXT DEFAULT 'fact',
                    metadata TEXT,
                    created_at TEXT,
                    accessed_at TEXT,
                    access_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_type ON memories(memory_type)
            """)

    def store(
        self,
        content: str,
        importance: float = 0.5,
        memory_type: str = "fact",
        metadata: Dict = None,
    ) -> str:
        """Store a memory"""
        # Generate ID from content hash
        mem_id = hashlib.md5(content.encode()).hexdigest()[:16]
        now = datetime.now().isoformat()

        with db_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories 
                (id, content, importance, memory_type, metadata, created_at, accessed_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, 
                    COALESCE((SELECT access_count FROM memories WHERE id = ?), 0) + 1)
            """,
                (
                    mem_id,
                    content,
                    importance,
                    memory_type,
                    json.dumps(metadata or {}),
                    now,
                    now,
                    mem_id,
                ),
            )

        return mem_id

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> List[Dict]:
        """Retrieve relevant memories"""
        conn = get_connection(self.db_path)

        # Simple keyword-based retrieval (can be upgraded to embeddings)
        query_words = query.lower().split()

        sql = """
            SELECT id, content, importance, memory_type, metadata, created_at, access_count
            FROM memories
            WHERE importance >= ?
        """
        params = [min_importance]

        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)

        sql += " ORDER BY importance DESC, access_count DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(sql, params)
        results = []

        for row in cursor.fetchall():
            content = row[1]
            # Simple relevance scoring
            content_lower = content.lower()
            relevance = sum(1 for word in query_words if word in content_lower)

            results.append(
                {
                    "id": row[0],
                    "content": content,
                    "importance": row[2],
                    "type": row[3],
                    "metadata": json.loads(row[4] or "{}"),
                    "created_at": row[5],
                    "access_count": row[6],
                    "relevance": relevance,
                }
            )

        # Sort by combined score
        results.sort(key=lambda x: x["importance"] * 0.5 + x["relevance"], reverse=True)

        return results[:limit]

    def update_access(self, mem_id: str):
        """Update access timestamp and count"""
        with db_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE memories 
                SET accessed_at = ?, access_count = access_count + 1
                WHERE id = ?
            """,
                (datetime.now().isoformat(), mem_id),
            )

    def delete(self, mem_id: str):
        """Delete a memory"""
        with db_connection(self.db_path) as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))


class SelfModel:
    """
    Self-Model - AURA's understanding of itself and user
    Stores:
    - User preferences and habits
    - AURA's capabilities and limitations
    - Relationship dynamics
    """

    def __init__(self, db_path: str = "data/memories/self_model.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize self-model database"""
        with db_connection(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS self_model (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    category TEXT,
                    confidence REAL DEFAULT 0.5,
                    updated_at TEXT
                )
            """)

    def set(
        self,
        key: str,
        value: Any,
        category: str = "preference",
        confidence: float = 0.5,
    ):
        """Set a self-model attribute"""
        with db_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO self_model (key, value, category, confidence, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    key,
                    json.dumps(value),
                    category,
                    confidence,
                    datetime.now().isoformat(),
                ),
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a self-model attribute"""
        conn = get_connection(self.db_path)
        cursor = conn.execute("SELECT value FROM self_model WHERE key = ?", (key,))
        row = cursor.fetchone()

        if row:
            try:
                return json.loads(row[0])
            except:
                return row[0]
        return default

    def get_category(self, category: str) -> Dict[str, Any]:
        """Get all attributes in a category"""
        conn = get_connection(self.db_path)
        cursor = conn.execute(
            "SELECT key, value, confidence FROM self_model WHERE category = ?",
            (category,),
        )

        results = {}
        for row in cursor.fetchall():
            try:
                results[row[0]] = {"value": json.loads(row[1]), "confidence": row[2]}
            except:
                results[row[0]] = {"value": row[1], "confidence": row[2]}

        return results

    def learn(self, key: str, value: Any, category: str = "preference"):
        """Learn and update with confidence based on repetition"""
        existing = self.get(key)

        if existing is not None:
            # Increase confidence on repeated observation
            current_confidence = 0.5  # Default
            conn = get_connection(self.db_path)
            cursor = conn.execute(
                "SELECT confidence FROM self_model WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            if row:
                current_confidence = min(1.0, row[0] + 0.1)  # Increase by 0.1

            self.set(key, value, category, current_confidence)
        else:
            self.set(key, value, category, 0.5)


class HierarchicalMemory:
    """
    Unified Hierarchical Memory System
    Coordinates all memory layers
    """

    def __init__(
        self,
        working_size: int = 10,
        short_term_size: int = 100,
        db_path: str = "data/memories/aura_memories.db",
        self_model_path: str = "data/memories/self_model.db",
    ):
        self.immediate = ImmediateMemory()
        self.working = WorkingMemory(max_size=working_size)
        self.short_term = ShortTermMemory(max_size=short_term_size)
        self.long_term = LongTermMemory(db_path=db_path)
        self.self_model = SelfModel(db_path=self_model_path)

    async def store(
        self,
        content: str,
        importance: float = 0.5,
        memory_type: str = "fact",
        metadata: Dict = None,
    ):
        """Store to appropriate memory layer"""

        # High importance -> long-term immediately
        if importance >= 0.7:
            self.long_term.store(
                content=content,
                importance=importance,
                memory_type=memory_type,
                metadata=metadata,
            )

        # Always add to short-term
        self.short_term.add(
            role="system" if memory_type == "interaction" else "memory",
            content=content,
            metadata=metadata,
        )

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        context: Dict = None,
        min_importance: float = 0.3,
    ) -> List[Dict]:
        """Retrieve from all memory layers"""
        results = []

        # Get from long-term
        long_term_results = self.long_term.retrieve(
            query=query, limit=limit, min_importance=min_importance
        )
        results.extend(long_term_results)

        # Get from short-term
        for item in self.short_term.get_history(limit=5):
            results.append(
                {
                    "content": item["content"],
                    "importance": 0.5,
                    "type": "short_term",
                    "timestamp": item["timestamp"],
                }
            )

        # Sort by importance
        results.sort(key=lambda x: x.get("importance", 0), reverse=True)

        return results[:limit]

    def get_context_for_llm(self, max_tokens: int = 2000) -> str:
        """Get all context for LLM"""
        parts = []

        # Working memory
        working_context = self.working.get_context()
        if working_context:
            parts.append(f"[Working Memory]\n{working_context}")

        # Short-term
        short_term_context = self.short_term.get_context_for_llm()
        if short_term_context:
            parts.append(f"[Recent History]\n{short_term_context}")

        return "\n\n".join(parts)

    def add_interaction(self, role: str, content: str):
        """Add a conversation interaction"""
        self.short_term.add(role, content)

    def clear_working(self):
        """Clear working memory (between tasks)"""
        self.working.clear()

    def learn_user_preference(self, preference: str, value: Any):
        """Learn a user preference"""
        self.self_model.learn(preference, value, "preference")

    async def initialize(self):
        """Initialize memory system - called at startup"""
        logger.info("Initializing Hierarchical Memory...")
        # Long-term memory initializes its DB in __init__
        # Self-model initializes in __init__
        # Pool handles directory creation
        logger.info("Hierarchical Memory initialized")

    async def persist(self):
        """Persist any pending data - called at shutdown"""
        logger.info("Persisting memory data...")
        # Short-term and working are in-memory, auto-cleared
        # Long-term is already persisted to SQLite
        # Self-model is already persisted to SQLite
        logger.info("Memory persisted")

    def get_stats(self) -> str:
        """Get memory statistics"""
        return f"Working: {len(self.working.items)}/{self.working.max_size}, Short-term: {len(self.short_term.items)}/{self.short_term.max_size}"

    def get_user_preference(self, preference: str, default: Any = None) -> Any:
        """Get a user preference"""
        return self.self_model.get(preference, default)

    def learn_contact_pattern(self, contact: str, pattern: str, value: Any):
        """Learn a pattern about a contact"""
        key = f"contact:{contact}:{pattern}"
        self.self_model.learn(key, value, "contact_pattern")

    def get_contact_pattern(self, contact: str, pattern: str) -> Any:
        """Get a pattern for a contact"""
        key = f"contact:{contact}:{pattern}"
        return self.self_model.get(key)
