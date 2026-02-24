"""
SQLite State Machine - Persistent State Management
Uses SQLite as a state machine for memory persistence with transaction-based operations
"""

import sqlite3
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class StateTransitionError(Exception):
    """Error during state transition"""

    pass


class StateMachineError(Exception):
    """General state machine error"""

    pass


class TransitionType(Enum):
    """Types of state transitions"""

    EXPLICIT = "explicit"  # Explicit user action
    AUTOMATIC = "automatic"  # Automatic system transition
    SCHEDULED = "scheduled"  # Time-based transition
    CONDITION = "condition"  # Condition-based transition


@dataclass
class State:
    """A state in the state machine"""

    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class Transition:
    """A state transition"""

    id: str
    from_state: str
    to_state: str
    trigger: str
    transition_type: TransitionType
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateSnapshot:
    """Snapshot of state for rollback"""

    id: str
    state_name: str
    data: str  # JSON serialized
    timestamp: float
    label: str = ""


class SQLiteStateMachine:
    """
    SQLite-based state machine for memory persistence

    Features:
    - Transaction-based operations
    - State versioning with rollback
    - Event sourcing for audit trail
    - Efficient querying with SQLite
    - Thread-safe operations
    - Works offline on Android/Termux
    """

    def __init__(
        self,
        db_path: str = "data/memory/state_machine.db",
        max_snapshots: int = 100,
        enable_wal: bool = True,
    ):
        self.db_path = db_path
        self.max_snapshots = max_snapshots
        self._lock = threading.RLock()

        self._init_db(enable_wal)

        self._transition_handlers: Dict[str, List[Callable]] = {}
        self._current_state_cache: Optional[State] = None

    def _init_db(self, enable_wal: bool):
        """Initialize SQLite database"""
        import os

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)

        if enable_wal:
            conn.execute("PRAGMA journal_mode=WAL")

        conn.execute("PRAGMA synchronous=NORMAL")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS states (
                name TEXT PRIMARY KEY,
                data TEXT,
                metadata TEXT,
                created_at REAL,
                updated_at REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS transitions (
                id TEXT PRIMARY KEY,
                from_state TEXT,
                to_state TEXT,
                trigger TEXT,
                transition_type TEXT,
                conditions TEXT,
                actions TEXT,
                timestamp REAL,
                metadata TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id TEXT PRIMARY KEY,
                state_name TEXT,
                data TEXT,
                timestamp REAL,
                label TEXT
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_transitions_timestamp 
            ON transitions(timestamp DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_transitions_states 
            ON transitions(from_state, to_state)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_state 
            ON snapshots(state_name, timestamp DESC)
        """)

        conn.commit()
        conn.close()

    @contextmanager
    def _transaction(self, conn: sqlite3.Connection):
        """Context manager for transactions"""
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise StateTransitionError(f"Transaction failed: {e}")

    def get_state(self, name: str) -> Optional[State]:
        """Get current state by name"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT name, data, metadata, created_at, updated_at FROM states WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return State(
                    name=row[0],
                    data=json.loads(row[1]) if row[1] else {},
                    metadata=json.loads(row[2]) if row[2] else {},
                    created_at=row[3],
                    updated_at=row[4],
                )
            return None

    def set_state(self, state: State, create_snapshot: bool = True) -> State:
        """Set state with optional snapshot"""
        with self._lock:
            state.updated_at = time.time()

            if create_snapshot and self._current_state_cache:
                self._create_snapshot(self._current_state_cache.name)

            conn = sqlite3.connect(self.db_path)
            try:
                with self._transaction(conn):
                    conn.execute(
                        """INSERT OR REPLACE INTO states 
                           (name, data, metadata, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            state.name,
                            json.dumps(state.data),
                            json.dumps(state.metadata),
                            state.created_at,
                            state.updated_at,
                        ),
                    )
            finally:
                conn.close()

            self._current_state_cache = state
            return state

    def transition(
        self,
        from_state: str,
        to_state: str,
        trigger: str,
        transition_type: TransitionType = TransitionType.EXPLICIT,
        conditions: Dict[str, Any] = None,
        actions: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> Transition:
        """Execute a state transition"""
        with self._lock:
            current = self.get_state(from_state)
            if not current:
                raise StateTransitionError(f"State '{from_state}' does not exist")

            if to_state:
                target = self.get_state(to_state)
                if not target:
                    target = State(name=to_state)
                    self.set_state(target, create_snapshot=False)

            transition = Transition(
                id=f"t_{int(time.time() * 1000)}",
                from_state=from_state,
                to_state=to_state,
                trigger=trigger,
                transition_type=transition_type,
                conditions=conditions or {},
                actions=actions or [],
                metadata=metadata or {},
            )

            conn = sqlite3.connect(self.db_path)
            try:
                with self._transaction(conn):
                    conn.execute(
                        """INSERT INTO transitions 
                           (id, from_state, to_state, trigger, transition_type, 
                            conditions, actions, timestamp, metadata)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            transition.id,
                            transition.from_state,
                            transition.to_state,
                            transition.trigger,
                            transition.transition_type.value,
                            json.dumps(transition.conditions),
                            json.dumps(transition.actions),
                            transition.timestamp,
                            json.dumps(transition.metadata),
                        ),
                    )
            finally:
                conn.close()

            handlers = self._transition_handlers.get(to_state, [])
            for handler in handlers:
                try:
                    handler(transition)
                except Exception as e:
                    logger.warning(f"Transition handler failed: {e}")

            return transition

    def register_handler(self, state: str, handler: Callable):
        """Register handler for state transitions"""
        if state not in self._transition_handlers:
            self._transition_handlers[state] = []
        self._transition_handlers[state].append(handler)

    def get_history(
        self,
        state_name: str = None,
        limit: int = 100,
    ) -> List[Transition]:
        """Get transition history"""
        conn = sqlite3.connect(self.db_path)

        if state_name:
            cursor = conn.execute(
                """SELECT id, from_state, to_state, trigger, transition_type,
                          conditions, actions, timestamp, metadata
                   FROM transitions 
                   WHERE from_state = ? OR to_state = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (state_name, state_name, limit),
            )
        else:
            cursor = conn.execute(
                """SELECT id, from_state, to_state, trigger, transition_type,
                          conditions, actions, timestamp, metadata
                   FROM transitions 
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (limit,),
            )

        results = []
        for row in cursor.fetchall():
            results.append(
                Transition(
                    id=row[0],
                    from_state=row[1],
                    to_state=row[2],
                    trigger=row[3],
                    transition_type=TransitionType(row[4]),
                    conditions=json.loads(row[5]) if row[5] else {},
                    actions=json.loads(row[6]) if row[6] else [],
                    timestamp=row[7],
                    metadata=json.loads(row[8]) if row[8] else {},
                )
            )

        conn.close()
        return results

    def _create_snapshot(self, state_name: str, label: str = ""):
        """Create state snapshot for rollback"""
        state = self.get_state(state_name)
        if not state:
            return

        snapshot = StateSnapshot(
            id=f"snap_{int(time.time() * 1000)}",
            state_name=state_name,
            data=json.dumps(state.data),
            timestamp=time.time(),
            label=label,
        )

        conn = sqlite3.connect(self.db_path)
        try:
            with self._transaction(conn):
                conn.execute(
                    """INSERT INTO snapshots (id, state_name, data, timestamp, label)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        snapshot.id,
                        snapshot.state_name,
                        snapshot.data,
                        snapshot.timestamp,
                        snapshot.label,
                    ),
                )

                conn.execute(
                    """DELETE FROM snapshots 
                       WHERE state_name = ? AND id NOT IN (
                           SELECT id FROM snapshots 
                           WHERE state_name = ? 
                           ORDER BY timestamp DESC 
                           LIMIT ?
                       )""",
                    (state_name, state_name, self.max_snapshots),
                )
        finally:
            conn.close()

    def rollback(self, state_name: str, snapshot_id: str = None) -> State:
        """Rollback to previous snapshot"""
        conn = sqlite3.connect(self.db_path)

        if snapshot_id:
            cursor = conn.execute(
                "SELECT id, state_name, data, timestamp, label FROM snapshots WHERE id = ?",
                (snapshot_id,),
            )
        else:
            cursor = conn.execute(
                """SELECT id, state_name, data, timestamp, label 
                   FROM snapshots 
                   WHERE state_name = ?
                   ORDER BY timestamp DESC 
                   LIMIT 1""",
                (state_name,),
            )

        row = cursor.fetchone()
        conn.close()

        if not row:
            raise StateMachineError(f"No snapshot found for state '{state_name}'")

        state = State(
            name=row[1],
            data=json.loads(row[2]),
            created_at=row[3],
            updated_at=time.time(),
        )

        self.set_state(state, create_snapshot=False)
        return state

    def get_snapshots(self, state_name: str) -> List[StateSnapshot]:
        """Get available snapshots for state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """SELECT id, state_name, data, timestamp, label 
               FROM snapshots 
               WHERE state_name = ?
               ORDER BY timestamp DESC""",
            (state_name,),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                StateSnapshot(
                    id=row[0],
                    state_name=row[1],
                    data=row[2],
                    timestamp=row[3],
                    label=row[4] or "",
                )
            )

        conn.close()
        return results

    def query_states(
        self,
        predicate: Callable[[State], bool] = None,
        limit: int = 100,
    ) -> List[State]:
        """Query states with optional predicate"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT name, data, metadata, created_at, updated_at FROM states LIMIT ?",
            (limit,),
        )

        results = []
        for row in cursor.fetchall():
            state = State(
                name=row[0],
                data=json.loads(row[1]) if row[1] else {},
                metadata=json.loads(row[2]) if row[2] else {},
                created_at=row[3],
                updated_at=row[4],
            )

            if predicate is None or predicate(state):
                results.append(state)

        conn.close()
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get state machine statistics"""
        conn = sqlite3.connect(self.db_path)

        states_count = conn.execute("SELECT COUNT(*) FROM states").fetchone()[0]
        transitions_count = conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[
            0
        ]
        snapshots_count = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]

        conn.close()

        return {
            "states": states_count,
            "transitions": transitions_count,
            "snapshots": snapshots_count,
        }


class MemoryStateMachine(SQLiteStateMachine):
    """
    Specialized state machine for memory persistence
    Provides high-level memory state management
    """

    MEMORY_STATES = {
        "active": "Memory is actively being used",
        "idle": "Memory is stored but not active",
        "archiving": "Memory is being archived",
        "archived": "Memory is archived for long-term storage",
        "consolidating": "Memory is being consolidated",
        "consolidated": "Memory has been consolidated to semantic",
        "forgetting": "Memory is being forgotten",
    }

    def __init__(self, db_path: str = "data/memory/memory_state.db"):
        super().__init__(db_path)
        self._init_memory_states()

    def _init_memory_states(self):
        """Initialize memory states"""
        for state_name in self.MEMORY_STATES:
            if not self.get_state(state_name):
                self.set_state(State(name=state_name), create_snapshot=False)

    def store_memory(
        self,
        memory_id: str,
        memory_type: str,
        data: Dict[str, Any],
        priority: str = "normal",
    ) -> State:
        """Store a memory in the state machine"""
        state = self.get_state(memory_id)

        if state:
            state.data.update(data)
            state.data["memory_type"] = memory_type
            state.data["priority"] = priority
            state.data["last_accessed"] = time.time()
        else:
            state = State(
                name=memory_id,
                data={
                    **data,
                    "memory_type": memory_type,
                    "priority": priority,
                    "created_at": time.time(),
                    "last_accessed": time.time(),
                },
            )

        return self.set_state(state)

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory data"""
        state = self.get_state(memory_id)
        if state:
            state.data["last_accessed"] = time.time()
            self.set_state(state, create_snapshot=False)
            return state.data
        return None

    def archive_memory(self, memory_id: str) -> Transition:
        """Archive a memory"""
        return self.transition(
            from_state=memory_id,
            to_state="archived",
            trigger="archive",
            transition_type=TransitionType.AUTOMATIC,
        )

    def consolidate_memory(self, memory_id: str) -> Transition:
        """Mark memory as consolidated"""
        return self.transition(
            from_state=memory_id,
            to_state="consolidated",
            trigger="consolidate",
            transition_type=TransitionType.AUTOMATIC,
        )

    def get_active_memories(self, limit: int = 100) -> List[State]:
        """Get active memories sorted by last access"""

        def is_active(state: State) -> bool:
            return state.data.get("last_accessed", 0) > time.time() - 86400

        return self.query_states(predicate=is_active, limit=limit)


def get_state_machine(
    db_path: str = "data/memory/state_machine.db",
) -> SQLiteStateMachine:
    """Get or create state machine instance"""
    return SQLiteStateMachine(db_path)


def get_memory_state_machine(
    db_path: str = "data/memory/memory_state.db",
) -> MemoryStateMachine:
    """Get or create memory state machine instance"""
    return MemoryStateMachine(db_path)


__all__ = [
    "SQLiteStateMachine",
    "MemoryStateMachine",
    "State",
    "Transition",
    "StateSnapshot",
    "TransitionType",
    "StateTransitionError",
    "StateMachineError",
    "get_state_machine",
    "get_memory_state_machine",
]
