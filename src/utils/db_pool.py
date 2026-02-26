"""
AURA v3 SQLite Connection Pool

Lightweight connection pool for SQLite databases. Eliminates the overhead of
opening/closing connections on every operation (94 call sites across 11 files).

Design decisions:
- threading.local() for thread safety — each thread gets its own connection
  per database file (SQLite connections are not thread-safe)
- WAL journal mode — allows concurrent reads while writing
- Connection reuse — same thread + same db_path = same connection
- Context manager interface — clean usage with automatic error handling
- Explicit close_all() for graceful shutdown
- No external dependencies — pure stdlib

Usage:
    from src.utils.db_pool import get_connection, connection

    # Context manager (preferred — handles commit/rollback):
    with connection(db_path) as conn:
        conn.execute("INSERT INTO ...")

    # Direct access (for read-only or custom transaction control):
    conn = get_connection(db_path)
    rows = conn.execute("SELECT ...").fetchall()

    # Shutdown:
    close_all()

Memory impact: ~50KB per open connection. With 9 databases on a single thread,
that's ~450KB — negligible vs the current overhead of 94 connect/close cycles.
"""

import os
import sqlite3
import threading
import logging
from contextlib import contextmanager
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Thread-local storage: each thread maintains its own dict of {db_path: connection}
_local = threading.local()

# Global registry of all connections across all threads (for shutdown cleanup)
_registry_lock = threading.Lock()
_registry: Dict[
    int, Dict[str, sqlite3.Connection]
] = {}  # thread_id -> {db_path -> conn}


def _get_thread_connections() -> Dict[str, sqlite3.Connection]:
    """Get the connection dict for the current thread, creating if needed."""
    if not hasattr(_local, "connections"):
        _local.connections = {}
        # Register in global registry for shutdown cleanup
        with _registry_lock:
            _registry[threading.get_ident()] = _local.connections
    return _local.connections


def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Get a pooled SQLite connection for the given database path.

    Returns the same connection for repeated calls with the same db_path
    from the same thread. The connection is configured with:
    - WAL journal mode (better concurrent access)
    - Foreign keys enabled
    - Busy timeout of 5 seconds

    Args:
        db_path: Path to the SQLite database file

    Returns:
        sqlite3.Connection — reusable, do NOT close manually
    """
    # Normalize the path so different representations match
    db_path = os.path.abspath(db_path)

    connections = _get_thread_connections()

    conn = connections.get(db_path)
    if conn is not None:
        # Verify connection is still alive
        try:
            conn.execute("SELECT 1")
            return conn
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            # Connection was closed or corrupted — remove and recreate
            logger.debug("Stale connection for %s, recreating", db_path)
            connections.pop(db_path, None)

    # Ensure directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # Create new connection with optimized settings
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    # Synchronous NORMAL is safe with WAL and ~2x faster than FULL
    conn.execute("PRAGMA synchronous=NORMAL")

    connections[db_path] = conn
    logger.debug(
        "Opened pooled connection for %s (thread %d)", db_path, threading.get_ident()
    )

    return conn


@contextmanager
def connection(db_path: str):
    """
    Context manager for database operations with automatic commit/rollback.

    Usage:
        with connection("data/memory/episodic.db") as conn:
            conn.execute("INSERT INTO ...")
            # Auto-commits on success, rolls back on exception

    The connection is NOT closed on exit — it stays pooled for reuse.
    Only commit/rollback is handled.
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except sqlite3.ProgrammingError:
            pass  # Connection already closed/invalid
        raise


def close_connection(db_path: str) -> None:
    """Close a specific pooled connection for the current thread."""
    db_path = os.path.abspath(db_path)
    connections = _get_thread_connections()
    conn = connections.pop(db_path, None)
    if conn is not None:
        try:
            conn.close()
            logger.debug("Closed pooled connection for %s", db_path)
        except Exception as e:
            logger.warning("Error closing connection for %s: %s", db_path, e)


def close_all() -> None:
    """
    Close ALL pooled connections across ALL threads.
    Call this during graceful shutdown.
    """
    with _registry_lock:
        total_closed = 0
        for thread_id, connections in list(_registry.items()):
            for db_path, conn in list(connections.items()):
                try:
                    conn.close()
                    total_closed += 1
                except Exception as e:
                    logger.warning(
                        "Error closing %s (thread %d): %s", db_path, thread_id, e
                    )
            connections.clear()
        _registry.clear()

    # Also clear current thread's local
    if hasattr(_local, "connections"):
        _local.connections = {}

    logger.info("Connection pool shutdown: closed %d connections", total_closed)


def get_pool_stats() -> dict:
    """Get statistics about the current connection pool state."""
    with _registry_lock:
        total_connections = sum(len(conns) for conns in _registry.values())
        active_threads = len(_registry)
        db_paths = set()
        for conns in _registry.values():
            db_paths.update(conns.keys())

    return {
        "active_threads": active_threads,
        "total_connections": total_connections,
        "unique_databases": len(db_paths),
        "databases": sorted(db_paths),
    }
