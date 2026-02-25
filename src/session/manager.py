"""Session management with encrypted storage for AURA."""

import asyncio
import hashlib
import json
import logging
import os
import uuid
from base64 import b64decode, b64encode
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning(
        "cryptography package not available. Sessions will not be encrypted."
    )


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        return cls(**data)


@dataclass
class Session:
    id: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Session":
        messages = [Message.from_dict(m) for m in data.get("messages", [])]
        return cls(
            id=data["id"],
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            messages=messages,
            metadata=data.get("metadata", {}),
        )


class EncryptedStorage:
    """AES-256 encrypted session storage"""

    def __init__(self, path: str, key: bytes):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        if not CRYPTO_AVAILABLE:
            self._fernet = None
            logger.warning("Encryption disabled - cryptography package not installed")
        else:
            self._fernet = self._derive_fernet_key(key)

        self._lock = asyncio.Lock()

    def _derive_fernet_key(self, key: bytes) -> "Fernet":
        if not CRYPTO_AVAILABLE:
            return None

        salt = b"AURA_SESSION_SALT_v1"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        derived_key = b64encode(kdf.derive(key))
        return Fernet(derived_key)

    def _get_file_path(self, key: str) -> Path:
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        return self.path / f"{safe_key}.enc"

    async def save(self, key: str, data: Dict):
        """Save encrypted data"""
        async with self._lock:
            file_path = self._get_file_path(key)
            json_data = json.dumps(data)

            if self._fernet:
                encrypted = self._fernet.encrypt(json_data.encode())
            else:
                encrypted = b64encode(json_data.encode())

            with open(file_path, "wb") as f:
                f.write(encrypted)

    async def load(self, key: str) -> Optional[Dict]:
        """Load and decrypt data"""
        async with self._lock:
            file_path = self._get_file_path(key)

            if not file_path.exists():
                return None

            try:
                with open(file_path, "rb") as f:
                    encrypted = f.read()

                if self._fernet:
                    decrypted = self._fernet.decrypt(encrypted)
                else:
                    decrypted = b64decode(encrypted)

                return json.loads(decrypted.decode())
            except Exception as e:
                logger.error(f"Failed to load session {key}: {e}")
                return None

    async def delete(self, key: str):
        """Delete stored data"""
        async with self._lock:
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()

    async def list_keys(self) -> List[str]:
        """List all stored keys"""
        keys = []
        async with self._lock:
            for file_path in self.path.glob("*.enc"):
                keys.append(file_path.stem)
        return keys

    async def clear_all(self):
        """Clear all stored data"""
        async with self._lock:
            for file_path in self.path.glob("*.enc"):
                file_path.unlink()


class SessionManager:
    """Manages conversation sessions with encryption"""

    def __init__(
        self,
        storage_path: str,
        encryption_key: bytes = None,
        encryption: bool = True,
        max_history: int = 100,
    ):
        self.storage_path = storage_path
        self.encryption_enabled = encryption
        self._max_history = max_history

        if encryption_key is None and encryption:
            encryption_key = self._generate_key()
        elif encryption_key is None:
            encryption_key = b"dummy"  # Placeholder when encryption disabled

        self._storage = EncryptedStorage(storage_path, encryption_key)
        self._sessions: Dict[str, Session] = {}
        self._active_session: Optional[str] = None
        self._session_ttl = timedelta(days=30)

    async def initialize(self):
        """Initialize session manager"""
        # Ensure storage directory exists
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        # Load persisted sessions
        persisted = await self._storage.list_keys()
        for session_id in persisted[:10]:  # Load last 10
            await self.load_session(session_id)

    async def save_all(self):
        """Save all active sessions"""
        for session_id in list(self._sessions.keys()):
            await self.persist_session(session_id)

    def _generate_key(self) -> bytes:
        key_file = Path(self.storage_path) / ".key"

        if key_file.exists():
            with open(key_file, "rb") as f:
                return f.read()

        key = os.urandom(32)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        with open(key_file, "wb") as f:
            f.write(key)

        return key

    def create_session(self) -> str:
        """Create new session, return session_id"""
        session_id = str(uuid.uuid4())
        session = Session(id=session_id)
        self._sessions[session_id] = session
        self._active_session = session_id
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        if session_id in self._sessions:
            return self._sessions[session_id].to_dict()
        return None

    async def load_session(self, session_id: str) -> Optional[Dict]:
        """Load session from storage"""
        data = await self._storage.load(session_id)
        if data:
            session = Session.from_dict(data)
            self._sessions[session_id] = session
            return session.to_dict()
        return None

    def add_message(
        self, session_id: str, role: str, content: str, metadata: Dict = None
    ):
        """Add message to session"""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )

        session = self._sessions[session_id]
        session.messages.append(message)
        session.updated_at = datetime.utcnow().isoformat()

        if len(session.messages) > self._max_history:
            session.messages = session.messages[-self._max_history :]

    def get_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Get conversation history"""
        if session_id not in self._sessions:
            return []

        session = self._sessions[session_id]
        messages = session.messages[-limit:]
        return [m.to_dict() for m in messages]

    def clear_session(self, session_id: str):
        """Clear session data"""
        if session_id in self._sessions:
            self._sessions[session_id].messages.clear()
            self._sessions[session_id].updated_at = datetime.utcnow().isoformat()

    def delete_session(self, session_id: str):
        """Delete session completely"""
        if session_id in self._sessions:
            del self._sessions[session_id]

        if self._active_session == session_id:
            self._active_session = None

    async def persist_session(self, session_id: str):
        """Save session to encrypted storage"""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self._sessions[session_id]
        await self._storage.save(session_id, session.to_dict())

    async def delete_persisted_session(self, session_id: str):
        """Delete session from storage"""
        await self._storage.delete(session_id)

    def list_sessions(self) -> List[str]:
        """List all session IDs in memory"""
        return list(self._sessions.keys())

    async def list_persisted_sessions(self) -> List[str]:
        """List all persisted session IDs"""
        return await self._storage.list_keys()

    def encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive data"""
        if self._storage._fernet:
            return self._storage._fernet.encrypt(data.encode())
        return b64encode(data.encode())

    def decrypt_data(self, encrypted: bytes) -> str:
        """Decrypt data"""
        if self._storage._fernet:
            return self._storage._fernet.decrypt(encrypted).decode()
        return b64decode(encrypted).decode()

    def export_session(self, session_id: str) -> str:
        """Export session as JSON"""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self._sessions[session_id]
        export_data = {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "session": session.to_dict(),
        }
        return json.dumps(export_data, indent=2)

    def import_session(self, json_data: str) -> str:
        """Import session from JSON"""
        data = json.loads(json_data)

        if "session" in data:
            session_data = data["session"]
        else:
            session_data = data

        session_id = session_data.get("id", str(uuid.uuid4()))
        session = Session.from_dict(session_data)

        self._sessions[session_id] = session
        return session_id

    def get_active_session(self) -> Optional[str]:
        """Get active session ID"""
        return self._active_session

    def set_active_session(self, session_id: str):
        """Set active session"""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")
        self._active_session = session_id

    def get_session_metadata(self, session_id: str, key: str = None) -> Any:
        """Get session metadata"""
        if session_id not in self._sessions:
            return None

        metadata = self._sessions[session_id].metadata
        if key:
            return metadata.get(key)
        return metadata

    def set_session_metadata(self, session_id: str, key: str, value: Any):
        """Set session metadata"""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        self._sessions[session_id].metadata[key] = value
        self._sessions[session_id].updated_at = datetime.utcnow().isoformat()

    def search_messages(self, query: str, session_id: str = None) -> List[Dict]:
        """Search messages across sessions"""
        results = []
        query_lower = query.lower()

        sessions_to_search = [session_id] if session_id else list(self._sessions.keys())

        for sid in sessions_to_search:
            if sid not in self._sessions:
                continue

            session = self._sessions[sid]
            for message in session.messages:
                if query_lower in message.content.lower():
                    results.append(
                        {
                            "session_id": sid,
                            "message": message.to_dict(),
                        }
                    )

        return results

    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions from memory"""
        expired = []
        now = datetime.utcnow()

        for session_id, session in self._sessions.items():
            updated = datetime.fromisoformat(session.updated_at)
            if now - updated > self._session_ttl:
                expired.append(session_id)

        for session_id in expired:
            del self._sessions[session_id]

        return len(expired)

    def get_stats(self) -> Dict:
        """Get session statistics"""
        total_messages = 0
        for session in self._sessions.values():
            total_messages += len(session.messages)

        return {
            "total_sessions": len(self._sessions),
            "total_messages": total_messages,
            "active_session": self._active_session,
            "encryption_enabled": self._storage._fernet is not None,
        }


class AsyncSessionManager(SessionManager):
    """Async wrapper for session manager"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = asyncio.Lock()

    async def create_session_async(self) -> str:
        async with self._lock:
            return self.create_session()

    async def add_message_async(
        self, session_id: str, role: str, content: str, metadata: Dict = None
    ):
        async with self._lock:
            self.add_message(session_id, role, content, metadata)

    async def get_history_async(self, session_id: str, limit: int = 20) -> List[Dict]:
        async with self._lock:
            return self.get_history(session_id, limit)

    async def clear_session_async(self, session_id: str):
        async with self._lock:
            self.clear_session(session_id)

    async def export_session_async(self, session_id: str) -> str:
        async with self._lock:
            return self.export_session(session_id)

    async def import_session_async(self, json_data: str) -> str:
        async with self._lock:
            return self.import_session(json_data)
