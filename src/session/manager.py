"""
AURA v3 Session Manager
Comprehensive session management for all interaction types
"""

import asyncio
import os
import base64
import json
import logging
import hashlib
import secrets
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class SessionType(Enum):
    """Types of sessions"""

    INTERACTION = "interaction"  # Regular text interaction
    VOICE = "voice"  # Voice conversation
    TASK = "task"  # Task execution
    LEARNING = "learning"  # Learning session
    PROACTIVE = "proactive"  # Proactive suggestion session
    EMERGENCY = "emergency"  # Emergency handling


class SessionState(Enum):
    """Session states"""

    ACTIVE = "active"
    IDLE = "idle"
    PAUSED = "paused"
    ENDED = "ended"


@dataclass
class SessionConfig:
    """Session configuration"""

    session_type: SessionType = SessionType.INTERACTION
    timeout_minutes: int = 30
    auto_save: bool = True
    max_history: int = 100
    encryption_enabled: bool = True


@dataclass
class InteractionTurn:
    """Single interaction in a session"""

    turn_id: str
    timestamp: datetime
    user_input: str
    aura_response: str
    context: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: int = 0
    tokens_used: int = 0
    success: bool = True
    user_correction: Optional[str] = None


@dataclass
class Session:
    """Complete session data"""

    session_id: str
    session_type: SessionType
    state: SessionState
    created_at: datetime
    last_activity: datetime
    ended_at: Optional[datetime] = None

    # Content
    turns: List[InteractionTurn] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    total_turns: int = 0
    successful_turns: int = 0
    total_tokens: int = 0
    avg_response_time_ms: float = 0.0

    # User info
    user_id: Optional[str] = None
    channel: Optional[str] = None


class SessionManager:
    """
    Comprehensive session manager
    Handles all types of sessions with proper lifecycle management
    """

    def __init__(self, session_dir: str = "data/sessions"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._salt_file = self.session_dir / ".salt"
        self._sessions: Dict[str, Session] = {}
        self._active_session_id: Optional[str] = None

        # Session configuration
        self._config = SessionConfig()

        # Session history for quick access
        self._recent_sessions: deque = deque(maxlen=50)

        self._load_or_generate_salt()
        self._load_recent_sessions()

    def _load_or_generate_salt(self) -> None:
        """Load existing salt or generate a new random one"""
        if self._salt_file.exists():
            try:
                with open(self._salt_file, "rb") as f:
                    self._salt = base64.b64decode(f.read())
                logger.info("Loaded existing session salt")
            except Exception as e:
                logger.warning(f"Failed to load salt, generating new: {e}")
                self._generate_new_salt()
        else:
            self._generate_new_salt()

    def _generate_new_salt(self) -> None:
        """Generate a new random salt for encryption"""
        self._salt = os.urandom(32)
        try:
            with open(self._salt_file, "wb") as f:
                f.write(base64.b64encode(self._salt))
            os.chmod(self._salt_file, 0o600)
            logger.info("Generated new session salt")
        except Exception as e:
            logger.error(f"Failed to save salt: {e}")

    def _load_recent_sessions(self):
        """Load recent sessions metadata"""
        path = self.session_dir / "recent_sessions.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                    for s in data.get("sessions", []):
                        self._recent_sessions.append(s)
            except Exception as e:
                logger.warning(f"Failed to load recent sessions: {e}")

    def _save_recent_sessions(self):
        """Save recent sessions metadata"""
        path = self.session_dir / "recent_sessions.json"
        try:
            with open(path, "w") as f:
                json.dump({"sessions": list(self._recent_sessions)}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save recent sessions: {e}")

    def get_salt(self) -> bytes:
        """Get the current encryption salt"""
        return self._salt

    def create_session(
        self,
        session_type: SessionType = SessionType.INTERACTION,
        user_id: str = None,
        channel: str = None,
        initial_context: Dict = None,
    ) -> str:
        """Create a new session"""
        session_id = self._generate_session_id()

        session = Session(
            session_id=session_id,
            session_type=session_type,
            state=SessionState.ACTIVE,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            user_id=user_id,
            channel=channel,
            context=initial_context or {},
        )

        self._sessions[session_id] = session
        self._active_session_id = session_id

        # Update recent sessions
        self._recent_sessions.append(
            {
                "session_id": session_id,
                "session_type": session_type.value,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
            }
        )
        self._save_recent_sessions()

        # Save to disk
        self._save_session(session)

        logger.info(f"Created session: {session_id} ({session_type.value})")
        return session_id

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random = secrets.token_hex(4)
        return f"sess_{timestamp}_{random}"

    def get_session(self, session_id: str = None) -> Optional[Session]:
        """Get session by ID"""
        if session_id is None:
            session_id = self._active_session_id

        if session_id not in self._sessions:
            if not self._load_session(session_id):
                return None

        if session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_activity = datetime.now()
            return session

        return None

    def add_turn(
        self,
        user_input: str,
        aura_response: str,
        context: Dict = None,
        processing_time_ms: int = 0,
        tokens_used: int = 0,
        success: bool = True,
        user_correction: str = None,
    ) -> Optional[str]:
        """Add an interaction turn to the current session"""
        session = self.get_session()
        if not session:
            logger.warning("No active session to add turn")
            return None

        turn = InteractionTurn(
            turn_id=secrets.token_hex(8),
            timestamp=datetime.now(),
            user_input=user_input,
            aura_response=aura_response,
            context=context or {},
            processing_time_ms=processing_time_ms,
            tokens_used=tokens_used,
            success=success,
            user_correction=user_correction,
        )

        session.turns.append(turn)
        session.total_turns += 1
        session.total_tokens += tokens_used

        if success:
            session.successful_turns += 1

        # Update average response time
        if session.total_turns > 1:
            session.avg_response_time_ms = (
                session.avg_response_time_ms * (session.total_turns - 1)
                + processing_time_ms
            ) / session.total_turns
        else:
            session.avg_response_time_ms = processing_time_ms

        session.last_activity = datetime.now()

        # Trim history if needed
        if len(session.turns) > self._config.max_history:
            session.turns = session.turns[-self._config.max_history :]

        # Auto-save
        if self._config.auto_save:
            self._save_session(session)

        return turn.turn_id

    def update_context(self, context: Dict):
        """Update session context"""
        session = self.get_session()
        if session:
            session.context.update(context)
            session.last_activity = datetime.now()
            if self._config.auto_save:
                self._save_session(session)

    def pause_session(self):
        """Pause current session"""
        session = self.get_session()
        if session:
            session.state = SessionState.IDLE
            self._save_session(session)

    def resume_session(self):
        """Resume paused session"""
        session = self.get_session()
        if session and session.state == SessionState.IDLE:
            session.state = SessionState.ACTIVE
            session.last_activity = datetime.now()
            self._save_session(session)

    def end_session(self) -> Optional[Dict]:
        """End current session and return summary"""
        session = self.get_session()
        if not session:
            return None

        session.state = SessionState.ENDED
        session.ended_at = datetime.now()

        # Calculate session summary
        summary = {
            "session_id": session.session_id,
            "duration_minutes": (session.ended_at - session.created_at).total_seconds()
            / 60,
            "total_turns": session.total_turns,
            "successful_turns": session.successful_turns,
            "success_rate": session.successful_turns / session.total_turns
            if session.total_turns > 0
            else 0,
            "total_tokens": session.total_tokens,
            "avg_response_time_ms": session.avg_response_time_ms,
        }

        self._save_session(session)

        # Clear active session
        self._active_session_id = None

        logger.info(f"Ended session {session.session_id}: {summary}")
        return summary

    def set_active_session(self, session_id: str) -> bool:
        """Set active session"""
        if session_id in self._sessions or self._load_session(session_id):
            self._active_session_id = session_id
            return True
        return False

    def _save_session(self, session: Session) -> bool:
        """Save session to disk"""
        try:
            session_file = self.session_dir / f"{session.session_id}.json"

            data = {
                "session_id": session.session_id,
                "session_type": session.session_type.value,
                "state": session.state.value,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                "context": session.context,
                "metadata": session.metadata,
                "total_turns": session.total_turns,
                "successful_turns": session.successful_turns,
                "total_tokens": session.total_tokens,
                "avg_response_time_ms": session.avg_response_time_ms,
                "user_id": session.user_id,
                "channel": session.channel,
                "turns": [
                    {
                        "turn_id": t.turn_id,
                        "timestamp": t.timestamp.isoformat(),
                        "user_input": t.user_input,
                        "aura_response": t.aura_response,
                        "context": t.context,
                        "processing_time_ms": t.processing_time_ms,
                        "tokens_used": t.tokens_used,
                        "success": t.success,
                        "user_correction": t.user_correction,
                    }
                    for t in session.turns
                ],
            }

            with open(session_file, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    def _load_session(self, session_id: str) -> bool:
        """Load session from disk"""
        try:
            session_file = self.session_dir / f"{session_id}.json"
            if not session_file.exists():
                return False

            with open(session_file, "r") as f:
                data = json.load(f)

            session = Session(
                session_id=data["session_id"],
                session_type=SessionType(data["session_type"]),
                state=SessionState(data["state"]),
                created_at=datetime.fromisoformat(data["created_at"]),
                last_activity=datetime.fromisoformat(data["last_activity"]),
                ended_at=datetime.fromisoformat(data["ended_at"])
                if data.get("ended_at")
                else None,
                context=data.get("context", {}),
                metadata=data.get("metadata", {}),
                total_turns=data.get("total_turns", 0),
                successful_turns=data.get("successful_turns", 0),
                total_tokens=data.get("total_tokens", 0),
                avg_response_time_ms=data.get("avg_response_time_ms", 0.0),
                user_id=data.get("user_id"),
                channel=data.get("channel"),
            )

            # Load turns
            for t_data in data.get("turns", []):
                turn = InteractionTurn(
                    turn_id=t_data["turn_id"],
                    timestamp=datetime.fromisoformat(t_data["timestamp"]),
                    user_input=t_data["user_input"],
                    aura_response=t_data["aura_response"],
                    context=t_data.get("context", {}),
                    processing_time_ms=t_data.get("processing_time_ms", 0),
                    tokens_used=t_data.get("tokens_used", 0),
                    success=t_data.get("success", True),
                    user_correction=t_data.get("user_correction"),
                )
                session.turns.append(turn)

            self._sessions[session_id] = session
            return True

        except Exception as e:
            logger.warning(f"Failed to load session {session_id}: {e}")
            return False

    def cleanup_expired_sessions(self, max_age_days: int = 30) -> int:
        """Remove sessions older than max_age_days"""
        removed = 0
        try:
            for session_file in self.session_dir.glob("sess_*.json"):
                try:
                    mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                    if datetime.now() - mtime > timedelta(days=max_age_days):
                        session_file.unlink()
                        removed += 1
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        return removed

    def get_session_history(
        self, session_type: SessionType = None, user_id: str = None, limit: int = 10
    ) -> List[Dict]:
        """Get session history"""
        sessions = []

        for session_id in list(self._recent_sessions):
            # Load session metadata
            if session_id not in self._sessions:
                if not self._load_session(session_id):
                    continue

            session = self._sessions.get(session_id)
            if not session:
                continue

            # Filter
            if session_type and session.session_type != session_type:
                continue
            if user_id and session.user_id != user_id:
                continue

            sessions.append(
                {
                    "session_id": session.session_id,
                    "session_type": session.session_type.value,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "total_turns": session.total_turns,
                    "successful_turns": session.successful_turns,
                    "state": session.state.value,
                }
            )

            if len(sessions) >= limit:
                break

        return sessions

    def get_active_session_id(self) -> Optional[str]:
        """Get current active session ID"""
        return self._active_session_id

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            "total_sessions": len(self._recent_sessions),
            "active_session": self._active_session_id,
            "loaded_sessions": len(self._sessions),
        }


# Global instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create session manager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
