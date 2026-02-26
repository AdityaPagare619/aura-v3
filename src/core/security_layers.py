"""
AURA v3 Security Layers
Default encryption, authentication, and session management

Security by default - all user data encrypted unless explicitly disabled
"""

import asyncio
import hashlib
import hmac
import os
import json
import logging
import secrets
import base64
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    NONE = "none"
    PIN = "pin"
    BIOMETRIC = "biometric"
    PASSWORD = "password"


@dataclass
class SecurityConfig:
    encrypt_by_default: bool = True
    require_auth: bool = False
    session_timeout_minutes: int = 30
    auto_lock_on_background: bool = True
    pin_length: int = 4
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    encryption_algorithm: str = "fernet"
    key_derivation_iterations: int = 100000


class EncryptionManager:
    """
    Manages encryption/decryption of user data
    Default: encryption enabled for all stored data
    """

    def __init__(
        self, storage_path: str = "data/security", config: SecurityConfig = None
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.config = config or SecurityConfig()
        self._fernet: Optional[Fernet] = None
        self._master_key: Optional[bytes] = None
        self._key_file = self.storage_path / "master.key"
        self._lock = Lock()

        if CRYPTO_AVAILABLE:
            self._initialize_key()

    def _initialize_key(self):
        """Initialize or load master encryption key"""
        if self._key_file.exists():
            try:
                with open(self._key_file, "rb") as f:
                    key_data = json.load(f)
                    self._master_key = base64.b64decode(key_data["key"])
                    salt = base64.b64decode(key_data["salt"])
                    self._fernet = Fernet(self._master_key)
                    logger.info("Encryption key loaded")
            except Exception as e:
                logger.warning(f"Failed to load encryption key: {e}")
                self._generate_new_key()
        else:
            self._generate_new_key()

    def _generate_new_key(self):
        """Generate new master encryption key"""
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography not available - encryption disabled")
            return

        salt = secrets.token_bytes(32)
        self._master_key = Fernet.generate_key()

        key_data = {
            "key": base64.b64encode(self._master_key).decode(),
            "salt": base64.b64encode(salt).decode(),
            "created": datetime.now().isoformat(),
        }

        try:
            with open(self._key_file, "w") as f:
                json.dump(key_data, f)
            os.chmod(self._key_file, 0o600)
            self._fernet = Fernet(self._master_key)
            logger.info("New encryption key generated")
        except Exception as e:
            logger.error(f"Failed to save encryption key: {e}")

    def is_encryption_available(self) -> bool:
        return CRYPTO_AVAILABLE and self._fernet is not None

    def encrypt(self, data: str) -> Optional[str]:
        """Encrypt string data"""
        if not self.is_encryption_available():
            return data

        if not data:
            return data

        try:
            with self._lock:
                encrypted = self._fernet.encrypt(data.encode())
                return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return None

    def decrypt(self, encrypted_data: str) -> Optional[str]:
        """Decrypt string data"""
        if not self.is_encryption_available():
            return encrypted_data

        if not encrypted_data:
            return encrypted_data

        try:
            with self._lock:
                decoded = base64.b64decode(encrypted_data.encode())
                decrypted = self._fernet.decrypt(decoded)
                return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None

    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> bool:
        """Encrypt a file"""
        if not self.is_encryption_available():
            return False

        try:
            with open(file_path, "rb") as f:
                data = f.read()

            encrypted = self._fernet.encrypt(data)

            output = output_path or file_path + ".enc"
            with open(output, "wb") as f:
                f.write(encrypted)

            return True
        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            return False

    def decrypt_file(self, file_path: str, output_path: Optional[str] = None) -> bool:
        """Decrypt a file"""
        if not self.is_encryption_available():
            return False

        try:
            with open(file_path, "rb") as f:
                encrypted = f.read()

            decrypted = self._fernet.decrypt(encrypted)

            output = output_path or file_path.replace(".enc", "")
            with open(output, "wb") as f:
                f.write(decrypted)

            return True
        except Exception as e:
            logger.error(f"File decryption failed: {e}")
            return False

    def derive_key_from_pin(self, pin: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from PIN"""
        if salt is None:
            salt = secrets.token_bytes(32)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.config.key_derivation_iterations,
            backend=default_backend(),
        )
        return base64.urlsafe_b64encode(kdf.derive(pin.encode()))


class SecureStorage:
    """
    Secure encrypted JSON storage for sensitive data.
    Automatically encrypts data before saving and decrypts on load.

    Usage:
        storage = SecureStorage('data/healthcare')
        storage.save('profile.json', {'name': 'John', 'age': 30})
        data = storage.load('profile.json')  # Returns decrypted data
    """

    _instance = None
    _encryption_manager = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, storage_path: str = "data/security", encrypt_by_default: bool = True
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.encrypt_by_default = encrypt_by_default
        self._initialized = True

        # Use singleton encryption manager
        if SecureStorage._encryption_manager is None:
            SecureStorage._encryption_manager = EncryptionManager(storage_path)
        self._encryption = SecureStorage._encryption_manager

    def _is_encrypted_content(self, content: bytes) -> bool:
        """Check if content appears to be encrypted (base64 encoded Fernet)"""
        try:
            decoded = base64.b64decode(content)
            return len(decoded) > 1 and decoded[0] == 0x80
        except Exception:
            return False

    def save(self, filename: str, data: Dict[str, Any], encrypt: bool = None) -> bool:
        """Save data to file with optional encryption."""
        if encrypt is None:
            encrypt = self.encrypt_by_default

        file_path = self.storage_path / filename

        try:
            json_data = json.dumps(data, indent=2, default=str)

            if encrypt and self._encryption.is_encryption_available():
                encrypted = self._encryption.encrypt(json_data)
                if encrypted is None:
                    logger.error(f"Encryption failed for {filename}")
                    return False
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("ENC:" + encrypted)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(json_data)

            logger.info(f"Data saved to {filename} (encrypted={encrypt})")
            return True
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
            return False

    def load(self, filename: str, encrypt: bool = None) -> Optional[Dict[str, Any]]:
        """Load data from file with automatic decryption detection."""
        file_path = self.storage_path / filename

        if not file_path.exists():
            logger.warning(f"File not found: {filename}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if encrypt is None:
                encrypt = content.startswith("ENC:")

            if encrypt and content.startswith("ENC:"):
                encrypted_data = content[4:]
                decrypted = self._encryption.decrypt(encrypted_data)
                if decrypted is None:
                    logger.error(f"Decryption failed for {filename}")
                    return None
                return json.loads(decrypted)
            else:
                return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return None

    def save_raw(self, filename: str, content: str, encrypt: bool = True) -> bool:
        """Save raw string content with encryption"""
        file_path = self.storage_path / filename

        try:
            if encrypt and self._encryption.is_encryption_available():
                encrypted = self._encryption.encrypt(content)
                if encrypted:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write("ENC:" + encrypted)
                else:
                    return False
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to save raw {filename}: {e}")
            return False

    def load_raw(self, filename: str) -> Optional[str]:
        """Load raw string content with automatic decryption"""
        file_path = self.storage_path / filename

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if content.startswith("ENC:"):
                return self._encryption.decrypt(content[4:])
            return content
        except Exception as e:
            logger.error(f"Failed to load raw {filename}: {e}")
            return None

    def exists(self, filename: str) -> bool:
        """Check if file exists"""
        return (self.storage_path / filename).exists()

    def delete(self, filename: str) -> bool:
        """Delete a file"""
        try:
            file_path = self.storage_path / filename
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete {filename}: {e}")
            return False

    def list_files(self, pattern: str = "*.json") -> List[str]:
        """List files matching pattern"""
        try:
            return [f.name for f in self.storage_path.glob(pattern)]
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []

    def is_encryption_available(self) -> bool:
        """Check if encryption is available"""
        return self._encryption.is_encryption_available()


class AuthManager:
    """
    Manages PIN/biometric authentication
    Optional: require PIN to unlock AURA
    """

    def __init__(
        self, storage_path: str = "data/security", config: SecurityConfig = None
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.config = config or SecurityConfig()
        self._auth_file = self.storage_path / "auth.json"
        self._pin_hash: Optional[str] = None
        self._salt: Optional[str] = None
        self._auth_method: Optional[str] = None
        self._manually_locked: bool = False
        self._failed_attempts: int = 0
        self._locked_until: Optional[datetime] = None
        self._encryption_key: Optional[bytes] = None

        self._lock = Lock()
        self._load_auth()

    def _load_auth(self):
        """Load authentication configuration"""
        if self._auth_file.exists():
            try:
                with open(self._auth_file, "r") as f:
                    data = json.load(f)
                    self._pin_hash = data.get("pin_hash")
                    self._salt = data.get("salt")
                    self._auth_method = data.get("method")
            except Exception as e:
                logger.warning(f"Failed to load auth config: {e}")

    def _save_auth(self):
        """Save authentication configuration"""
        try:
            data = {
                "pin_hash": self._pin_hash,
                "salt": self._salt,
                "method": self._auth_method,
                "updated": datetime.now().isoformat(),
            }
            with open(self._auth_file, "w") as f:
                json.dump(data, f, indent=2)
            os.chmod(self._auth_file, 0o600)
        except Exception as e:
            logger.error(f"Failed to save auth config: {e}")

    def _hash_pin(self, pin: str) -> bytes:
        """Hash PIN with salt"""
        salt = base64.b64decode(self._salt) if self._salt else secrets.token_bytes(32)
        return hashlib.pbkdf2_hmac(
            "sha256", pin.encode(), salt, self.config.key_derivation_iterations
        )

    def has_pin_setup(self) -> bool:
        """Check if PIN is configured"""
        return self._pin_hash is not None

    def get_auth_method(self) -> Optional[str]:
        """Get configured authentication method"""
        return self._auth_method

    async def setup_pin(self, pin: str) -> bool:
        """Set up PIN authentication"""
        if len(pin) < self.config.pin_length or len(pin) > 6:
            logger.warning(f"PIN must be {self.config.pin_length}-6 digits")
            return False

        try:
            salt = secrets.token_bytes(32)
            pin_hash = self._hash_pin_with_salt(pin, salt)

            self._salt = base64.b64encode(salt).decode()
            self._pin_hash = base64.b64encode(pin_hash).decode()
            self._auth_method = "pin"
            self._save_auth()

            logger.info("PIN setup complete")
            return True
        except Exception as e:
            logger.error(f"PIN setup failed: {e}")
            return False

    def _hash_pin_with_salt(self, pin: str, salt: bytes) -> bytes:
        """Hash PIN with provided salt"""
        return hashlib.pbkdf2_hmac(
            "sha256", pin.encode(), salt, self.config.key_derivation_iterations
        )

    async def verify_pin(self, pin: str) -> bool:
        """Verify PIN - returns True if correct"""
        if self._is_locked():
            return False

        if not self.has_pin_setup():
            return False

        try:
            salt = base64.b64decode(self._salt)
            stored_hash = base64.b64decode(self._pin_hash)
            input_hash = self._hash_pin_with_salt(pin, salt)

            if hmac.compare_digest(stored_hash, input_hash):
                self._failed_attempts = 0
                self._encryption_key = EncryptionManager().derive_key_from_pin(
                    pin, salt
                )
                logger.info("PIN verification successful")
                return True
            else:
                with self._lock:
                    self._failed_attempts += 1
                    if self._failed_attempts >= self.config.max_failed_attempts:
                        self._locked_until = datetime.now() + timedelta(
                            minutes=self.config.lockout_duration_minutes
                        )
                        logger.warning(
                            f"Too many failed attempts - locked until {self._locked_until}"
                        )
                    return False
        except Exception as e:
            logger.error(f"PIN verification error: {e}")
            return False

    def _is_locked(self) -> bool:
        """Check if account is locked due to failed attempts"""
        if self._locked_until and datetime.now() < self._locked_until:
            return True
        elif self._locked_until:
            self._locked_until = None
            self._failed_attempts = 0
        return False

    def is_locked(self) -> bool:
        """Check if currently locked"""
        return self._is_locked()

    def get_lock_time_remaining(self) -> Optional[int]:
        """Get remaining lock time in seconds"""
        if self._locked_until and datetime.now() < self._locked_until:
            delta = self._locked_until - datetime.now()
            return int(delta.total_seconds())
        return None

    def change_pin(self, old_pin: str, new_pin: str) -> bool:
        """Change PIN - requires current PIN verification"""
        if not asyncio.run(self.verify_pin(old_pin)):
            return False

        if len(new_pin) < self.config.pin_length or len(new_pin) > 6:
            return False

        asyncio.run(self.setup_pin(new_pin))
        return True

    def remove_pin(self, pin: str) -> bool:
        """Remove PIN - requires current PIN verification"""
        if not asyncio.run(self.verify_pin(pin)):
            return False

        self._pin_hash = None
        self._salt = None
        self._auth_method = None
        self._save_auth()
        logger.info("PIN removed")
        return True

    def get_encryption_key(self) -> Optional[bytes]:
        """Get derived encryption key after successful auth"""
        return self._encryption_key

    def lock(self):
        """Manually lock AURA"""
        self._manually_locked = True
        logger.info("AURA locked")

    def unlock(self):
        """Manually unlock AURA"""
        self._manually_locked = False
        logger.info("AURA unlocked")

    def is_manual_lock(self) -> bool:
        """Check if manually locked"""
        return self._manually_locked


class SessionManager:
    """
    Manages secure sessions with timeout
    Auto-lock on background/timeout
    """

    def __init__(
        self, storage_path: str = "data/security", config: SecurityConfig = None
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.config = config or SecurityConfig()
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._session_file = self.storage_path / "sessions.json"
        self._lock = Lock()

        self._load_sessions()

    def _load_sessions(self):
        """Load sessions from disk"""
        if self._session_file.exists():
            try:
                with open(self._session_file, "r") as f:
                    data = json.load(f)
                    self._cleanup_expired_sessions(data)
            except Exception as e:
                logger.warning(f"Failed to load sessions: {e}")

    def _save_sessions(self):
        """Save sessions to disk"""
        try:
            with open(self._session_file, "w") as f:
                json.dump(self._sessions, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")

    def _cleanup_expired_sessions(self, data: Dict):
        """Remove expired sessions"""
        now = datetime.now()
        expired = []

        for session_id, info in data.items():
            if "expires_at" in info:
                expires = datetime.fromisoformat(info["expires_at"])
                if expires < now:
                    expired.append(session_id)

        for sid in expired:
            del data[sid]

        self._sessions = data

    def create_session(self, user_id: str, session_data: Optional[Dict] = None) -> str:
        """Create new session"""
        session_id = secrets.token_urlsafe(32)

        with self._lock:
            self._sessions[session_id] = {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "expires_at": (
                    datetime.now()
                    + timedelta(minutes=self.config.session_timeout_minutes)
                ).isoformat(),
                "last_activity": datetime.now().isoformat(),
                "data": session_data or {},
                "active": True,
            }
            self._save_sessions()

        logger.info(f"Session created for user {user_id}")
        return session_id

    def validate_session(self, session_id: str) -> bool:
        """Validate session is active and not expired"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            if not session.get("active", False):
                return False

            expires = datetime.fromisoformat(session["expires_at"])
            if datetime.now() > expires:
                self._end_session(session_id)
                return False

            session["last_activity"] = datetime.now().isoformat()
            session["expires_at"] = (
                datetime.now() + timedelta(minutes=self.config.session_timeout_minutes)
            ).isoformat()
            self._save_sessions()

            return True

    def _end_session(self, session_id: str):
        """End a session"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._save_sessions()

    def end_session(self, session_id: str):
        """End session (public)"""
        self._end_session(session_id)
        logger.info(f"Session {session_id} ended")

    def end_all_user_sessions(self, user_id: str):
        """End all sessions for a user"""
        with self._lock:
            to_remove = [
                sid
                for sid, info in self._sessions.items()
                if info.get("user_id") == user_id
            ]
            for sid in to_remove:
                del self._sessions[sid]
            self._save_sessions()

        if to_remove:
            logger.info(f"Ended {len(to_remove)} sessions for user {user_id}")

    def refresh_session(self, session_id: str) -> bool:
        """Refresh session timeout"""
        return self.validate_session(session_id)

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        return self._sessions.get(session_id)

    def get_active_sessions(self, user_id: Optional[str] = None) -> List[Dict]:
        """Get all active sessions"""
        sessions = []
        now = datetime.now()

        with self._lock:
            for sid, info in self._sessions.items():
                if not info.get("active", False):
                    continue

                expires = datetime.fromisoformat(info["expires_at"])
                if now > expires:
                    continue

                if user_id and info.get("user_id") != user_id:
                    continue

                sessions.append({"session_id": sid, **info})

        return sessions

    def set_timeout(self, minutes: int):
        """Update session timeout"""
        self.config.session_timeout_minutes = minutes

    def auto_lock_check(self):
        """Check and expire sessions based on timeout"""
        with self._lock:
            now = datetime.now()
            expired = []

            for session_id, info in self._sessions.items():
                expires = datetime.fromisoformat(info["expires_at"])
                if now > expires:
                    expired.append(session_id)

            for sid in expired:
                del self._sessions[sid]

            if expired:
                self._save_sessions()


class SecurityLayers:
    """
    Unified security interface combining all security managers
    """

    def __init__(self, config_path: str = "config/security.yaml"):
        self.config = self._load_config(config_path)

        storage_path = self.config.get("storage_path", "data/security")

        self.encryption = EncryptionManager(storage_path, self._get_security_config())
        self.auth = AuthManager(storage_path, self._get_security_config())
        self.session = SessionManager(storage_path, self._get_security_config())

    def _load_config(self, config_path: str) -> Dict:
        """Load security configuration"""
        if os.path.exists(config_path):
            try:
                import yaml

                with open(config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load security config: {e}")
        return {}

    def _get_security_config(self) -> SecurityConfig:
        """Get SecurityConfig from loaded config"""
        sec = self.config.get("security", {})
        return SecurityConfig(
            encrypt_by_default=sec.get("encrypt_by_default", True),
            require_auth=sec.get("require_auth", False),
            session_timeout_minutes=sec.get("session_timeout_minutes", 30),
            auto_lock_on_background=sec.get("auto_lock_on_background", True),
            max_failed_attempts=sec.get("max_failed_attempts", 5),
            lockout_duration_minutes=sec.get("lockout_duration_minutes", 15),
        )

    def is_encryption_enabled(self) -> bool:
        """Check if encryption is enabled"""
        return self.encryption.is_encryption_available()

    def requires_authentication(self) -> bool:
        """Check if authentication is required"""
        return self.config.get("security", {}).get("require_auth", False)

    def is_locked(self) -> bool:
        """Check if AURA is locked"""
        return self.auth.is_locked() or self.auth.is_manual_lock()

    def lock(self):
        """Lock AURA"""
        self.auth.lock()
        self.session.end_all_user_sessions("default")

    def unlock(self, pin: str) -> bool:
        """Unlock AURA with PIN"""
        return asyncio.run(self.auth.verify_pin(pin))


_security_layers: Optional[SecurityLayers] = None


def get_security_layers(config_path: str = "config/security.yaml") -> SecurityLayers:
    """Get or create security layers instance"""
    global _security_layers
    if _security_layers is None:
        _security_layers = SecurityLayers(config_path)
    return _security_layers


__all__ = [
    "SecurityLayers",
    "EncryptionManager",
    "SecureStorage",
    "AuthManager",
    "SessionManager",
    "SecurityConfig",
    "SecurityLevel",
    "get_security_layers",
]
