"""
AURA v3 Security Package
Handles encryption, authentication, and privacy controls
100% offline - all security operations local

CRITICAL: This is AURA's security layer - must be production quality
"""

import asyncio
import hashlib
import hmac
import os
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import secrets
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations"""

    NONE = 0
    PIN = 1
    BIOMETRIC = 2
    PASSWORD = 3


@dataclass
class SecurityPolicy:
    """Security policy configuration"""

    require_auth_for_start: bool = False
    require_auth_for_sensitive: bool = True
    auto_lock_minutes: int = 5
    max_failed_attempts: int = 3
    lockout_duration_minutes: int = 15
    encrypt_storage: bool = True
    allow_rooted: bool = True  # Allow on rooted devices with warning


@dataclass
class AuthAttempt:
    """Authentication attempt record"""

    timestamp: datetime
    method: str  # "pin", "biometric", "password"
    success: bool
    device_id: Optional[str] = None


class LocalAuthenticator:
    """
    Local authentication system
    Supports PIN, biometric (via Android), and password
    """

    def __init__(self, storage_path: str = "data/security"):
        self.storage_path = storage_path
        self._policy = SecurityPolicy()
        self._failed_attempts: List[AuthAttempt] = []
        self._locked_until: Optional[datetime] = None
        self._authenticated_sessions: Dict[str, datetime] = {}

        # Encryption key (derived from user password/pin)
        self._encryption_key: Optional[bytes] = None
        self._fernet: Optional[Fernet] = None

        os.makedirs(storage_path, exist_ok=True)
        self._load_config()

    def _load_config(self):
        """Load security configuration"""
        config_path = os.path.join(self.storage_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    data = json.load(f)
                    self._policy = SecurityPolicy(**data.get("policy", {}))
            except Exception as e:
                logger.warning(f"Failed to load security config: {e}")

    def _save_config(self):
        """Save security configuration"""
        config_path = os.path.join(self.storage_path, "config.json")
        try:
            with open(config_path, "w") as f:
                json.dump(
                    {
                        "policy": {
                            "require_auth_for_start": self._policy.require_auth_for_start,
                            "require_auth_for_sensitive": self._policy.require_auth_for_sensitive,
                            "auto_lock_minutes": self._policy.auto_lock_minutes,
                            "max_failed_attempts": self._policy.max_failed_attempts,
                            "lockout_duration_minutes": self._policy.lockout_duration_minutes,
                            "encrypt_storage": self._policy.encrypt_storage,
                            "allow_rooted": self._policy.allow_rooted,
                        }
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save security config: {e}")

    def set_policy(self, policy: SecurityPolicy):
        """Update security policy"""
        self._policy = policy
        self._save_config()

    async def setup_pin(self, pin: str) -> bool:
        """Set up PIN authentication"""
        try:
            # Hash PIN with salt
            salt = secrets.token_bytes(32)
            pin_hash = self._hash_input(pin, salt)

            # Store hashed PIN
            auth_path = os.path.join(self.storage_path, "auth.json")
            with open(auth_path, "w") as f:
                json.dump(
                    {
                        "pin_hash": base64.b64encode(pin_hash).decode(),
                        "salt": base64.b64encode(salt).decode(),
                        "method": "pin",
                    },
                    f,
                )

            # Derive encryption key from PIN
            self._encryption_key = self._derive_key(pin, salt)
            self._fernet = Fernet(self._encryption_key)

            logger.info("PIN setup complete")
            return True

        except Exception as e:
            logger.error(f"PIN setup failed: {e}")
            return False

    async def verify_pin(self, pin: str) -> bool:
        """Verify PIN"""
        if self._is_locked():
            return False

        auth_path = os.path.join(self.storage_path, "auth.json")
        if not os.path.exists(auth_path):
            return False  # No PIN set

        try:
            with open(auth_path) as f:
                auth_data = json.load(f)

            salt = base64.b64decode(auth_data["salt"])
            stored_hash = base64.b64decode(auth_data["pin_hash"])
            input_hash = self._hash_input(pin, salt)

            if hmac.compare_digest(stored_hash, input_hash):
                # Success - derive encryption key
                self._encryption_key = self._derive_key(pin, salt)
                self._fernet = Fernet(self._encryption_key)

                self._failed_attempts = []  # Reset failed attempts
                logger.info("PIN verification successful")
                return True
            else:
                self._record_failed_attempt("pin")
                return False

        except Exception as e:
            logger.error(f"PIN verification error: {e}")
            return False

    async def setup_password(self, password: str) -> bool:
        """Set up password authentication (more secure than PIN)"""
        try:
            salt = secrets.token_bytes(32)
            password_hash = self._hash_input(password, salt)

            auth_path = os.path.join(self.storage_path, "auth.json")
            with open(auth_path, "w") as f:
                json.dump(
                    {
                        "password_hash": base64.b64encode(password_hash).decode(),
                        "salt": base64.b64decode(salt).decode(),
                        "method": "password",
                    },
                    f,
                )

            # Derive encryption key
            self._encryption_key = self._derive_key(password, salt)
            self._fernet = Fernet(self._encryption_key)

            logger.info("Password setup complete")
            return True

        except Exception as e:
            logger.error(f"Password setup failed: {e}")
            return False

    async def verify_password(self, password: str) -> bool:
        """Verify password"""
        if self._is_locked():
            return False

        auth_path = os.path.join(self.storage_path, "auth.json")
        if not os.path.exists(auth_path):
            return False

        try:
            with open(auth_path) as f:
                auth_data = json.load(f)

            salt = base64.b64decode(auth_data["salt"])
            stored_hash = base64.b64decode(auth_data["password_hash"])
            input_hash = self._hash_input(password, salt)

            if hmac.compare_digest(stored_hash, input_hash):
                self._encryption_key = self._derive_key(password, salt)
                self._fernet = Fernet(self._encryption_key)
                self._failed_attempts = []
                return True
            else:
                self._record_failed_attempt("password")
                return False

        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    def _hash_input(self, input_str: str, salt: bytes) -> bytes:
        """Hash input with salt"""
        return hashlib.pbkdf2_hmac(
            "sha256",
            input_str.encode(),
            salt,
            100000,  # High iteration count for security
        )

    def _derive_key(self, input_str: str, salt: bytes) -> bytes:
        """Derive encryption key from input"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        return base64.urlsafe_b64encode(kdf.derive(input_str.encode()))

    def _record_failed_attempt(self, method: str):
        """Record failed authentication attempt"""
        self._failed_attempts.append(
            AuthAttempt(timestamp=datetime.now(), method=method, success=False)
        )

        # Check if should lock
        recent_failures = [
            a
            for a in self._failed_attempts
            if (datetime.now() - a.timestamp).total_seconds() < 300  # Last 5 minutes
        ]

        if len(recent_failures) >= self._policy.max_failed_attempts:
            self._locked_until = datetime.now() + timedelta(
                minutes=self._policy.lockout_duration_minutes
            )
            logger.warning(f"Account locked until {self._locked_until}")

    def _is_locked(self) -> bool:
        """Check if account is locked"""
        if self._locked_until and datetime.now() < self._locked_until:
            return True
        elif self._locked_until:
            self._locked_until = None  # Lock expired
        return False

    def is_authenticated(self, session_id: str = "default") -> bool:
        """Check if session is authenticated"""
        if session_id not in self._authenticated_sessions:
            return False

        session_time = self._authenticated_sessions[session_id]
        elapsed = (datetime.now() - session_time).total_seconds() / 60

        if elapsed > self._policy.auto_lock_minutes:
            del self._authenticated_sessions[session_id]
            return False

        return True

    def create_session(self, session_id: str = "default"):
        """Create authenticated session"""
        self._authenticated_sessions[session_id] = datetime.now()

    def end_session(self, session_id: str = "default"):
        """End authenticated session"""
        if session_id in self._authenticated_sessions:
            del self._authenticated_sessions[session_id]

    def encrypt(self, data: str) -> Optional[str]:
        """Encrypt data"""
        if not self._fernet:
            logger.warning("No encryption key, cannot encrypt")
            return None
        try:
            return self._fernet.encrypt(data.encode()).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return None

    def decrypt(self, encrypted_data: str) -> Optional[str]:
        """Decrypt data"""
        if not self._fernet:
            logger.warning("No encryption key, cannot decrypt")
            return None
        try:
            return self._fernet.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None

    def has_auth_setup(self) -> bool:
        """Check if authentication is set up"""
        auth_path = os.path.join(self.storage_path, "auth.json")
        return os.path.exists(auth_path)

    def get_auth_method(self) -> Optional[str]:
        """Get configured auth method"""
        auth_path = os.path.join(self.storage_path, "auth.json")
        if not os.path.exists(auth_path):
            return None
        try:
            with open(auth_path) as f:
                return json.load(f).get("method")
        except:
            return None


class PrivacyManager:
    """
    Manages data privacy and consent
    """

    def __init__(self, storage_path: str = "data/security"):
        self.storage_path = storage_path
        self._consents: Dict[str, bool] = {}
        self._load_consents()

    def _load_consents(self):
        """Load consent settings"""
        path = os.path.join(self.storage_path, "consents.json")
        if os.path.exists(path):
            with open(path) as f:
                self._consents = json.load(f)

    def _save_consents(self):
        """Save consent settings"""
        path = os.path.join(self.storage_path, "consents.json")
        with open(path, "w") as f:
            json.dump(self._consents, f, indent=2)

    def set_consent(self, category: str, granted: bool):
        """Set consent for a category"""
        self._consents[category] = granted
        self._save_consents()

    def has_consent(self, category: str) -> bool:
        """Check if consent is granted"""
        return self._consents.get(category, False)

    def get_all_consents(self) -> Dict[str, bool]:
        """Get all consent settings"""
        return self._consents.copy()

    # Common consent categories
    CONSENT_CATEGORIES = {
        "location": "Access to location data",
        "contacts": "Access to contacts",
        "calendar": "Access to calendar",
        "sensors": "Access to device sensors",
        "voice": "Voice recording and processing",
        "learning": "Learn from user behavior",
        "analytics": "Anonymous usage analytics",
        "cloud_backup": "Cloud backup of data",
    }


class PermissionManager:
    """
    Manages runtime permissions
    """

    def __init__(self):
        self._permissions: Dict[str, bool] = {}

    async def check_permission(self, permission: str) -> bool:
        """Check if permission is granted"""
        # In production, would check actual Android permissions
        return self._permissions.get(permission, False)

    async def request_permission(self, permission: str) -> bool:
        """Request permission (would trigger Android dialog)"""
        # Would use termux-permission or Android API
        logger.info(f"Requesting permission: {permission}")
        return True

    def grant_permission(self, permission: str):
        """Grant permission (for testing/development)"""
        self._permissions[permission] = True

    def revoke_permission(self, permission: str):
        """Revoke permission"""
        self._permissions[permission] = False

    def get_all_permissions(self) -> Dict[str, bool]:
        """Get all permission states"""
        return self._permissions.copy()


class SecurityAuditor:
    """
    Security audit and monitoring
    """

    def __init__(self, storage_path: str = "data/security"):
        self.storage_path = storage_path
        self._audit_log: List[Dict] = []
        os.makedirs(storage_path, exist_ok=True)
        self._load_log()

    def _load_log(self):
        """Load audit log"""
        path = os.path.join(self.storage_path, "audit.log")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self._audit_log = json.load(f)
            except:
                self._audit_log = []

    def _save_log(self):
        """Save audit log"""
        path = os.path.join(self.storage_path, "audit.log")
        # Keep last 1000 entries
        self._audit_log = self._audit_log[-1000:]
        with open(path, "w") as f:
            json.dump(self._audit_log, f)

    def log_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "details": details,
        }
        self._audit_log.append(event)
        self._save_log()

    def get_events(
        self, event_type: Optional[str] = None, since: Optional[datetime] = None
    ) -> List[Dict]:
        """Get audit events"""
        events = self._audit_log

        if event_type:
            events = [e for e in events if e.get("type") == event_type]

        if since:
            events = [
                e for e in events if datetime.fromisoformat(e["timestamp"]) >= since
            ]

        return events

    def check_anomalies(self) -> List[Dict]:
        """Check for security anomalies"""
        anomalies = []

        # Check for multiple failed attempts
        failed_events = [e for e in self._audit_log if e.get("type") == "auth_failed"]

        if len(failed_events) > 5:
            anomalies.append(
                {
                    "type": "multiple_failed_auth",
                    "count": len(failed_events),
                    "recent": failed_events[-5:],
                }
            )

        return anomalies


# Global instances
_authenticator: Optional[LocalAuthenticator] = None
_privacy_manager: Optional[PrivacyManager] = None
_permission_manager: Optional[PermissionManager] = None
_auditor: Optional[SecurityAuditor] = None


def get_authenticator() -> LocalAuthenticator:
    """Get authenticator instance"""
    global _authenticator
    if _authenticator is None:
        _authenticator = LocalAuthenticator()
    return _authenticator


def get_privacy_manager() -> PrivacyManager:
    """Get privacy manager instance"""
    global _privacy_manager
    if _privacy_manager is None:
        _privacy_manager = PrivacyManager()
    return _privacy_manager


def get_permission_manager() -> PermissionManager:
    """Get permission manager instance"""
    global _permission_manager
    if _permission_manager is None:
        _permission_manager = PermissionManager()
    return _permission_manager


def get_security_auditor() -> SecurityAuditor:
    """Get security auditor instance"""
    global _auditor
    if _auditor is None:
        _auditor = SecurityAuditor()
    return _auditor
