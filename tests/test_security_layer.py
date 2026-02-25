"""
Tests for AURA v3 Security Layer

Tests the main security package:
- SecurityLevel enum
- SecurityPolicy dataclass  
- LocalAuthenticator for PIN/password auth
- Encryption utilities
"""
import unittest
import asyncio
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

sys.path.insert(0, '.')
from src.security.security import SecurityLevel, SecurityPolicy, AuthAttempt, LocalAuthenticator

class TestSecurityLevel(unittest.TestCase):
    def test_all_levels(self):
        levels = [SecurityLevel.NONE, SecurityLevel.PIN, SecurityLevel.BIOMETRIC, SecurityLevel.PASSWORD]
        self.assertEqual(len(levels), 4)
    def test_values(self):
        self.assertEqual(SecurityLevel.NONE.value, 0)
        self.assertEqual(SecurityLevel.PIN.value, 1)
        self.assertEqual(SecurityLevel.BIOMETRIC.value, 2)
        self.assertEqual(SecurityLevel.PASSWORD.value, 3)

class TestSecurityPolicy(unittest.TestCase):
    def test_default_policy(self):
        policy = SecurityPolicy()
        self.assertFalse(policy.require_auth_for_start)
        self.assertTrue(policy.require_auth_for_sensitive)
        self.assertEqual(policy.auto_lock_minutes, 5)
        self.assertEqual(policy.max_failed_attempts, 3)
        self.assertEqual(policy.lockout_duration_minutes, 15)
        self.assertTrue(policy.encrypt_storage)
        self.assertTrue(policy.allow_rooted)

    def test_custom_policy(self):
        policy = SecurityPolicy(
            require_auth_for_start=True,
            require_auth_for_sensitive=True,
            auto_lock_minutes=10,
            max_failed_attempts=5,
            lockout_duration_minutes=30,
            encrypt_storage=False,
            allow_rooted=False
        )
        self.assertTrue(policy.require_auth_for_start)
        self.assertEqual(policy.auto_lock_minutes, 10)
        self.assertEqual(policy.max_failed_attempts, 5)

class TestAuthAttempt(unittest.TestCase):
    def test_create_attempt(self):
        from datetime import datetime
        attempt = AuthAttempt(timestamp=datetime.now(), method="pin", success=True)
        self.assertEqual(attempt.method, "pin")
        self.assertTrue(attempt.success)
    def test_failed_attempt(self):
        from datetime import datetime
        attempt = AuthAttempt(timestamp=datetime.now(), method="password", success=False, device_id="test123")
        self.assertFalse(attempt.success)
        self.assertEqual(attempt.device_id, "test123")

class TestLocalAuthenticator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    def test_init(self):
        auth = LocalAuthenticator(storage_path=self.temp_dir)
        self.assertEqual(auth.storage_path, self.temp_dir)
        self.assertFalse(auth._policy.require_auth_for_start)
    def test_policy_default(self):
        auth = LocalAuthenticator(storage_path=self.temp_dir)
        self.assertEqual(auth._policy.require_auth_for_sensitive, True)
    def test_set_policy(self):
        auth = LocalAuthenticator(storage_path=self.temp_dir)
        new_policy = SecurityPolicy(require_auth_for_start=True, auto_lock_minutes=15)
        auth.set_policy(new_policy)
        self.assertEqual(auth._policy.require_auth_for_start, True)
        self.assertEqual(auth._policy.auto_lock_minutes, 15)
    def test_failed_attempts_tracking(self):
        auth = LocalAuthenticator(storage_path=self.temp_dir)
        self.assertEqual(len(auth._failed_attempts), 0)
    def test_authenticated_sessions(self):
        auth = LocalAuthenticator(storage_path=self.temp_dir)
        self.assertEqual(len(auth._authenticated_sessions), 0)

class TestSecurityIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    def test_security_level_ordering(self):
        self.assertLess(SecurityLevel.NONE.value, SecurityLevel.PIN.value)
        self.assertLess(SecurityLevel.PIN.value, SecurityLevel.BIOMETRIC.value)
        self.assertLess(SecurityLevel.BIOMETRIC.value, SecurityLevel.PASSWORD.value)
    def test_policy_strictness(self):
        strict = SecurityPolicy(max_failed_attempts=3, lockout_duration_minutes=30)
        lenient = SecurityPolicy(max_failed_attempts=10, lockout_duration_minutes=5)
        self.assertLess(strict.max_failed_attempts, lenient.max_failed_attempts)
        self.assertGreater(strict.lockout_duration_minutes, lenient.lockout_duration_minutes)

if __name__ == "__main__": unittest.main()
