"""
AURA Security Permission System
Implements L1-L4 permission levels with banking app protection and audit logging
"""

from enum import Enum
from typing import Dict, Optional, List, Any
from datetime import datetime
import json
import os
import hashlib
import re


class PermissionLevel(Enum):
    L1_FULL = 1
    L2_NORMAL = 2
    L3_RESTRICTED = 3
    L4_MAXIMUM = 4


class SecurityDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    NEEDS_APPROVAL = "needs_approval"


BANKING_APPS = {
    "paytm",
    "phonepe",
    "gpay",
    "upi",
    "bhim",
    "google pay",
    "amazon pay",
    "mobikwik",
    "freecharge",
    "paypal",
    "venmo",
    "cashapp",
    "zelle",
    "hdfc",
    "icici",
    "sbi",
    "axis",
    "kotak",
    "yesbank",
    "idfc",
    "rbl",
    "federal",
    "indusind",
    "zerodha",
    "groww",
    "upstox",
    "angel",
    "5paisa",
    "coinbase",
    "binance",
    "wazirx",
    "coindcx",
}

BANKING_KEYWORDS = [
    "bank",
    "banking",
    "finance",
    "payment",
    "wallet",
    "pay",
    "upi",
    "trading",
    "stocks",
    "invest",
    "crypto",
    "bitcoin",
    "forex",
    "insurance",
    "loan",
]

TOOL_PERMISSIONS = {
    "read_messages": PermissionLevel.L1_FULL,
    "read_notifications": PermissionLevel.L1_FULL,
    "read_contacts": PermissionLevel.L1_FULL,
    "read_calendar": PermissionLevel.L1_FULL,
    "search_web": PermissionLevel.L1_FULL,
    "get_weather": PermissionLevel.L1_FULL,
    "play_music": PermissionLevel.L1_FULL,
    "set_alarm": PermissionLevel.L1_FULL,
    "create_reminder": PermissionLevel.L1_FULL,
    "send_message": PermissionLevel.L2_NORMAL,
    "make_call": PermissionLevel.L2_NORMAL,
    "send_email": PermissionLevel.L2_NORMAL,
    "open_app": PermissionLevel.L2_NORMAL,
    "share_location": PermissionLevel.L2_NORMAL,
    "post_social": PermissionLevel.L2_NORMAL,
    "modify_settings": PermissionLevel.L3_RESTRICTED,
    "delete_file": PermissionLevel.L3_RESTRICTED,
    "install_app": PermissionLevel.L3_RESTRICTED,
    "uninstall_app": PermissionLevel.L3_RESTRICTED,
    "make_payment": PermissionLevel.L4_MAXIMUM,
    "access_banking": PermissionLevel.L4_MAXIMUM,
    "share_sensitive_data": PermissionLevel.L4_MAXIMUM,
}

RISK_LEVELS = {
    PermissionLevel.L1_FULL: "low",
    PermissionLevel.L2_NORMAL: "medium",
    PermissionLevel.L3_RESTRICTED: "high",
    PermissionLevel.L4_MAXIMUM: "critical",
}


class SecurityLayer:
    def __init__(self, default_level: str = "L2", audit_log_path: Optional[str] = None):
        level_map = {
            "L1": PermissionLevel.L1_FULL,
            "L2": PermissionLevel.L2_NORMAL,
            "L3": PermissionLevel.L3_RESTRICTED,
            "L4": PermissionLevel.L4_MAXIMUM,
        }
        self._default_level = level_map.get(default_level, PermissionLevel.L2_NORMAL)
        self._current_level = self._default_level
        self._audit_log_path = audit_log_path or os.path.join(
            os.path.dirname(__file__), "..", "logs", "security_audit.log"
        )
        self._audit_log: List[Dict] = []
        self._session_overrides: Dict[str, PermissionLevel] = {}
        self._sandboxed_tools: set = set()
        self._blocked_apps_cache: Dict[str, bool] = {}
        self._user_id: Optional[str] = None
        self._initialize_audit_log()

    def _initialize_audit_log(self):
        os.makedirs(os.path.dirname(self._audit_log_path), exist_ok=True)
        if os.path.exists(self._audit_log_path):
            try:
                with open(self._audit_log_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._audit_log.append(json.loads(line))
            except (json.JSONDecodeError, IOError):
                self._audit_log = []

    def get_required_level(self, tool_name: str) -> PermissionLevel:
        if tool_name in TOOL_PERMISSIONS:
            return TOOL_PERMISSIONS[tool_name]
        if any(kw in tool_name.lower() for kw in ["read", "get", "search", "list"]):
            return PermissionLevel.L1_FULL
        if any(kw in tool_name.lower() for kw in ["send", "create", "open", "share"]):
            return PermissionLevel.L2_NORMAL
        if any(
            kw in tool_name.lower() for kw in ["modify", "delete", "update", "remove"]
        ):
            return PermissionLevel.L3_RESTRICTED
        if any(
            kw in tool_name.lower()
            for kw in ["payment", "bank", "transfer", "sensitive"]
        ):
            return PermissionLevel.L4_MAXIMUM
        return PermissionLevel.L2_NORMAL

    def get_current_user_level(self) -> PermissionLevel:
        return self._current_level

    def set_permission_level(self, level: str) -> bool:
        level_map = {
            "L1": PermissionLevel.L1_FULL,
            "L2": PermissionLevel.L2_NORMAL,
            "L3": PermissionLevel.L3_RESTRICTED,
            "L4": PermissionLevel.L4_MAXIMUM,
        }
        if level in level_map:
            old_level = self._current_level
            self._current_level = level_map[level]
            self.log_security_event(
                "permission_level_changed",
                {
                    "old_level": old_level.name,
                    "new_level": self._current_level.name,
                },
            )
            return True
        return False

    def set_session_override(self, tool_name: str, level: PermissionLevel):
        self._session_overrides[tool_name] = level
        self.log_security_event(
            "session_override_set",
            {
                "tool": tool_name,
                "level": level.name,
            },
        )

    def clear_session_override(self, tool_name: str):
        if tool_name in self._session_overrides:
            del self._session_overrides[tool_name]

    async def check_permission(self, tool: str, params: Dict) -> Dict:
        result = {
            "allowed": False,
            "needs_approval": False,
            "reason": "",
            "risk_level": "unknown",
            "tool": tool,
            "params_sanitized": self._sanitize_params(params),
        }
        if tool in self._session_overrides:
            required_level = self._session_overrides[tool]
        else:
            required_level = self.get_required_level(tool)
        result["risk_level"] = RISK_LEVELS.get(required_level, "unknown")
        if params.get("app_name") or params.get("package_name"):
            app_name = params.get("app_name") or params.get("package_name", "")
            if self.is_banking_app(app_name):
                result["allowed"] = False
                result["needs_approval"] = True
                result["reason"] = (
                    f"Banking/financial app '{app_name}' is blocked for safety"
                )
                result["is_banking_block"] = True
                self.log_security_event(
                    "banking_app_blocked",
                    {
                        "tool": tool,
                        "app_name": app_name,
                        "required_level": required_level.name,
                    },
                )
                return result
        if self._current_level.value <= required_level.value:
            result["allowed"] = True
            result["needs_approval"] = False
            result["reason"] = "Permission granted"
            self.log_security_event(
                "permission_granted",
                {
                    "tool": tool,
                    "user_level": self._current_level.name,
                    "required_level": required_level.name,
                },
            )
            return result
        if self._current_level == PermissionLevel.L4_MAXIMUM:
            if tool in [
                "read_messages",
                "read_notifications",
                "read_contacts",
                "search_web",
                "get_weather",
            ]:
                result["allowed"] = True
                result["needs_approval"] = False
                result["reason"] = "Read-only operation allowed in maximum security"
                self.log_security_event(
                    "read_only_allowed",
                    {
                        "tool": tool,
                        "user_level": self._current_level.name,
                    },
                )
                return result
        result["allowed"] = False
        result["needs_approval"] = True
        result["reason"] = (
            f"Tool requires {required_level.name}, current level is {self._current_level.name}"
        )
        self.log_security_event(
            "permission_denied_needs_approval",
            {
                "tool": tool,
                "user_level": self._current_level.name,
                "required_level": required_level.name,
            },
        )
        return result

    async def request_override(
        self, tool: str, params: Dict, user_confirmation: bool
    ) -> Dict:
        if not user_confirmation:
            self.log_security_event(
                "override_denied_by_user",
                {
                    "tool": tool,
                },
            )
            return {
                "allowed": False,
                "reason": "User denied the override request",
            }
        self.log_security_event(
            "override_approved_by_user",
            {
                "tool": tool,
            },
        )
        self.set_session_override(tool, PermissionLevel.L1_FULL)
        return await self.check_permission(tool, params)

    def is_banking_app(self, app_name: str) -> bool:
        if not app_name:
            return False
        app_lower = app_name.lower().strip()
        if app_lower in self._blocked_apps_cache:
            return self._blocked_apps_cache[app_lower]
        is_banking = False
        if app_lower in BANKING_APPS:
            is_banking = True
        else:
            for keyword in BANKING_KEYWORDS:
                if keyword in app_lower:
                    is_banking = True
                    break
        self._blocked_apps_cache[app_lower] = is_banking
        return is_banking

    def add_banking_app(self, app_name: str):
        BANKING_APPS.add(app_name.lower().strip())
        self._blocked_apps_cache.clear()
        self.log_security_event("banking_app_added", {"app_name": app_name})

    def remove_banking_app(self, app_name: str) -> bool:
        app_lower = app_name.lower().strip()
        if app_lower in BANKING_APPS:
            BANKING_APPS.remove(app_lower)
            self._blocked_apps_cache.clear()
            self.log_security_event("banking_app_removed", {"app_name": app_name})
            return True
        return False

    def get_banking_apps_list(self) -> List[str]:
        return sorted(list(BANKING_APPS))

    def log_security_event(self, event: str, details: Dict):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "details": details,
            "user_level": self._current_level.name,
            "session_id": self._get_session_hash(),
        }
        self._audit_log.append(entry)
        self._persist_audit_entry(entry)
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]

    def _persist_audit_entry(self, entry: Dict):
        try:
            with open(self._audit_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except IOError:
            pass

    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        return self._audit_log[-limit:]

    def get_audit_log_filtered(
        self,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict]:
        filtered = self._audit_log
        if event_type:
            filtered = [e for e in filtered if e.get("event") == event_type]
        if since:
            filtered = [
                e for e in filtered if datetime.fromisoformat(e["timestamp"]) >= since
            ]
        return filtered[-limit:]

    def _sanitize_params(self, params: Dict) -> Dict:
        sensitive_keys = {"password", "token", "api_key", "secret", "credential", "pin"}
        sanitized = {}
        for key, value in params.items():
            if any(sk in key.lower() for sk in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 100:
                sanitized[key] = value[:100] + "..."
            else:
                sanitized[key] = value
        return sanitized

    def _get_session_hash(self) -> str:
        if not self._user_id:
            self._user_id = hashlib.sha256(
                datetime.now().isoformat().encode()
            ).hexdigest()[:16]
        return self._user_id

    def create_sandbox_context(self, tool: str) -> Dict:
        return {
            "sandbox_id": hashlib.sha256(
                f"{tool}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16],
            "tool": tool,
            "created_at": datetime.now().isoformat(),
            "is_sandboxed": True,
            "allowed_operations": ["read"],
            "network_access": False,
            "file_access": "temp_only",
        }

    def mark_tool_sandboxed(self, tool: str):
        self._sandboxed_tools.add(tool)
        self.log_security_event("tool_sandboxed", {"tool": tool})

    def is_tool_sandboxed(self, tool: str) -> bool:
        return tool in self._sandboxed_tools

    def get_security_summary(self) -> Dict:
        return {
            "current_level": self._current_level.name,
            "level_description": self._get_level_description(self._current_level),
            "sandboxed_tools": list(self._sandboxed_tools),
            "session_overrides": {
                k: v.name for k, v in self._session_overrides.items()
            },
            "banking_apps_count": len(BANKING_APPS),
            "audit_log_entries": len(self._audit_log),
        }

    def _get_level_description(self, level: PermissionLevel) -> str:
        descriptions = {
            PermissionLevel.L1_FULL: "Full access - No restrictions, AURA has complete trust",
            PermissionLevel.L2_NORMAL: "Normal - Ask for high-risk actions (calls, new contacts)",
            PermissionLevel.L3_RESTRICTED: "Restricted - Ask for medium+ risk, block sensitive apps",
            PermissionLevel.L4_MAXIMUM: "Maximum - Ask for everything except reading",
        }
        return descriptions.get(level, "Unknown level")

    def get_tools_by_permission_level(self, level: PermissionLevel) -> List[str]:
        return [
            tool for tool, req_level in TOOL_PERMISSIONS.items() if req_level == level
        ]

    def evaluate_risk_score(self, tool: str, params: Dict) -> int:
        score = 0
        required = self.get_required_level(tool)
        score += required.value * 25
        if params.get("app_name") and self.is_banking_app(params["app_name"]):
            score += 100
        if any(sk in str(params).lower() for sk in ["password", "token", "secret"]):
            score += 50
        if params.get("recipient") or params.get("to"):
            score += 20
        return min(score, 100)

    def should_sandbox(self, tool: str, params: Dict) -> bool:
        risk_score = self.evaluate_risk_score(tool, params)
        return risk_score >= 75 or self.is_banking_app(params.get("app_name", ""))
