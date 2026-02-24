"""
AURA v3 Privacy Tier System
============================

User-controlled privacy tiers that determine AURA's proactivity:
- PRIVATE: Gallery, messages, personal docs - Asks MORE before acting
- SENSITIVE: Financial apps, health data - Asks BEFORE acting
- PUBLIC: News apps, general files - Acts proactively with suggestions
"""

import asyncio
import json
import logging
import os
import yaml
from typing import Dict, Optional, List, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class PrivacyTier(Enum):
    """Privacy tiers with increasing restriction"""

    PUBLIC = "public"  # Proactive suggestions allowed
    SENSITIVE = "sensitive"  # Confirm for writes, warn for reads
    PRIVATE = "private"  # Always confirm before acting

    def __lt__(self, other):
        """Define ordering: PUBLIC < SENSITIVE < PRIVATE"""
        order = {
            PrivacyTier.PUBLIC: 0,
            PrivacyTier.SENSITIVE: 1,
            PrivacyTier.PRIVATE: 2,
        }
        return order[self] < order[other]

    def requires_confirmation(self, action: str) -> bool:
        """Check if this tier requires confirmation for an action"""
        if self == PrivacyTier.PRIVATE:
            return True  # Always confirm for private
        elif self == PrivacyTier.SENSITIVE:
            return action in ("write", "modify", "delete")  # Confirm for writes
        return False  # No confirmation for public


class CategoryType(Enum):
    """Category types for data classification"""

    MEDIA = "media"
    MESSAGES = "messages"
    PERSONAL_DOCS = "personal_docs"
    FINANCIAL = "financial"
    HEALTH = "health"
    WORK = "work"
    PUBLIC = "public"
    GENERAL = "general"


@dataclass
class PermissionGrant:
    """Record of a permission grant"""

    category: str
    tier: PrivacyTier
    granted_at: datetime
    expires_at: Optional[datetime] = None
    scope: str = "once"  # "once", "session", "always"
    trusted: bool = False


@dataclass
class PermissionRequest:
    """A permission request that needs user confirmation"""

    tool_name: str
    category: str
    action: str  # "read", "write", "analyze", "delete"
    tier: PrivacyTier
    context: Dict[str, Any] = field(default_factory=dict)
    user_response: Optional[bool] = None
    requested_at: datetime = field(default_factory=datetime.now)


@dataclass
class CategoryConfig:
    """Configuration for a data category"""

    name: str
    type: CategoryType
    default_tier: PrivacyTier
    description: str = ""
    keywords: List[str] = field(default_factory=list)


class CategoryRegistry:
    """Registry of categories and their default tiers"""

    DEFAULT_CATEGORIES: List[CategoryConfig] = [
        # PRIVATE Categories
        CategoryConfig(
            name="gallery_photos",
            type=CategoryType.MEDIA,
            default_tier=PrivacyTier.PRIVATE,
            description="Photos in device gallery",
            keywords=["photo", "gallery", "image", "picture", "camera"],
        ),
        CategoryConfig(
            name="gallery_videos",
            type=CategoryType.MEDIA,
            default_tier=PrivacyTier.PRIVATE,
            description="Videos in device gallery",
            keywords=["video", "movie", "recording"],
        ),
        CategoryConfig(
            name="screenshots",
            type=CategoryType.MEDIA,
            default_tier=PrivacyTier.PRIVATE,
            description="Device screenshots",
            keywords=["screenshot", "screen capture"],
        ),
        CategoryConfig(
            name="sms",
            type=CategoryType.MESSAGES,
            default_tier=PrivacyTier.PRIVATE,
            description="SMS messages",
            keywords=["sms", "text", "message"],
        ),
        CategoryConfig(
            name="whatsapp",
            type=CategoryType.MESSAGES,
            default_tier=PrivacyTier.PRIVATE,
            description="WhatsApp messages",
            keywords=["whatsapp", "whats app"],
        ),
        CategoryConfig(
            name="telegram_chats",
            type=CategoryType.MESSAGES,
            default_tier=PrivacyTier.PRIVATE,
            description="Telegram chats",
            keywords=["telegram", "tg"],
        ),
        CategoryConfig(
            name="email",
            type=CategoryType.MESSAGES,
            default_tier=PrivacyTier.PRIVATE,
            description="Email messages",
            keywords=["email", "gmail", "mail"],
        ),
        CategoryConfig(
            name="notes",
            type=CategoryType.PERSONAL_DOCS,
            default_tier=PrivacyTier.PRIVATE,
            description="Personal notes",
            keywords=["note", "keep", "memo"],
        ),
        CategoryConfig(
            name="calendar",
            type=CategoryType.PERSONAL_DOCS,
            default_tier=PrivacyTier.PRIVATE,
            description="Calendar events and appointments",
            keywords=["calendar", "event", "appointment", "schedule"],
        ),
        CategoryConfig(
            name="contacts",
            type=CategoryType.PERSONAL_DOCS,
            default_tier=PrivacyTier.PRIVATE,
            description="Contact information",
            keywords=["contact", "address book", "phone book"],
        ),
        # SENSITIVE Categories
        CategoryConfig(
            name="banking_apps",
            type=CategoryType.FINANCIAL,
            default_tier=PrivacyTier.SENSITIVE,
            description="Banking applications",
            keywords=["bank", "banking", "finance"],
        ),
        CategoryConfig(
            name="payment_apps",
            type=CategoryType.FINANCIAL,
            default_tier=PrivacyTier.SENSITIVE,
            description="Payment applications",
            keywords=["payment", "pay", "venmo", "cashapp", "paypal"],
        ),
        CategoryConfig(
            name="crypto",
            type=CategoryType.FINANCIAL,
            default_tier=PrivacyTier.SENSITIVE,
            description="Cryptocurrency apps",
            keywords=["crypto", "bitcoin", "ethereum", "wallet"],
        ),
        CategoryConfig(
            name="fitness_data",
            type=CategoryType.HEALTH,
            default_tier=PrivacyTier.SENSITIVE,
            description="Fitness tracking data",
            keywords=["fitness", "workout", "exercise", "steps"],
        ),
        CategoryConfig(
            name="medical_apps",
            type=CategoryType.HEALTH,
            default_tier=PrivacyTier.SENSITIVE,
            description="Medical and health applications",
            keywords=["medical", "health", "doctor"],
        ),
        CategoryConfig(
            name="sleep_data",
            type=CategoryType.HEALTH,
            default_tier=PrivacyTier.SENSITIVE,
            description="Sleep tracking data",
            keywords=["sleep", "rest", "dream"],
        ),
        CategoryConfig(
            name="slack",
            type=CategoryType.WORK,
            default_tier=PrivacyTier.SENSITIVE,
            description="Slack messages",
            keywords=["slack", "workspace"],
        ),
        CategoryConfig(
            name="teams",
            type=CategoryType.WORK,
            default_tier=PrivacyTier.SENSITIVE,
            description="Microsoft Teams",
            keywords=["teams", "ms teams", "microsoft teams"],
        ),
        CategoryConfig(
            name="work_documents",
            type=CategoryType.WORK,
            default_tier=PrivacyTier.SENSITIVE,
            description="Work-related documents",
            keywords=["work", "office", "document", "doc"],
        ),
        # PUBLIC Categories
        CategoryConfig(
            name="news_apps",
            type=CategoryType.PUBLIC,
            default_tier=PrivacyTier.PUBLIC,
            description="News applications",
            keywords=["news", "article", "headline"],
        ),
        CategoryConfig(
            name="weather",
            type=CategoryType.PUBLIC,
            default_tier=PrivacyTier.PUBLIC,
            description="Weather information",
            keywords=["weather", "forecast", "temperature"],
        ),
        CategoryConfig(
            name="social_media",
            type=CategoryType.PUBLIC,
            default_tier=PrivacyTier.PUBLIC,
            description="Social media content",
            keywords=["social", "facebook", "twitter", "instagram"],
        ),
        CategoryConfig(
            name="general_files",
            type=CategoryType.GENERAL,
            default_tier=PrivacyTier.PUBLIC,
            description="General files and downloads",
            keywords=["file", "download", "folder"],
        ),
    ]

    def __init__(self):
        self._categories: Dict[str, CategoryConfig] = {}
        self._category_keywords: Dict[str, List[str]] = {}
        self._load_defaults()

    def _load_defaults(self):
        """Load default category configurations"""
        for cat in self.DEFAULT_CATEGORIES:
            self._categories[cat.name] = cat
            self._category_keywords[cat.name] = cat.keywords

    def get_category(self, name: str) -> Optional[CategoryConfig]:
        """Get category by name"""
        return self._categories.get(name.lower())

    def get_tier(self, category: str) -> PrivacyTier:
        """Get default tier for a category"""
        cat = self.get_category(category)
        return cat.default_tier if cat else PrivacyTier.PUBLIC

    def find_category_by_keyword(self, keyword: str) -> Optional[str]:
        """Find category by keyword"""
        keyword_lower = keyword.lower()
        for cat_name, keywords in self._category_keywords.items():
            if any(keyword_lower in k for k in keywords):
                return cat_name
        return None

    def get_all_categories(self) -> List[CategoryConfig]:
        """Get all registered categories"""
        return list(self._categories.values())


class UserPermissionManager:
    """
    Manages user permissions for different privacy tiers

    Provides:
    - Permission checking before actions
    - Permission granting (once/session/always)
    - User override handling
    - Permission persistence
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/privacy.yaml"
        self.category_registry = CategoryRegistry()

        # User overrides: category -> tier
        self._tier_overrides: Dict[str, PrivacyTier] = {}

        # Active permission grants: category -> PermissionGrant
        self._grants: Dict[str, PermissionGrant] = {}

        # Permission history for learning
        self._permission_history: List[PermissionRequest] = []

        # Default tier
        self._default_tier = PrivacyTier.SENSITIVE

        # Settings
        self.settings = {
            "confirm_private_always": True,
            "confirm_sensitive_writes": True,
            "public_proactive": True,
            "log_permissions": True,
        }

        self._load_config()

    def _load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config = yaml.safe_load(f)

                if not config:
                    return

                privacy_config = config.get("privacy", {})

                # Load default tier
                tier_str = privacy_config.get("default_tier", "sensitive")
                self._default_tier = PrivacyTier(tier_str)

                # Load category overrides
                categories = privacy_config.get("categories", {})
                for cat_name, tier_str in categories.items():
                    try:
                        self._tier_overrides[cat_name] = PrivacyTier(tier_str)
                    except ValueError:
                        logger.warning(
                            f"Invalid tier '{tier_str}' for category '{cat_name}'"
                        )

                # Load settings
                confirm = privacy_config.get("confirm", {})
                self.settings.update(confirm)

                logger.info(f"Loaded privacy config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load privacy config: {e}")

    def save_config(self):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        config = {
            "privacy": {
                "default_tier": self._default_tier.value,
                "categories": {
                    cat: tier.value for cat, tier in self._tier_overrides.items()
                },
                "confirm": self.settings,
            }
        }

        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Saved privacy config to {self.config_path}")

    def get_effective_tier(self, category: str) -> PrivacyTier:
        """Get effective tier for a category (considering overrides)"""
        # Check user override first
        if category in self._tier_overrides:
            return self._tier_overrides[category]

        # Check for active grant
        if category in self._grants:
            grant = self._grants[category]
            if grant.expires_at and datetime.now() > grant.expires_at:
                del self._grants[category]
            elif grant.trusted:
                return PrivacyTier.PUBLIC  # Trusted = full access

        # Fall back to default
        return self.category_registry.get_tier(category)

    def check_permission(
        self, category: str, action: str, tool_name: str = ""
    ) -> PermissionRequest:
        """
        Check if permission is needed for an action

        Returns PermissionRequest with:
        - tier: The effective privacy tier
        - Whether confirmation is needed
        """
        tier = self.get_effective_tier(category)

        request = PermissionRequest(
            tool_name=tool_name, category=category, action=action, tier=tier
        )

        # Determine if confirmation is needed
        if tier == PrivacyTier.PRIVATE:
            request.context["needs_confirmation"] = True
            request.context["confirmation_reason"] = (
                "Private data - explicit permission required"
            )
        elif tier == PrivacyTier.SENSITIVE:
            if action in ("write", "modify", "delete"):
                request.context["needs_confirmation"] = True
                request.context["confirmation_reason"] = (
                    "Sensitive data - write operation"
                )
            else:
                request.context["needs_warning"] = True
                request.context["warning_message"] = "Reading sensitive data"
        else:
            # Public - no confirmation needed
            request.context["needs_confirmation"] = False

        self._permission_history.append(request)
        return request

    def grant_permission(
        self,
        category: str,
        scope: str = "session",
        duration_minutes: Optional[int] = None,
    ) -> bool:
        """Grant permission for a category"""
        tier = self.get_effective_tier(category)

        expires_at = None
        if scope == "once":
            expires_at = datetime.now() + timedelta(minutes=5)
        elif scope == "session" and duration_minutes:
            expires_at = datetime.now() + timedelta(minutes=duration_minutes)

        self._grants[category] = PermissionGrant(
            category=category,
            tier=tier,
            granted_at=datetime.now(),
            expires_at=expires_at,
            scope=scope,
        )

        logger.info(f"Granted {scope} permission for {category}")
        return True

    def revoke_permission(self, category: str) -> bool:
        """Revoke permission for a category"""
        if category in self._grants:
            del self._grants[category]
            logger.info(f"Revoked permission for {category}")
            return True
        return False

    def set_tier_override(self, category: str, tier: PrivacyTier):
        """Set a permanent tier override for a category"""
        self._tier_overrides[category] = tier
        self.save_config()
        logger.info(f"Set tier override for {category}: {tier.value}")

    def set_default_tier(self, tier: PrivacyTier):
        """Set the default tier"""
        self._default_tier = tier
        self.save_config()

    def clear_override(self, category: str) -> bool:
        """Clear tier override for a category"""
        if category in self._tier_overrides:
            del self._tier_overrides[category]
            self.save_config()
            return True
        return False

    def get_permission_status(self) -> Dict[str, Any]:
        """Get current permission status"""
        categories = self.category_registry.get_all_categories()

        status = {
            "default_tier": self._default_tier.value,
            "settings": self.settings,
            "categories": {},
        }

        for cat in categories:
            effective = self.get_effective_tier(cat.name)
            override = self._tier_overrides.get(cat.name)
            grant = self._grants.get(cat.name)

            status["categories"][cat.name] = {
                "default": cat.default_tier.value,
                "effective": effective.value,
                "override": override.value if override else None,
                "grant": grant.scope if grant else None,
            }

        return status

    def parse_natural_permission(self, text: str) -> Optional[tuple]:
        """
        Parse natural language permission grants

        Examples:
        - "You can access my photos" -> ("gallery_photos", "always")
        - "Don't read my messages without asking" -> ("messages", "always")
        """
        text_lower = text.lower()

        # Permission keywords
        grant_patterns = {
            "can access": "grant",
            "you can": "grant",
            "feel free to": "grant",
            "go ahead and": "grant",
            "it's okay to": "grant",
            "i give you permission": "grant",
        }

        restrict_patterns = {
            "don't": "restrict",
            "do not": "restrict",
            "never": "restrict",
            "without asking": "restrict",
            "always ask": "restrict",
            "require confirmation": "restrict",
        }

        # Category detection
        category = self.category_registry.find_category_by_keyword(text_lower)

        if not category:
            # Try broader categories
            if any(k in text_lower for k in ["photo", "gallery", "image"]):
                category = "gallery_photos"
            elif any(k in text_lower for k in ["message", "chat", "sms"]):
                category = "messages"
            elif any(k in text_lower for k in ["calendar", "event"]):
                category = "calendar"
            elif any(k in text_lower for k in ["note", "note"]):
                category = "notes"
            elif any(
                k in text_lower for k in ["bank", "banking", "finance", "financial"]
            ):
                category = "banking_apps"
            elif any(k in text_lower for k in ["fitness", "health", "medical"]):
                category = "medical_apps"

        if not category:
            return None

        # Determine if granting or restricting
        is_grant = any(p in text_lower for p in grant_patterns.keys())
        is_restrict = any(p in text_lower for p in restrict_patterns.keys())

        if is_restrict:
            return (category, "restrict")
        elif is_grant:
            # Check for scope
            if "always" in text_lower or "whenever" in text_lower:
                return (category, "always")
            return (category, "session")

        return None

    def handle_natural_permission(self, text: str) -> Optional[str]:
        """Handle natural language permission statement"""
        result = self.parse_natural_permission(text)

        if not result:
            return None

        category, action = result

        if action == "restrict":
            self.set_tier_override(category, PrivacyTier.PRIVATE)
            return f"Understood. I'll always ask before accessing your {category.replace('_', ' ')}."
        else:
            self.grant_permission(category, scope="always")
            return (
                f"Got it! I'll access your {category.replace('_', ' ')} when helpful."
            )


class ToolPrivacyMixin:
    """Mixin to add privacy tier checking to tools"""

    def __init__(self):
        self._permission_manager: Optional[UserPermissionManager] = None

    def init_privacy(self, permission_manager: UserPermissionManager):
        """Initialize with permission manager"""
        self._permission_manager = permission_manager

    def check_tool_permission(
        self, tool_name: str, category: str, action: str = "read"
    ) -> "tuple[bool, Optional[str]]":
        """
        Check if tool can execute

        Returns:
            (allowed, response_message)
        """
        if not self._permission_manager:
            return True, None  # No permission manager = allow all

        request = self._permission_manager.check_permission(
            category=category, action=action, tool_name=tool_name
        )

        if request.context.get("needs_confirmation"):
            reason = request.context.get("confirmation_reason", "Permission required")
            return False, f"Permission needed: {reason}. Please confirm to proceed."

        if request.context.get("needs_warning"):
            logger.warning(f"Accessing sensitive data: {category}")

        return True, None


# Global instance
_permission_manager: Optional[UserPermissionManager] = None


def get_permission_manager() -> UserPermissionManager:
    """Get or create global permission manager"""
    global _permission_manager
    if _permission_manager is None:
        _permission_manager = UserPermissionManager()
    return _permission_manager


def init_permission_manager(
    config_path: str = "config/privacy.yaml",
) -> UserPermissionManager:
    """Initialize permission manager with config"""
    global _permission_manager
    _permission_manager = UserPermissionManager(config_path)
    return _permission_manager
