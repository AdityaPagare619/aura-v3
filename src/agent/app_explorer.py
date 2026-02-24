"""
AURA v3 App Explorer System
===========================

Explores installed apps, analyzes usage patterns, and provides
intelligent suggestions to help organize the user's digital life.

Features:
- App inventory and usage tracking
- Cross-app pattern detection
- Security analysis (spam, suspicious links)
- Productivity suggestions
- PrivacyGuard warnings

Run modes:
- Background exploration (periodic, respects privacy)
- Manual trigger via /explore command
- View suggestions via /suggestions command
"""

import asyncio
import logging
import re
import os
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of analysis the explorer can perform"""

    APP_INVENTORY = "app_inventory"
    USAGE_PATTERNS = "usage_patterns"
    DATA_ACCESS = "data_access"
    CROSS_APP_PATTERNS = "cross_app_patterns"
    PRODUCTIVITY = "productivity"
    NOTIFICATION_ANALYSIS = "notification_analysis"
    TIME_ANALYSIS = "time_analysis"
    REDUNDANT_APPS = "redundant_apps"
    SECURITY_SCAN = "security_scan"
    PRIVACY_GUARD = "privacy_guard"


class SuggestionPriority(Enum):
    """Priority levels for suggestions"""

    CRITICAL = 3  # Security warnings
    HIGH = 2  # Important suggestions
    MEDIUM = 1  # Helpful suggestions
    LOW = 0  # Nice-to-have


@dataclass
class AppInfo:
    """Information about an installed app"""

    app_id: str
    name: str
    category: str
    permissions: List[str] = field(default_factory=list)
    installed_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    usage_duration_minutes: int = 0
    notifications_received: int = 0
    is_system_app: bool = False


@dataclass
class AppUsagePattern:
    """Usage pattern for an app"""

    app_id: str
    app_name: str
    frequency_per_day: float
    avg_duration_minutes: float
    time_of_day_peaks: List[str]
    day_of_week_pattern: Dict[str, float]
    typical_actions: List[str]


@dataclass
class CrossAppPattern:
    """Pattern showing relationship between apps"""

    app_ids: List[str]
    app_names: List[str]
    pattern_type: str  # "sequential", "concurrent", "complementary"
    strength: float  # 0-1
    description: str


@dataclass
class Suggestion:
    """An actionable suggestion for the user"""

    id: str
    title: str
    message: str
    priority: SuggestionPriority
    category: str
    action_type: str
    related_apps: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    is_dismissed: bool = False
    is_implemented: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "priority": self.priority.name,
            "category": self.category,
            "action_type": self.action_type,
            "related_apps": self.related_apps,
            "created_at": self.created_at.isoformat(),
            "is_dismissed": self.is_dismissed,
            "is_implemented": self.is_implemented,
        }


@dataclass
class SecurityAlert:
    """Security-related alert"""

    id: str
    alert_type: str  # "spam", "suspicious_link", "malicious_app", "data_exposure"
    severity: SuggestionPriority
    title: str
    description: str
    source_app: Optional[str] = None
    source_content: Optional[str] = None
    recommendation: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    is_resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "alert_type": self.alert_type,
            "severity": self.severity.name,
            "title": self.title,
            "description": self.description,
            "source_app": self.source_app,
            "source_content": self.source_content[:200]
            if self.source_content
            else None,
            "recommendation": self.recommendation,
            "created_at": self.created_at.isoformat(),
            "is_resolved": self.is_resolved,
        }


class SecurityAnalyzer:
    """Analyzes content for security threats"""

    SPAM_PATTERNS = [
        r"(?i)(you won|congratulations|free gift|claim now)",
        r"(?i)(urgent|immediate action required|act now|limited time)",
        r"(?i)(click here|visit now|buy now|order now)",
        r"(?i)(verify your account|confirm your identity|suspend)",
        r"(?i)(nigeria|prince|inheritance|lottery)",
    ]

    SUSPICIOUS_LINK_PATTERNS = [
        r"(?i)(bit\.ly|tinyurl|goo\.gl|t\.co|is\.gd)",
        r"(?i)(login|signin|account|verify|password)",
        r"(?i)(bank|paypal|ebay|amazon)\.[a-z]{2,}",
        r"(?i)(update|secure|confirm|unusual)",
    ]

    PHISHING_KEYWORDS = [
        "verify your account",
        "confirm your password",
        "unusual activity",
        "account suspended",
        "click to login",
        "secure your account",
    ]

    def __init__(self):
        self._spam_regexes = [re.compile(p) for p in self.SPAM_PATTERNS]
        self._link_regex = re.compile(r"https?://[^\s]+")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for spam and threats"""
        result = {
            "is_spam": False,
            "spam_score": 0.0,
            "suspicious_links": [],
            "phishing_keywords": [],
            "threats_detected": [],
        }

        if not text:
            return result

        # Check for spam patterns
        spam_matches = 0
        for regex in self._spam_regexes:
            if regex.search(text):
                spam_matches += 1

        result["spam_score"] = min(spam_matches / 3.0, 1.0)
        result["is_spam"] = result["spam_score"] > 0.5

        # Extract and analyze links
        links = self._link_regex.findall(text)
        for link in links:
            link_lower = link.lower()
            for pattern in self.SUSPICIOUS_LINK_PATTERNS:
                if re.search(pattern, link_lower):
                    result["suspicious_links"].append(link)
                    break

        # Check for phishing keywords
        text_lower = text.lower()
        for keyword in self.PHISHING_KEYWORDS:
            if keyword in text_lower:
                result["phishing_keywords"].append(keyword)

        # Determine threats
        if result["is_spam"]:
            result["threats_detected"].append("spam_detected")
        if result["suspicious_links"]:
            result["threats_detected"].append("suspicious_links")
        if len(result["phishing_keywords"]) >= 2:
            result["threats_detected"].append("phishing_attempt")

        return result

    def analyze_app_permissions(self, permissions: List[str]) -> Dict[str, Any]:
        """Analyze app permissions for privacy concerns"""
        sensitive_permissions = [
            "READ_CONTACTS",
            "WRITE_CONTACTS",
            "READ_SMS",
            "SEND_SMS",
            "RECEIVE_SMS",
            "READ_CALL_LOG",
            "READ_PHONE_STATE",
            "ACCESS_FINE_LOCATION",
            "ACCESS_COARSE_LOCATION",
            "RECORD_AUDIO",
            "CAMERA",
            "READ_EXTERNAL_STORAGE",
            "WRITE_EXTERNAL_STORAGE",
        ]

        dangerous_combos = [
            {"permissions": ["READ_SMS", "INTERNET"], "risk": "sms_theft"},
            {"permissions": ["READ_CONTACTS", "INTERNET"], "risk": "contact_theft"},
            {"permissions": ["READ_CALL_LOG", "INTERNET"], "risk": "call_log_theft"},
            {
                "permissions": ["ACCESS_FINE_LOCATION", "INTERNET"],
                "risk": "location_tracking",
            },
        ]

        result = {
            "sensitive_permissions": [],
            "dangerous_combos": [],
            "privacy_score": 1.0,
            "warnings": [],
        }

        for perm in permissions:
            perm_upper = perm.upper()
            if perm_upper in sensitive_permissions:
                result["sensitive_permissions"].append(perm)
                result["privacy_score"] -= 0.15

        for combo in dangerous_combos:
            if all(p in [p.upper() for p in permissions] for p in combo["permissions"]):
                result["dangerous_combos"].append(combo["risk"])
                result["warnings"].append(f"Risk: {combo['risk']}")
                result["privacy_score"] -= 0.2

        result["privacy_score"] = max(0, result["privacy_score"])
        return result


class PrivacyGuard:
    """Monitors and warns about data exposure"""

    EXPOSURE_PATTERNS = {
        "contacts_shared": {
            "keywords": ["contact", "address book", "contacts shared"],
            "severity": SuggestionPriority.HIGH,
        },
        "location_exposed": {
            "keywords": ["location", "gps", "tracking"],
            "severity": SuggestionPriority.HIGH,
        },
        "messages_exposed": {
            "keywords": ["messages", "sms", "chats"],
            "severity": SuggestionPriority.CRITICAL,
        },
        "photos_exposed": {
            "keywords": ["photos", "gallery", "media"],
            "severity": SuggestionPriority.MEDIUM,
        },
    }

    def __init__(self):
        self._exposure_history: List[Dict] = []

    def check_data_access(
        self, app: AppInfo, accessed_data: List[str]
    ) -> List[SecurityAlert]:
        """Check if app is accessing sensitive data inappropriately"""
        alerts = []

        for data in accessed_data:
            data_lower = data.lower()

            for pattern_name, pattern_info in self.EXPOSURE_PATTERNS.items():
                if any(kw in data_lower for kw in pattern_info["keywords"]):
                    # Check if app should have this access
                    should_warn = True

                    if app.category in ["social", "messaging"]:
                        if pattern_name in ["contacts_shared", "messages_exposed"]:
                            should_warn = False  # Expected
                    elif app.category in ["navigation", "fitness"]:
                        if pattern_name == "location_exposed":
                            should_warn = False  # Expected

                    if should_warn:
                        alert = SecurityAlert(
                            id=f"exposure_{app.app_id}_{pattern_name}",
                            alert_type="data_exposure",
                            severity=pattern_info["severity"],
                            title=f"Data Exposure: {app.name}",
                            description=f"{app.name} is accessing {data}",
                            source_app=app.name,
                            recommendation=f"Review {app.name}'s permissions",
                        )
                        alerts.append(alert)

        return alerts


class AppExplorer:
    """
    App Explorer - Analyzes installed apps and usage patterns

    Provides:
    - App inventory management
    - Usage pattern analysis
    - Cross-app relationship detection
    - Security scanning
    - PrivacyGuard
    - Intelligent suggestions
    """

    # App categories for classification
    APP_CATEGORIES = {
        "social": [
            "facebook",
            "instagram",
            "twitter",
            "tiktok",
            "snapchat",
            "linkedin",
        ],
        "messaging": [
            "whatsapp",
            "telegram",
            "messenger",
            "signal",
            "discord",
            "slack",
        ],
        "productivity": [
            "notion",
            "evernote",
            "todoist",
            "asana",
            "office",
            "google docs",
        ],
        "entertainment": ["netflix", "youtube", "spotify", "twitch", "prime video"],
        "news": ["news", "flipboard", "feedly", "reddit"],
        "fitness": ["fitbit", "myfitnesspal", "strava", "garmin", "health"],
        "finance": ["bank", "paypal", "venmo", "cashapp", "coinbase", "mint"],
        "shopping": ["amazon", "ebay", "etsy", "shopify", "walmart"],
        "navigation": ["maps", "google maps", "waze", "uber", "lyft"],
        "utilities": ["calculator", "files", "settings", "clock", "calendar"],
    }

    COMPLEMENTARY_APP_PAIRS = [
        (["notion", "google calendar"], "sync notes with calendar events"),
        (["todoist", "google calendar"], "block time for tasks"),
        (["spotify", "nordvpn"], None),  # Just example
        (["evernote", "google drive"], "attach files to notes"),
        (["slack", "zoom"], "start video call from chat"),
        (["notion", "todoist"], "track projects in both apps"),
        (["whatsapp", "google calendar"], "share events via chat"),
    ]

    def __init__(self, privacy_manager=None):
        # Core data
        self._apps: Dict[str, AppInfo] = {}
        self._usage_patterns: Dict[str, AppUsagePattern] = {}
        self._cross_app_patterns: List[CrossAppPattern] = []

        # Security
        self._security_analyzer = SecurityAnalyzer()
        self._privacy_guard = PrivacyGuard()
        self._security_alerts: List[SecurityAlert] = []

        # Suggestions
        self._suggestions: List[Suggestion] = []

        # State
        self._is_enabled = True
        self._privacy_manager = privacy_manager
        self._last_exploration: Optional[datetime] = None
        self._exploration_interval = timedelta(hours=1)

        # Settings
        self._settings = {
            "exploration_enabled": True,
            "background_exploration": True,
            "security_scan_enabled": True,
            "suggestion_frequency": "balanced",  # "minimal", "balanced", "proactive"
            "max_suggestions": 10,
        }

        # Tracking
        self._message_history: List[Dict] = []
        self._notification_history: List[Dict] = []

    # =========================================================================
    # CORE EXPLORATION
    # =========================================================================

    async def explore_apps(self) -> Dict[str, Any]:
        """Perform full app exploration"""
        if not self._settings["exploration_enabled"]:
            return {"status": "disabled", "message": "Exploration is disabled"}

        results = {
            "timestamp": datetime.now().isoformat(),
            "apps_analyzed": 0,
            "new_suggestions": 0,
            "security_alerts": 0,
        }

        try:
            # Analyze app inventory
            await self._analyze_app_inventory()
            results["apps_analyzed"] = len(self._apps)

            # Analyze usage patterns
            await self._analyze_usage_patterns()

            # Detect cross-app patterns
            await self._detect_cross_app_patterns()

            # Generate suggestions
            await self._generate_suggestions()

            # Security scan
            if self._settings["security_scan_enabled"]:
                await self._security_scan()
                results["security_alerts"] = len(self._security_alerts)

            self._last_exploration = datetime.now()
            results["new_suggestions"] = len(
                [
                    s
                    for s in self._suggestions
                    if (datetime.now() - s.created_at).total_seconds() < 60
                ]
            )

            logger.info(f"Exploration complete: {results}")
            return results

        except Exception as e:
            logger.error(f"Exploration error: {e}")
            return {"status": "error", "message": str(e)}

    async def _analyze_app_inventory(self):
        """Build and analyze app inventory"""
        # This would integrate with actual app listing on the device
        # For now, we work with tracked apps

        for app_id, app in self._apps.items():
            # Categorize if not already done
            if not app.category:
                app.category = self._categorize_app(app.name)

    def _categorize_app(self, app_name: str) -> str:
        """Categorize an app based on its name"""
        name_lower = app_name.lower()

        for category, keywords in self.APP_CATEGORIES.items():
            if any(kw in name_lower for kw in keywords):
                return category

        return "other"

    async def _analyze_usage_patterns(self):
        """Analyze usage patterns across apps"""
        # Group apps by time of day
        time_patterns = defaultdict(list)

        for app_id, pattern in self._usage_patterns.items():
            for peak_time in pattern.time_of_day_peaks:
                time_patterns[peak_time].append(app_id)

        # Identify complementary patterns
        self._cross_app_patterns.clear()

        for apps, description in self.COMPLEMENTARY_APP_PAIRS:
            matching_apps = []
            for app_id, app in self._apps.items():
                if any(app_name in app.name.lower() for app_name in apps):
                    matching_apps.append(app_id)

            if len(matching_apps) >= 2:
                pattern = CrossAppPattern(
                    app_ids=matching_apps,
                    app_names=[self._apps[a].name for a in matching_apps],
                    pattern_type="complementary",
                    strength=0.7,
                    description=description
                    or f"{' + '.join([self._apps[a].name for a in matching_apps])} work well together",
                )
                self._cross_app_patterns.append(pattern)

    async def _detect_cross_app_patterns(self):
        """Detect relationships between apps"""
        # Sequential patterns: user often uses app Y after app X
        # Concurrent patterns: user uses multiple apps at once
        # Complementary patterns: apps serve related purposes

        pass  # Implemented in _analyze_usage_patterns

    async def _generate_suggestions(self):
        """Generate actionable suggestions"""
        new_suggestions = []

        # Cross-app integration suggestions
        for pattern in self._cross_app_patterns:
            existing = any(
                s.action_type == "cross_app_integration"
                and set(s.related_apps) == set(pattern.app_ids)
                for s in self._suggestions
            )

            if not existing:
                suggestion = Suggestion(
                    id=f"suggestion_{len(self._suggestions)}_{datetime.now().timestamp()}",
                    title=f"Connect {pattern.app_names[0]} and {pattern.app_names[1]}",
                    message=f"I noticed you use {pattern.app_names[0]} and {pattern.app_names[1]} - they could work together. {pattern.description}",
                    priority=SuggestionPriority.MEDIUM,
                    category="productivity",
                    action_type="cross_app_integration",
                    related_apps=pattern.app_ids,
                )
                new_suggestions.append(suggestion)

        # Time analysis suggestions
        await self._generate_time_suggestions(new_suggestions)

        # Redundant app detection
        await self._generate_redundancy_suggestions(new_suggestions)

        # Add new suggestions
        self._suggestions.extend(new_suggestions)

        # Limit suggestions
        if len(self._suggestions) > self._settings["max_suggestions"]:
            # Remove oldest/lowest priority
            self._suggestions.sort(key=lambda s: (s.priority.value, s.created_at))
            self._suggestions = self._suggestions[-self._settings["max_suggestions"] :]

    async def _generate_time_suggestions(self, suggestions: List[Suggestion]):
        """Generate time management suggestions"""
        high_usage_apps = []

        for app_id, pattern in self._usage_patterns.items():
            if pattern.avg_duration_minutes > 120:  # 2+ hours
                high_usage_apps.append((app_id, pattern))

        for app_id, pattern in high_usage_apps[:3]:
            existing = any(
                s.action_type == "time_warning" and app_id in s.related_apps
                for s in self._suggestions
            )

            if not existing:
                suggestion = Suggestion(
                    id=f"time_{app_id}_{datetime.now().timestamp()}",
                    title=f"You're spending a lot of time on {pattern.app_name}",
                    message=f"You're averaging {pattern.avg_duration_minutes:.0f} min/day on {pattern.app_name}. Want me to set a break reminder?",
                    priority=SuggestionPriority.LOW,
                    category="wellbeing",
                    action_type="time_warning",
                    related_apps=[app_id],
                    data={"avg_duration": pattern.avg_duration_minutes},
                )
                suggestions.append(suggestion)

    async def _generate_redundancy_suggestions(self, suggestions: List[Suggestion]):
        """Detect and suggest removal of redundant apps"""
        # Find apps in same category with similar usage
        category_apps = defaultdict(list)

        for app_id, app in self._apps.items():
            if app.category and app.usage_count > 0:
                category_apps[app.category].append(app)

        for category, apps in category_apps.items():
            if len(apps) >= 2:
                # Suggest keeping the most-used one
                apps.sort(key=lambda a: a.usage_count, reverse=True)

                for app in apps[1:]:
                    existing = any(
                        s.action_type == "remove_redundant"
                        and app.app_id in s.related_apps
                        for s in self._suggestions
                    )

                    if not existing and app.usage_count < 10:  # Rarely used
                        suggestion = Suggestion(
                            id=f"redundant_{app.app_id}_{datetime.now().timestamp()}",
                            title=f"Consider removing {app.name}?",
                            message=f"You have multiple {category} apps. {app.name} is rarely used compared to {apps[0].name}. Want me to show you how to remove it?",
                            priority=SuggestionPriority.LOW,
                            category="productivity",
                            action_type="remove_redundant",
                            related_apps=[app.app_id, apps[0].app_id],
                        )
                        suggestions.append(suggestion)

    async def _security_scan(self):
        """Perform security analysis"""
        # Analyze recent messages/notifications for threats
        for item in self._message_history[-50:]:
            content = item.get("content", "")
            if content:
                analysis = self._security_analyzer.analyze_text(content)

                if analysis["threats_detected"]:
                    alert = SecurityAlert(
                        id=f"security_{len(self._security_alerts)}_{datetime.now().timestamp()}",
                        alert_type=analysis["threats_detected"][0],
                        severity=SuggestionPriority.CRITICAL,
                        title="Suspicious content detected",
                        description=f"Found: {', '.join(analysis['threats_detected'])}",
                        source_content=content,
                        recommendation="Don't click any links. Delete this message.",
                    )
                    self._security_alerts.append(alert)

        # Analyze app permissions
        for app_id, app in self._apps.items():
            if app.permissions:
                analysis = self._security_analyzer.analyze_app_permissions(
                    app.permissions
                )

                if analysis["dangerous_combos"]:
                    alert = SecurityAlert(
                        id=f"app_security_{app_id}_{datetime.now().timestamp()}",
                        alert_type="malicious_app",
                        severity=SuggestionPriority.HIGH,
                        title=f"Risky permissions: {app.name}",
                        description=f"App has dangerous permission combination: {', '.join(analysis['dangerous_combos'])}",
                        source_app=app.name,
                        recommendation="Review app permissions in Settings",
                    )
                    self._security_alerts.append(alert)

    # =========================================================================
    # DATA MANAGEMENT
    # =========================================================================

    def register_app(self, app_info: AppInfo):
        """Register an app with the explorer"""
        self._apps[app_info.app_id] = app_info
        logger.debug(f"Registered app: {app_info.name}")

    def update_app_usage(self, app_id: str, duration_minutes: int = 0):
        """Update usage data for an app"""
        if app_id not in self._apps:
            return

        app = self._apps[app_id]
        app.usage_count += 1
        app.usage_duration_minutes += duration_minutes
        app.last_used = datetime.now()

        # Update or create pattern
        if app_id not in self._usage_patterns:
            self._usage_patterns[app_id] = AppUsagePattern(
                app_id=app_id,
                app_name=app.name,
                frequency_per_day=0,
                avg_duration_minutes=0,
                time_of_day_peaks=[],
                day_of_week_pattern={},
                typical_actions=[],
            )

        pattern = self._usage_patterns[app_id]
        pattern.frequency_per_day = app.usage_count / max(
            1, (datetime.now() - (app.installed_at or datetime.now())).days
        )
        pattern.avg_duration_minutes = app.usage_duration_minutes / max(
            1, app.usage_count
        )

    def add_notification(self, app_id: str, content: str = ""):
        """Add a notification to analyze"""
        if app_id in self._apps:
            self._apps[app_id].notifications_received += 1

        self._notification_history.append(
            {
                "app_id": app_id,
                "content": content,
                "timestamp": datetime.now(),
            }
        )

        # Limit history
        if len(self._notification_history) > 1000:
            self._notification_history = self._notification_history[-500:]

        # Immediate threat check
        if content and self._settings["security_scan_enabled"]:
            analysis = self._security_analyzer.analyze_text(content)
            if analysis["threats_detected"]:
                return analysis

        return None

    def add_message(self, app_id: str, content: str, is_incoming: bool = True):
        """Add a message to analyze"""
        self._message_history.append(
            {
                "app_id": app_id,
                "content": content,
                "is_incoming": is_incoming,
                "timestamp": datetime.now(),
            }
        )

        # Limit history
        if len(self._message_history) > 1000:
            self._message_history = self._message_history[-500:]

        # Immediate threat check
        if is_incoming and content and self._settings["security_scan_enabled"]:
            analysis = self._security_analyzer.analyze_text(content)
            if analysis["threats_detected"]:
                return analysis

        return None

    # =========================================================================
    # SUGGESTIONS
    # =========================================================================

    def get_suggestions(
        self,
        category: Optional[str] = None,
        priority: Optional[SuggestionPriority] = None,
        include_implemented: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get current suggestions"""
        suggestions = self._suggestions

        if category:
            suggestions = [s for s in suggestions if s.category == category]
        if priority:
            suggestions = [s for s in suggestions if s.priority == priority]
        if not include_implemented:
            suggestions = [s for s in suggestions if not s.is_implemented]

        suggestions.sort(key=lambda s: (s.priority.value, -s.created_at.timestamp()))

        return [s.to_dict() for s in suggestions]

    def dismiss_suggestion(self, suggestion_id: str) -> bool:
        """Dismiss a suggestion"""
        for suggestion in self._suggestions:
            if suggestion.id == suggestion_id:
                suggestion.is_dismissed = True
                return True
        return False

    def implement_suggestion(self, suggestion_id: str) -> bool:
        """Mark suggestion as implemented"""
        for suggestion in self._suggestions:
            if suggestion.id == suggestion_id:
                suggestion.is_implemented = True
                return True
        return False

    def clear_suggestions(self, category: Optional[str] = None):
        """Clear suggestions"""
        if category:
            self._suggestions = [s for s in self._suggestions if s.category != category]
        else:
            self._suggestions.clear()

    # =========================================================================
    # SECURITY ALERTS
    # =========================================================================

    def get_security_alerts(
        self,
        alert_type: Optional[str] = None,
        include_resolved: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get security alerts"""
        alerts = self._security_alerts

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if not include_resolved:
            alerts = [a for a in alerts if not a.is_resolved]

        alerts.sort(key=lambda a: (a.severity.value, -a.created_at.timestamp()))

        return [a.to_dict() for a in alerts]

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved"""
        for alert in self._security_alerts:
            if alert.id == alert_id:
                alert.is_resolved = True
                return True
        return False

    # =========================================================================
    # APP INVENTORY
    # =========================================================================

    def get_app_inventory(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all registered apps"""
        apps = list(self._apps.values())

        if category:
            apps = [a for a in apps if a.category == category]

        return [
            {
                "app_id": a.app_id,
                "name": a.name,
                "category": a.category,
                "permissions": a.permissions,
                "usage_count": a.usage_count,
                "usage_duration_minutes": a.usage_duration_minutes,
                "notifications_received": a.notifications_received,
                "last_used": a.last_used.isoformat() if a.last_used else None,
            }
            for a in apps
        ]

    def get_app(self, app_id: str) -> Optional[AppInfo]:
        """Get specific app info"""
        return self._apps.get(app_id)

    # =========================================================================
    # SETTINGS
    # =========================================================================

    def enable(self):
        """Enable the explorer"""
        self._is_enabled = True
        self._settings["exploration_enabled"] = True
        logger.info("App Explorer enabled")

    def disable(self):
        """Disable the explorer"""
        self._is_enabled = False
        self._settings["exploration_enabled"] = False
        logger.info("App Explorer disabled")

    def is_enabled(self) -> bool:
        """Check if explorer is enabled"""
        return self._is_enabled

    def update_settings(self, **kwargs):
        """Update explorer settings"""
        for key, value in kwargs.items():
            if key in self._settings:
                self._settings[key] = value
        logger.info(f"Updated settings: {kwargs}")

    def get_settings(self) -> Dict[str, Any]:
        """Get current settings"""
        return self._settings.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get explorer status"""
        return {
            "is_enabled": self._is_enabled,
            "total_apps": len(self._apps),
            "total_suggestions": len(self._suggestions),
            "pending_suggestions": len(
                [s for s in self._suggestions if not s.is_implemented]
            ),
            "security_alerts": len(
                [a for a in self._security_alerts if not a.is_resolved]
            ),
            "last_exploration": self._last_exploration.isoformat()
            if self._last_exploration
            else None,
            "settings": self._settings,
        }

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    def get_usage_report(self) -> Dict[str, Any]:
        """Get usage analysis report"""
        total_usage = sum(a.usage_duration_minutes for a in self._apps.values())
        top_apps = sorted(
            self._apps.values(),
            key=lambda a: a.usage_duration_minutes,
            reverse=True,
        )[:10]

        return {
            "total_usage_minutes": total_usage,
            "total_usage_hours": round(total_usage / 60, 1),
            "app_count": len(self._apps),
            "top_apps": [
                {
                    "name": a.name,
                    "usage_minutes": a.usage_duration_minutes,
                    "usage_hours": round(a.usage_duration_minutes / 60, 1),
                }
                for a in top_apps
            ],
            "category_breakdown": self._get_category_breakdown(),
        }

    def _get_category_breakdown(self) -> Dict[str, int]:
        """Get usage by category"""
        breakdown = defaultdict(int)
        for app in self._apps.values():
            breakdown[app.category] += app.usage_duration_minutes
        return dict(breakdown)


class ExplorationCommandHandler:
    """Handles /explore and /suggestions commands"""

    def __init__(self, app_explorer: AppExplorer):
        self._explorer = app_explorer

    async def handle_explore(self, args: str = "") -> str:
        """Handle /explore command"""
        parts = args.strip().lower().split()

        if not parts:
            # Full exploration
            result = await self._explorer.explore_apps()
            return self._format_explore_result(result)

        command = parts[0]

        if command == "apps":
            return await self._handle_apps(parts[1:])
        elif command == "security":
            return await self._handle_security()
        elif command == "usage":
            return await self._handle_usage()
        elif command == "patterns":
            return await self._handle_patterns()
        elif command == "enable":
            self._explorer.enable()
            return "Exploration enabled"
        elif command == "disable":
            self._explorer.disable()
            return "Exploration disabled"
        else:
            return f"Unknown explore command: {command}. Use: apps, security, usage, patterns"

    async def handle_suggestions(self, args: str = "") -> str:
        """Handle /suggestions command"""
        parts = args.strip().lower().split()

        category = None
        priority = None

        for part in parts:
            if part in ["productivity", "security", "wellbeing"]:
                category = part
            elif part in ["high", "medium", "low", "critical"]:
                try:
                    priority = SuggestionPriority[part.upper()]
                except KeyError:
                    pass

        suggestions = self._explorer.get_suggestions(
            category=category,
            priority=priority,
        )

        if not suggestions:
            return "No suggestions at the moment. Run /explore to analyze your apps."

        return self._format_suggestions(suggestions)

    async def _handle_apps(self, args: List[str]) -> str:
        """Handle explore apps"""
        category = args[0] if args else None
        apps = self._explorer.get_app_inventory(category=category)

        if not apps:
            return "No apps registered yet."

        lines = ["📱 App Inventory:"]
        for app in apps[:10]:
            lines.append(
                f"  • {app['name']} ({app['category']}) - {app['usage_count']} uses"
            )

        return "\n".join(lines)

    async def _handle_security(self) -> str:
        """Handle explore security"""
        alerts = self._explorer.get_security_alerts()

        if not alerts:
            return "🔒 No security alerts. Your apps look safe!"

        lines = ["⚠️ Security Alerts:"]
        for alert in alerts[:5]:
            lines.append(f"  • [{alert['severity']}] {alert['title']}")
            lines.append(f"    {alert['description']}")

        return "\n".join(lines)

    async def _handle_usage(self) -> str:
        """Handle explore usage"""
        report = self._explorer.get_usage_report()

        lines = [
            f"⏱️ Usage Report",
            f"  Total: {report['total_usage_hours']} hours",
            f"  Apps tracked: {report['app_count']}",
            "",
            "  Top apps:",
        ]

        for app in report["top_apps"][:5]:
            lines.append(f"    • {app['name']}: {app['usage_hours']}h")

        return "\n".join(lines)

    async def _handle_patterns(self) -> str:
        """Handle explore patterns"""
        # This would return cross-app patterns
        return "Cross-app pattern analysis coming soon."

    def _format_explore_result(self, result: Dict) -> str:
        """Format exploration result"""
        if result.get("status") == "disabled":
            return "⚠️ Exploration is disabled. Use /explore enable to enable."

        lines = [
            "🔍 Exploration Complete",
            f"  Apps analyzed: {result.get('apps_analyzed', 0)}",
            f"  New suggestions: {result.get('new_suggestions', 0)}",
            f"  Security alerts: {result.get('security_alerts', 0)}",
        ]

        return "\n".join(lines)

    def _format_suggestions(self, suggestions: List[Dict]) -> str:
        """Format suggestions for display"""
        lines = [f"💡 You have {len(suggestions)} suggestions:"]

        priority_icons = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢",
        }

        for suggestion in suggestions[:10]:
            icon = priority_icons.get(suggestion["priority"], "💡")
            lines.append(f"\n{icon} {suggestion['title']}")
            lines.append(f"   {suggestion['message']}")

        lines.append("\n\nUse /suggestions [category] to filter")
        return "\n".join(lines)


# Global instance
_explorer: Optional[AppExplorer] = None
_handler: Optional[ExplorationCommandHandler] = None


def get_app_explorer() -> AppExplorer:
    """Get or create the app explorer"""
    global _explorer
    if _explorer is None:
        _explorer = AppExplorer()
    return _explorer


def get_exploration_handler() -> ExplorationCommandHandler:
    """Get the exploration command handler"""
    global _handler
    if _handler is None:
        _handler = ExplorationCommandHandler(get_app_explorer())
    return _handler


async def run_exploration() -> Dict[str, Any]:
    """Run a full exploration cycle"""
    explorer = get_app_explorer()
    return await explorer.explore_apps()
