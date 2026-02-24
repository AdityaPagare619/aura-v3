"""
AURA v3 Dynamic App Discovery System
Discovers and manages apps on the Android device

Based on OpenClaw's skills architecture but adapted for:
- Mobile-first design (Termux on Android)
- Offline-first operation
- Biologically inspired learning
"""

import asyncio
import json
import logging
import os
import subprocess
import yaml
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AppCategory(Enum):
    """App categories for organization"""

    COMMUNICATION = "communication"
    PRODUCTIVITY = "productivity"
    MEDIA = "media"
    SOCIAL = "social"
    UTILITY = "utility"
    SYSTEM = "system"
    CUSTOM = "custom"


class AppCapability(Enum):
    """What an app can do - for capability gap detection"""

    IMAGE_ANALYSIS = "image_analysis"
    VOICE_INPUT = "voice_input"
    VOICE_OUTPUT = "voice_output"
    TEXT_TO_SPEECH = "text_to_speech"
    SPEECH_TO_TEXT = "speech_to_text"
    VIDEO_ANALYSIS = "video_analysis"
    FILE_ACCESS = "file_access"
    CAMERA_ACCESS = "camera_access"
    LOCATION_ACCESS = "location_access"
    NOTIFICATIONS = "notifications"
    API_CALLS = "api_calls"
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"


@dataclass
class AppMetadata:
    """App manifest metadata (YAML frontmatter parsed)"""

    name: str
    description: str
    icon: str = "📱"
    platforms: List[str] = field(default_factory=lambda: ["android"])
    requires: Dict[str, List[str]] = field(default_factory=dict)
    always_load: bool = False
    category: AppCategory = AppCategory.UTILITY
    capabilities: List[AppCapability] = field(default_factory=list)
    fallback_apps: List[str] = field(default_factory=list)

    # Intent patterns that trigger this app
    trigger_patterns: List[str] = field(default_factory=list)


@dataclass
class AppEntry:
    """A discovered app entry"""

    id: str
    name: str
    metadata: AppMetadata
    manifest_path: Path

    # How to invoke this app
    invoke_command: Optional[str] = None
    api_endpoint: Optional[str] = None

    # Execution status
    is_available: bool = True
    last_used: Optional[datetime] = None
    use_count: int = 0


@dataclass
class CapabilityGap:
    """Represents a capability that AURA doesn't have natively"""

    missing_capability: AppCapability
    attempted_apps: List[str] = field(default_factory=list)
    fallback_strategies: List[str] = field(default_factory=list)
    can_partially_fulfill: bool = False
    partial_solution: Optional[str] = None


class AppDiscovery:
    """
    Dynamic app discovery system - discovers apps on the device
    and creates tool bindings automatically

    Architecture inspired by OpenClaw skills:
    - Progressive disclosure (metadata always, full manifest on trigger)
    - Eligibility filtering (platform, binaries, env vars)
    - Hot reload support

    Key difference: Mobile-focused with Termux integration
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._apps: Dict[str, AppEntry] = {}
        self._capability_map: Dict[AppCapability, List[str]] = {}
        self._watchers: List[Callable] = []
        self._discovery_paths: List[Path] = []

        # Track capability gaps and fallback attempts
        self._capability_gaps: Dict[AppCapability, CapabilityGap] = {}

    async def initialize(self):
        """Initialize app discovery"""
        logger.info("Initializing AURA app discovery...")

        # Set up discovery paths
        await self._setup_discovery_paths()

        # Discover all apps
        await self.discover_apps()

        # Build capability map
        self._build_capability_map()

        # Start watching for changes
        await self._start_watchers()

        logger.info(f"Discovered {len(self._apps)} apps")

    async def _setup_discovery_paths(self):
        """Set up app discovery paths with priority"""
        base_paths = [
            # Bundled apps (default)
            Path(__file__).parent / "bundled",
            # Workspace apps
            Path.cwd() / "aura_apps",
            # User apps
            Path.home() / ".aura" / "apps",
            # Termux apps (Android specific)
            Path.home() / ".aura" / "termux_apps",
        ]

        for path in base_paths:
            if path.exists():
                self._discovery_paths.append(path)
                logger.info(f"Added discovery path: {path}")
            else:
                # Create default bundled path with some starter apps
                if "bundled" in str(path):
                    path.mkdir(parents=True, exist_ok=True)
                    self._discovery_paths.append(path)

    async def discover_apps(self):
        """Discover all apps from discovery paths"""
        for discovery_path in self._discovery_paths:
            await self._discover_from_path(discovery_path)

    async def _discover_from_path(self, path: Path):
        """Discover apps from a specific path"""
        if not path.exists():
            return

        for manifest_file in path.glob("*/APP.yaml"):
            try:
                app = await self._parse_app_manifest(manifest_file)
                if app:
                    self._apps[app.id] = app
                    logger.debug(f"Discovered app: {app.name}")
            except Exception as e:
                logger.warning(f"Failed to parse app manifest {manifest_file}: {e}")

    async def _parse_app_manifest(self, manifest_path: Path) -> Optional[AppEntry]:
        """Parse an app manifest (APP.yaml)"""
        try:
            content = manifest_path.read_text(encoding="utf-8")

            # Parse YAML frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    body = parts[2].strip() if len(parts) > 2 else ""
                else:
                    frontmatter = yaml.safe_load(content)
                    body = ""
            else:
                frontmatter = yaml.safe_load(content)
                body = ""

            # Build metadata
            meta = (
                frontmatter.get("metadata", {}) if isinstance(frontmatter, dict) else {}
            )

            metadata = AppMetadata(
                name=frontmatter.get("name", manifest_path.parent.name),
                description=frontmatter.get("description", ""),
                icon=meta.get("icon", "📱"),
                platforms=meta.get("platforms", ["android"]),
                requires=meta.get("requires", {}),
                always_load=meta.get("always_load", False),
                category=AppCategory(meta.get("category", "utility")),
                capabilities=[AppCapability(c) for c in meta.get("capabilities", [])],
                fallback_apps=meta.get("fallback_apps", []),
                trigger_patterns=meta.get("trigger_patterns", []),
            )

            return AppEntry(
                id=metadata.name.lower().replace(" ", "_"),
                name=metadata.name,
                metadata=metadata,
                manifest_path=manifest_path,
                invoke_command=frontmatter.get("invoke_command"),
                api_endpoint=frontmatter.get("api_endpoint"),
            )

        except Exception as e:
            logger.error(f"Error parsing manifest {manifest_path}: {e}")
            return None

    def _build_capability_map(self):
        """Build map of capabilities to apps that provide them"""
        self._capability_map.clear()

        for app_id, app in self._apps.items():
            for cap in app.metadata.capabilities:
                if cap not in self._capability_map:
                    self._capability_map[cap] = []
                self._capability_map[cap].append(app_id)

    async def _start_watchers(self):
        """Start file watchers for hot reload"""
        # For now, simple polling-based watching
        # In production, use watchdog or similar
        pass

    async def get_app_for_intent(self, user_intent: str) -> Optional[AppEntry]:
        """
        Find the best app for a user intent

        This is the CORE of dynamic discovery - matching user needs to apps
        """
        intent_lower = user_intent.lower()

        # Check trigger patterns first
        best_match = None
        best_score = 0

        for app_id, app in self._apps.items():
            score = 0

            # Check trigger patterns
            for pattern in app.metadata.trigger_patterns:
                if pattern.lower() in intent_lower:
                    score += 10

            # Check description keywords
            desc_words = app.metadata.description.lower().split()
            for word in desc_words:
                if word in intent_lower:
                    score += 1

            # Check name
            if app.metadata.name.lower() in intent_lower:
                score += 5

            if score > best_score:
                best_score = score
                best_match = app

        return best_match if best_score > 0 else None

    async def get_apps_by_capability(self, capability: AppCapability) -> List[AppEntry]:
        """Get all apps that provide a specific capability"""
        app_ids = self._capability_map.get(capability, [])
        return [self._apps[aid] for aid in app_ids if aid in self._apps]

    async def check_capability_gap(
        self, required_capability: AppCapability
    ) -> CapabilityGap:
        """
        Check if AURA has a capability gap and find solutions

        This is the KEY to AURA "finding ways" - the fallback system
        """
        if required_capability not in self._capability_map:
            # No app provides this natively
            gap = CapabilityGap(
                missing_capability=required_capability,
                can_partially_fulfill=False,
            )

            # Find fallback strategies
            gap.fallback_strategies = await self._find_fallback_strategies(
                required_capability
            )

            self._capability_gaps[required_capability] = gap
            return gap

        # Capability exists, return success
        return CapabilityGap(
            missing_capability=required_capability,
            can_partially_fulfill=True,
        )

    async def _find_fallback_strategies(self, capability: AppCapability) -> List[str]:
        """
        Find alternative ways to achieve a capability

        This is where AURA's "intelligence" comes from - finding creative solutions
        """
        strategies = []

        # Map capabilities to potential workarounds
        fallback_map = {
            AppCapability.IMAGE_ANALYSIS: [
                "Check if any gallery app has image metadata",
                "Use file system to find images by date/location",
                "Check cloud backups for image references",
                "Query chat history for shared images",
            ],
            AppCapability.VOICE_INPUT: [
                "Use text input as fallback",
                "Check for voice message files in chat apps",
            ],
            AppCapability.TEXT_TO_SPEECH: [
                "Use notification with vibration pattern",
                "Display text on screen prominently",
            ],
            AppCapability.CAMERA_ACCESS: [
                "Check existing photos in gallery",
                "Use screen capture of camera feed if available",
            ],
            AppCapability.WEB_SEARCH: [
                "Use cached data from previous searches",
                "Check local knowledge base",
            ],
        }

        return fallback_map.get(capability, ["No fallback available"])

    async def attempt_with_fallback(
        self, capability: AppCapability, user_request: str
    ) -> Dict[str, Any]:
        """
        Attempt to fulfill a request, using fallbacks if primary fails

        This implements the OpenClaw pattern of "finding ways"
        """
        # First check if we have the capability
        gap = await self.check_capability_gap(capability)

        if gap.can_partially_fulfill:
            apps = await self.get_apps_by_capability(capability)
            if apps:
                # Try the most used app first
                app = max(apps, key=lambda a: a.use_count)
                return await self._execute_app(app, user_request)

        # Gap exists - try fallback strategies
        result = {
            "success": False,
            "capability_gap": capability.value,
            "strategies_tried": [],
            "partial_result": None,
        }

        for strategy in gap.fallback_strategies:
            result["strategies_tried"].append(strategy)

            # Try each strategy
            # This would be expanded with actual implementation
            if "gallery" in strategy.lower():
                result["partial_result"] = "Checked gallery for related images"

        return result

    async def _execute_app(self, app: AppEntry, user_request: str) -> Dict[str, Any]:
        """Execute an app with the given request"""
        try:
            # Update usage stats
            app.use_count += 1
            app.last_used = datetime.now()

            if app.invoke_command:
                # Execute via Termux
                result = await self._execute_termux_command(
                    app.invoke_command, user_request
                )
                return {"success": True, "result": result}

            elif app.api_endpoint:
                # Execute via API
                result = await self._execute_api_call(app.api_endpoint, user_request)
                return {"success": True, "result": result}

            return {"success": False, "error": "No invoke method defined"}

        except Exception as e:
            logger.error(f"Error executing app {app.name}: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_termux_command(self, command: str, args: str) -> str:
        """Execute a Termux command"""
        full_cmd = f"{command} {args}"

        try:
            result = subprocess.run(
                full_cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.stdout if result.returncode == 0 else result.stderr
        except subprocess.TimeoutExpired:
            return "Command timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _execute_api_call(self, endpoint: str, data: str) -> str:
        """Execute an API call (for future HTTP-based apps)"""
        # Placeholder for HTTP API calls
        return f"API call to {endpoint} with data: {data}"

    def get_all_apps(self) -> List[AppEntry]:
        """Get all discovered apps"""
        return list(self._apps.values())

    def get_available_apps(self) -> List[AppEntry]:
        """Get apps that are currently available"""
        return [a for a in self._apps.values() if a.is_available]


# Global instance
_app_discovery: Optional[AppDiscovery] = None


async def get_app_discovery() -> AppDiscovery:
    """Get or create app discovery instance"""
    global _app_discovery
    if _app_discovery is None:
        _app_discovery = AppDiscovery()
        await _app_discovery.initialize()
    return _app_discovery
