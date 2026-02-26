"""
AURA v3 Capability Gap Handler
Intelligent fallback system for when capabilities are missing

This is the CORE of "AURA finding ways" - when AURA can't do something
directly, it finds alternative paths to achieve the user's goal

Inspired by OpenClaw's fallback system but adapted for mobile/offline:
- Detect capability gaps
- Find alternative solutions
- Try multiple strategies
- Learn from failures
"""

import asyncio
import json
import logging
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Threshold constants
REPEATED_GAP_THRESHOLD = 3  # After N failures, suggest solutions proactively
GAP_HISTORY_RETENTION_DAYS = 30  # Keep history for N days
MAX_GAP_HISTORY_ENTRIES = 1000  # Limit history size


class CapabilityType(Enum):
    """Categories of capabilities AURA might need"""

    # Vision/Image
    IMAGE_RECOGNITION = "image_recognition"
    IMAGE_GENERATION = "image_generation"
    VIDEO_ANALYSIS = "video_analysis"

    # Audio
    SPEECH_RECOGNITION = "speech_recognition"
    SPEECH_SYNTHESIS = "speech_synthesis"
    VOICE_ID = "voice_identification"

    # Language
    TEXT_SUMMARIZATION = "text_summarization"
    TRANSLATION = "translation"
    TEXT_ANALYSIS = "text_analysis"

    # Actions
    WEB_ACCESS = "web_access"
    API_CALL = "api_call"
    FILE_ANALYSIS = "file_analysis"

    # Mobile-specific
    CAMERA_ACCESS = "camera_access"
    LOCATION_ACCESS = "location_access"
    NOTIFICATION_ACCESS = "notification_access"
    CONTACTS_ACCESS = "contacts_access"
    SMS_ACCESS = "sms_access"


@dataclass
class Strategy:
    """A strategy to fulfill a capability gap"""

    name: str
    description: str
    # How to attempt this strategy
    attempt_fn: Optional[Callable] = None
    # Prerequisites for this strategy
    prerequisites: List[str] = field(default_factory=list)
    # Required tool from registry (actual tool name)
    required_tool: Optional[str] = None
    # Required permission (from security/permission system)
    required_permission: Optional[str] = None
    # Required service (e.g., termux, network, gps)
    required_service: Optional[str] = None
    # Expected success rate (0-1)
    estimated_success: float = 0.5
    # Cost (computational, time, etc.)
    cost: float = 0.5
    # Whether this is a last resort
    is_last_resort: bool = False


@dataclass
class PrerequisiteResult:
    """Result of checking prerequisites for a strategy"""

    satisfied: bool
    tool_available: bool = True
    permission_granted: bool = True
    service_running: bool = True
    missing: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GapHistoryEntry:
    """A record of a capability gap occurrence"""

    capability: str
    user_goal: str
    timestamp: str  # ISO format for JSON serialization
    resolved: bool
    strategy_used: Optional[str] = None
    attempts: int = 0
    error: Optional[str] = None


@dataclass
class CapabilityGap:
    """A capability that AURA doesn't have"""

    capability: CapabilityType
    # What the user wanted
    user_goal: str
    # Strategies tried
    attempted_strategies: List[str] = field(default_factory=list)
    # Results
    results: List[Dict[str, Any]] = field(default_factory=list)
    # Whether any strategy succeeded
    resolved: bool = False
    # Partial success
    partial_result: Optional[str] = None


@dataclass
class GapResolution:
    """Result of attempting to resolve a gap"""

    success: bool
    result: Any = None
    strategy_used: Optional[str] = None
    message: str = ""
    partial: bool = False


class CapabilityGapHandler:
    """
    Intelligent Capability Gap Handler

    Key behaviors:
    1. DETECT gaps - recognize when AURA can't do something
    2. STRATEGIZE - find multiple ways to achieve goal
    3. EXECUTE - try strategies with fallback
    4. LEARN - remember what works/doesn't

    This is NOT magic - it's explicit code paths that try alternatives.
    The "intelligence" comes from having many well-designed fallback paths.
    """

    def __init__(self, storage_path: str = "data/capability_gaps"):
        # Registered capability strategies
        self._strategies: Dict[CapabilityType, List[Strategy]] = defaultdict(list)

        # History of gap resolutions
        self._gap_history: List[CapabilityGap] = []
        self._successful_patterns: Dict[str, int] = defaultdict(int)

        # Persistent gap history tracking
        self._storage_path = storage_path
        self._persistent_gap_history: List[GapHistoryEntry] = []
        self._gap_frequency: Dict[str, List[str]] = defaultdict(
            list
        )  # capability -> timestamps

        # External integrations (lazy loaded)
        self._tool_registry = None
        self._permission_manager = None
        self._termux_bridge = None

        # Service status cache (to avoid repeated expensive checks)
        self._service_cache: Dict[str, Tuple[bool, datetime]] = {}
        self._service_cache_ttl = 60  # seconds

        # Load persistent data
        self._load_gap_history()

        # Initialize default strategies
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default fallback strategies"""

        # Image recognition fallback strategies
        self.register_strategy(
            CapabilityType.IMAGE_RECOGNITION,
            Strategy(
                name="file_search",
                description="Search gallery for similar file names",
                prerequisites=["gallery_access"],
                required_tool="find_files",
                required_permission="storage",
                estimated_success=0.3,
                cost=0.2,
            ),
        )

        self.register_strategy(
            CapabilityType.IMAGE_RECOGNITION,
            Strategy(
                name="chat_history_search",
                description="Search chat apps for shared images",
                prerequisites=["messaging_access"],
                required_permission="messaging",
                estimated_success=0.4,
                cost=0.3,
            ),
        )

        self.register_strategy(
            CapabilityType.IMAGE_RECOGNITION,
            Strategy(
                name="metadata_analysis",
                description="Analyze image file metadata for clues",
                prerequisites=["file_access"],
                required_tool="read_file",
                required_permission="storage",
                estimated_success=0.2,
                cost=0.1,
            ),
        )

        # Speech recognition fallback
        self.register_strategy(
            CapabilityType.SPEECH_RECOGNITION,
            Strategy(
                name="text_fallback",
                description="Ask user to type instead",
                prerequisites=[],
                estimated_success=1.0,
                cost=0.0,
            ),
        )

        # Web access fallback
        self.register_strategy(
            CapabilityType.WEB_ACCESS,
            Strategy(
                name="cached_data",
                description="Use previously cached web content",
                prerequisites=["cache_access"],
                required_service="cache",
                estimated_success=0.3,
                cost=0.1,
            ),
        )

        self.register_strategy(
            CapabilityType.WEB_ACCESS,
            Strategy(
                name="local_knowledge",
                description="Use AURA's trained knowledge",
                prerequisites=[],
                estimated_success=0.5,
                cost=0.0,
            ),
        )

        # Location access fallback
        self.register_strategy(
            CapabilityType.LOCATION_ACCESS,
            Strategy(
                name="last_known",
                description="Use last known location from logs",
                prerequisites=["location_history"],
                required_service="location_history",
                estimated_success=0.6,
                cost=0.1,
            ),
        )

        self.register_strategy(
            CapabilityType.LOCATION_ACCESS,
            Strategy(
                name="manual_confirmation",
                description="Ask user to confirm location",
                prerequisites=[],
                estimated_success=1.0,
                cost=0.0,
            ),
        )

        # Camera access fallback
        self.register_strategy(
            CapabilityType.CAMERA_ACCESS,
            Strategy(
                name="use_existing_photos",
                description="Use existing photos from gallery",
                prerequisites=["gallery_access"],
                required_tool="list_images",
                required_permission="storage",
                estimated_success=0.5,
                cost=0.2,
            ),
        )

        self.register_strategy(
            CapabilityType.CAMERA_ACCESS,
            Strategy(
                name="request_user_upload",
                description="Ask user to manually take and share photo",
                prerequisites=[],
                estimated_success=0.8,
                cost=0.0,
                is_last_resort=True,
            ),
        )

    def register_strategy(self, capability: CapabilityType, strategy: Strategy):
        """Register a strategy for a capability"""
        self._strategies[capability].append(strategy)
        # Sort by estimated success / cost ratio
        self._strategies[capability].sort(
            key=lambda s: s.estimated_success / max(s.cost, 0.1), reverse=True
        )

    async def check_capability(self, capability: CapabilityType) -> bool:
        """
        Check if AURA has a capability

        Returns True if capability is available, False if there's a gap
        """
        # This would check:
        # - Whether required tools/apps are available
        # - Whether required APIs are accessible
        # - Whether required permissions are granted

        # For now, simplified - assume certain capabilities exist
        available = {
            CapabilityType.FILE_ANALYSIS: True,
            CapabilityType.NOTIFICATION_ACCESS: True,
            CapabilityType.CONTACTS_ACCESS: True,
        }

        return available.get(capability, False)

    async def handle_gap(
        self,
        user_goal: str,
        required_capability: CapabilityType,
        context: Optional[Dict[str, Any]] = None,
    ) -> GapResolution:
        """
        Main entry point: handle a capability gap

        This is where AURA "finds ways" - tries multiple strategies
        until one works or all are exhausted
        """
        logger.info(f"Handling capability gap: {required_capability.value}")
        logger.info(f"User goal: {user_goal}")

        # Check for repeated gaps and suggest proactive solutions
        proactive_suggestions = self._check_repeated_gap(required_capability)
        if proactive_suggestions:
            logger.info(f"Repeated gap detected, suggestions: {proactive_suggestions}")

        # Create gap record
        gap = CapabilityGap(
            capability=required_capability,
            user_goal=user_goal,
        )

        # Get available strategies
        strategies = self._strategies.get(required_capability, [])

        if not strategies:
            self._record_gap_history(
                required_capability,
                user_goal,
                False,
                None,
                0,
                "No strategies available",
            )
            return GapResolution(
                success=False,
                message=f"No strategies available for {required_capability.value}",
            )

        # Try each strategy
        attempts = 0
        for strategy in strategies:
            attempts += 1
            logger.info(f"Trying strategy: {strategy.name}")
            gap.attempted_strategies.append(strategy.name)

            # Check prerequisites with REAL checks
            prereq_result = await self._check_prerequisites(strategy)

            if not prereq_result.satisfied:
                logger.info(
                    f"Prerequisites not met for {strategy.name}: {prereq_result.missing}"
                )
                gap.results.append(
                    {
                        "strategy": strategy.name,
                        "result": {"success": False, "reason": "prerequisites_not_met"},
                        "prerequisites": {
                            "missing": prereq_result.missing,
                            "suggestions": prereq_result.suggestions,
                            "details": prereq_result.details,
                        },
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                continue

            # Execute strategy
            result = await self._execute_strategy(strategy, user_goal, context)

            gap.results.append(
                {
                    "strategy": strategy.name,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            if result.get("success", False):
                # Success!
                gap.resolved = True
                self._record_success(strategy.name, required_capability)
                self._record_gap_history(
                    required_capability, user_goal, True, strategy.name, attempts
                )

                return GapResolution(
                    success=True,
                    result=result.get("data") or result.get("result"),
                    strategy_used=strategy.name,
                    message=f"Succeeded using {strategy.name}",
                    partial=result.get("partial", False),
                )

        # All strategies failed
        self._record_failure(strategies, required_capability)
        self._record_gap_history(
            required_capability,
            user_goal,
            False,
            None,
            attempts,
            "All strategies failed",
        )

        # Return best partial result if any
        partials = [r for r in gap.results if r.get("result", {}).get("partial")]
        if partials:
            best = partials[0]
            return GapResolution(
                success=False,
                result=best.get("result", {}).get("data"),
                strategy_used=best.get("strategy"),
                message="Could not fully complete, but found partial information",
                partial=True,
            )

        # Include proactive suggestions if this is a repeated failure
        message = "Could not find a way to accomplish this with available capabilities"
        if proactive_suggestions:
            message += f". Suggested fixes: {', '.join(proactive_suggestions[:3])}"

        return GapResolution(
            success=False,
            message=message,
        )

    async def _check_prerequisites(self, strategy: Strategy) -> PrerequisiteResult:
        """
        Check if prerequisites for a strategy are met

        This is the REAL implementation that:
        1. Checks tool availability against registry
        2. Checks permissions against security system
        3. Checks service availability (termux, network, etc.)
        """
        missing = []
        suggestions = []
        details = {}

        tool_available = True
        permission_granted = True
        service_running = True

        # Check required tool availability
        if strategy.required_tool:
            tool_available = await self._check_tool_available(strategy.required_tool)
            details["tool"] = {
                "name": strategy.required_tool,
                "available": tool_available,
            }
            if not tool_available:
                missing.append(f"tool:{strategy.required_tool}")
                suggestions.append(
                    f"Install or enable the '{strategy.required_tool}' tool"
                )

        # Check required permission
        if strategy.required_permission:
            permission_granted = await self._check_permission_granted(
                strategy.required_permission
            )
            details["permission"] = {
                "name": strategy.required_permission,
                "granted": permission_granted,
            }
            if not permission_granted:
                missing.append(f"permission:{strategy.required_permission}")
                suggestions.append(
                    f"Grant '{strategy.required_permission}' permission in device settings"
                )

        # Check required service
        if strategy.required_service:
            service_running = await self._check_service_running(
                strategy.required_service
            )
            details["service"] = {
                "name": strategy.required_service,
                "running": service_running,
            }
            if not service_running:
                missing.append(f"service:{strategy.required_service}")
                suggestions.append(
                    f"Enable or start the '{strategy.required_service}' service"
                )

        # Check generic prerequisites (legacy support)
        for prereq in strategy.prerequisites:
            prereq_met = await self._check_generic_prerequisite(prereq)
            if not prereq_met:
                missing.append(f"prereq:{prereq}")
                suggestions.append(f"Enable '{prereq}' access")

        satisfied = (
            tool_available
            and permission_granted
            and service_running
            and len(missing) == len([m for m in missing if m.startswith("prereq:")])
        )

        return PrerequisiteResult(
            satisfied=satisfied,
            tool_available=tool_available,
            permission_granted=permission_granted,
            service_running=service_running,
            missing=missing,
            suggestions=suggestions,
            details=details,
        )

    async def _check_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is available in the registry"""
        try:
            registry = self._get_tool_registry()
            if registry is None:
                # No registry available, assume tool exists
                logger.debug(
                    f"Tool registry not available, assuming {tool_name} exists"
                )
                return True

            tool_def = registry.get_tool_definition(tool_name)
            if tool_def is None:
                logger.debug(f"Tool '{tool_name}' not found in registry")
                return False

            # Check if tool has a handler bound
            if tool_def.handler is not None:
                return True

            # Tool defined but no handler - partial availability
            logger.debug(f"Tool '{tool_name}' defined but no handler bound")
            return True  # Consider available if defined

        except Exception as e:
            logger.warning(f"Error checking tool availability: {e}")
            return True  # Fail open for mobile efficiency

    async def _check_permission_granted(self, permission: str) -> bool:
        """Check if a permission is granted"""
        try:
            perm_manager = self._get_permission_manager()
            if perm_manager is None:
                # No permission manager, assume granted
                logger.debug(
                    f"Permission manager not available, assuming {permission} granted"
                )
                return True

            return await perm_manager.check_permission(permission)

        except Exception as e:
            logger.warning(f"Error checking permission: {e}")
            return True  # Fail open

    async def _check_service_running(self, service: str) -> bool:
        """Check if a service is running with caching"""
        # Check cache first (mobile efficiency)
        if service in self._service_cache:
            cached_result, cached_time = self._service_cache[service]
            if (datetime.now() - cached_time).total_seconds() < self._service_cache_ttl:
                return cached_result

        result = await self._check_service_running_impl(service)
        self._service_cache[service] = (result, datetime.now())
        return result

    async def _check_service_running_impl(self, service: str) -> bool:
        """Actually check if a service is running"""
        try:
            # Service-specific checks
            service_checks = {
                "termux": self._check_termux_available,
                "network": self._check_network_available,
                "gps": self._check_gps_available,
                "location_history": self._check_location_history_available,
                "cache": self._check_cache_available,
            }

            check_fn = service_checks.get(service)
            if check_fn:
                return await check_fn()

            # Unknown service - try termux bridge
            bridge = await self._get_termux_bridge()
            if bridge:
                result = await bridge.run_command(["pgrep", "-x", service])
                return result.success

            # Can't check, assume available
            return True

        except Exception as e:
            logger.warning(f"Error checking service {service}: {e}")
            return True

    async def _check_termux_available(self) -> bool:
        """Check if Termux is available"""
        try:
            bridge = await self._get_termux_bridge()
            if bridge:
                return await bridge.check_availability()
            return False
        except:
            return False

    async def _check_network_available(self) -> bool:
        """Check if network is available"""
        try:
            bridge = await self._get_termux_bridge()
            if bridge:
                result = await bridge.run_command(
                    ["ping", "-c", "1", "-W", "2", "8.8.8.8"]
                )
                return result.success
            # Fallback: assume available if no bridge
            return True
        except:
            return True

    async def _check_gps_available(self) -> bool:
        """Check if GPS is available"""
        try:
            bridge = await self._get_termux_bridge()
            if bridge:
                result = await bridge.run_command(["termux-location", "-p", "gps"])
                return result.success
            return False
        except:
            return False

    async def _check_location_history_available(self) -> bool:
        """Check if location history is available"""
        # Check for location history file
        history_path = Path(self._storage_path) / "location_history.json"
        return history_path.exists()

    async def _check_cache_available(self) -> bool:
        """Check if cache service is available"""
        cache_path = Path(self._storage_path).parent / "cache"
        return cache_path.exists()

    async def _check_generic_prerequisite(self, prereq: str) -> bool:
        """Check a generic prerequisite"""
        # Map prerequisites to actual checks
        prereq_map = {
            "gallery_access": "storage",
            "file_access": "storage",
            "messaging_access": "messaging",
            "cache_access": "cache",
            "location_history": "location_history",
        }

        if prereq in prereq_map:
            mapped = prereq_map[prereq]
            # Check as permission or service
            perm_granted = await self._check_permission_granted(mapped)
            if perm_granted:
                return True
            service_running = await self._check_service_running(mapped)
            return service_running

        # Unknown prerequisite - assume met
        return True

    def _get_tool_registry(self):
        """Lazy load tool registry"""
        if self._tool_registry is None:
            try:
                from src.tools.registry import ToolRegistry

                self._tool_registry = ToolRegistry()
            except ImportError:
                logger.debug("Tool registry not available")
                return None
        return self._tool_registry

    def _get_permission_manager(self):
        """Lazy load permission manager"""
        if self._permission_manager is None:
            try:
                from src.security.security import get_permission_manager

                self._permission_manager = get_permission_manager()
            except ImportError:
                logger.debug("Permission manager not available")
                return None
        return self._permission_manager

    async def _get_termux_bridge(self):
        """Lazy load termux bridge"""
        if self._termux_bridge is None:
            try:
                from src.addons.termux_bridge import get_termux_bridge

                self._termux_bridge = await get_termux_bridge()
            except ImportError:
                logger.debug("Termux bridge not available")
                return None
        return self._termux_bridge

    async def _execute_strategy(
        self, strategy: Strategy, user_goal: str, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute a specific strategy"""

        # If strategy has custom function, use it
        if strategy.attempt_fn:
            try:
                return await strategy.attempt_fn(user_goal, context or {})
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed: {e}")
                return {"success": False, "error": str(e)}

        # Otherwise, use built-in strategy handlers
        handlers = {
            "file_search": self._strategy_file_search,
            "chat_history_search": self._strategy_chat_search,
            "metadata_analysis": self._strategy_metadata,
            "cached_data": self._strategy_cached,
            "local_knowledge": self._strategy_local_knowledge,
            "last_known": self._strategy_last_location,
            "manual_confirmation": self._strategy_manual,
            "text_fallback": self._strategy_text_fallback,
        }

        handler = handlers.get(strategy.name)
        if handler:
            return await handler(user_goal, context or {})

        return {"success": False, "error": f"No handler for strategy: {strategy.name}"}

    # =========================================================================
    # GAP HISTORY TRACKING
    # =========================================================================

    def _load_gap_history(self):
        """Load persistent gap history from storage"""
        os.makedirs(self._storage_path, exist_ok=True)
        history_path = os.path.join(self._storage_path, "gap_history.json")

        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    data = json.load(f)
                    self._persistent_gap_history = [
                        GapHistoryEntry(**entry) for entry in data.get("history", [])
                    ]
                    self._gap_frequency = defaultdict(list, data.get("frequency", {}))
                    self._successful_patterns = defaultdict(
                        int, data.get("patterns", {})
                    )
            except Exception as e:
                logger.warning(f"Failed to load gap history: {e}")
                self._persistent_gap_history = []
                self._gap_frequency = defaultdict(list)

    def _save_gap_history(self):
        """Save gap history to persistent storage"""
        try:
            os.makedirs(self._storage_path, exist_ok=True)
            history_path = os.path.join(self._storage_path, "gap_history.json")

            # Prune old entries
            cutoff = datetime.now() - timedelta(days=GAP_HISTORY_RETENTION_DAYS)
            self._persistent_gap_history = [
                entry
                for entry in self._persistent_gap_history
                if datetime.fromisoformat(entry.timestamp) > cutoff
            ][-MAX_GAP_HISTORY_ENTRIES:]

            # Prune old frequency entries
            cutoff_str = cutoff.isoformat()
            for cap in list(self._gap_frequency.keys()):
                self._gap_frequency[cap] = [
                    ts for ts in self._gap_frequency[cap] if ts > cutoff_str
                ]
                if not self._gap_frequency[cap]:
                    del self._gap_frequency[cap]

            data = {
                "history": [
                    {
                        "capability": entry.capability,
                        "user_goal": entry.user_goal,
                        "timestamp": entry.timestamp,
                        "resolved": entry.resolved,
                        "strategy_used": entry.strategy_used,
                        "attempts": entry.attempts,
                        "error": entry.error,
                    }
                    for entry in self._persistent_gap_history
                ],
                "frequency": dict(self._gap_frequency),
                "patterns": dict(self._successful_patterns),
            }

            with open(history_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save gap history: {e}")

    def _record_gap_history(
        self,
        capability: CapabilityType,
        user_goal: str,
        resolved: bool,
        strategy_used: Optional[str],
        attempts: int,
        error: Optional[str] = None,
    ):
        """Record a gap occurrence in history"""
        entry = GapHistoryEntry(
            capability=capability.value,
            user_goal=user_goal,
            timestamp=datetime.now().isoformat(),
            resolved=resolved,
            strategy_used=strategy_used,
            attempts=attempts,
            error=error,
        )

        self._persistent_gap_history.append(entry)
        self._gap_frequency[capability.value].append(entry.timestamp)

        # Save periodically (every 10 entries)
        if len(self._persistent_gap_history) % 10 == 0:
            self._save_gap_history()

    def _check_repeated_gap(self, capability: CapabilityType) -> List[str]:
        """
        Check if this gap has been hit repeatedly and return proactive suggestions

        After N repeated failures for the same capability, suggest solutions
        """
        cap_value = capability.value
        recent_timestamps = self._gap_frequency.get(cap_value, [])

        # Count failures in last 24 hours
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        recent_failures = [ts for ts in recent_timestamps if ts > cutoff]

        if len(recent_failures) >= REPEATED_GAP_THRESHOLD:
            # Get suggestions based on capability type
            suggestions = self._get_proactive_suggestions(capability)
            logger.warning(
                f"Repeated gap for {cap_value}: {len(recent_failures)} times in 24h"
            )
            return suggestions

        return []

    def _get_proactive_suggestions(self, capability: CapabilityType) -> List[str]:
        """Get proactive suggestions for a capability gap"""
        suggestions_map = {
            CapabilityType.IMAGE_RECOGNITION: [
                "Install a local image recognition model (e.g., MobileNet)",
                "Enable cloud vision API when network is available",
                "Grant storage permission for better file search",
            ],
            CapabilityType.SPEECH_RECOGNITION: [
                "Install Vosk offline speech recognition",
                "Check microphone permissions in Termux",
                "Consider using text input as primary mode",
            ],
            CapabilityType.WEB_ACCESS: [
                "Cache frequently accessed web content when online",
                "Enable offline mode with pre-downloaded knowledge",
                "Check network connectivity settings",
            ],
            CapabilityType.LOCATION_ACCESS: [
                "Enable GPS in device settings",
                "Grant location permission to Termux",
                "Set a default/home location for offline use",
            ],
            CapabilityType.CAMERA_ACCESS: [
                "Grant camera permission to Termux",
                "Install termux-api package for camera access",
                "Use existing photos as alternative",
            ],
        }

        return suggestions_map.get(
            capability,
            [
                "Check system permissions",
                "Verify required tools are installed",
                "Try alternative approaches",
            ],
        )

    def get_gap_statistics(self) -> Dict[str, Any]:
        """Get statistics about capability gaps"""
        stats = {
            "total_gaps": len(self._persistent_gap_history),
            "resolved_count": sum(
                1 for e in self._persistent_gap_history if e.resolved
            ),
            "unresolved_count": sum(
                1 for e in self._persistent_gap_history if not e.resolved
            ),
            "by_capability": {},
            "most_successful_strategies": dict(
                sorted(self._successful_patterns.items(), key=lambda x: -x[1])[:10]
            ),
        }

        # Count by capability
        for entry in self._persistent_gap_history:
            cap = entry.capability
            if cap not in stats["by_capability"]:
                stats["by_capability"][cap] = {"total": 0, "resolved": 0}
            stats["by_capability"][cap]["total"] += 1
            if entry.resolved:
                stats["by_capability"][cap]["resolved"] += 1

        return stats

    # =========================================================================
    # STRATEGY IMPLEMENTATIONS
    # =========================================================================

    async def _strategy_file_search(
        self, user_goal: str, context: Dict
    ) -> Dict[str, Any]:
        """Search gallery for files matching user goal"""
        try:
            bridge = await self._get_termux_bridge()
            if not bridge:
                return {
                    "success": False,
                    "error": "Termux bridge not available",
                }

            # Extract search terms from user goal
            search_terms = self._extract_search_terms(user_goal)
            search_dir = context.get("search_directory", "/sdcard/DCIM")

            all_results = []
            for term in search_terms[:3]:  # Limit to 3 terms for efficiency
                files = await bridge.fs.find_files(search_dir, f"*{term}*")
                all_results.extend(files[:10])  # Limit results per term

            if all_results:
                return {
                    "success": True,
                    "partial": True,
                    "data": all_results[:20],  # Return top 20
                    "description": f"Found {len(all_results)} files matching search",
                }

            return {
                "success": False,
                "error": "No matching files found",
            }

        except Exception as e:
            logger.error(f"File search strategy failed: {e}")
            return {"success": False, "error": str(e)}

    async def _strategy_chat_search(
        self, user_goal: str, context: Dict
    ) -> Dict[str, Any]:
        """Search chat apps for relevant content"""
        try:
            bridge = await self._get_termux_bridge()
            if not bridge:
                return {"success": False, "error": "Termux bridge not available"}

            # Search common chat app directories
            chat_dirs = [
                "/sdcard/WhatsApp/Media",
                "/sdcard/Telegram",
                "/sdcard/Android/media/com.whatsapp/WhatsApp/Media",
            ]

            search_terms = self._extract_search_terms(user_goal)
            all_results = []

            for chat_dir in chat_dirs:
                for term in search_terms[:2]:
                    try:
                        files = await bridge.fs.find_files(chat_dir, f"*{term}*")
                        all_results.extend(files[:5])
                    except:
                        continue

            if all_results:
                return {
                    "success": True,
                    "partial": True,
                    "data": all_results[:15],
                    "description": "Found related content in messaging apps",
                }

            return {"success": False, "error": "No matching content in chat apps"}

        except Exception as e:
            logger.error(f"Chat search strategy failed: {e}")
            return {"success": False, "error": str(e)}

    async def _strategy_metadata(self, user_goal: str, context: Dict) -> Dict[str, Any]:
        """Analyze file metadata"""
        try:
            file_path = context.get("file_path")
            if not file_path:
                return {"success": False, "error": "No file path provided"}

            bridge = await self._get_termux_bridge()
            if not bridge:
                return {"success": False, "error": "Termux bridge not available"}

            # Get file metadata using stat and exiftool if available
            result = await bridge.run_command(["stat", file_path])
            metadata = {"stat": result.stdout if result.success else None}

            # Try exiftool for image metadata
            exif_result = await bridge.run_command(["exiftool", file_path])
            if exif_result.success:
                metadata["exif"] = exif_result.stdout

            if metadata.get("stat") or metadata.get("exif"):
                return {
                    "success": True,
                    "partial": True,
                    "data": metadata,
                    "description": "Extracted information from file metadata",
                }

            return {"success": False, "error": "Could not extract metadata"}

        except Exception as e:
            logger.error(f"Metadata strategy failed: {e}")
            return {"success": False, "error": str(e)}

    async def _strategy_cached(self, user_goal: str, context: Dict) -> Dict[str, Any]:
        """Use cached web data"""
        try:
            cache_path = Path(self._storage_path).parent / "cache" / "web"
            if not cache_path.exists():
                return {"success": False, "error": "Cache directory not found"}

            # Search cache for relevant content
            search_terms = self._extract_search_terms(user_goal)
            cache_files = list(cache_path.glob("*.json")) + list(
                cache_path.glob("*.txt")
            )

            relevant_cached = []
            for cache_file in cache_files[:50]:  # Limit search
                try:
                    content = cache_file.read_text(errors="ignore")
                    for term in search_terms:
                        if term.lower() in content.lower():
                            relevant_cached.append(
                                {
                                    "file": str(cache_file),
                                    "matched_term": term,
                                    "preview": content[:200],
                                }
                            )
                            break
                except:
                    continue

            if relevant_cached:
                return {
                    "success": True,
                    "partial": True,
                    "data": relevant_cached[:10],
                    "description": "Found relevant cached web content",
                }

            return {"success": False, "error": "No relevant cached content"}

        except Exception as e:
            logger.error(f"Cache strategy failed: {e}")
            return {"success": False, "error": str(e)}

    async def _strategy_local_knowledge(
        self, user_goal: str, context: Dict
    ) -> Dict[str, Any]:
        """Use AURA's built knowledge"""
        # This strategy always "succeeds" as a fallback
        # The actual response quality depends on the LLM
        return {
            "success": True,
            "data": {
                "type": "knowledge_query",
                "query": user_goal,
                "source": "local_model",
            },
            "description": "Will answer using trained knowledge base",
        }

    async def _strategy_last_location(
        self, user_goal: str, context: Dict
    ) -> Dict[str, Any]:
        """Use last known location"""
        try:
            history_path = Path(self._storage_path) / "location_history.json"

            if history_path.exists():
                data = json.loads(history_path.read_text())
                if data.get("last_location"):
                    return {
                        "success": True,
                        "partial": True,
                        "data": data["last_location"],
                        "description": f"Using location from {data.get('timestamp', 'unknown')}",
                    }

            return {"success": False, "error": "No location history available"}

        except Exception as e:
            logger.error(f"Last location strategy failed: {e}")
            return {"success": False, "error": str(e)}

    async def _strategy_manual(self, user_goal: str, context: Dict) -> Dict[str, Any]:
        """Ask user for manual input"""
        return {
            "success": True,
            "data": {
                "type": "user_input_required",
                "prompt": f"Please provide: {user_goal}",
                "input_type": context.get("input_type", "text"),
            },
            "description": "Requesting user input to proceed",
        }

    async def _strategy_text_fallback(
        self, user_goal: str, context: Dict
    ) -> Dict[str, Any]:
        """Offer text input instead of voice"""
        return {
            "success": True,
            "data": {
                "type": "fallback_mode",
                "original_mode": "voice",
                "fallback_mode": "text",
                "message": "Please type your request instead of speaking",
            },
            "description": "Switched to text input mode",
        }

    def _extract_search_terms(self, user_goal: str) -> List[str]:
        """Extract search terms from user goal"""
        # Remove common words and extract meaningful terms
        stop_words = {
            "find",
            "search",
            "look",
            "for",
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "from",
            "with",
            "my",
            "me",
            "i",
            "want",
            "need",
            "please",
            "can",
            "you",
            "help",
            "show",
            "get",
            "where",
            "is",
            "are",
            "was",
        }

        words = user_goal.lower().split()
        terms = [w for w in words if w not in stop_words and len(w) > 2]

        return terms[:5]  # Return top 5 terms

    # =========================================================================
    # LEARNING
    # =========================================================================

    def _record_success(self, strategy_name: str, capability: CapabilityType):
        """Record successful strategy"""
        key = f"{capability.value}:{strategy_name}"
        self._successful_patterns[key] += 1
        logger.info(f"Recorded success: {key}")

    def _record_failure(self, strategies: List[Strategy], capability: CapabilityType):
        """Record failed attempts"""
        logger.warning(f"All strategies failed for {capability.value}")

    def get_successful_patterns(self) -> Dict[str, int]:
        """Get learned successful patterns"""
        return dict(self._successful_patterns)

    def suggest_alternatives(self, capability: CapabilityType) -> List[str]:
        """Suggest alternative approaches for a capability"""
        strategies = self._strategies.get(capability, [])
        return [s.name for s in strategies[:3]]  # Top 3


# Global instance
_gap_handler: Optional[CapabilityGapHandler] = None


def get_capability_gap_handler() -> CapabilityGapHandler:
    """Get or create capability gap handler"""
    global _gap_handler
    if _gap_handler is None:
        _gap_handler = CapabilityGapHandler()
    return _gap_handler
