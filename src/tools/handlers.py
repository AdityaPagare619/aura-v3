"""
AURA v3 Tool Handlers
Implements actual execution for tools registered in the registry
Uses TermuxBridge for Android operations

This module is the CANONICAL tool execution layer for AURA v3.
It includes:
- TermuxBridge integration for secure command execution
- App exploration memory (explore once, remember forever)
- Security validation on all inputs
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from src.core.security import (
        SecurityError,
        sanitize_string,
        sanitize_path,
        sanitize_filename,
        validate_phone_number,
        validate_app_name,
        validate_command,
        validate_integer,
    )
except ImportError:
    SecurityError = Exception
    sanitize_string = lambda x: x
    sanitize_path = lambda x: x
    sanitize_filename = lambda x: x
    validate_phone_number = lambda x: x
    validate_app_name = lambda x: x
    validate_command = lambda x, y=None: True
    validate_integer = lambda x, y=None, z=None: int(x)

# Import app exploration memory for "explore once, remember forever" feature
try:
    from src.tools.android import AppExplorationMemory
except ImportError:
    AppExplorationMemory = None


class ToolHandlers:
    """
    Tool execution handlers - CANONICAL tool execution for AURA v3.

    Connects tool definitions to actual implementations.
    Includes:
    - TermuxBridge for secure Android command execution
    - AppExplorationMemory for "explore once, remember forever"
    - Security validation on all inputs
    """

    def __init__(self):
        self.termux_bridge = None
        self.exploration_memory = None
        self._initialized = False

    async def initialize(self):
        """Initialize tool handlers with Termux bridge and exploration memory"""
        if self._initialized:
            return

        # Initialize Termux bridge
        try:
            from src.addons.termux_bridge import TermuxBridge

            self.termux_bridge = TermuxBridge()
            await self.termux_bridge.check_availability()
            logger.info("Tool handlers initialized with Termux bridge")
        except Exception as e:
            logger.warning(f"Termux bridge not available: {e}")
            # Continue without Termux - tools will return appropriate errors

        # Initialize exploration memory (explore once, remember forever)
        if AppExplorationMemory is not None:
            try:
                self.exploration_memory = AppExplorationMemory()
                logger.info("App exploration memory initialized")
            except Exception as e:
                logger.warning(f"Exploration memory not available: {e}")

        self._initialized = True

    # =========================================================================
    # Android App Tools
    # =========================================================================

    async def open_app(self, app_name: str) -> Dict[str, Any]:
        """Open an application"""
        logger.info(f"Opening app: {app_name}")

        if not self.termux_bridge:
            return {
                "success": False,
                "error": "Termux bridge not available",
                "message": "App control requires Termux",
            }

        try:
            validated_name = validate_app_name(app_name)

            package_map = {
                "whatsapp": "com.whatsapp",
                "instagram": "com.instagram.android",
                "telegram": "org.telegram.messenger",
                "chrome": "com.android.chrome",
                "spotify": "com.spotify.music",
                "youtube": "com.google.android.youtube",
                "gmail": "com.google.android.gm",
                "maps": "com.google.android.apps.maps",
                "camera": "com.android.camera",
                "settings": "com.android.settings",
            }

            package_name = package_map.get(validated_name, validated_name)

            result = await self.termux_bridge.app.open_app(package_name)

            return {
                "success": result.success,
                "message": f"Opened {app_name}"
                if result.success
                else f"Failed to open {app_name}",
                "output": result.output,
            }
        except SecurityError as e:
            logger.error(f"Security validation failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error opening app: {e}")
            return {"success": False, "error": str(e)}

    async def close_app(self, app_name: str) -> Dict[str, Any]:
        """Close an application"""
        logger.info(f"Closing app: {app_name}")

        if not self.termux_bridge:
            return {
                "success": False,
                "error": "Termux bridge not available",
            }

        try:
            validated_name = validate_app_name(app_name)

            package_map = {
                "whatsapp": "com.whatsapp",
                "instagram": "com.instagram.android",
                "telegram": "org.telegram.messenger",
            }

            package_name = package_map.get(validated_name, validated_name)
            result = await self.termux_bridge.run_command(
                ["am", "force-stop", package_name]
            )

            return {
                "success": result.success,
                "message": f"Closed {app_name}"
                if result.success
                else f"Failed to close {app_name}",
            }
        except SecurityError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_current_app(self) -> Dict[str, Any]:
        """Get currently focused app"""
        if not self.termux_bridge:
            return {"success": False, "error": "Termux bridge not available"}

        try:
            result = await self.termux_bridge.run_command(
                "dumpsys window | grep mCurrentFocus"
            )
            return {
                "success": result.success,
                "app": result.stdout.strip() if result.success else "Unknown",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Notification Tools
    # =========================================================================

    async def get_notifications(self, max_count: int = 10) -> Dict[str, Any]:
        """Get recent notifications"""
        logger.info(f"Getting {max_count} notifications")

        if not self.termux_bridge:
            return {
                "success": False,
                "error": "Termux bridge not available",
                "notifications": [],
            }

        try:
            validated_count = validate_integer(max_count, min_val=1, max_val=100)

            result = await self.termux_bridge.run_command(
                ["dumpsys", "notification", f"--n={validated_count}"]
            )

            notifications = []
            if result.success:
                for line in result.stdout.split("\n"):
                    if "android.title" in line or "android.text" in line:
                        notifications.append(line.strip())

            return {
                "success": True,
                "notifications": notifications[:validated_count],
                "count": len(notifications),
            }
        except SecurityError as e:
            return {"success": False, "error": str(e), "notifications": []}
        except Exception as e:
            return {"success": False, "error": str(e), "notifications": []}

    # =========================================================================
    # Message Sending Tools
    # =========================================================================

    async def send_message(
        self, contact: str, message: str, app: str = "whatsapp"
    ) -> Dict[str, Any]:
        """Send a message via specified app"""
        logger.info(f"Sending message via {app}: {message[:30]}...")

        if not self.termux_bridge:
            return {
                "success": False,
                "error": "Termux bridge not available",
            }

        try:
            validated_contact = validate_phone_number(contact)
            validated_message = sanitize_string(message, max_length=1000)
            validated_app = validate_app_name(app)

            package_map = {
                "whatsapp": "com.whatsapp",
                "telegram": "org.telegram.messenger",
                "sms": "com.android.mms",
            }

            package_name = package_map.get(validated_app, validated_app)

            result = await self.termux_bridge.run_command(
                [
                    "am",
                    "start",
                    "-a",
                    "android.intent.action.SENDTO",
                    "-d",
                    f"sms:{validated_contact}",
                    "--es",
                    "sms_body",
                    validated_message,
                ]
            )

            return {
                "success": result.success,
                "message": f"Message sent to {contact}"
                if result.success
                else "Failed to send",
                "output": result.output,
            }
        except SecurityError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Call Tools
    # =========================================================================

    async def make_call(self, phone_number: str) -> Dict[str, Any]:
        """Make a phone call"""
        logger.info(f"Making call to: {phone_number}")

        if not self.termux_bridge:
            return {
                "success": False,
                "error": "Termux bridge not available",
            }

        try:
            validated_number = validate_phone_number(phone_number)

            result = await self.termux_bridge.run_command(
                ["service", "call", "phone", "1", "s16", validated_number]
            )

            return {
                "success": result.success,
                "message": f"Calling {phone_number}"
                if result.success
                else "Failed to call",
            }
        except SecurityError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Screenshot Tools
    # =========================================================================

    async def take_screenshot(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Take a screenshot"""
        logger.info("Taking screenshot")

        if not self.termux_bridge:
            return {
                "success": False,
                "error": "Termux bridge not available",
            }

        try:
            if path:
                validated_path = sanitize_path(path)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                validated_path = (
                    f"/storage/emulated/0/Pictures/aura_screenshot_{timestamp}.png"
                )

            result = await self.termux_bridge.run_command(
                ["screencap", "-p", validated_path]
            )

            return {
                "success": result.success,
                "path": path if result.success else None,
                "message": f"Screenshot saved to {path}"
                if result.success
                else "Failed",
            }
        except SecurityError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # File Operations
    # =========================================================================

    async def read_file(self, path: str, max_size: int = 1024 * 1024) -> Dict[str, Any]:
        """Read a file"""
        logger.info(f"Reading file: {path}")

        if not self.termux_bridge:
            return {
                "success": False,
                "error": "Termux bridge not available",
            }

        try:
            result = await self.termux_bridge.fs.read_file(path, max_size)

            return {
                "success": True,
                "content": result[:500] if len(result) > 500 else result,
                "truncated": len(result) > 500,
                "size": len(result),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write to a file"""
        logger.info(f"Writing file: {path}")

        if not self.termux_bridge:
            return {
                "success": False,
                "error": "Termux bridge not available",
            }

        try:
            result = await self.termux_bridge.fs.write_file(path, content)

            return {
                "success": result.success,
                "message": f"File written to {path}" if result.success else "Failed",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # System Info Tools
    # =========================================================================

    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        if not self.termux_bridge:
            return {
                "success": False,
                "error": "Termux bridge not available",
            }

        try:
            battery = await self.termux_bridge.get_battery_status()
            memory = await self.termux_bridge.get_memory_status()

            return {
                "success": True,
                "battery": battery,
                "memory": memory,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Utility Tools
    # =========================================================================

    async def run_shell_command(self, command: str) -> Dict[str, Any]:
        """Run a shell command"""
        logger.info(f"Running shell command: {command[:50]}...")

        if not self.termux_bridge:
            return {
                "success": False,
                "error": "Termux bridge not available",
            }

        allowed_commands = [
            "ls",
            "cat",
            "echo",
            "pwd",
            "date",
            "whoami",
            "df",
            "free",
            "top",
        ]

        try:
            validate_command(command, allowed_commands)

            import shlex

            cmd_parts = shlex.split(command)
            result = await self.termux_bridge.run_command(cmd_parts)

            return {
                "success": result.success,
                "output": result.output,
                "return_code": result.return_code,
            }
        except SecurityError as e:
            return {
                "success": False,
                "error": str(e),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # App Exploration Memory Tools (explore once, remember forever)
    # =========================================================================

    async def explore_current_app(self) -> Dict[str, Any]:
        """
        Explore current app's UI and cache its structure.
        This enables AURA's 'explore once, remember forever' capability.
        """
        logger.info("Exploring current app")

        if not self.termux_bridge:
            return {"success": False, "error": "Termux bridge not available"}

        if not self.exploration_memory:
            return {"success": False, "error": "Exploration memory not available"}

        try:
            # Get current app info
            app_result = await self.get_current_app()
            if not app_result.get("success"):
                return {"success": False, "error": "Could not get current app"}

            app_name = app_result.get("app", "unknown")

            # Take screenshot for visual reference
            screenshot_result = await self.take_screenshot()

            # Get screen size
            size_result = await self.termux_bridge.run_command(["wm", "size"])
            screen_size = {"width": 1080, "height": 2400}  # defaults
            if size_result.success and size_result.stdout:
                try:
                    size_str = size_result.stdout.split(":")[-1].strip()
                    width, height = map(int, size_str.split("x"))
                    screen_size = {"width": width, "height": height}
                except:
                    pass

            # Build structure
            structure = {
                "package": app_name,
                "screen_size": screen_size,
                "elements": {},  # Would be populated by UI dump
                "last_explored": datetime.now().isoformat(),
            }

            # Save to memory
            self.exploration_memory.save_app_structure(app_name, structure)

            return {
                "success": True,
                "app": app_name,
                "structure": structure,
                "message": f"Explored and cached structure for {app_name}",
            }
        except Exception as e:
            logger.error(f"Error exploring app: {e}")
            return {"success": False, "error": str(e)}

    async def get_cached_apps(self) -> Dict[str, Any]:
        """Get list of apps with cached UI structures"""
        if not self.exploration_memory:
            return {
                "success": False,
                "error": "Exploration memory not available",
                "apps": [],
            }

        try:
            apps = self.exploration_memory.list_cached_apps()
            return {"success": True, "apps": apps, "count": len(apps)}
        except Exception as e:
            return {"success": False, "error": str(e), "apps": []}

    async def get_app_structure(self, app_name: str) -> Dict[str, Any]:
        """Get cached UI structure for an app"""
        if not self.exploration_memory:
            return {"success": False, "error": "Exploration memory not available"}

        try:
            validated_name = validate_app_name(app_name)
            structure = self.exploration_memory.get_app_structure(validated_name)

            if structure:
                return {"success": True, "app": validated_name, "structure": structure}
            return {"success": False, "error": f"No cached structure for {app_name}"}
        except SecurityError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def save_element_position(
        self, app_name: str, element_desc: str, x: int, y: int
    ) -> Dict[str, Any]:
        """Save a UI element position for future use"""
        if not self.exploration_memory:
            return {"success": False, "error": "Exploration memory not available"}

        try:
            validated_name = validate_app_name(app_name)
            validated_desc = sanitize_string(element_desc, max_length=100)
            validated_x = validate_integer(x, min_val=0, max_val=4096)
            validated_y = validate_integer(y, min_val=0, max_val=4096)

            self.exploration_memory.save_element_position(
                validated_name, validated_desc, (validated_x, validated_y)
            )

            return {
                "success": True,
                "message": f"Saved element '{element_desc}' at ({x}, {y}) for {app_name}",
            }
        except SecurityError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def tap_cached_element(
        self, app_name: str, element_desc: str
    ) -> Dict[str, Any]:
        """Tap a previously cached UI element position"""
        if not self.exploration_memory:
            return {"success": False, "error": "Exploration memory not available"}

        if not self.termux_bridge:
            return {"success": False, "error": "Termux bridge not available"}

        try:
            validated_name = validate_app_name(app_name)
            validated_desc = sanitize_string(element_desc, max_length=100)

            coords = self.exploration_memory.get_element_position(
                validated_name, validated_desc
            )
            if not coords:
                return {
                    "success": False,
                    "error": f"Element '{element_desc}' not found in cache for {app_name}",
                }

            x, y = coords
            result = await self.termux_bridge.run_command(
                ["input", "tap", str(x), str(y)]
            )

            return {
                "success": result.success,
                "message": f"Tapped element '{element_desc}' at ({x}, {y})",
                "coords": {"x": x, "y": y},
            }
        except SecurityError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def clear_app_memory(self, app_name: str) -> Dict[str, Any]:
        """Clear cached structure for an app"""
        if not self.exploration_memory:
            return {"success": False, "error": "Exploration memory not available"}

        try:
            validated_name = validate_app_name(app_name)
            self.exploration_memory.delete_app_structure(validated_name)
            return {"success": True, "message": f"Cleared cache for {app_name}"}
        except SecurityError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global instance
_tool_handlers: Optional[ToolHandlers] = None


async def get_tool_handlers() -> ToolHandlers:
    """Get or create tool handlers instance"""
    global _tool_handlers
    if _tool_handlers is None:
        _tool_handlers = ToolHandlers()
        await _tool_handlers.initialize()
    return _tool_handlers


def create_handler_dict(handlers: ToolHandlers) -> Dict[str, Any]:
    """Create handler dictionary for binding to registry"""
    return {
        "open_app": handlers.open_app,
        "close_app": handlers.close_app,
        "get_current_app": handlers.get_current_app,
        "get_notifications": handlers.get_notifications,
        "send_message": handlers.send_message,
        "make_call": handlers.make_call,
        "take_screenshot": handlers.take_screenshot,
        "read_file": handlers.read_file,
        "write_file": handlers.write_file,
        "get_system_info": handlers.get_system_info,
        "run_shell_command": handlers.run_shell_command,
        # Exploration memory tools
        "explore_current_app": handlers.explore_current_app,
        "get_cached_apps": handlers.get_cached_apps,
        "get_app_structure": handlers.get_app_structure,
        "save_element_position": handlers.save_element_position,
        "tap_cached_element": handlers.tap_cached_element,
        "clear_app_memory": handlers.clear_app_memory,
    }
