"""
AURA v3 Termux Bridge
Android device control through Termux

Provides shell command execution, app control, and system access
on Android devices running Termux

Key differences from OpenClaw:
- Mobile-constrained (battery, RAM, network)
- Termux-specific APIs
- Offline-first operation
"""

import asyncio
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TermuxError(Exception):
    """Termux-specific errors"""

    pass


class CommandResult:
    """Result of a Termux command execution"""

    def __init__(
        self, success: bool, stdout: str = "", stderr: str = "", return_code: int = 0
    ):
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        self.timestamp = datetime.now()

    @property
    def output(self) -> str:
        return self.stdout or self.stderr

    def __str__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return f"[{status}] {self.output[:200]}"


class AppControl:
    """App control operations on Android"""

    def __init__(self, bridge: "TermuxBridge"):
        self.bridge = bridge

    async def list_apps(
        self, package_pattern: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """List installed apps"""
        if package_pattern:
            cmd = ["pm", "list", "packages", package_pattern]
        else:
            cmd = ["pm", "list", "packages"]

        result = await self.bridge.run_command(cmd)

        if not result.success:
            return []

        apps = []
        for line in result.stdout.strip().split("\n"):
            if line.startswith("package:"):
                pkg = line.replace("package:", "")
                apps.append({"package": pkg, "name": pkg.split(".")[-1]})
        return apps

    async def open_app(self, package_name: str) -> CommandResult:
        """Open an app by package name"""
        cmd = ["am", "start", "-n", f"{package_name}/.MainActivity"]
        result = await self.bridge.run_command(cmd)

        if not result.success:
            cmd = [
                "am",
                "start",
                "-n",
                f"{package_name}/com.{package_name.split('.')[-1]}.MainActivity",
            ]
            result = await self.bridge.run_command(cmd)

        return result

    async def get_app_info(self, package_name: str) -> Dict[str, Any]:
        """Get app information"""
        cmd = ["dumpsys", "package", package_name]
        result = await self.bridge.run_command(cmd)

        if not result.success:
            return {}

        info = {"package": package_name}
        for line in result.stdout.split("\n"):
            if "versionName" in line:
                info["version"] = line.split("=")[-1].strip()
            elif "firstInstallTime" in line:
                info["installed"] = line.split("=")[-1].strip()

        return info


class FileSystem:
    """File system operations"""

    def __init__(self, bridge: "TermuxBridge"):
        self.bridge = bridge

    async def list_directory(self, path: str) -> List[Dict[str, Any]]:
        """List directory contents"""
        cmd = ["ls", "-la", path]
        result = await self.bridge.run_command(cmd)

        if not result.success:
            return []

        items = []
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split()
            if len(parts) >= 9:
                items.append(
                    {
                        "permissions": parts[0],
                        "size": parts[4],
                        "date": " ".join(parts[5:8]),
                        "name": " ".join(parts[8:]),
                    }
                )
        return items

    async def find_files(self, directory: str, pattern: str) -> List[str]:
        """Find files matching pattern"""
        cmd = ["find", directory, "-name", pattern]
        result = await self.bridge.run_command(cmd)

        if not result.success:
            return []

        return [f for f in result.stdout.strip().split("\n") if f]

    async def read_file(self, path: str, max_size: int = 1024 * 1024) -> str:
        """Read file contents (with size limit for mobile)"""
        size_cmd = ["stat", "-c", "%s", path]
        size_result = await self.bridge.run_command(size_cmd)

        if size_result.success:
            try:
                size = int(size_result.stdout.strip())
                if size > max_size:
                    return f"[File too large: {size} bytes, limit: {max_size}]"
            except:
                pass

        cmd = ["cat", path]
        result = await self.bridge.run_command(cmd)
        return result.output if result.success else result.stderr

    async def search_content(self, path: str, query: str) -> List[str]:
        """Search file contents"""
        cmd = ["grep", "-r", query, path]
        result = await self.bridge.run_command(cmd)

        if not result.success:
            return []

        return [line for line in result.stdout.strip().split("\n") if line]


class MediaControl:
    """Media and camera control"""

    def __init__(self, bridge: "TermuxBridge"):
        self.bridge = bridge

    async def take_photo(self, save_path: str = None) -> CommandResult:
        """Take a photo using camera"""
        if not save_path:
            save_path = "/sdcard/DCIM/AURA/aura_photo.jpg"

        cmd = ["termux-camera-photo", "-c", "0", save_path]
        result = await self.bridge.run_command(cmd)
        return result

    async def list_images(self, directory: str = "/sdcard/DCIM") -> List[str]:
        """List images in directory"""
        cmd = [
            "find",
            directory,
            "-type",
            "f",
            "-name",
            "*.jpg",
            "-o",
            "-name",
            "*.png",
        ]
        result = await self.bridge.run_command(cmd)

        if not result.success:
            return []

        return [f for f in result.stdout.strip().split("\n") if f]

    async def get_location(self) -> Optional[Dict[str, float]]:
        """Get current GPS location"""
        cmd = ["termux-location"]
        result = await self.bridge.run_command(cmd)

        if not result.success:
            return None

        try:
            import json

            return json.loads(result.stdout)
        except:
            return None


class NotificationControl:
    """Android notification control"""

    def __init__(self, bridge: "TermuxBridge"):
        self.bridge = bridge

    async def send_notification(
        self, title: str, content: str, urgency: str = "normal"
    ) -> CommandResult:
        """Send a notification"""
        cmd = ["termux-notification", "-t", title, "-c", content, "--urgency", urgency]
        return await self.bridge.run_command(cmd)

    async def vibrate(self, duration: int = 500) -> CommandResult:
        """Vibrate the device"""
        cmd = ["termux-vibrate", "-d", str(duration)]
        return await self.bridge.run_command(cmd)


class TermuxBridge:
    """
    Main bridge for Termux operations on Android

    This is AURA's interface to control the Android device

    Architecture:
    - Command execution with timeout limits (mobile constraint)
    - Resource-aware operations
    - Offline-capable where possible
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._is_available = None

        # Sub-controllers
        self.apps = AppControl(self)
        self.fs = FileSystem(self)
        self.media = MediaControl(self)
        self.notifications = NotificationControl(self)

    async def check_availability(self) -> bool:
        """Check if Termux is available"""
        if self._is_available is not None:
            return self._is_available

        result = await self.run_command(["echo", "AURA_TERMUX_TEST"])
        self._is_available = result.success and "AURA_TERMUX_TEST" in result.stdout
        return self._is_available

    async def run_command(
        self, command: Union[str, List[str]], timeout: Optional[int] = None
    ) -> CommandResult:
        """Execute a Termux command"""
        if timeout is None:
            timeout = self.timeout

        try:
            if isinstance(command, list):
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )

                success = process.returncode == 0
                return CommandResult(
                    success=success,
                    stdout=stdout.decode("utf-8", errors="ignore"),
                    stderr=stderr.decode("utf-8", errors="ignore"),
                    return_code=process.returncode or 0,
                )

            except asyncio.TimeoutError:
                process.kill()
                return CommandResult(
                    success=False, stderr=f"Command timed out after {timeout}s"
                )

        except Exception as e:
            logger.error(f"Termux command error: {e}")
            return CommandResult(success=False, stderr=str(e))

    async def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        info = {}

        result = await self.run_command(["getprop", "ro.product.model"])
        info["model"] = result.stdout.strip() if result.success else "Unknown"

        result = await self.run_command(["getprop", "ro.build.version.release"])
        info["android_version"] = result.stdout.strip() if result.success else "Unknown"

        result = await self.run_command(["termux-battery-status"])
        if result.success:
            try:
                import json

                info["battery"] = json.loads(result.stdout)
            except:
                pass

        result = await self.run_command(["df", "-h", "/storage/emulated/0"])
        if result.success:
            info["storage"] = result.stdout.strip()

        return info

    async def execute_tool(
        self, tool_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool through Termux

        This is the main interface for the agent loop to use tools
        """
        tool_handlers = {
            "open_app": lambda: self.apps.open_app(params.get("package_name", "")),
            "list_apps": lambda: self.apps.list_apps(params.get("pattern")),
            "list_images": lambda: self.media.list_images(
                params.get("directory", "/sdcard/DCIM")
            ),
            "take_photo": lambda: self.media.take_photo(params.get("save_path")),
            "get_location": lambda: self.media.get_location(),
            "send_notification": lambda: self.notifications.send_notification(
                params.get("title", "AURA"),
                params.get("content", ""),
                params.get("urgency", "normal"),
            ),
            "vibrate": lambda: self.notifications.vibrate(params.get("duration", 500)),
            "read_file": lambda: self.fs.read_file(params.get("path", "")),
            "find_files": lambda: self.fs.find_files(
                params.get("directory", "/sdcard"), params.get("pattern", "*")
            ),
            "search_content": lambda: self.fs.search_content(
                params.get("path", "/sdcard"), params.get("query", "")
            ),
        }

        handler = tool_handlers.get(tool_name)
        if not handler:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        try:
            result = await handler()
            if isinstance(result, CommandResult):
                return {
                    "success": result.success,
                    "output": result.output,
                    "return_code": result.return_code,
                }
            return result
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"success": False, "error": str(e)}


# Global instance
_termux_bridge: Optional[TermuxBridge] = None


async def get_termux_bridge() -> TermuxBridge:
    """Get or create Termux bridge instance"""
    global _termux_bridge
    if _termux_bridge is None:
        _termux_bridge = TermuxBridge()

    # Check availability
    available = await _termux_bridge.check_availability()
    if not available:
        logger.warning("Termux not available - running in simulation mode")

    return _termux_bridge
