"""
AURA v3 Termux Widget Integration
================================

Provides integration with Termux for:
- Widget support (home screen widgets)
- Floating bubble/head functionality
- Notification-based HUD
- Background service management

This enables Aura to be visible and accessible without
opening the full app - like a floating companion.
"""

import asyncio
import logging
import subprocess
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class WidgetConfig:
    """Configuration for an Aura widget"""

    widget_type: str  # bubble, hud, status, quick_action
    title: str
    content: str
    icon: Optional[str] = None
    action: Optional[str] = None


class TermuxWidgetBridge:
    """
    Bridge to Termux APIs for widgets and floating elements.

    Uses Termux APIs:
    - termux-widget: For home screen widgets
    - termux-notification: For notifications/HUD
    - termux-toast: For quick toasts
    - termux-vibrate: For haptic feedback
    """

    def __init__(self):
        self._available = self._check_termux_available()
        self._notification_callbacks: List[Callable] = []

    def _check_termux_available(self) -> bool:
        """Check if running in Termux"""
        try:
            result = subprocess.run(["termux-info"], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def is_available(self) -> bool:
        """Check if Termux integration is available"""
        return self._available

    async def show_notification(
        self,
        title: str,
        content: str,
        urgency: str = "normal",
        actions: Optional[List[Dict]] = None,
    ) -> bool:
        """Show a notification"""
        if not self._available:
            logger.warning("Termux not available")
            return False

        try:
            cmd = [
                "termux-notification",
                "-t",
                title,
                "-c",
                content,
            ]

            if urgency == "high":
                cmd.extend(["--priority", "high"])
            elif urgency == "low":
                cmd.extend(["--priority", "low"])

            if actions:
                for i, action in enumerate(actions):
                    cmd.extend([f"--button{i}", action.get("label", "")])
                    cmd.extend([f"--button-action{i}", action.get("action", "")])

            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Notification error: {e}")
            return False

    async def show_toast(self, message: str, duration: str = "short") -> bool:
        """Show a toast message"""
        if not self._available:
            return False

        try:
            cmd = ["termux-toast", "-s", message]
            if duration == "long":
                cmd.extend(["-l"])

            result = subprocess.run(cmd, capture_output=True, timeout=5)
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Toast error: {e}")
            return False

    async def vibrate(self, duration: int = 100) -> bool:
        """Trigger vibration feedback"""
        if not self._available:
            return False

        try:
            cmd = ["termux-vibrate", "-d", str(duration)]
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Vibrate error: {e}")
            return False

    def register_notification_callback(self, callback: Callable):
        """Register callback for notification actions"""
        self._notification_callbacks.append(callback)

    async def update_widget(self, config: WidgetConfig) -> bool:
        """Update a widget with new content"""
        # Widgets in Termux work via shortcuts
        # This is a simplified implementation
        logger.info(f"Widget update: {config.title} - {config.content}")
        return True

    async def create_quick_action_shortcut(
        self, action_id: str, label: str, intent: str
    ) -> bool:
        """Create a quick action shortcut"""
        if not self._available:
            return False

        try:
            # Create a shell script as a shortcut
            script_content = f"""#!/bin/bash
am start -n org.aura.v3/.MainActivity --ei action {action_id}
"""
            script_path = f"~/.shortcuts/{action_id}.sh"

            # Write shortcut script
            with open(script_path, "w") as f:
                f.write(script_content)

            logger.info(f"Created shortcut: {action_id}")
            return True

        except Exception as e:
            logger.error(f"Shortcut error: {e}")
            return False


class FloatingBubbleManager:
    """
    Manages floating bubble/head display for Aura.

    On Android, this can be implemented via:
    - Floating bubble library
    - Picture-in-picture mode
    - Custom notification with expand
    """

    def __init__(self, termux_bridge: TermuxWidgetBridge):
        self.termux = termux_bridge
        self._active = False

    async def show_bubble(self, status: str, thought: Optional[str] = None) -> bool:
        """Show a floating bubble with Aura's status"""
        # Use notification as a "bubble" alternative
        title = f"🟣 Aura: {status}"
        content = thought if thought else "Tap to open Aura"

        return await self.termux.show_notification(
            title=title,
            content=content,
            urgency="low",
            actions=[
                {"label": "Open", "action": "open_aura"},
                {"label": "Quiet", "action": "toggle_quiet"},
            ],
        )

    async def update_thinking_bubble(self, thought: str) -> bool:
        """Show bubble while Aura is thinking"""
        return await self.show_bubble("Thinking...", thought)

    async def show_result_bubble(self, result: str) -> bool:
        """Show bubble with result"""
        return await self.show_bubble("Done", result)

    async def dismiss(self) -> bool:
        """Dismiss the floating bubble"""
        # In a real implementation, this would cancel the notification
        self._active = False
        return True


class HUDOverlayManager:
    """
    Manages minimal HUD overlay for Mission Control mode.

    Shows at top of screen:
    - Aura's current state
    - Quick status indicators
    - Mini orb visualization
    """

    def __init__(self, termux_bridge: TermuxWidgetBridge):
        self.termux = termux_bridge
        self._visible = False

    async def show_hud(self, state: Dict[str, Any]) -> bool:
        """Show minimal HUD overlay"""
        if not self._visible:
            self._visible = True

        status = state.get("status", "Ready")
        task = state.get("current_task", "")

        title = f"AURA • {status}"
        content = task if task else "All systems normal"

        return await self.termux.show_notification(
            title=title, content=content, urgency="low"
        )

    async def update_state(self, state: Dict[str, Any]) -> bool:
        """Update HUD with new state"""
        if self._visible:
            return await self.show_hud(state)
        return False

    async def hide(self) -> bool:
        """Hide the HUD"""
        self._visible = False
        return True


# Global instance
_termux_bridge: Optional[TermuxWidgetBridge] = None


def get_termux_bridge() -> TermuxWidgetBridge:
    """Get or create the Termux bridge"""
    global _termux_bridge
    if _termux_bridge is None:
        _termux_bridge = TermuxWidgetBridge()
    return _termux_bridge
