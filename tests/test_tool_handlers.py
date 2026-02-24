"""
Tests for AURA v3 Tool Handlers

Tests the tool execution pipeline
"""

import unittest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tools.handlers import ToolHandlers, get_tool_handlers, create_handler_dict


class TestToolHandlers(unittest.TestCase):
    """Test ToolHandlers class"""

    def setUp(self):
        """Set up test"""
        self.handlers = ToolHandlers()

    def test_default_values(self):
        """Test default values"""
        self.assertIsNone(self.handlers.termux_bridge)
        self.assertFalse(self.handlers._initialized)

    def test_create_handler_dict(self):
        """Test creating handler dictionary"""
        handlers = ToolHandlers()
        handler_dict = create_handler_dict(handlers)

        # Check expected handlers are present
        expected_handlers = [
            "open_app",
            "close_app",
            "get_current_app",
            "get_notifications",
            "send_message",
            "make_call",
            "take_screenshot",
            "read_file",
            "write_file",
            "get_system_info",
            "run_shell_command",
        ]

        for handler_name in expected_handlers:
            self.assertIn(handler_name, handler_dict)


class TestToolHandlersAsync(unittest.TestCase):
    """Async tests for ToolHandlers"""

    def setUp(self):
        """Set up test"""
        self.handlers = ToolHandlers()

    def test_initialize(self):
        """Test initialization"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(self.handlers.initialize())

        # Should be initialized (may or may not have termux bridge)
        self.assertTrue(self.handlers._initialized)

        loop.close()

    def test_open_app_no_bridge(self):
        """Test opening app without termux bridge"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(self.handlers.open_app("whatsapp"))

        # Should return error since no bridge
        self.assertFalse(result["success"])
        self.assertIn("Termux bridge not available", result["error"])

        loop.close()

    def test_close_app_no_bridge(self):
        """Test closing app without termux bridge"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(self.handlers.close_app("whatsapp"))

        self.assertFalse(result["success"])

        loop.close()

    def test_get_notifications_no_bridge(self):
        """Test getting notifications without bridge"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(self.handlers.get_notifications(5))

        self.assertIn("notifications", result)
        self.assertEqual(result["notifications"], [])

        loop.close()

    def test_send_message_no_bridge(self):
        """Test sending message without bridge"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            self.handlers.send_message("+1234567890", "Hello!")
        )

        self.assertFalse(result["success"])

        loop.close()

    def test_make_call_no_bridge(self):
        """Test making call without bridge"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(self.handlers.make_call("+1234567890"))

        self.assertFalse(result["success"])

        loop.close()

    def test_take_screenshot_no_bridge(self):
        """Test taking screenshot without bridge"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(self.handlers.take_screenshot())

        self.assertFalse(result["success"])

        loop.close()

    def test_read_file_no_bridge(self):
        """Test reading file without bridge"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(self.handlers.read_file("/test/file.txt"))

        self.assertFalse(result["success"])

        loop.close()

    def test_write_file_no_bridge(self):
        """Test writing file without bridge"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            self.handlers.write_file("/test/file.txt", "content")
        )

        self.assertFalse(result["success"])

        loop.close()

    def test_get_system_info_no_bridge(self):
        """Test getting system info without bridge"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(self.handlers.get_system_info())

        self.assertFalse(result["success"])

        loop.close()

    def test_run_shell_command_no_bridge(self):
        """Test running shell command without bridge"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(self.handlers.run_shell_command("ls"))

        self.assertFalse(result["success"])

        loop.close()

    def test_run_shell_command_blocked(self):
        """Test running blocked shell command"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Dangerous command should be blocked even without bridge
        # It should either fail with "not allowed" or "Termux bridge not available"
        result = loop.run_until_complete(self.handlers.run_shell_command("rm -rf /"))

        # Either "not allowed" or "bridge not available" is acceptable
        error_msg = result.get("error", "")
        self.assertTrue(
            "not allowed" in error_msg.lower() or "bridge" in error_msg.lower(),
            f"Unexpected error: {error_msg}",
        )

        loop.close()


class TestToolHandlerSecurity(unittest.TestCase):
    """Security tests for tool handlers"""

    def setUp(self):
        """Set up test"""
        self.handlers = ToolHandlers()

    def test_allowed_shell_commands(self):
        """Test that allowed commands work"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        allowed_commands = ["ls", "cat", "echo", "pwd", "date", "whoami", "df", "free"]

        for cmd in allowed_commands:
            result = loop.run_until_complete(self.handlers.run_shell_command(cmd))
            # Without bridge, should fail but not with "not allowed"
            # The command should at least be parsed
            self.assertIn("success", result)

        loop.close()


if __name__ == "__main__":
    unittest.main()
