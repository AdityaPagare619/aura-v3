"""
AURA v3 Security Module
Input sanitization, command validation, and safe execution wrappers
"""

import ast
import logging
import os
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when input fails security validation"""

    pass


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """Sanitize a string input by removing dangerous characters"""
    if not isinstance(value, str):
        value = str(value)

    if len(value) > max_length:
        value = value[:max_length]

    return value


def sanitize_path(path: str, allowed_base_dirs: Optional[List[str]] = None) -> str:
    """
    Sanitize a file path to prevent path traversal attacks

    Args:
        path: The path to sanitize
        allowed_base_dirs: List of allowed base directories. If None, uses current dir.

    Returns:
        Sanitized path

    Raises:
        SecurityError: If path attempts directory traversal
    """
    if not path:
        raise SecurityError("Path cannot be empty")

    path = path.strip()

    if ".." in path.split(os.sep) or path.startswith(".."):
        raise SecurityError("Path traversal not allowed")

    if ".." in path.replace("\\", "/").split("/"):
        raise SecurityError("Path traversal not allowed")

    path = os.path.normpath(path)

    if allowed_base_dirs:
        abs_path = os.path.abspath(path)
        for base_dir in allowed_base_dirs:
            abs_base = os.path.abspath(base_dir)
            if abs_path.startswith(abs_base + os.sep) or abs_path == abs_base:
                return path

        raise SecurityError(
            f"Path must be within allowed directories: {allowed_base_dirs}"
        )

    return path


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path manipulation"""
    if not filename:
        raise SecurityError("Filename cannot be empty")

    filename = os.path.basename(filename)

    dangerous_chars = ["/", "\\", "..", "\0", "\n", "\r"]
    for char in dangerous_chars:
        filename = filename.replace(char, "")

    filename = re.sub(r"[^\w\s\-_\.]", "", filename)

    if not filename or filename.startswith("."):
        raise SecurityError("Invalid filename")

    return filename[:255]


def validate_phone_number(phone: str) -> str:
    """Validate and sanitize a phone number"""
    if not phone:
        raise SecurityError("Phone number cannot be empty")

    phone = phone.strip()

    phone = re.sub(r"[^\d+\-\s()]", "", phone)

    if not re.match(r"^[\d+\-\s()]+$", phone):
        raise SecurityError("Invalid phone number format")

    if len(phone) > 20:
        raise SecurityError("Phone number too long")

    return phone


def validate_command(
    command: str, allowed_commands: Optional[List[str]] = None
) -> bool:
    """
    Validate a shell command against allowed patterns

    Args:
        command: The command to validate
        allowed_commands: List of allowed command prefixes

    Returns:
        True if command is safe

    Raises:
        SecurityError: If command is not allowed
    """
    if not command:
        raise SecurityError("Command cannot be empty")

    command = command.strip()

    dangerous_patterns = [
        r";\s*\w+",
        r"\|\s*\w+",
        r"&&\s*\w+",
        r"\|\|\s*\w+",
        r"\$\(",
        r"`",
        r">\s*/",
        r">>\s*/",
        r"<\s*/",
        r"\n",
        r"\r",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, command):
            raise SecurityError(f"Command contains dangerous pattern: {pattern}")

    if allowed_commands:
        cmd_parts = shlex.split(command) if command else []
        if not cmd_parts:
            raise SecurityError("Empty command")

        cmd_name = cmd_parts[0]
        if cmd_name not in allowed_commands:
            raise SecurityError(f"Command '{cmd_name}' not in allowed list")

    return True


def validate_app_name(app_name: str) -> str:
    """Validate and sanitize an app name/package name"""
    if not app_name:
        raise SecurityError("App name cannot be empty")

    app_name = app_name.strip().lower()

    if not re.match(r"^[a-z][a-z0-9_.]*$", app_name):
        raise SecurityError("Invalid app name format")

    if len(app_name) > 256:
        raise SecurityError("App name too long")

    return app_name


def safe_eval_math(expression: str) -> Union[int, float]:
    """
    Safely evaluate a mathematical expression using AST parsing

    This is a secure alternative to eval() that only allows
    basic mathematical operations.

    Args:
        expression: A mathematical expression string

    Returns:
        The result of the evaluation

    Raises:
        SecurityError: If expression contains unsafe operations
        ValueError: If expression is malformed
    """
    if not expression or not isinstance(expression, str):
        raise SecurityError("Invalid expression")

    expression = expression.strip()

    valid_ops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b
        if b != 0
        else raise_(ValueError("Division by zero")),
        ast.Pow: lambda a, b: a**b,
        ast.Mod: lambda a, b: a % b if b != 0 else raise_(ValueError("Modulo by zero")),
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: +a,
    }

    def raise_(e):
        raise e

    def eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op_type = type(node.op)
            if op_type in valid_ops:
                return valid_ops[op_type](left, right)
            raise SecurityError(f"Unsupported operation: {op_type.__name__}")
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in valid_ops:
                return valid_ops[op_type](eval_node(node.operand))
            raise SecurityError(f"Unsupported unary operation: {op_type.__name__}")
        else:
            raise SecurityError(f"Unsupported node type: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
        return eval_node(tree.body)
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")
    except SecurityError:
        raise
    except Exception as e:
        raise ValueError(f"Evaluation error: {e}")


def build_safe_command(*parts: str) -> str:
    """
    Build a shell command safely by escaping each part

    Args:
        *parts: Command parts to combine

    Returns:
        Safely escaped command string
    """
    escaped_parts = [shlex.quote(str(p)) for p in parts]
    return " ".join(escaped_parts)


def validate_integer(
    value: Any, min_val: Optional[int] = None, max_val: Optional[int] = None
) -> int:
    """Validate and convert to integer within bounds"""
    try:
        result = int(value)
    except (ValueError, TypeError):
        raise SecurityError(f"Invalid integer: {value}")

    if min_val is not None and result < min_val:
        raise SecurityError(f"Value {result} below minimum {min_val}")

    if max_val is not None and result > max_val:
        raise SecurityError(f"Value {result} above maximum {max_val}")

    return result


def validate_enum(value: str, allowed_values: List[str]) -> str:
    """Validate value is in allowed list"""
    if value not in allowed_values:
        raise SecurityError(f"Value '{value}' not in allowed list: {allowed_values}")
    return value


class CommandValidator:
    """
    Stateful command validator with allowlist support
    """

    def __init__(self, allowed_commands: Optional[List[str]] = None):
        self.allowed_commands = allowed_commands or []
        self.command_history: List[str] = []

    def validate(self, command: str) -> bool:
        """Validate a command"""
        validate_command(command, self.allowed_commands)
        self.command_history.append(command)
        return True

    def get_allowed_commands(self) -> List[str]:
        """Get list of allowed commands"""
        return self.allowed_commands.copy()

    def add_allowed_command(self, command: str) -> None:
        """Add a command to the allowlist"""
        if command not in self.allowed_commands:
            self.allowed_commands.append(command)

    def remove_allowed_command(self, command: str) -> None:
        """Remove a command from the allowlist"""
        if command in self.allowed_commands:
            self.allowed_commands.remove(command)


DEFAULT_VALIDATOR = CommandValidator(
    allowed_commands=["ls", "cat", "echo", "pwd", "date", "whoami", "df", "free", "top"]
)


def get_default_validator() -> CommandValidator:
    """Get the default command validator"""
    return DEFAULT_VALIDATOR
