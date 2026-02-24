"""
Security Module Tests
Tests for input sanitization, command validation, and safe execution
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.security import (
    SecurityError,
    sanitize_string,
    sanitize_path,
    sanitize_filename,
    validate_phone_number,
    validate_app_name,
    validate_command,
    validate_integer,
    validate_enum,
    safe_eval_math,
    build_safe_command,
    CommandValidator,
)


class TestSanitizeString:
    def test_basic_string(self):
        result = sanitize_string("hello world")
        assert result == "hello world"

    def test_truncate_long_string(self):
        long_string = "a" * 2000
        result = sanitize_string(long_string, max_length=100)
        assert len(result) == 100

    def test_non_string_input(self):
        result = sanitize_string(123)
        assert result == "123"


class TestSanitizePath:
    def test_valid_path(self):
        result = sanitize_path("/home/user/file.txt")
        assert "home" in result

    def test_path_traversal_blocked(self):
        with pytest.raises(SecurityError):
            sanitize_path("../../../etc/passwd")

    def test_path_with_traversal(self):
        with pytest.raises(SecurityError):
            sanitize_path("/sdcard/../etc/passwd")

    def test_empty_path(self):
        with pytest.raises(SecurityError):
            sanitize_path("")


class TestSanitizeFilename:
    def test_valid_filename(self):
        result = sanitize_filename("photo.jpg")
        assert result == "photo.jpg"

    def test_strips_directory(self):
        result = sanitize_filename("/path/to/photo.jpg")
        assert result == "photo.jpg"

    def test_removes_dangerous_chars(self):
        result = sanitize_filename("photo;rm-rf.jpg")
        assert ";" not in result

    def test_empty_filename_raises(self):
        with pytest.raises(SecurityError):
            sanitize_filename("")

    def test_only_extension_raises(self):
        with pytest.raises(SecurityError):
            sanitize_filename(".jpg")


class TestValidatePhoneNumber:
    def test_valid_phone(self):
        result = validate_phone_number("+1234567890")
        assert result == "+1234567890"

    def test_valid_with_dashes(self):
        result = validate_phone_number("123-456-7890")
        assert result == "123-456-7890"

    def test_strips_invalid_chars(self):
        result = validate_phone_number("+1 234 567-8900")
        assert "+12345678900" in result.replace("-", "").replace(" ", "")

    def test_empty_phone_raises(self):
        with pytest.raises(SecurityError):
            validate_phone_number("")

    def test_too_long_raises(self):
        with pytest.raises(SecurityError):
            validate_phone_number("1" * 30)


class TestValidateAppName:
    def test_valid_app_name(self):
        result = validate_app_name("whatsapp")
        assert result == "whatsapp"

    def test_valid_package_name(self):
        result = validate_app_name("com.whatsapp")
        assert result == "com.whatsapp"

    def test_uppercase_converted(self):
        result = validate_app_name("WhatsApp")
        assert result == "whatsapp"

    def test_invalid_chars_raises(self):
        with pytest.raises(SecurityError):
            validate_app_name("whatsapp;rm -rf")

    def test_empty_raises(self):
        with pytest.raises(SecurityError):
            validate_app_name("")


class TestValidateCommand:
    def test_valid_simple_command(self):
        result = validate_command("ls -la")
        assert result is True

    def test_valid_echo_command(self):
        result = validate_command("echo hello")
        assert result is True

    def test_command_with_pipe_blocked(self):
        with pytest.raises(SecurityError):
            validate_command("ls | cat /etc/passwd")

    def test_command_with_semicolon_blocked(self):
        with pytest.raises(SecurityError):
            validate_command("ls; rm -rf /")

    def test_command_with_substitution_blocked(self):
        with pytest.raises(SecurityError):
            validate_command("ls $(cat /etc/passwd)")

    def test_command_with_backtick_blocked(self):
        with pytest.raises(SecurityError):
            validate_command("ls `cat /etc/passwd`")

    def test_command_redirect_blocked(self):
        with pytest.raises(SecurityError):
            validate_command("ls > /tmp/output")

    def test_command_with_ampersand_blocked(self):
        with pytest.raises(SecurityError):
            validate_command("ls && cat /etc/passwd")

    def test_allowed_commands_list(self):
        result = validate_command("ls", allowed_commands=["ls", "cat"])
        assert result is True

    def test_not_allowed_command_raises(self):
        with pytest.raises(SecurityError):
            validate_command("rm", allowed_commands=["ls", "cat"])


class TestValidateInteger:
    def test_valid_integer(self):
        result = validate_integer("42")
        assert result == 42

    def test_with_min_value(self):
        result = validate_integer("10", min_val=5)
        assert result == 10

    def test_with_max_value(self):
        result = validate_integer("100", max_val=200)
        assert result == 100

    def test_below_min_raises(self):
        with pytest.raises(SecurityError):
            validate_integer("1", min_val=5)

    def test_above_max_raises(self):
        with pytest.raises(SecurityError):
            validate_integer("500", max_val=100)

    def test_invalid_raises(self):
        with pytest.raises(SecurityError):
            validate_integer("not_a_number")


class TestValidateEnum:
    def test_valid_value(self):
        result = validate_enum("apple", ["apple", "banana", "cherry"])
        assert result == "apple"

    def test_invalid_value_raises(self):
        with pytest.raises(SecurityError):
            validate_enum("grape", ["apple", "banana"])


class TestSafeEvalMath:
    def test_basic_addition(self):
        result = safe_eval_math("2 + 3")
        assert result == 5

    def test_basic_subtraction(self):
        result = safe_eval_math("10 - 4")
        assert result == 6

    def test_basic_multiplication(self):
        result = safe_eval_math("3 * 4")
        assert result == 12

    def test_basic_division(self):
        result = safe_eval_math("15 / 3")
        assert result == 5

    def test_power(self):
        result = safe_eval_math("2 ** 3")
        assert result == 8

    def test_modulo(self):
        result = safe_eval_math("10 % 3")
        assert result == 1

    def test_negative_numbers(self):
        result = safe_eval_math("-5 + 3")
        assert result == -2

    def test_complex_expression(self):
        result = safe_eval_math("2 + 3 * 4")
        assert result == 14

    def test_parentheses(self):
        result = safe_eval_math("(2 + 3) * 4")
        assert result == 20

    def test_division_by_zero_raises(self):
        with pytest.raises(ValueError):
            safe_eval_math("1 / 0")

    def test_invalid_syntax_raises(self):
        with pytest.raises(ValueError):
            safe_eval_math("2 +")

    def test_empty_raises(self):
        with pytest.raises(SecurityError):
            safe_eval_math("")


class TestBuildSafeCommand:
    def test_single_part(self):
        result = build_safe_command("ls")
        assert result == "ls"

    def test_multiple_parts(self):
        result = build_safe_command("ls", "-la", "/path")
        assert "ls" in result
        assert "-la" in result

    def test_escapes_special_chars(self):
        result = build_safe_command("echo", "hello world")
        assert "hello world" in result


class TestCommandValidator:
    def test_init_with_allowed_commands(self):
        validator = CommandValidator(["ls", "cat"])
        assert "ls" in validator.get_allowed_commands()

    def test_validate_adds_to_history(self):
        validator = CommandValidator(["ls"])
        validator.validate("ls")
        assert "ls" in validator.command_history

    def test_add_allowed_command(self):
        validator = CommandValidator()
        validator.add_allowed_command("ps")
        assert "ps" in validator.get_allowed_commands()

    def test_remove_allowed_command(self):
        validator = CommandValidator(["ls", "cat"])
        validator.remove_allowed_command("ls")
        assert "ls" not in validator.get_allowed_commands()


class TestIntegration:
    def test_full_validation_chain(self):
        phone = validate_phone_number("+1234567890")
        assert phone

        app = validate_app_name("whatsapp")
        assert app

        cmd = validate_command("ls -la", allowed_commands=["ls", "cat"])
        assert cmd

        math_result = safe_eval_math("2 + 2")
        assert math_result == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
