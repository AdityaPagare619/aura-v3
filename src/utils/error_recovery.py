"""
AURA v3 Error Recovery System
Provides automatic error recovery, retry logic, and fallback mechanisms
Optimized for mobile - handles transient failures gracefully
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any, Optional, Callable, List, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    TRANSIENT = "transient"
    RESOURCE = "resource"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information about an error"""

    error: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime
    component: str
    operation: str
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """A recovery action to attempt"""

    name: str
    action_func: Callable
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff: float = 30.0


@dataclass
class ErrorRecoveryResult:
    """Result of error recovery attempt"""

    success: bool
    recovered: bool
    error: Optional[Exception]
    attempts: int
    final_result: Any = None
    actions_taken: List[str] = field(default_factory=list)


class UnauthorizedError(Exception):
    """Raised when authentication/authorization fails"""

    pass


# =============================================================================
# RepairableException Pattern (Inspired by Agent Zero)
# Self-healing errors that provide LLM-readable context for recovery
# =============================================================================


class RepairableException(Exception):
    """
    Base exception that provides LLM-readable context for self-healing.

    Inspired by Agent Zero's error handling pattern - errors become prompts
    that guide the agent to fix itself. When caught, these exceptions provide
    rich context that can be forwarded to the LLM for reasoning about recovery.

    Usage:
        raise RepairableException(
            "File not found: config.yaml",
            recovery_hint="Check if config file exists or create with defaults",
            suggested_actions=["create_default_config", "check_path_permissions"],
            context={"path": "config.yaml", "operation": "read_config"}
        )
    """

    def __init__(
        self,
        message: str,
        recovery_hint: str = "",
        context: Dict[str, Any] = None,
        suggested_actions: List[str] = None,
        is_retriable: bool = True,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
    ):
        super().__init__(message)
        self.message = message
        self.recovery_hint = recovery_hint
        self.context = context or {}
        self.suggested_actions = suggested_actions or []
        self.is_retriable = is_retriable
        self.severity = severity
        self.category = category
        self.timestamp = datetime.now()

    def to_llm_prompt(self) -> str:
        """
        Convert to LLM-readable error context for self-healing.

        Returns a structured prompt that helps the LLM understand what went wrong
        and how to recover, enabling Agent Zero-style self-repair.
        """
        import json

        parts = [
            f"ERROR OCCURRED: {self.message}",
            f"SEVERITY: {self.severity.value}",
            f"CATEGORY: {self.category.value}",
        ]

        if self.recovery_hint:
            parts.append(f"RECOVERY HINT: {self.recovery_hint}")

        if self.context:
            # Sanitize sensitive keys
            safe_context = {
                k: (
                    "***"
                    if any(
                        s in k.lower()
                        for s in ["password", "token", "secret", "key", "pin"]
                    )
                    else v
                )
                for k, v in self.context.items()
            }
            parts.append(f"CONTEXT: {json.dumps(safe_context, indent=2, default=str)}")

        if self.suggested_actions:
            parts.append(f"SUGGESTED ACTIONS: {', '.join(self.suggested_actions)}")

        parts.append(f"IS RETRIABLE: {self.is_retriable}")

        parts.append("""
Based on this error, please:
1. Analyze what went wrong
2. Consider the suggested actions  
3. Choose an alternative approach or fix the issue
4. If not retriable, inform the user gracefully""")

        return "\n".join(parts)

    def to_observation(self) -> Dict[str, Any]:
        """
        Convert to a structured observation for the ReAct loop.

        This format integrates with AURA's ReAct agent loop, allowing the error
        to be processed as an observation that triggers reflection and adaptation.
        """
        return {
            "type": "error",
            "success": False,
            "error": self.message,
            "error_type": self.__class__.__name__,
            "severity": self.severity.value,
            "category": self.category.value,
            "recovery_hint": self.recovery_hint,
            "suggested_actions": self.suggested_actions,
            "is_retriable": self.is_retriable,
            "context": self.context,
            "llm_prompt": self.to_llm_prompt(),
        }


class ToolExecutionError(RepairableException):
    """
    Tool execution failed - LLM can try alternative tools or parameters.

    Example:
        raise ToolExecutionError(
            "send_whatsapp failed: Contact not found",
            recovery_hint="Try searching for the contact or ask user for correct name",
            suggested_actions=["search_contacts", "ask_clarification"],
            context={"tool": "send_whatsapp", "contact": "John"}
        )
    """

    def __init__(
        self,
        message: str,
        tool_name: str = "",
        parameters: Dict[str, Any] = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        context.update(
            {
                "tool_name": tool_name,
                "parameters": parameters or {},
            }
        )
        super().__init__(
            message, category=ErrorCategory.SYSTEM, context=context, **kwargs
        )
        self.tool_name = tool_name
        self.parameters = parameters or {}


class ResourceError(RepairableException):
    """
    Resource unavailable - LLM can wait, use alternatives, or reduce load.

    Example:
        raise ResourceError(
            "Insufficient memory to load model",
            recovery_hint="Try unloading other models or using smaller variant",
            suggested_actions=["unload_models", "use_smaller_model", "wait_and_retry"]
        )
    """

    def __init__(self, message: str, resource_type: str = "", **kwargs):
        context = kwargs.pop("context", {})
        context["resource_type"] = resource_type
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs,
        )
        self.resource_type = resource_type


class NetworkError(RepairableException):
    """
    Network operation failed - LLM can retry, use cached data, or go offline.

    Example:
        raise NetworkError(
            "API request timed out",
            recovery_hint="Check connectivity, retry with backoff, or use cached response",
            suggested_actions=["retry_with_backoff", "use_cached", "notify_offline"]
        )
    """

    def __init__(self, message: str, endpoint: str = "", **kwargs):
        context = kwargs.pop("context", {})
        context["endpoint"] = endpoint
        super().__init__(
            message,
            category=ErrorCategory.TRANSIENT,
            severity=ErrorSeverity.LOW,
            is_retriable=True,
            context=context,
            **kwargs,
        )
        self.endpoint = endpoint


class PermissionDeniedError(RepairableException):
    """
    Permission denied - LLM can request permission or use workaround.

    Example:
        raise PermissionDeniedError(
            "Cannot access camera without permission",
            recovery_hint="Request camera permission or explain why it's needed",
            suggested_actions=["request_permission", "explain_need", "use_alternative"]
        )
    """

    def __init__(self, message: str, permission: str = "", **kwargs):
        context = kwargs.pop("context", {})
        context["permission"] = permission
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.MEDIUM,
            is_retriable=False,
            context=context,
            **kwargs,
        )
        self.permission = permission


class ValidationError(RepairableException):
    """
    Input validation failed - LLM can fix input or ask for clarification.

    Example:
        raise ValidationError(
            "Invalid phone number format",
            recovery_hint="Phone number should be 10 digits or include country code",
            suggested_actions=["fix_format", "ask_clarification"],
            context={"input": "123-abc", "expected_format": "10 digits"}
        )
    """

    def __init__(self, message: str, field: str = "", expected: str = "", **kwargs):
        context = kwargs.pop("context", {})
        context.update({"field": field, "expected_format": expected})
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            is_retriable=True,
            context=context,
            **kwargs,
        )
        self.field = field
        self.expected = expected


class HandledException(Exception):
    """
    Fatal error that should stop the ReAct loop gracefully.

    Inspired by Agent Zero - this exception indicates the error has been handled
    and the loop should terminate with a user-friendly message, not retry.
    """

    def __init__(self, message: str, user_message: str = ""):
        super().__init__(message)
        self.user_message = user_message or message

    def get_user_message(self) -> str:
        """Get the user-friendly message to display"""
        return self.user_message


class InterventionRequired(Exception):
    """
    User intervention is required to proceed.

    Inspired by Agent Zero - this exception pauses the loop and asks the user
    for input or confirmation before continuing.
    """

    def __init__(self, message: str, question: str = "", options: List[str] = None):
        super().__init__(message)
        self.question = question or message
        self.options = options or []

    def get_prompt(self) -> str:
        """Get the prompt to show the user"""
        if self.options:
            options_str = "\n".join(
                f"  {i + 1}. {opt}" for i, opt in enumerate(self.options)
            )
            return f"{self.question}\n{options_str}"
        return self.question


class ErrorClassifier:
    """Classifies errors to determine recovery strategy"""

    TRANSIENT_EXCEPTIONS = (
        asyncio.TimeoutError,
        ConnectionError,
        ConnectionResetError,
        ConnectionRefusedError,
        ConnectionAbortedError,
    )

    RESOURCE_EXCEPTIONS = (
        MemoryError,
        OSError,
        ResourceWarning,
    )

    AUTH_EXCEPTIONS = (
        PermissionError,
        UnauthorizedError,
    )

    @staticmethod
    def classify(error: Exception) -> ErrorCategory:
        """Classify error by category"""
        if isinstance(error, ErrorClassifier.TRANSIENT_EXCEPTIONS):
            return ErrorCategory.TRANSIENT
        elif isinstance(error, ErrorClassifier.RESOURCE_EXCEPTIONS):
            return ErrorCategory.RESOURCE
        elif isinstance(error, ErrorClassifier.AUTH_EXCEPTIONS):
            return ErrorCategory.AUTHENTICATION
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorCategory.VALIDATION
        else:
            return ErrorCategory.UNKNOWN

    @staticmethod
    def get_severity(error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity"""
        if isinstance(error, MemoryError):
            return ErrorSeverity.CRITICAL
        elif category == ErrorCategory.RESOURCE:
            return ErrorSeverity.HIGH
        elif category == ErrorCategory.TRANSIENT:
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM


class ErrorRecovery:
    """
    Error recovery system with automatic retry and fallback logic

    Features:
    - Exponential backoff for retries
    - Fallback mechanisms
    - Error classification
    - Recovery action chains
    - Mobile-optimized (handles low memory, network issues)
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_backoff: float = 1.0,
        max_backoff: float = 30.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.jitter = jitter

        self._recovery_actions: Dict[ErrorCategory, List[RecoveryAction]] = {}
        self._fallback_handlers: Dict[Type[Exception], Callable] = {}
        self._error_history: deque = deque(maxlen=100)

    def register_recovery_action(self, category: ErrorCategory, action: RecoveryAction):
        """Register a recovery action for an error category"""
        if category not in self._recovery_actions:
            self._recovery_actions[category] = []
        self._recovery_actions[category].append(action)
        logger.info(f"Registered recovery action for {category.value}: {action.name}")

    def register_fallback(
        self, exception_type: Type[Exception], fallback_func: Callable
    ):
        """Register a fallback handler for an exception type"""
        self._fallback_handlers[exception_type] = fallback_func
        logger.info(f"Registered fallback for {exception_type.__name__}")

    async def recover(
        self,
        error: Exception,
        component: str,
        operation: str,
        default_result: Any = None,
    ) -> ErrorRecoveryResult:
        """Attempt to recover from an error"""
        category = ErrorClassifier.classify(error)
        severity = ErrorClassifier.get_severity(error, category)

        context = ErrorContext(
            error=error,
            category=category,
            severity=severity,
            timestamp=datetime.now(),
            component=component,
            operation=operation,
        )

        self._error_history.append(context)
        logger.warning(
            f"Error in {component}.{operation}: {error} "
            f"(category: {category.value}, severity: {severity.value})"
        )

        if severity == ErrorSeverity.CRITICAL:
            return await self._handle_critical(context, default_result)

        attempts = 0
        actions_taken = []
        last_error = error

        while attempts < self.max_retries:
            attempts += 1
            context.retry_count = attempts

            if category in self._recovery_actions:
                for action in self._recovery_actions[category]:
                    if attempts <= action.max_retries:
                        try:
                            logger.info(f"Attempting recovery action: {action.name}")
                            await action.action_func()
                            actions_taken.append(action.name)
                        except Exception as e:
                            logger.warning(f"Recovery action failed: {e}")

            backoff = self._calculate_backoff(attempts)
            logger.info(
                f"Retrying after {backoff:.2f}s (attempt {attempts}/{self.max_retries})"
            )

            await asyncio.sleep(backoff)

        if default_result is not None:
            logger.info(f"Returning default result: {default_result}")
            return ErrorRecoveryResult(
                success=False,
                recovered=False,
                error=last_error,
                attempts=attempts,
                final_result=default_result,
                actions_taken=actions_taken,
            )

        return ErrorRecoveryResult(
            success=False,
            recovered=False,
            error=last_error,
            attempts=attempts,
            actions_taken=actions_taken,
        )

    async def _handle_critical(
        self, context: ErrorContext, default_result: Any
    ) -> ErrorRecoveryResult:
        """Handle critical errors that may require system intervention"""
        logger.error(f"CRITICAL error in {context.component}.{context.operation}")

        if context.error.__class__ in self._fallback_handlers:
            fallback = self._fallback_handlers[context.error.__class__]
            try:
                result = fallback()
                if asyncio.iscoroutine(result):
                    result = await result
                return ErrorRecoveryResult(
                    success=True,
                    recovered=True,
                    error=context.error,
                    attempts=1,
                    final_result=result,
                    actions_taken=["fallback"],
                )
            except Exception as e:
                logger.error(f"Fallback also failed: {e}")

        return ErrorRecoveryResult(
            success=False,
            recovered=False,
            error=context.error,
            attempts=0,
            final_result=default_result,
            actions_taken=[],
        )

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with optional jitter"""
        backoff = self.base_backoff * (2 ** (attempt - 1))
        backoff = min(backoff, self.max_backoff)

        if self.jitter:
            backoff *= 0.5 + random.random()

        return backoff

    def get_error_history(
        self,
        since: Optional[datetime] = None,
        component: Optional[str] = None,
    ) -> List[ErrorContext]:
        """Get error history with optional filters"""
        history = list(self._error_history)

        if since:
            history = [e for e in history if e.timestamp >= since]

        if component:
            history = [e for e in history if e.component == component]

        return history

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        history = list(self._error_history)

        if not history:
            return {"total_errors": 0}

        categories = {}
        severities = {}
        components = {}

        for ctx in history:
            categories[ctx.category.value] = categories.get(ctx.category.value, 0) + 1
            severities[ctx.severity.value] = severities.get(ctx.severity.value, 0) + 1
            components[ctx.component] = components.get(ctx.component, 0) + 1

        return {
            "total_errors": len(history),
            "by_category": categories,
            "by_severity": severities,
            "by_component": components,
            "oldest_error": history[0].timestamp.isoformat() if history else None,
            "newest_error": history[-1].timestamp.isoformat() if history else None,
        }


def with_retry(
    max_retries: int = 3,
    backoff_base: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_backoff: float = 30.0,
    retry_on: Optional[List[Type[Exception]]] = None,
):
    """Decorator to add retry logic to async functions"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            jitter = True

            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if retry_on and not isinstance(e, tuple(retry_on)):
                        raise

                    if attempt < max_retries:
                        backoff = backoff_base * (backoff_multiplier ** (attempt - 1))
                        backoff = min(backoff, max_backoff)
                        if jitter:
                            backoff *= 0.5 + random.random()
                        logger.warning(
                            f"Retry {attempt}/{max_retries} for {func.__name__}: {e}. "
                            f"Waiting {backoff:.2f}s"
                        )
                        await asyncio.sleep(backoff)

            raise last_error

        return wrapper

    return decorator


class RecoveryStateManager:
    """Manages recovery state for long-running operations"""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self._checkpoints: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def save_checkpoint(self, operation_id: str, state: Dict[str, Any]):
        """Save a checkpoint for an operation"""
        async with self._lock:
            self._checkpoints[operation_id] = {
                "state": state,
                "timestamp": datetime.now().isoformat(),
            }
            logger.info(f"Checkpoint saved for operation: {operation_id}")

    async def load_checkpoint(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint for an operation"""
        async with self._lock:
            checkpoint = self._checkpoints.get(operation_id)
            if checkpoint:
                logger.info(f"Checkpoint loaded for operation: {operation_id}")
                return checkpoint["state"]
            return None

    async def clear_checkpoint(self, operation_id: str):
        """Clear a checkpoint after successful completion"""
        async with self._lock:
            if operation_id in self._checkpoints:
                del self._checkpoints[operation_id]
                logger.info(f"Checkpoint cleared for operation: {operation_id}")
