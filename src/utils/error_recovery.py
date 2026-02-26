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
