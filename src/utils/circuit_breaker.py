"""
AURA v3 Circuit Breaker
Prevents cascading failures and provides fault tolerance
Optimized for mobile - lightweight state machine
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0
    half_open_max_calls: int = 3
    excluded_exceptions: List[type] = field(default_factory=list)


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker"""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state: CircuitState = CircuitState.CLOSED
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreaker:
    """
    Circuit Breaker implementation for AURA

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold reached, requests are rejected
    - HALF_OPEN: Testing if service recovered

    Mobile-optimized:
    - Minimal memory footprint
    - Simple state machine
    - Configurable thresholds
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback

        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        self._last_state_change = time.time()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        return self._stats

    def _should_exclude(self, exception: Exception) -> bool:
        """Check if exception should be excluded from failure counting"""
        return any(
            isinstance(exception, exc_type)
            for exc_type in self.config.excluded_exceptions
        )

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            if not self._can_execute():
                self._stats.rejected_calls += 1
                logger.warning(f"Circuit {self.name} is OPEN, rejecting call")

                if self.fallback:
                    return await self._execute_fallback(*args, **kwargs)
                raise CircuitBreakerOpenError(f"Circuit {self.name} is OPEN")

            self._stats.total_calls += 1

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self._on_success()
            return result

        except Exception as e:
            if not self._should_exclude(e):
                await self._on_failure(e)
            raise

    async def _execute_fallback(self, *args, **kwargs) -> Any:
        """Execute fallback function"""
        if self.fallback:
            if asyncio.iscoroutinefunction(self.fallback):
                return await self.fallback(*args, **kwargs)
            return self.fallback(*args, **kwargs)
        raise CircuitBreakerOpenError(f"Circuit {self.name} is OPEN")

    def _can_execute(self) -> bool:
        """Check if execution is allowed based on current state"""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            if self._timeout_reached():
                logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
                self._stats.consecutive_successes = 0
                self._last_state_change = time.time()
                return True
            return False

        if self._state == CircuitState.HALF_OPEN:
            return self._stats.total_calls < self.config.half_open_max_calls

        return False

    def _timeout_reached(self) -> bool:
        """Check if timeout has passed since circuit opened"""
        elapsed = time.time() - self._last_state_change
        return elapsed >= self.config.timeout

    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            self._stats.successful_calls += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self._stats.consecutive_successes += 1
                if self._stats.consecutive_successes >= self.config.success_threshold:
                    logger.info(f"Circuit {self.name} transitioning to CLOSED")
                    self._state = CircuitState.CLOSED
                    self._stats.state = CircuitState.CLOSED
                    self._last_state_change = time.time()

            self._stats.state = self._state

    async def _on_failure(self, exception: Exception):
        """Handle failed call"""
        async with self._lock:
            self._stats.failed_calls += 1
            self._stats.consecutive_failures += 1
            self._stats.last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"Circuit {self.name} transitioning to OPEN (half-open failure)"
                )
                self._state = CircuitState.OPEN
                self._last_state_change = time.time()

            elif self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.config.failure_threshold:
                    logger.warning(f"Circuit {self.name} transitioning to OPEN")
                    self._state = CircuitState.OPEN
                    self._last_state_change = time.time()

            self._stats.state = self._state

    def reset(self):
        """Manually reset the circuit breaker"""
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._last_state_change = time.time()
        logger.info(f"Circuit {self.name} manually reset")

    def get_health_report(self) -> Dict[str, Any]:
        """Get health report for monitoring"""
        return {
            "name": self.name,
            "state": self._state.value,
            "total_calls": self._stats.total_calls,
            "successful_calls": self._stats.successful_calls,
            "failed_calls": self._stats.failed_calls,
            "rejected_calls": self._stats.rejected_calls,
            "success_rate": (
                self._stats.successful_calls / self._stats.total_calls
                if self._stats.total_calls > 0
                else 0
            ),
            "last_failure": self._stats.last_failure_time.isoformat()
            if self._stats.last_failure_time
            else None,
            "last_success": self._stats.last_success_time.isoformat()
            if self._stats.last_success_time
            else None,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""

    pass


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different AURA components
    """

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def create_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable] = None,
    ) -> CircuitBreaker:
        """Create a new circuit breaker"""
        breaker = CircuitBreaker(name, config, fallback)
        self._breakers[name] = breaker
        logger.info(f"Created circuit breaker: {name}")
        return breaker

    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get an existing circuit breaker"""
        return self._breakers.get(name)

    def remove_breaker(self, name: str):
        """Remove a circuit breaker"""
        if name in self._breakers:
            del self._breakers[name]
            logger.info(f"Removed circuit breaker: {name}")

    def get_all_health_reports(self) -> List[Dict[str, Any]]:
        """Get health reports for all circuit breakers"""
        return [breaker.get_health_report() for breaker in self._breakers.values()]

    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")


def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    fallback: Optional[Callable] = None,
):
    """Decorator to add circuit breaker to a function"""
    _breaker = CircuitBreaker(name, config, fallback)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await _breaker.call(func, *args, **kwargs)

        wrapper._circuit_breaker = _breaker
        return wrapper

    return decorator
