"""
Retry logic with exponential backoff and circuit breaker pattern.

Provides decorators and utilities for resilient upstream calls.
"""

import asyncio
import functools
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, TypeVar, Union

from errors import (
    CircuitBreakerOpenError,
    GatewayTimeoutError,
    ServiceUnavailableError,
    UpstreamError,
)
from logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Failing, reject requests
    HALF_OPEN = auto()  # Testing if recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for failing upstreams."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("circuit_breaker_half_open", provider="upstream")
                else:
                    retry_after = self._get_retry_after()
                    raise CircuitBreakerOpenError(
                        "Circuit breaker is open - too many failures",
                        retry_after=retry_after,
                    )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        "Circuit breaker half-open limit reached",
                        retry_after=int(self.recovery_timeout),
                    )
                self._half_open_calls += 1

        # Execute the call
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try resetting."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.recovery_timeout

    def _get_retry_after(self) -> int:
        """Calculate seconds until circuit might close."""
        if self._last_failure_time is None:
            return 0
        elapsed = time.time() - self._last_failure_time
        remaining = max(0, self.recovery_timeout - elapsed)
        return int(remaining) + 1

    async def _on_success(self):
        """Record successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls -= 1
                if self._half_open_calls <= 0:
                    self._state = CircuitState.CLOSED
                    self._failures = 0
                    self._last_failure_time = None
                    logger.info("circuit_breaker_closed", provider="upstream")
            else:
                self._failures = max(0, self._failures - 1)

    async def _on_failure(self):
        """Record failed call."""
        async with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    "circuit_breaker_opened",
                    provider="upstream",
                    reason="half_open_call_failed",
                )
            elif self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "circuit_breaker_opened",
                    provider="upstream",
                    reason=f"{self._failures}_consecutive_failures",
                )


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        retryable_statuses: Optional[set[int]] = None,
        retryable_exceptions: Optional[tuple[type[Exception], ...]] = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_statuses = retryable_statuses or {502, 503, 504, 429}
        self.retryable_exceptions = retryable_exceptions or (
            ServiceUnavailableError,
            GatewayTimeoutError,
            UpstreamError,
        )

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay,
        )
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return delay + jitter


async def with_retry(
    func: Callable[..., T],
    config: Optional[RetryConfig] = None,
    *args,
    **kwargs,
) -> T:
    """
    Execute function with retry logic.

    Args:
        func: Function to execute
        config: Retry configuration
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        Last exception if all retries exhausted
    """
    cfg = config or RetryConfig()
    last_exception: Optional[Exception] = None

    for attempt in range(cfg.max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        except Exception as e:
            last_exception = e

            # Check if we should retry this exception
            should_retry = False

            # Check by status code for HTTP errors
            if hasattr(e, "status_code") and e.status_code in cfg.retryable_statuses:
                should_retry = True
            elif hasattr(e, "response") and hasattr(e.response, "status_code"):
                if e.response.status_code in cfg.retryable_statuses:
                    should_retry = True

            # Check by exception type
            if isinstance(e, cfg.retryable_exceptions):
                should_retry = True

            # Don't retry on last attempt
            if attempt >= cfg.max_retries:
                break

            if should_retry:
                delay = cfg.calculate_delay(attempt)
                logger.warning(
                    "retry_attempt",
                    attempt=attempt + 1,
                    max_retries=cfg.max_retries,
                    delay_ms=round(delay * 1000, 2),
                    error_type=type(e).__name__,
                )
                await asyncio.sleep(delay)
            else:
                # Don't retry non-retryable exceptions
                break

    # All retries exhausted or non-retryable error
    raise last_exception or Exception("Unknown error in retry logic")


# Global circuit breaker instance for upstream calls
_upstream_circuit_breaker: Optional[CircuitBreaker] = None


def get_circuit_breaker() -> CircuitBreaker:
    """Get or create the global circuit breaker instance."""
    global _upstream_circuit_breaker
    if _upstream_circuit_breaker is None:
        _upstream_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            half_open_max_calls=3,
        )
    return _upstream_circuit_breaker


def reset_circuit_breaker():
    """Reset the circuit breaker to closed state."""
    global _upstream_circuit_breaker
    _upstream_circuit_breaker = None
