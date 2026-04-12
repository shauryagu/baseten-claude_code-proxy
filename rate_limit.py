"""
Rate limiting with token bucket algorithm.

Provides per-API-key and per-model rate limiting with proper headers.
Supports:
  - RATE_LIMIT_ENABLED=false  → bypass all limiting (useful for local single-user setups)
  - RATE_LIMIT_BY_KEY=false   → use a single global bucket instead of per-caller buckets
  - RATE_LIMIT_WINDOW         → now actually read from config (was hardcoded to 60)
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from errors import RateLimitError
from logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: int   # Maximum tokens
    refill_rate: float  # Tokens per second
    _tokens: float = field(default=0.0, init=False)
    _last_refill: float = field(default_factory=time.time, init=False)

    def __post_init__(self):
        self._tokens = float(self.capacity)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.capacity,
            self._tokens + elapsed * self.refill_rate,
        )
        self._last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    @property
    def tokens(self) -> int:
        """Get current token count."""
        self._refill()
        return int(self._tokens)

    @property
    def reset_time(self) -> float:
        """Get timestamp when bucket will be full."""
        tokens_needed = self.capacity - self._tokens
        return self._last_refill + (tokens_needed / self.refill_rate)


# ---------------------------------------------------------------------------
# Unlimited pass-through (used when RATE_LIMIT_ENABLED=false)
# ---------------------------------------------------------------------------


class UnlimitedRateLimiter:
    """
    No-op rate limiter returned when rate limiting is disabled.

    Always allows requests and returns an empty headers dict so
    callers never need to branch on whether limiting is active.
    """

    def check_rate_limit(
        self,
        api_key: str,
        model: Optional[str] = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Always allow; return empty rate-limit headers."""
        return True, {}


# ---------------------------------------------------------------------------
# Real rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """
    Rate limiter with per-key (or global) and per-model token buckets.

    When by_key=True  → each distinct api_key gets its own bucket.
    When by_key=False → all callers share a single '__global__' bucket,
                        useful for enforcing a server-wide ceiling.
    """

    _GLOBAL_KEY = "__global__"

    def __init__(
        self,
        requests_per_window: int = 100,
        window_seconds: int = 60,
        by_key: bool = True,
        per_model_limits: Optional[dict[str, tuple[int, int]]] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_window: Default requests allowed per window
            window_seconds: Duration of the sliding window in seconds
            by_key: True = per-caller buckets; False = one shared global bucket
            per_model_limits: Optional dict of model_id -> (requests, window_seconds)
        """
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.by_key = by_key
        self.per_model_limits = per_model_limits or {}

        # Buckets keyed by caller API key (or _GLOBAL_KEY when by_key=False)
        self._key_buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=self.requests_per_window,
                refill_rate=self.requests_per_window / self.window_seconds,
            )
        )

        # Buckets keyed by (caller, model) for per-model limits
        self._model_buckets: dict[tuple[str, str], TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=self.requests_per_window,
                refill_rate=self.requests_per_window / self.window_seconds,
            )
        )

    def _resolve_key(self, api_key: str) -> str:
        """Return the bucket key to use based on the by_key setting."""
        return api_key if self.by_key else self._GLOBAL_KEY

    def _get_model_bucket(self, bucket_key: str, model: str) -> TokenBucket:
        """Get or create bucket for model-specific limits."""
        key = (bucket_key, model)
        if key not in self._model_buckets:
            if model in self.per_model_limits:
                requests, window = self.per_model_limits[model]
                self._model_buckets[key] = TokenBucket(
                    capacity=requests,
                    refill_rate=requests / window,
                )
            else:
                self._model_buckets[key] = TokenBucket(
                    capacity=self.requests_per_window,
                    refill_rate=self.requests_per_window / self.window_seconds,
                )
        return self._model_buckets[key]

    def check_rate_limit(
        self,
        api_key: str,
        model: Optional[str] = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if request is within rate limit.

        Args:
            api_key: Caller's API key (ignored when by_key=False)
            model: Optional model for model-specific limits

        Returns:
            Tuple of (allowed, rate-limit headers dict)
        """
        bucket_key = self._resolve_key(api_key)
        key_bucket = self._key_buckets[bucket_key]
        key_allowed = key_bucket.consume()

        # Check optional model-specific bucket
        model_allowed = True
        model_bucket = None
        if model:
            model_bucket = self._get_model_bucket(bucket_key, model)
            model_allowed = model_bucket.consume()

        # Build standard rate-limit response headers
        headers: dict[str, Any] = {
            "X-RateLimit-Limit": str(key_bucket.capacity),
            "X-RateLimit-Remaining": str(key_bucket.tokens),
            "X-RateLimit-Reset": str(int(key_bucket.reset_time)),
        }

        if model_bucket:
            headers["X-RateLimit-Model-Limit"] = str(model_bucket.capacity)
            headers["X-RateLimit-Model-Remaining"] = str(model_bucket.tokens)

        allowed = key_allowed and model_allowed

        if not allowed:
            # Calculate how long until the relevant bucket refills
            retry_after = 1
            if not key_allowed:
                retry_after = max(
                    retry_after,
                    int(key_bucket.reset_time - time.time()) + 1,
                )
            if model_bucket and not model_allowed:
                retry_after = max(
                    retry_after,
                    int(model_bucket.reset_time - time.time()) + 1,
                )
            headers["Retry-After"] = str(retry_after)

            logger.warning(
                "rate_limit_exceeded",
                source="proxy_local",
                bucket_key=bucket_key,
                model=model,
                retry_after=retry_after,
                by_key=self.by_key,
            )

        return allowed, headers


# ---------------------------------------------------------------------------
# Factory and global singleton
# ---------------------------------------------------------------------------


def create_rate_limiter(
    requests_per_window: int = 100,
    window_seconds: int = 60,
    by_key: bool = True,
    per_model_limits: Optional[dict[str, tuple[int, int]]] = None,
) -> RateLimiter:
    """
    Create a rate limiter with the specified configuration.

    Args:
        requests_per_window: Requests allowed per window
        window_seconds: Window duration in seconds
        by_key: True = per-caller buckets; False = one shared global bucket
        per_model_limits: Dict of model_id -> (requests, window_seconds)

    Returns:
        Configured RateLimiter instance
    """
    return RateLimiter(
        requests_per_window=requests_per_window,
        window_seconds=window_seconds,
        by_key=by_key,
        per_model_limits=per_model_limits,
    )


# Global rate limiter instance (RateLimiter or UnlimitedRateLimiter)
_rate_limiter: Optional[Any] = None


def get_rate_limiter() -> Any:
    """
    Get or create global rate limiter.

    Reads RATE_LIMIT_ENABLED, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW,
    and RATE_LIMIT_BY_KEY from CONFIG on first call so that env-var
    overrides are actually honoured at runtime.
    """
    global _rate_limiter
    if _rate_limiter is None:
        # Import here to avoid circular imports at module load time
        from config import CONFIG

        if not CONFIG.rate_limit_enabled:
            logger.info("rate_limiting_disabled", reason="RATE_LIMIT_ENABLED=false")
            _rate_limiter = UnlimitedRateLimiter()
        else:
            _rate_limiter = create_rate_limiter(
                requests_per_window=CONFIG.rate_limit_requests,
                window_seconds=CONFIG.rate_limit_window,
                by_key=CONFIG.rate_limit_by_key,
            )
            logger.info(
                "rate_limiter_initialised",
                requests_per_window=CONFIG.rate_limit_requests,
                window_seconds=CONFIG.rate_limit_window,
                by_key=CONFIG.rate_limit_by_key,
            )

    return _rate_limiter


def set_rate_limiter(limiter: Any) -> None:
    """Set global rate limiter (used in tests)."""
    global _rate_limiter
    _rate_limiter = limiter
