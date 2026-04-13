"""
Performance optimizations for the proxy.

Provides connection pooling, caching, and other performance enhancements.
"""

import functools
import hashlib
import re
from collections.abc import Callable
from typing import Any, Optional, TypeVar, Union

import httpx

from config import CONFIG
from logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class ConnectionPool:
    """HTTP connection pool manager."""

    _instance: Optional[httpx.AsyncClient] = None
    _lock = None

    @classmethod
    def get_client(cls) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if cls._instance is None:
            # No timeout — tool calls and long streaming responses must not be cut off.
            # This matches the behaviour of the proven app_optimized.py client.
            cls._instance = httpx.AsyncClient(
                timeout=httpx.Timeout(None),
                limits=httpx.Limits(
                    max_keepalive_connections=CONFIG.keepalive_connections,
                    max_connections=CONFIG.max_connections,
                ),
                http2=True,
            )
            logger.info(
                "http_client_initialized",
                max_connections=CONFIG.max_connections,
                keepalive=CONFIG.keepalive_connections,
                http2=True,
            )
        return cls._instance

    @classmethod
    async def close(cls) -> None:
        """Close the HTTP client."""
        if cls._instance is not None:
            await cls._instance.aclose()
            cls._instance = None
            logger.info("http_client_closed")


class Cache:
    """Simple in-memory cache with TTL."""

    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache.

        Args:
            default_ttl: Default TTL in seconds
        """
        self._cache: dict[str, tuple[Any, float]] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]
        if time.time() > expiry:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        expiry = time.time() + (ttl or self._default_ttl)
        self._cache[key] = (value, expiry)

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def cleanup(self) -> int:
        """Remove expired entries and return count removed."""
        now = time.time()
        expired = [k for k, (_, expiry) in self._cache.items() if now > expiry]
        for k in expired:
            del self._cache[k]
        return len(expired)


# Global cache instances
_tool_cache: Optional[Cache] = None
_model_cache: Optional[Cache] = None


def get_tool_cache() -> Cache:
    """Get or create global tool cache."""
    global _tool_cache
    if _tool_cache is None:
        _tool_cache = Cache(default_ttl=3600)  # 1 hour TTL for tools
    return _tool_cache


def get_model_cache() -> Cache:
    """Get or create global model cache."""
    global _model_cache
    if _model_cache is None:
        _model_cache = Cache(default_ttl=300)  # 5 min TTL for model metadata
    return _model_cache


def memoize(ttl: int = 300) -> Callable:
    """
    Decorator for memoizing function results.

    Args:
        ttl: Cache TTL in seconds

    Returns:
        Decorator function
    """
    cache = Cache(default_ttl=ttl)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            for arg in args:
                key_parts.append(_make_hashable(arg))
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={_make_hashable(v)}")

            key = hashlib.sha256(
                "|".join(str(p) for p in key_parts).encode()
            ).hexdigest()

            # Try cache first
            result = cache.get(key)
            if result is not None:
                return result

            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        return wrapper

    return decorator


def _make_hashable(obj: Any) -> str:
    """Convert object to hashable string."""
    if isinstance(obj, (str, int, float, bool)):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_make_hashable(x) for x in obj) + "]"
    if isinstance(obj, dict):
        return "{" + ",".join(f"{k}:{_make_hashable(v)}" for k, v in sorted(obj.items())) + "}"
    return str(obj)


# Pre-compiled regex patterns for performance
class Patterns:
    """Pre-compiled regex patterns."""

    # Tool call patterns
    TOOL_CALL_BEGIN = re.compile(r'   <|tool_calls_section_begin|>', re.UNICODE)
    TOOL_CALL_END = re.compile(r'   <|tool_calls_section_end|>', re.UNICODE)
    TOOL_CALL_FUNCTION = re.compile(r'functions\.(\w+)', re.UNICODE)

    # JSON patterns
    JSON_OBJECT = re.compile(r'\{[^{}]*\}', re.DOTALL)
    JSON_ARRAY = re.compile(r'\[[^\[\]]*\]', re.DOTALL)

    # Content patterns
    WHITESPACE = re.compile(r'\s+')
    NEWLINES = re.compile(r'\n{3,}')


# Compile patterns on module load
PATTERNS = Patterns()


# Import at end to avoid circular imports
import asyncio  # noqa: E402
import time  # noqa: E402
