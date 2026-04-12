"""
Middleware for request logging, metrics, and tracing.

Provides FastAPI middleware for:
- Request timing and latency metrics
- Token usage tracking
- Error rate monitoring
- Request ID propagation
"""

import time
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from logging_config import get_logger, set_request_id

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests with timing and metrics."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        """Process request and log metrics."""
        # Set request ID for this context
        request_id = request.headers.get("x-request-id") or request.headers.get("x-correlation-id")
        set_request_id(request_id)

        start_time = time.time()
        path = request.url.path
        method = request.method

        # Log request start
        logger.info(
            "request_started",
            method=method,
            path=path,
            client_ip=request.client.host if request.client else None,
        )

        try:
            response = await call_next(request)
            duration = time.time() - start_time

            # Add request ID to response headers
            response.headers["x-request-id"] = get_request_id()

            # Log successful response
            logger.info(
                "request_completed",
                method=method,
                path=path,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
            )

            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(
                "request_failed",
                method=method,
                path=path,
                error_type=type(e).__name__,
                duration_ms=round(duration * 1000, 2),
            )
            raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting metrics about requests."""

    def __init__(self, app: Any):
        super().__init__(app)
        self._request_count = 0
        self._error_count = 0
        self._total_latency = 0.0

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        """Track metrics for this request."""
        start_time = time.time()

        response = await call_next(request)

        # Update metrics
        self._request_count += 1
        self._total_latency += time.time() - start_time

        if response.status_code >= 400:
            self._error_count += 1

        # Add metrics headers
        response.headers["x-proxy-request-count"] = str(self._request_count)
        response.headers["x-proxy-error-count"] = str(self._error_count)

        return response

    @property
    def metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        avg_latency = (
            self._total_latency / self._request_count if self._request_count > 0 else 0
        )
        error_rate = (
            self._error_count / self._request_count if self._request_count > 0 else 0
        )
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "average_latency_ms": round(avg_latency * 1000, 2),
            "error_rate": round(error_rate, 4),
        }


def get_request_id() -> str:
    """Get current request ID from context."""
    from logging_config import get_request_id as _get_request_id

    return _get_request_id()
