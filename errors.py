"""
Exception hierarchy for Anthropic Proxy.

Provides structured error handling with proper status codes and error types
for consistent API error responses.
"""

from typing import Any, Optional


class ProxyError(Exception):
    """Base proxy error with status code and error type."""

    status_code: int = 500
    error_type: str = "api_error"
    error_code: Optional[str] = None

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_type: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        if status_code:
            self.status_code = status_code
        if error_type:
            self.error_type = error_type
        if error_code:
            self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for JSON response."""
        result = {
            "error": {
                "type": self.error_type,
                "message": self.message,
            }
        }
        if self.error_code:
            result["error"]["code"] = self.error_code
        if self.details:
            result["error"]["details"] = self.details
        return result


class AuthenticationError(ProxyError):
    """Authentication failed."""

    status_code = 401
    error_type = "authentication_error"


class PermissionError(ProxyError):
    """Permission denied."""

    status_code = 403
    error_type = "permission_error"


class RateLimitError(ProxyError):
    """Rate limit exceeded."""

    status_code = 429
    error_type = "rate_limit_error"

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        limit: Optional[int] = None,
        remaining: int = 0,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining

    def to_dict(self) -> dict[str, Any]:
        """Add rate limit headers info to error dict."""
        result = super().to_dict()
        if self.retry_after:
            result["error"]["retry_after"] = self.retry_after
        return result


class ValidationError(ProxyError):
    """Request validation failed."""

    status_code = 400
    error_type = "invalid_request_error"


class NotFoundError(ProxyError):
    """Resource not found."""

    status_code = 404
    error_type = "not_found_error"


class UpstreamError(ProxyError):
    """Upstream provider error."""

    status_code = 502
    error_type = "upstream_error"

    def __init__(
        self,
        message: str,
        upstream_status: Optional[int] = None,
        upstream_body: Optional[dict[str, Any]] = None,
        provider: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.upstream_status = upstream_status
        self.upstream_body = upstream_body or {}
        self.provider = provider

    def to_dict(self) -> dict[str, Any]:
        """Add upstream details to error dict."""
        result = super().to_dict()
        if self.upstream_status:
            result["error"]["upstream_status"] = self.upstream_status
        if self.provider:
            result["error"]["provider"] = self.provider
        return result


class ServiceUnavailableError(ProxyError):
    """Service temporarily unavailable."""

    status_code = 503
    error_type = "service_unavailable_error"


class GatewayTimeoutError(ProxyError):
    """Gateway timeout."""

    status_code = 504
    error_type = "gateway_timeout_error"


class CircuitBreakerOpenError(ProxyError):
    """Circuit breaker is open."""

    status_code = 503
    error_type = "circuit_breaker_open"

    def __init__(self, message: str, retry_after: int = 30, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


def create_error_response(error: ProxyError) -> dict[str, Any]:
    """Create a standardized error response dictionary."""
    return error.to_dict()
