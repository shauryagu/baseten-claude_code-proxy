"""
Comprehensive health checking system.

Provides multi-layer health checks for the proxy and upstream providers.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import httpx

from config import CONFIG, load_model_registry
from errors import UpstreamError
from logging_config import get_logger
from performance import ConnectionPool

logger = get_logger(__name__)


class HealthStatusEnum(Enum):
    """Health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class CheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatusEnum
    latency_ms: float
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """Overall health status."""

    status: HealthStatusEnum
    checks: dict[str, CheckResult]
    latency_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp,
            "checks": {
                name: {
                    "status": check.status.value,
                    "latency_ms": round(check.latency_ms, 2),
                    "message": check.message,
                    "details": check.details,
                }
                for name, check in self.checks.items()
            },
        }


class HealthChecker:
    """Multi-layer health checking system."""

    def __init__(self):
        self._last_check: Optional[HealthStatus] = None
        self._check_interval = 10.0  # Minimum seconds between full checks
        self._last_check_time = 0.0
        self._cache: dict[str, CheckResult] = {}

    async def check(self, include_upstream: bool = True) -> HealthStatus:
        """
        Run comprehensive health checks.

        Args:
            include_upstream: Whether to check upstream connectivity

        Returns:
            HealthStatus with all check results
        """
        start_time = time.time()

        # Use cached result if recent
        if (
            self._last_check
            and time.time() - self._last_check_time < self._check_interval
            and not include_upstream
        ):
            return self._last_check

        checks: dict[str, CheckResult] = {}

        # Basic connectivity check
        checks["basic"] = await self._check_basic_connectivity()

        # Memory usage check
        checks["memory"] = await self._check_memory_usage()

        # Upstream connectivity check (if requested)
        if include_upstream:
            checks["upstream"] = await self._check_upstream_health()

        # Aggregate status
        overall_status = self._aggregate_status(list(checks.values()))

        health_status = HealthStatus(
            status=overall_status,
            checks=checks,
            latency_ms=(time.time() - start_time) * 1000,
        )

        # Cache result
        self._last_check = health_status
        self._last_check_time = time.time()

        return health_status

    async def _check_basic_connectivity(self) -> CheckResult:
        """Check basic proxy connectivity."""
        start = time.time()
        try:
            # Simple self-check
            latency = (time.time() - start) * 1000
            return CheckResult(
                name="basic",
                status=HealthStatusEnum.HEALTHY,
                latency_ms=latency,
                message="Proxy is operational",
            )
        except Exception as e:
            return CheckResult(
                name="basic",
                status=HealthStatusEnum.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=f"Basic check failed: {e}",
            )

    async def _check_memory_usage(self) -> CheckResult:
        """Check memory usage."""
        start = time.time()
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            # Determine status based on memory usage
            if memory_percent > 90:
                status = HealthStatusEnum.UNHEALTHY
                message = f"Critical memory usage: {memory_percent:.1f}%"
            elif memory_percent > 75:
                status = HealthStatusEnum.DEGRADED
                message = f"High memory usage: {memory_percent:.1f}%"
            else:
                status = HealthStatusEnum.HEALTHY
                message = f"Memory usage: {memory_percent:.1f}% ({memory_info.rss / 1024 / 1024:.1f} MB)"

            return CheckResult(
                name="memory",
                status=status,
                latency_ms=(time.time() - start) * 1000,
                message=message,
                details={
                    "rss_mb": memory_info.rss / 1024 / 1024,
                    "vms_mb": memory_info.vms / 1024 / 1024,
                    "percent": memory_percent,
                },
            )
        except ImportError:
            return CheckResult(
                name="memory",
                status=HealthStatusEnum.HEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message="Memory check skipped (psutil not installed)",
            )
        except Exception as e:
            return CheckResult(
                name="memory",
                status=HealthStatusEnum.DEGRADED,
                latency_ms=(time.time() - start) * 1000,
                message=f"Memory check failed: {e}",
            )

    async def _check_upstream_health(self) -> CheckResult:
        """Check upstream provider health with lightweight probe."""
        start = time.time()
        try:
            # Get HTTP client
            from performance import ConnectionPool

            client = ConnectionPool.get_client()

            # Build lightweight health check request
            # Use a minimal request to check if upstream is responsive
            url = f"{CONFIG.baseten_base_url}/models"
            headers = {"Authorization": f"Bearer {CONFIG.baseten_api_key}"}

            # Short timeout for health check
            response = await client.get(url, headers=headers, timeout=5.0)

            latency = (time.time() - start) * 1000

            if response.status_code < 500:
                return CheckResult(
                    name="upstream",
                    status=HealthStatusEnum.HEALTHY,
                    latency_ms=latency,
                    message=f"Upstream is healthy ({response.status_code})",
                    details={
                        "status_code": response.status_code,
                        "upstream_url": CONFIG.baseten_base_url,
                    },
                )
            else:
                return CheckResult(
                    name="upstream",
                    status=HealthStatusEnum.DEGRADED,
                    latency_ms=latency,
                    message=f"Upstream returned error: {response.status_code}",
                    details={
                        "status_code": response.status_code,
                        "upstream_url": CONFIG.baseten_base_url,
                    },
                )

        except Exception as e:
            return CheckResult(
                name="upstream",
                status=HealthStatusEnum.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=f"Upstream health check failed: {e}",
                details={"error": str(e)},
            )

    def _aggregate_status(self, checks: list[CheckResult]) -> HealthStatusEnum:
        """Aggregate individual check results into overall status."""
        statuses = [c.status for c in checks]

        if any(s == HealthStatusEnum.UNHEALTHY for s in statuses):
            return HealthStatusEnum.UNHEALTHY
        if any(s == HealthStatusEnum.DEGRADED for s in statuses):
            return HealthStatusEnum.DEGRADED
        return HealthStatusEnum.HEALTHY


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get or create global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def reset_health_checker() -> None:
    """Reset health checker."""
    global _health_checker
    _health_checker = None
