"""
Lifecycle management for the proxy.

Provides CLI commands for starting, stopping, and managing the proxy server.
"""

import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import uvicorn

from config import CONFIG
from logging_config import get_logger, setup_logging
from performance import ConnectionPool

logger = get_logger(__name__)

# PID file for daemon mode
PID_FILE = Path("/tmp/anthropic_proxy.pid")
STATE_FILE = Path("/tmp/anthropic_proxy.state")


@dataclass
class ProxyState:
    """Proxy runtime state."""

    pid: int
    host: str
    port: int
    start_time: float
    request_count: int = 0
    error_count: int = 0
    active_connections: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pid": self.pid,
            "host": self.host,
            "port": self.port,
            "start_time": self.start_time,
            "uptime_seconds": time.time() - self.start_time,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "active_connections": self.active_connections,
        }


class ProxyManager:
    """Manager for proxy lifecycle."""

    def __init__(self):
        self._state: Optional[ProxyState] = None
        self._shutdown_event = asyncio.Event()
        self._server = None

    async def start(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 4,
        daemon: bool = False,
        reload: bool = False,
    ) -> None:
        """
        Start the proxy server.

        Args:
            host: Host to bind to
            port: Port to bind to
            workers: Number of worker processes
            daemon: Run as daemon process
            reload: Enable auto-reload on code changes
        """
        if daemon:
            self._daemonize()

        # Setup logging
        setup_logging()

        logger.info(
            "proxy_starting",
            host=host,
            port=port,
            workers=workers,
            daemon=daemon,
        )

        # Create state
        self._state = ProxyState(
            pid=os.getpid(),
            host=host,
            port=port,
            start_time=time.time(),
        )
        self._save_state()

        # Setup signal handlers
        self._setup_signal_handlers()

        # Import the modular app (app.py is frozen legacy code)
        from main import app

        # Configure uvicorn
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level=CONFIG.log_level.lower(),
            access_log=False,  # We handle logging ourselves
        )

        # Start server
        server = uvicorn.Server(config)
        self._server = server

        try:
            await server.serve()
        except asyncio.CancelledError:
            logger.info("proxy_cancelled")
        finally:
            await self._cleanup()

    async def stop(self, graceful: bool = True, timeout: float = 30.0) -> None:
        """
        Stop the proxy server.

        Args:
            graceful: Whether to wait for in-flight requests
            timeout: Timeout for graceful shutdown
        """
        logger.info("proxy_stopping", graceful=graceful, timeout=timeout)

        if self._server:
            self._server.should_exit = True

        if graceful:
            # Wait for connections to drain
            start = time.time()
            while time.time() - start < timeout:
                if self._state and self._state.active_connections == 0:
                    break
                await asyncio.sleep(0.1)

        await self._cleanup()

    async def restart(self, zero_downtime: bool = False) -> None:
        """
        Restart the proxy server.

        Args:
            zero_downtime: Whether to use zero-downtime restart
        """
        if zero_downtime:
            logger.info("zero_downtime_restart_not_implemented")
            # TODO: Implement zero-downtime restart with process handoff

        await self.stop(graceful=True)
        # Server will be restarted by process manager

    def status(self) -> dict[str, Any]:
        """Get current proxy status."""
        if self._state:
            return {
                "running": True,
                **self._state.to_dict(),
            }

        # Try to load state from file
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
                    # Check if process is still running
                    pid = state.get("pid")
                    if pid and self._is_process_running(pid):
                        return {"running": True, **state}
            except Exception:
                pass

        return {"running": False}

    def _daemonize(self) -> None:
        """Daemonize the process."""
        # First fork
        pid = os.fork()
        if pid > 0:
            # Parent exits
            sys.exit(0)

        # Decouple from parent environment
        os.chdir("/")
        os.setsid()
        os.umask(0)

        # Second fork
        pid = os.fork()
        if pid > 0:
            # Parent exits
            sys.exit(0)

        # Redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()

        with open("/dev/null", "r") as f:
            os.dup2(f.fileno(), sys.stdin.fileno())
        with open("/dev/null", "a+") as f:
            os.dup2(f.fileno(), sys.stdout.fileno())
            os.dup2(f.fileno(), sys.stderr.fileno())

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"received_signal_{sig_name}")
            self._shutdown_event.set()

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("proxy_cleanup_starting")

        # Close HTTP connections
        await ConnectionPool.close()

        # Remove state file
        if STATE_FILE.exists():
            STATE_FILE.unlink()

        # Remove PID file
        if PID_FILE.exists():
            PID_FILE.unlink()

        logger.info("proxy_cleanup_complete")

    def _save_state(self) -> None:
        """Save current state to file."""
        if self._state:
            with open(STATE_FILE, "w") as f:
                json.dump(self._state.to_dict(), f)

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is running."""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


# Global manager instance
_proxy_manager: Optional[ProxyManager] = None


def get_proxy_manager() -> ProxyManager:
    """Get or create global proxy manager."""
    global _proxy_manager
    if _proxy_manager is None:
        _proxy_manager = ProxyManager()
    return _proxy_manager
