#!/usr/bin/env python3
"""
CLI entry point for Anthropic Proxy lifecycle management.

Usage:
    proxy start [--config PATH] [--port PORT] [--daemon] [--workers N]
    proxy stop [--graceful-timeout SECONDS]
    proxy restart [--zero-downtime]
    proxy status
    proxy health [--verbose]
    proxy config validate
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from config import CONFIG, load_model_registry
from health import HealthChecker, HealthStatusEnum
from lifecycle import get_proxy_manager
from logging_config import setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="proxy",
        description="Anthropic Proxy lifecycle management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  proxy start --port 8080 --daemon
  proxy status
  proxy health --verbose
  proxy stop --graceful-timeout 30
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the proxy server")
    start_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=CONFIG.proxy_port,
        help=f"Port to bind to (default: {CONFIG.proxy_port})",
    )
    start_parser.add_argument(
        "--host",
        type=str,
        default=CONFIG.proxy_host,
        help=f"Host to bind to (default: {CONFIG.proxy_host})",
    )
    start_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)",
    )
    start_parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon process",
    )
    start_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes",
    )

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the proxy server")
    stop_parser.add_argument(
        "--graceful-timeout",
        type=float,
        default=30.0,
        help="Timeout for graceful shutdown in seconds",
    )

    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart the proxy server")
    restart_parser.add_argument(
        "--zero-downtime",
        action="store_true",
        help="Use zero-downtime restart (if supported)",
    )
    restart_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )

    # Status command
    subparsers.add_parser("status", help="Get proxy status")

    # Health command
    health_parser = subparsers.add_parser("health", help="Check proxy health")
    health_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed health information",
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    config_subparsers.add_parser("validate", help="Validate configuration")
    config_subparsers.add_parser("show", help="Show current configuration")

    return parser


async def start_proxy(args: argparse.Namespace) -> int:
    """Start the proxy server."""
    manager = get_proxy_manager()

    try:
        await manager.start(
            host=args.host,
            port=args.port,
            workers=args.workers,
            daemon=args.daemon,
            reload=args.reload,
        )
        return 0
    except Exception as e:
        print(f"Failed to start proxy: {e}", file=sys.stderr)
        return 1


async def stop_proxy(args: argparse.Namespace) -> int:
    """Stop the proxy server."""
    manager = get_proxy_manager()

    try:
        await manager.stop(graceful=True, timeout=args.graceful_timeout)
        print("Proxy stopped successfully")
        return 0
    except Exception as e:
        print(f"Failed to stop proxy: {e}", file=sys.stderr)
        return 1


async def restart_proxy(args: argparse.Namespace) -> int:
    """Restart the proxy server."""
    manager = get_proxy_manager()

    try:
        await manager.restart(zero_downtime=args.zero_downtime)
        print("Proxy restarted successfully")
        return 0
    except Exception as e:
        print(f"Failed to restart proxy: {e}", file=sys.stderr)
        return 1


def show_status() -> int:
    """Show proxy status."""
    manager = get_proxy_manager()
    status = manager.status()

    if status["running"]:
        print(f"Status: Running")
        print(f"PID: {status.get('pid', 'N/A')}")
        print(f"Host: {status.get('host', 'N/A')}")
        print(f"Port: {status.get('port', 'N/A')}")

        uptime = status.get('uptime_seconds', 0)
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        print(f"Uptime: {hours}h {minutes}m {seconds}s")
    else:
        print("Status: Not running")

    return 0


async def check_health(verbose: bool = False) -> int:
    """Check proxy health."""
    checker = get_health_checker()
    health = await checker.check(include_upstream=True)

    if verbose:
        print(json.dumps(health.to_dict(), indent=2))
    else:
        print(f"Status: {health.status.value}")
        print(f"Latency: {health.latency_ms:.2f}ms")
        for name, check in health.checks.items():
            status_symbol = "✓" if check.status == HealthStatusEnum.HEALTHY else "✗"
            print(f"  {status_symbol} {name}: {check.status.value}")

    return 0 if health.status == HealthStatusEnum.HEALTHY else 1


def validate_config() -> int:
    """Validate configuration."""
    try:
        # Try to load model registry
        registry = load_model_registry()
        print(f"✓ Model registry loaded: {len(registry.models)} models")

        # Validate current config
        print(f"✓ Configuration valid")
        print(f"  Default model: {CONFIG.default_model}")
        print(f"  Proxy host: {CONFIG.proxy_host}")
        print(f"  Proxy port: {CONFIG.proxy_port}")

        # Check API keys
        if CONFIG.baseten_api_key:
            print("  ✓ Baseten API key configured")
        else:
            print("  ⚠ Baseten API key not configured")

        if CONFIG.openai_api_key:
            print("  ✓ OpenAI API key configured")
        else:
            print("  ⚠ OpenAI API key not configured")

        return 0
    except Exception as e:
        print(f"✗ Configuration invalid: {e}")
        return 1


def show_config() -> int:
    """Show current configuration."""
    config_dict = {
        "proxy": {
            "host": CONFIG.proxy_host,
            "port": CONFIG.proxy_port,
            "auth_key_configured": bool(CONFIG.proxy_auth_key),
        },
        "models": {
            "default": CONFIG.default_model,
            "registry_path": CONFIG.model_registry_path,
        },
        "providers": {
            "baseten": {
                "base_url": CONFIG.baseten_base_url,
                "api_key_configured": bool(CONFIG.baseten_api_key),
            },
            "openai": {
                "base_url": CONFIG.openai_base_url,
                "api_key_configured": bool(CONFIG.openai_api_key),
            },
        },
        "performance": {
            "max_retries": CONFIG.max_retries,
            "request_timeout": CONFIG.request_timeout,
            "max_connections": CONFIG.max_connections,
        },
        "rate_limiting": {
            "requests_per_minute": CONFIG.rate_limit_requests,
            "window_seconds": CONFIG.rate_limit_window,
            "by_key": CONFIG.rate_limit_by_key,
        },
        "logging": {
            "level": CONFIG.log_level,
            "format": CONFIG.log_format,
            "path": CONFIG.log_path,
        },
    }

    print(json.dumps(config_dict, indent=2))
    return 0


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Handle commands
    if args.command == "start":
        return asyncio.run(start_proxy(args))
    elif args.command == "stop":
        return asyncio.run(stop_proxy(args))
    elif args.command == "restart":
        return asyncio.run(restart_proxy(args))
    elif args.command == "status":
        return show_status()
    elif args.command == "health":
        return asyncio.run(check_health(args.verbose))
    elif args.command == "config":
        if args.config_command == "validate":
            return validate_config()
        elif args.config_command == "show":
            return show_config()
        else:
            print("Usage: proxy config [validate|show]")
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
