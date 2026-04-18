"""
Configuration management for Anthropic Proxy.

Uses Pydantic for type-safe configuration with environment variable overrides.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelCapability(BaseModel):
    """Capabilities for a specific model."""

    provider: str
    model_id: str
    capabilities: list[str] = Field(default_factory=list)
    tool_format: Literal["native", "embedded"] = "native"
    max_tokens: int = 4096
    context_window: int = 128000


class ProviderConfig(BaseModel):
    """Configuration for a provider."""

    base_url: str
    auth_header: str = "Authorization"
    auth_prefix: Optional[str] = "Bearer"
    supports_streaming: bool = True
    supports_tools: bool = True


class ModelRegistry(BaseModel):
    """Registry of models and their configurations."""

    models: dict[str, ModelCapability]
    providers: dict[str, ProviderConfig]

    def get_model(self, alias: str) -> Optional[ModelCapability]:
        """Get model configuration by alias."""
        return self.models.get(alias)

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get provider configuration by name."""
        return self.providers.get(name)

    def list_models(self) -> list[str]:
        """List all available model aliases."""
        return list(self.models.keys())


class ProxyConfig(BaseSettings):
    """Main proxy configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    baseten_api_key: Optional[str] = Field(default=None, alias="BASETEN_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    # Proxy settings
    proxy_auth_key: Optional[str] = Field(default=None, alias="PROXY_AUTH_KEY")
    proxy_host: str = Field(default="0.0.0.0", alias="PROXY_HOST")
    proxy_port: int = Field(default=8000, alias="PROXY_PORT")

    # Model configuration
    default_model: str = Field(default="kimi-k2.5", alias="DEFAULT_MODEL")
    model_registry_path: str = Field(default="models.yaml", alias="MODEL_REGISTRY_PATH")

    # Provider overrides
    baseten_base_url: str = Field(
        default="https://inference.baseten.co/v1", alias="BASETEN_BASE_URL"
    )
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")

    # Performance settings
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    retry_delay: float = Field(default=1.0, alias="RETRY_DELAY")
    request_timeout: float = Field(default=300.0, alias="REQUEST_TIMEOUT")
    max_connections: int = Field(default=200, alias="MAX_CONNECTIONS")
    keepalive_connections: int = Field(default=100, alias="KEEPALIVE_CONNECTIONS")

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, alias="RATE_LIMIT_WINDOW")
    rate_limit_by_key: bool = Field(default=True, alias="RATE_LIMIT_BY_KEY")

    # Circuit breaker (can be disabled since Baseten handles this)
    circuit_breaker_enabled: bool = Field(default=False, alias="CIRCUIT_BREAKER_ENABLED")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")
    log_path: Optional[str] = Field(default=None, alias="LOG_PATH")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the allowed values."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        return v.upper() if v.upper() in allowed else "INFO"

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format is one of the allowed values."""
        return v if v in {"json", "text"} else "json"


@lru_cache(maxsize=1)
def load_model_registry(path: Optional[str] = None) -> ModelRegistry:
    """Load model registry from YAML file (cached)."""
    config = ProxyConfig()
    registry_path = path or config.model_registry_path

    # Try to find the file
    possible_paths = [
        Path(registry_path),
        Path(__file__).parent / registry_path,
        Path("/app") / registry_path,
    ]

    for p in possible_paths:
        if p.exists():
            with open(p, "r") as f:
                data = yaml.safe_load(f)
                return ModelRegistry(**data)

    # Return default registry if file not found
    return ModelRegistry(
        models={
            "kimi-k2.5": ModelCapability(
                provider="baseten",
                model_id="moonshotai/Kimi-K2.5",
                capabilities=["tools", "streaming"],
                tool_format="embedded",
                max_tokens=8192,
                context_window=256000,
            ),
            "gpt-4o": ModelCapability(
                provider="openai",
                model_id="gpt-4o-2024-08-06",
                capabilities=["tools", "vision", "streaming", "json_mode"],
                max_tokens=4096,
                context_window=128000,
            ),
        },
        providers={
            "baseten": ProviderConfig(
                base_url="https://inference.baseten.co/v1",
                auth_header="Authorization",
                auth_prefix="Bearer",
                supports_streaming=True,
                supports_tools=True,
            ),
            "openai": ProviderConfig(
                base_url="https://api.openai.com/v1",
                auth_header="Authorization",
                auth_prefix="Bearer",
                supports_streaming=True,
                supports_tools=True,
            ),
        },
    )


# Global configuration instance
CONFIG = ProxyConfig()
