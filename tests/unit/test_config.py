"""Unit tests for configuration module."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from config import (
    CONFIG,
    ModelCapability,
    ModelRegistry,
    ProxyConfig,
    load_model_registry,
)


class TestProxyConfig:
    """Tests for ProxyConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProxyConfig()
        assert config.proxy_host == "0.0.0.0"
        assert config.proxy_port == 8000
        assert config.default_model == "kimi-k2.5"
        assert config.max_retries == 3

    def test_log_level_validation(self):
        """Test log level validation."""
        config = ProxyConfig(log_level="debug")
        assert config.log_level == "DEBUG"

        config = ProxyConfig(log_level="invalid")
        assert config.log_level == "INFO"  # Falls back to default

    def test_log_format_validation(self):
        """Test log format validation."""
        config = ProxyConfig(log_format="json")
        assert config.log_format == "json"

        config = ProxyConfig(log_format="invalid")
        assert config.log_format == "json"  # Falls back to default


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_load_from_yaml(self, tmp_path):
        """Test loading registry from YAML file."""
        registry_data = {
            "models": {
                "test-model": {
                    "provider": "test",
                    "model_id": "test-model-v1",
                    "capabilities": ["streaming"],
                    "max_tokens": 4096,
                    "context_window": 128000,
                }
            },
            "providers": {
                "test": {
                    "base_url": "https://test.example.com",
                    "auth_header": "Authorization",
                    "supports_streaming": True,
                }
            },
        }

        yaml_path = tmp_path / "test_models.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(registry_data, f)

        # Temporarily override the registry path
        original_path = CONFIG.model_registry_path
        try:
            CONFIG.model_registry_path = str(yaml_path)
            registry = load_model_registry(str(yaml_path))

            assert "test-model" in registry.models
            model = registry.models["test-model"]
            assert model.provider == "test"
            assert model.max_tokens == 4096
        finally:
            CONFIG.model_registry_path = original_path

    def test_get_model(self):
        """Test getting model from registry."""
        registry = ModelRegistry(
            models={
                "gpt-4": ModelCapability(
                    provider="openai",
                    model_id="gpt-4",
                    capabilities=["streaming"],
                    max_tokens=4096,
                    context_window=128000,
                )
            },
            providers={},
        )

        model = registry.get_model("gpt-4")
        assert model is not None
        assert model.provider == "openai"

        missing = registry.get_model("nonexistent")
        assert missing is None

    def test_list_models(self):
        """Test listing models in registry."""
        registry = ModelRegistry(
            models={
                "model-a": ModelCapability(
                    provider="test",
                    model_id="model-a-v1",
                    capabilities=[],
                    max_tokens=1000,
                    context_window=10000,
                ),
                "model-b": ModelCapability(
                    provider="test",
                    model_id="model-b-v1",
                    capabilities=[],
                    max_tokens=2000,
                    context_window=20000,
                ),
            },
            providers={},
        )

        models = registry.list_models()
        assert len(models) == 2
        assert "model-a" in models
        assert "model-b" in models


class TestGlobalConfig:
    """Tests for global CONFIG."""

    def test_config_is_loaded(self):
        """Test that global config is loaded."""
        assert CONFIG is not None
        assert isinstance(CONFIG, ProxyConfig)
