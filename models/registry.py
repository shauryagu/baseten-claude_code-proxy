"""
Model handler registry with pattern matching.

Provides a registry for model-specific handlers with pattern matching support.
"""

import fnmatch
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Type, Union

from config import ModelCapability, load_model_registry
from models import DefaultHandler, ModelHandler

if TYPE_CHECKING:
    from models.kimi import KimiHandler
    from models.openai import OpenAIHandler


class ModelHandlerRegistry:
    """Registry for model-specific handlers."""

    _handlers: dict[str, Type[ModelHandler]] = {}
    _instance_cache: dict[str, ModelHandler] = {}

    @classmethod
    def register(cls, model_pattern: str, handler: Type[ModelHandler]) -> None:
        """
        Register a handler for models matching pattern.

        Args:
            model_pattern: Glob pattern to match model IDs
            handler: Handler class for matching models
        """
        cls._handlers[model_pattern] = handler
        # Clear cache when new handler registered
        cls._instance_cache.clear()

    @classmethod
    @lru_cache(maxsize=128)
    def get_handler(cls, model_id: str) -> Type[ModelHandler]:
        """
        Get appropriate handler class for model.

        Args:
            model_id: Model identifier

        Returns:
            Handler class for the model
        """
        for pattern, handler_class in cls._handlers.items():
            if fnmatch.fnmatch(model_id.lower(), pattern.lower()):
                return handler_class
        return DefaultHandler

    @classmethod
    def get_handler_instance(
        cls, model_id: str, capability: Optional[ModelCapability] = None
    ) -> ModelHandler:
        """
        Get handler instance for model.

        Args:
            model_id: Model identifier
            capability: Optional model capability info

        Returns:
            Handler instance for the model
        """
        cache_key = f"{model_id}:{id(capability)}"
        if cache_key not in cls._instance_cache:
            handler_class = cls.get_handler(model_id)
            cls._instance_cache[cache_key] = handler_class(capability)
        return cls._instance_cache[cache_key]

    @classmethod
    def list_patterns(cls) -> list[str]:
        """List all registered patterns."""
        return list(cls._handlers.keys())

    @classmethod
    def clear_cache(cls) -> None:
        """Clear handler instance cache."""
        cls._instance_cache.clear()
        cls.get_handler.cache_clear()


def _register_handlers() -> None:
    """Register model handlers - called after all imports are resolved."""
    # Import here to avoid circular imports
    from models.kimi import KimiHandler
    from models.openai import OpenAIHandler

    ModelHandlerRegistry.register("*kimi*", KimiHandler)
    ModelHandlerRegistry.register("*moonshot*", KimiHandler)
    ModelHandlerRegistry.register("*gpt*", OpenAIHandler)
    ModelHandlerRegistry.register("*openai*", OpenAIHandler)


# Register handlers on module load
_register_handlers()
