"""
Model-specific handlers for different LLM providers.

Each model may need custom handling for:
- Message formatting quirks
- Tool call formats
- System prompt handling
- Token counting
- Stop sequences
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from config import ModelCapability  # noqa: F401


class ModelHandler(ABC):
    """Base class for model-specific handling."""

    def __init__(self, capability: Optional[ModelCapability] = None):
        self.capability = capability

    @abstractmethod
    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform messages for this model's quirks."""
        pass

    @abstractmethod
    def prepare_tools(
        self, tools: Optional[list[dict[str, Any]]]
    ) -> Optional[list[dict[str, Any]]]:
        """Transform tool definitions for this model."""
        pass

    @abstractmethod
    def prepare_system(self, system: Optional[Union[str, list[dict[str, Any]]]]) -> Optional[str]:
        """Transform system prompt for this model."""
        pass

    @abstractmethod
    def parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Parse model-specific response format."""
        pass

    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if this model supports streaming."""
        pass

    @abstractmethod
    def supports_tools(self) -> bool:
        """Check if this model supports tools."""
        pass

    def get_stop_sequences(self) -> Optional[list[str]]:
        """Get model-specific stop sequences."""
        return None

    def get_max_tokens(self) -> int:
        """Get maximum tokens for this model."""
        return self.capability.max_tokens if self.capability else 4096


class DefaultHandler(ModelHandler):
    """Default handler for standard OpenAI-compatible models."""

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pass through messages unchanged."""
        return messages

    def prepare_tools(
        self, tools: Optional[list[dict[str, Any]]]
    ) -> Optional[list[dict[str, Any]]]:
        """Pass through tools unchanged."""
        return tools

    def prepare_system(self, system: Optional[Union[str, list[dict[str, Any]]]]) -> Optional[str]:
        """Convert system to string."""
        if system is None:
            return None
        if isinstance(system, str):
            return system
        # Convert list of blocks to string
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n\n".join(parts)

    def parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Pass through response unchanged."""
        return response

    def supports_streaming(self) -> bool:
        """Check if streaming is supported."""
        return self.capability.supports_streaming if self.capability else True

    def supports_tools(self) -> bool:
        """Check if tools are supported."""
        return self.capability.supports_tools if self.capability else True


# Import specific handlers
from .kimi import KimiHandler
from .openai import OpenAIHandler

__all__ = [
    "ModelHandler",
    "DefaultHandler",
    "KimiHandler",
    "OpenAIHandler",
]
