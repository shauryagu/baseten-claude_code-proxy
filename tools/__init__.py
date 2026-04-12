"""
Tool format adapters for different model providers.

This module provides adapters for different tool calling formats:
- Native: Standard OpenAI structured tool_calls
- Embedded: Kimi's text-embedded tool syntax ( <|tool_calls_section_begin|>... <|tool_calls_section_end|>)
"""

from abc import ABC, abstractmethod
from typing import Any


class ToolFormatAdapter(ABC):
    """Abstract base class for tool format adapters."""

    @abstractmethod
    def detect(self, text: str) -> bool:
        """Detect if text contains tool calls in this format."""
        pass

    @abstractmethod
    def parse(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """
        Extract tool calls from text.

        Returns:
            Tuple of (clean_text, tool_uses)
        """
        pass

    @abstractmethod
    def format_request(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format tools for upstream request.

        Args:
            tools: List of tool definitions

        Returns:
            Formatted tools for the specific provider
        """
        pass

    @abstractmethod
    def format_response(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format tool calls for Anthropic response.

        Args:
            tool_calls: Raw tool calls from upstream

        Returns:
            Formatted tool_use blocks for Anthropic format
        """
        pass


# Import implementations
from .kimi_adapter import KimiAdapter
from .openai_adapter import OpenAIAdapter

__all__ = [
    "ToolFormatAdapter",
    "KimiAdapter",
    "OpenAIAdapter",
]
