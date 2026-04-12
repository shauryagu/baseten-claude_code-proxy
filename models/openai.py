"""
OpenAI GPT models handler.

Handles OpenAI's standard format without special quirks.
"""

import json
import uuid
from typing import Any, Optional, Union

from config import ModelCapability
from models import ModelHandler


class OpenAIHandler(ModelHandler):
    """OpenAI GPT models handler."""

    def __init__(self, capability: Optional[ModelCapability] = None):
        super().__init__(capability)

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Transform messages for OpenAI.

        OpenAI uses standard message format, no transformation needed.
        """
        return messages

    def prepare_tools(
        self, tools: Optional[list[dict[str, Any]]]
    ) -> Optional[list[dict[str, Any]]]:
        """
        Transform tool definitions for OpenAI.

        OpenAI uses standard function format.
        """
        if not tools:
            return None

        formatted = []
        for tool in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return formatted

    def prepare_system(self, system: Optional[Union[str, list[dict[str, Any]]]]) -> Optional[str]:
        """
        Transform system prompt for OpenAI.

        OpenAI supports system messages as a string or list of content blocks.
        """
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
        """
        Parse OpenAI response format.

        OpenAI uses standard format, no transformation needed.
        """
        return response

    def supports_streaming(self) -> bool:
        """Check if OpenAI supports streaming."""
        return self.capability.supports_streaming if self.capability else True

    def supports_tools(self) -> bool:
        """Check if OpenAI supports tools."""
        return self.capability.supports_tools if self.capability else True

    def supports_vision(self) -> bool:
        """Check if OpenAI supports vision."""
        if self.capability and "vision" in self.capability.capabilities:
            return True
        return False

    def supports_json_mode(self) -> bool:
        """Check if OpenAI supports JSON mode."""
        if self.capability and "json_mode" in self.capability.capabilities:
            return True
        return False
