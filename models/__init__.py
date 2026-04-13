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
    """Default handler that converts Anthropic message format to OpenAI-compatible format."""

    # ---- helpers for Anthropic → OpenAI content conversion ----

    @staticmethod
    def _content_blocks_to_text(content: Any) -> str:
        """Flatten Anthropic content blocks into a plain text string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts) if parts else ""
        return str(content) if content else ""

    @staticmethod
    def _extract_tool_uses(content: Any) -> list[dict[str, Any]]:
        """Extract tool_use blocks from Anthropic content and return OpenAI tool_calls."""
        if not isinstance(content, list):
            return []
        tool_calls = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                import json as _json
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": _json.dumps(block.get("input", {})),
                    },
                })
        return tool_calls

    @staticmethod
    def _extract_tool_results(content: Any) -> list[dict[str, Any]]:
        """Convert Anthropic tool_result content blocks into OpenAI tool messages."""
        if not isinstance(content, list):
            return []
        results = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, list):
                    # Nested content blocks inside tool_result
                    inner = DefaultHandler._content_blocks_to_text(inner)
                results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": str(inner),
                })
        return results

    # ---- interface implementations ----

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic-format messages to OpenAI-compatible messages."""
        formatted: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # User messages may contain tool_result blocks mixed with text.
            # Tool results must come BEFORE any user text — this ordering matches
            # app_optimized.py and is required for Kimi to correctly associate
            # results with the preceding assistant tool_calls.
            if role == "user" and isinstance(content, list):
                tool_results = self._extract_tool_results(content)
                text = self._content_blocks_to_text(content)

                if tool_results:
                    formatted.extend(tool_results)
                    if text.strip():
                        formatted.append({"role": "user", "content": text})
                else:
                    formatted.append({"role": "user", "content": text})
                continue

            # Assistant messages may contain tool_use blocks.
            # Only include the "content" key when there is actual text — omitting
            # it entirely (rather than setting None) matches the OpenAI spec and
            # avoids confusing upstream models.
            if role == "assistant":
                tool_calls = self._extract_tool_uses(content)
                text = self._content_blocks_to_text(content)
                openai_msg: dict[str, Any] = {"role": "assistant"}
                if text.strip():
                    openai_msg["content"] = text
                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls
                formatted.append(openai_msg)
                continue

            # Anything else (system messages forwarded via the messages array, etc.)
            formatted.append({
                "role": role,
                "content": self._content_blocks_to_text(content),
            })

        return formatted

    def prepare_tools(
        self, tools: Optional[list[dict[str, Any]]]
    ) -> Optional[list[dict[str, Any]]]:
        """Convert Anthropic tool definitions to OpenAI function-calling format."""
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
        """Convert Anthropic system prompt to plain string."""
        if system is None:
            return None
        return self._content_blocks_to_text(system)

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
