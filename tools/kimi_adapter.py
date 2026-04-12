"""
Kimi tool format adapter for embedded tool syntax.

Kimi uses a special embedded syntax for tool calls:
  <|tool_calls_section_begin|>functions.tool_name:0{"arg": "value"}  <|tool_calls_section_end|>
"""

import json
import re
import uuid
from typing import Any

from tools import ToolFormatAdapter


class KimiAdapter(ToolFormatAdapter):
    """Adapter for Kimi's embedded tool syntax."""

    # Pattern for Kimi tool calls
    # Matches:  <|tool_calls_section_begin|>functions.tool_name:0{"arg": "value"}  <|tool_calls_section_end|>
    _KIMI_CALL_RE = re.compile(
        r' <|tool_calls_section_begin|>(?:functions\.)?(\w+)(?::\d+)?\s*({.*?})\s*  <|tool_calls_section_end|>',
        re.DOTALL,
    )

    # Alternative pattern for simpler format
    _KIMI_SIMPLE_RE = re.compile(
        r' <|tool_calls_section_begin|>(?:functions\.)?(\w+)(?::\d+)?\s*([^}]*)\s* <|tool_calls_section_end|>',
        re.DOTALL,
    )

    def detect(self, text: str) -> bool:
        """Detect if text contains Kimi tool calls."""
        return "  <|tool_calls_section_begin|>" in text and "  <|tool_calls_section_end|>" in text

    def parse(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """
        Extract tool calls from Kimi's embedded syntax.

        Returns:
            Tuple of (clean_text, tool_uses)
        """
        # Extract text outside tool sections
        clean_parts = []
        last_end = 0

        tool_uses = []

        # Find all tool calls
        for match in self._KIMI_CALL_RE.finditer(text):
            # Add text before this tool call
            before = text[last_end:match.start()]
            if before.strip():
                clean_parts.append(before)

            # Parse tool call
            tool_name = match.group(1)
            args_str = match.group(2).strip()

            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                # Try to parse as simple string
                args = {"raw": args_str}

            tool_uses.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": tool_name,
                "input": args,
            })

            last_end = match.end()

        # Add remaining text
        remaining = text[last_end:]
        if remaining.strip():
            clean_parts.append(remaining)

        clean_text = "".join(clean_parts).strip()

        return clean_text, tool_uses

    def format_request(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format tools for Kimi upstream request.

        Kimi uses standard OpenAI format for tool definitions.
        """
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

    def format_response(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format tool calls for Anthropic response.

        Args:
            tool_calls: Raw tool calls from upstream

        Returns:
            Formatted tool_use blocks for Anthropic format
        """
        formatted = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"raw": args_str}

            formatted.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": fn.get("name", ""),
                "input": args,
            })
        return formatted
