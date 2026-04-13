"""
Kimi K2.5 specific handler.

Handles Kimi's quirks:
- Embedded tool syntax ( <|tool_calls_section_begin|>... <|tool_calls_section_end|>)
- System prompt formatting
- Message structure
"""

import json
import re
import uuid
from typing import Any, Optional

from config import ModelCapability
from models import DefaultHandler


class KimiHandler(DefaultHandler):
    """Kimi K2.5 specific handling — inherits Anthropic→OpenAI conversion from DefaultHandler."""

    # Pattern for Kimi tool calls in embedded format
    _KIMI_CALL_RE = re.compile(
        r'  <|tool_calls_section_begin|>(?:functions\.)?(\w+)(?::\d+)?\s*({.*?})\s*  <|tool_calls_section_end|>',
        re.DOTALL,
    )

    def __init__(self, capability: Optional[ModelCapability] = None):
        super().__init__(capability)
        self._tool_format = capability.tool_format if capability else "embedded"

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Transform messages for Kimi.

        Kimi K2.5 via Baseten uses a standard OpenAI-compatible endpoint,
        so standard role:"tool" messages work correctly. No rewriting needed.
        """
        return super().prepare_messages(messages)

    def parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """
        Parse Kimi response format.

        Kimi may return tool calls in embedded format within content.
        """
        choices = response.get("choices", [])
        if not choices:
            return response

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")

        # Check for embedded tool calls
        if "  <|tool_calls_section_begin|>" in content and "  <|tool_calls_section_end|>" in content:
            clean_text, tool_uses = self._parse_embedded_tools(content)

            # Update message
            if clean_text:
                message["content"] = clean_text
            else:
                message["content"] = None

            # Add tool_calls in OpenAI format
            if tool_uses:
                message["tool_calls"] = [
                    {
                        "id": tu["id"],
                        "type": "function",
                        "function": {
                            "name": tu["name"],
                            "arguments": json.dumps(tu["input"]),
                        },
                    }
                    for tu in tool_uses
                ]

            choice["message"] = message
            response["choices"] = [choice]

        return response

    def _parse_embedded_tools(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Parse embedded tool calls from Kimi response text."""
        clean_parts = []
        last_end = 0
        tool_uses = []

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

