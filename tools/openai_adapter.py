"""
OpenAI tool format adapter for native structured tool_calls.

OpenAI uses a structured format for tool calls in the response:
{
    "tool_calls": [
        {
            "id": "call_xxx",
            "type": "function",
            "function": {
                "name": "tool_name",
                "arguments": '{"arg": "value"}'
            }
        }
    ]
}
"""

import json
import uuid
from typing import Any

from tools import ToolFormatAdapter


class OpenAIAdapter(ToolFormatAdapter):
    """Adapter for OpenAI's native structured tool_calls format."""

    def detect(self, text: str) -> bool:
        """
        Detect if text contains OpenAI-style tool calls.

        OpenAI uses structured tool_calls, not embedded text,
        so this always returns False for text content.
        """
        return False

    def parse(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """
        Parse tool calls from text.

        OpenAI doesn't embed tools in text, so this returns the text unchanged.
        """
        return text, []

    def format_request(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format tools for OpenAI upstream request.

        OpenAI uses standard function format.
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
            tool_calls: Raw tool_calls from OpenAI response

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
                # Handle partial/invalid JSON
                args = {"raw": args_str}

            formatted.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": fn.get("name", ""),
                "input": args,
            })
        return formatted

    def parse_streaming_tool_call(
        self,
        delta: dict[str, Any],
        accumulator: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """
        Parse streaming tool call delta.

        Args:
            delta: The tool_calls delta from streaming response
            accumulator: Accumulator for partial tool calls

        Returns:
            List of completed tool calls or None if still accumulating
        """
        if not delta or not delta.get("tool_calls"):
            return None

        tool_calls = delta["tool_calls"]
        completed = []

        for tc in tool_calls:
            idx = tc.get("index", 0)

            if idx not in accumulator:
                accumulator[idx] = {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }

            acc = accumulator[idx]

            # Update ID if provided
            if tc.get("id"):
                acc["id"] = tc["id"]

            # Update function data
            fn = tc.get("function", {})
            if fn.get("name"):
                acc["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                acc["function"]["arguments"] += fn["arguments"]

            # Check if complete (has ID, name, and complete arguments JSON)
            if acc["id"] and acc["function"]["name"] and acc["function"]["arguments"]:
                try:
                    json.loads(acc["function"]["arguments"])
                    completed.append(acc)
                except json.JSONDecodeError:
                    # Arguments not complete yet
                    pass

        return completed if completed else None
