"""Unit tests for tool adapters."""

import json

import pytest

from tools import KimiAdapter, OpenAIAdapter


class TestKimiAdapter:
    """Tests for Kimi tool adapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = KimiAdapter()

    def test_detect_embedded_tools(self):
        """Test detection of embedded tool calls."""
        text_with_tools = "Let me call a function   <|tool_calls_section_begin|>functions.search:0{\"query\": \"test\"}     <|tool_calls_section_end|> for you."
        assert self.adapter.detect(text_with_tools) is True

        text_without_tools = "This is just regular text."
        assert self.adapter.detect(text_without_tools) is False

    def test_parse_embedded_tools(self):
        """Test parsing embedded tool calls."""
        text = 'I will search for that.    <|tool_calls_section_begin|>functions.search:0{"query": "python"}     troublesome'
        clean_text, tool_uses = self.adapter.parse(text)

        assert "I will search for that." in clean_text
        assert "troublesome" in clean_text
        assert "    <|tool_calls_section_begin|>" not in clean_text
        assert "     <|tool_calls_section_end|>" not in clean_text

        assert len(tool_uses) == 1
        assert tool_uses[0]["type"] == "tool_use"
        assert tool_uses[0]["name"] == "search"
        assert tool_uses[0]["input"] == {"query": "python"}

    def test_format_request(self):
        """Test formatting tools for Kimi request."""
        tools = [
            {
                "name": "search",
                "description": "Search for information",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]

        formatted = self.adapter.format_request(tools)

        assert len(formatted) == 1
        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "search"
        assert "description" in formatted[0]["function"]

    def test_format_response(self):
        """Test formatting tool calls for Anthropic response."""
        tool_calls = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "test"}',
                },
            }
        ]

        formatted = self.adapter.format_response(tool_calls)

        assert len(formatted) == 1
        assert formatted[0]["type"] == "tool_use"
        assert formatted[0]["name"] == "search"
        assert formatted[0]["input"] == {"query": "test"}

    def test_parse_multiple_tools(self):
        """Test parsing multiple embedded tool calls."""
        text = 'First   <|tool_calls_section_begin|>functions.search:0{"q": "a"}     troublesome then   <|tool_calls_section_begin|>functions.calc:1{"x": 1}     troublesome done'
        clean_text, tool_uses = self.adapter.parse(text)

        assert len(tool_uses) == 2
        assert tool_uses[0]["name"] == "search"
        assert tool_uses[1]["name"] == "calc"


class TestOpenAIAdapter:
    """Tests for OpenAI tool adapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = OpenAIAdapter()

    def test_detect_always_false(self):
        """Test that detect always returns False for text."""
        assert self.adapter.detect("any text") is False
        assert self.adapter.detect("") is False

    def test_parse_returns_unchanged(self):
        """Test that parse returns text unchanged."""
        text = "some text"
        clean, tools = self.adapter.parse(text)
        assert clean == text
        assert tools == []

    def test_format_request(self):
        """Test formatting tools for OpenAI request."""
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather information",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        ]

        formatted = self.adapter.format_request(tools)

        assert len(formatted) == 1
        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "get_weather"

    def test_format_response(self):
        """Test formatting tool calls for Anthropic response."""
        tool_calls = [
            {
                "id": "call_xyz789",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco"}',
                },
            }
        ]

        formatted = self.adapter.format_response(tool_calls)

        assert len(formatted) == 1
        assert formatted[0]["type"] == "tool_use"
        assert formatted[0]["name"] == "get_weather"
        assert formatted[0]["input"] == {"location": "San Francisco"}

    def test_format_response_with_invalid_json(self):
        """Test formatting response with invalid JSON arguments."""
        tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "test",
                    "arguments": "invalid json",
                },
            }
        ]

        formatted = self.adapter.format_response(tool_calls)

        assert len(formatted) == 1
        assert formatted[0]["input"] == {"raw": "invalid json"}
