"""
Security utilities for input validation and sanitization.

Provides Pydantic models for request validation, content filtering,
and security-related utilities.
"""

import re
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from errors import ValidationError


class ToolDefinition(BaseModel):
    """Validated tool definition."""

    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(default="", max_length=1024)
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate tool name format."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Tool name must be alphanumeric with underscores/hyphens only")
        return v


class ContentBlock(BaseModel):
    """Validated content block."""

    type: str = Field(..., pattern="^(text|image|tool_use|tool_result)$")
    text: Optional[str] = None
    source: Optional[dict[str, Any]] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict[str, Any]] = None
    tool_use_id: Optional[str] = None
    content: Optional[Union[str, list[dict[str, Any]]]] = None
    is_error: Optional[bool] = None

    @model_validator(mode="after")
    def validate_content(self):
        """Validate content based on type."""
        if self.type == "text" and self.text is None:
            raise ValueError("Text blocks must have 'text' field")
        if self.type == "tool_use":
            if not self.id or not self.name:
                raise ValueError("Tool use blocks must have 'id' and 'name' fields")
        if self.type == "tool_result":
            if not self.tool_use_id:
                raise ValueError("Tool result blocks must have 'tool_use_id' field")
        return self


class MessageRequest(BaseModel):
    """Validated Anthropic message request."""

    model: str = Field(..., min_length=1)
    messages: list[dict[str, Any]] = Field(..., min_length=1)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=128000)
    system: Optional[Union[str, list[dict[str, Any]]]] = None
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[dict[str, Any]] = None
    stream: bool = False
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate message format."""
        if not v:
            raise ValueError("At least one message is required")

        for i, msg in enumerate(v):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be an object")
            if "role" not in msg:
                raise ValueError(f"Message {i} missing 'role' field")
            if "content" not in msg:
                raise ValueError(f"Message {i} missing 'content' field")
            if msg["role"] not in {"user", "assistant", "system"}:
                raise ValueError(f"Message {i} has invalid role: {msg['role']}")

        return v

    @field_validator("tools")
    @classmethod
    def validate_tools(
        cls, v: Optional[list[ToolDefinition]]
    ) -> Optional[list[ToolDefinition]]:
        """Validate tool definitions."""
        if not v:
            return v

        if len(v) > 256:
            raise ValueError("Maximum 256 tools allowed")

        names = [tool.name for tool in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate tool names not allowed")

        return v


class SecurityConfig(BaseModel):
    """Security configuration."""

    max_request_size: int = 50 * 1024 * 1024  # 50MB
    max_message_size: int = 10 * 1024 * 1024  # 10MB per message
    max_tools: int = 256
    max_tool_description_length: int = 1024
    max_content_length: int = 100000  # characters
    allowed_content_types: set[str] = {"text", "image", "tool_use", "tool_result"}
    blocked_patterns: list[str] = Field(default_factory=list)

    @field_validator("blocked_patterns")
    @classmethod
    def compile_patterns(cls, v: list[str]) -> list[str]:
        """Validate regex patterns."""
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        return v


def sanitize_content(content: str, config: Optional[SecurityConfig] = None) -> str:
    """
    Sanitize content by removing potentially harmful patterns.

    Args:
        content: Content to sanitize
        config: Security configuration

    Returns:
        Sanitized content
    """
    cfg = config or SecurityConfig()

    # Check content length
    if len(content) > cfg.max_content_length:
        raise ValidationError(
            f"Content exceeds maximum length of {cfg.max_content_length} characters"
        )

    # Apply blocked patterns
    for pattern in cfg.blocked_patterns:
        content = re.sub(pattern, "[REDACTED]", content)

    return content


def validate_request_size(body: bytes, config: Optional[SecurityConfig] = None) -> None:
    """
    Validate request body size.

    Args:
        body: Request body bytes
        config: Security configuration

    Raises:
        ValidationError: If request is too large
    """
    cfg = config or SecurityConfig()

    if len(body) > cfg.max_request_size:
        raise ValidationError(
            f"Request body exceeds maximum size of {cfg.max_request_size} bytes",
            details={"max_size": cfg.max_request_size, "actual_size": len(body)},
        )


# Global security config instance
_security_config: Optional[SecurityConfig] = None


def get_security_config() -> SecurityConfig:
    """Get or create global security config."""
    global _security_config
    if _security_config is None:
        _security_config = SecurityConfig()
    return _security_config


def set_security_config(config: SecurityConfig) -> None:
    """Set global security config."""
    global _security_config
    _security_config = config
