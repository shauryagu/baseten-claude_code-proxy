"""
OpenAI GPT models handler.

Inherits Anthropic → OpenAI conversion from DefaultHandler.
Only adds OpenAI-specific capability checks.
"""

from typing import Any, Optional

from config import ModelCapability
from models import DefaultHandler


class OpenAIHandler(DefaultHandler):
    """OpenAI GPT models handler — inherits all conversion from DefaultHandler."""

    def __init__(self, capability: Optional[ModelCapability] = None):
        super().__init__(capability)

    def supports_vision(self) -> bool:
        """Check if this model supports vision inputs."""
        if self.capability and "vision" in self.capability.capabilities:
            return True
        return False

    def supports_json_mode(self) -> bool:
        """Check if this model supports structured JSON output mode."""
        if self.capability and "json_mode" in self.capability.capabilities:
            return True
        return False
