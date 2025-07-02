"""Anthropic provider for handling Anthropic API specifics.

Description:
    This module provides specialised handling for Anthropic API functions,
    including message format conversion, tool calling support, and proper
    parameter handling. It manages conversion between LLMShield's standard
    format and Anthropic's expected parameters.

Classes:
    AnthropicProvider: Specialised provider for Anthropic API integration

Author:
    LLMShield by brainpolo, 2025
"""

# Standard Library Imports
import json
from collections.abc import Callable
from typing import Any

# Local Imports
from .base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude APIs."""

    def __init__(self, llm_func: Callable):
        """Initialise Anthropic provider."""
        super().__init__(llm_func)

    def prepare_single_message_params(
        self, cloaked_text: str, input_param: str, stream: bool, **kwargs
    ) -> tuple[dict[str, Any], bool]:
        """Prepare parameters for Anthropic single message calls."""
        # Remove original parameter and convert to messages format
        prepared_kwargs = kwargs.copy()
        prepared_kwargs.pop(input_param, None)
        prepared_kwargs["messages"] = [
            {"role": "user", "content": cloaked_text}
        ]

        # Anthropic supports streaming
        prepared_kwargs["stream"] = stream
        return prepared_kwargs, stream

    def prepare_multi_message_params(
        self, cloaked_messages: list[dict], stream: bool, **kwargs
    ) -> tuple[dict[str, Any], bool]:
        """Prepare parameters for Anthropic multi-message calls."""
        prepared_kwargs = kwargs.copy()

        # Convert messages to Anthropic format
        anthropic_messages = self._convert_messages_to_anthropic_format(
            cloaked_messages
        )
        prepared_kwargs["messages"] = anthropic_messages

        # Anthropic supports streaming
        prepared_kwargs["stream"] = stream
        return prepared_kwargs, stream

    def _convert_messages_to_anthropic_format(
        self, messages: list[dict]
    ) -> list[dict]:
        """Convert messages to Anthropic's expected format.

        Anthropic has some specific requirements:
        - Messages must alternate between user and assistant
        - Tool use has specific content block format
        """
        converted_messages = []

        for msg in messages:
            converted_msg = {
                "role": msg["role"],
                "content": msg.get("content"),
            }

            # Handle tool calls for assistant messages
            if (
                msg.get("role") == "assistant"
                and "tool_calls" in msg
                and msg["tool_calls"] is not None
            ):
                # Convert OpenAI-style tool calls to Anthropic format
                content_blocks = []

                # Add text content if present
                if msg.get("content") is not None:
                    content_blocks.append(
                        {"type": "text", "text": msg["content"]}
                    )

                # Add tool use blocks
                for tool_call in msg["tool_calls"]:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": self._parse_tool_arguments(
                                tool_call["function"]["arguments"]
                            ),
                        }
                    )

                converted_msg["content"] = content_blocks

            # Handle tool results
            elif msg.get("role") == "user" and "tool_call_id" in msg:
                converted_msg["content"] = [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg["tool_call_id"],
                        "content": msg.get("content", ""),
                    }
                ]

            converted_messages.append(converted_msg)

        return converted_messages

    def _parse_tool_arguments(self, arguments: str) -> dict:
        """Parse tool arguments from JSON string to dict."""
        try:
            return json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return {}

    @classmethod
    def can_handle(cls, llm_func: Callable) -> bool:
        """Check if this is an Anthropic API function."""
        func_name = getattr(llm_func, "__name__", "")
        func_qualname = getattr(llm_func, "__qualname__", "")
        func_module = getattr(llm_func, "__module__", "")

        return (
            "anthropic" in func_module.lower()
            or "claude" in func_module.lower()
            or "messages.create" in func_qualname
            or (
                "create" in func_name
                and any(x in func_module for x in ["anthropic", "claude"])
            )
        )
