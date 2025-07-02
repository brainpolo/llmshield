"""OpenAI provider for handling OpenAI API specifics.

Description:
    This module provides specialised handling for OpenAI API functions,
    including both standard and beta APIs. It manages parameter conversion
    to OpenAI's expected format and handles streaming capabilities based
    on API version detection.

Classes:
    OpenAIProvider: Specialised provider for OpenAI API integration

Author:
    LLMShield by brainpolo, 2025
"""

# Standard Library Imports
from collections.abc import Callable
from typing import Any

# Local Imports
from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI APIs (standard and beta)."""

    def __init__(self, llm_func: Callable):
        """Initialise OpenAI provider."""
        super().__init__(llm_func)
        self.is_beta_api = self._detect_beta_api()

    def _detect_beta_api(self) -> bool:
        """Detect if this is a beta API function."""
        return (
            "beta" in self.func_qualname
            or "beta" in self.func_module
            or "parse" in self.func_name
        )

    def prepare_single_message_params(
        self, cloaked_text: str, input_param: str, stream: bool, **kwargs
    ) -> tuple[dict[str, Any], bool]:
        """Prepare parameters for OpenAI single message calls."""
        # Remove original parameter and convert to messages format
        prepared_kwargs = kwargs.copy()
        prepared_kwargs.pop(input_param, None)
        prepared_kwargs["messages"] = [
            {"role": "user", "content": cloaked_text}
        ]

        # Handle streaming for beta APIs
        if self.is_beta_api:
            if stream:
                warning_msg = (
                    f"Warning: Beta API detected ({self.func_name}), "
                    "streaming not supported. Disabling stream."
                )
                print(warning_msg)
            # Beta APIs don't accept stream parameter at all
            prepared_kwargs.pop("stream", None)
            return prepared_kwargs, False

        # Standard APIs support streaming
        prepared_kwargs["stream"] = stream
        return prepared_kwargs, stream

    def prepare_multi_message_params(
        self, cloaked_messages: list[dict], stream: bool, **kwargs
    ) -> tuple[dict[str, Any], bool]:
        """Prepare parameters for OpenAI multi-message calls."""
        prepared_kwargs = kwargs.copy()
        prepared_kwargs["messages"] = cloaked_messages

        # Handle streaming for beta APIs
        if self.is_beta_api:
            if stream:
                warning_msg = (
                    f"Warning: Beta API detected ({self.func_name}), "
                    "streaming not supported. Disabling stream."
                )
                print(warning_msg)
            # Beta APIs don't accept stream parameter at all
            prepared_kwargs.pop("stream", None)
            return prepared_kwargs, False

        # Standard APIs support streaming
        prepared_kwargs["stream"] = stream
        return prepared_kwargs, stream

    @classmethod
    def can_handle(cls, llm_func: Callable) -> bool:
        """Check if this is an OpenAI API function."""
        func_name = getattr(llm_func, "__name__", "")
        func_qualname = getattr(llm_func, "__qualname__", "")

        return (
            "chat.completions.create" in func_qualname
            or "chat.completions.parse" in func_qualname
            or "create" in func_name
            or "parse" in func_name
        )
