"""Default provider for unknown/generic LLM functions.

Description:
    This module implements the default provider that serves as a fallback for
    handling generic LLM functions. It attempts to handle unknown LLM APIs by
    inspecting their parameter names and making reasonable assumptions about
    how to prepare the parameters.

Classes:
    DefaultProvider: Fallback provider for generic LLM function handling

Author:
    LLMShield by brainpolo, 2025
"""

# Standard Library Imports
from collections.abc import Callable
from typing import Any

# Local Imports
from .base import BaseLLMProvider


class DefaultProvider(BaseLLMProvider):
    """Default provider for unknown LLM functions.

    This provider attempts to handle generic LLM functions by inspecting
    their parameter names and making reasonable assumptions.
    """

    def prepare_single_message_params(
        self, cloaked_text: str, input_param: str, stream: bool, **kwargs
    ) -> tuple[dict[str, Any], bool]:
        """Prepare parameters for generic single message calls."""
        prepared_kwargs = kwargs.copy()

        # Try to determine the preferred parameter name for this function
        func_preferred_param = self._get_preferred_param_name()

        # Remove original parameter and add under preferred name
        prepared_kwargs.pop(input_param, None)
        prepared_kwargs[func_preferred_param] = cloaked_text
        prepared_kwargs["stream"] = stream

        return prepared_kwargs, stream

    def prepare_multi_message_params(
        self, cloaked_messages: list[dict], stream: bool, **kwargs
    ) -> tuple[dict[str, Any], bool]:
        """Prepare parameters for generic multi-message calls."""
        prepared_kwargs = kwargs.copy()
        prepared_kwargs["messages"] = cloaked_messages
        prepared_kwargs["stream"] = stream

        return prepared_kwargs, stream

    def _get_preferred_param_name(self) -> str:
        """Determine the preferred parameter name for this function."""
        try:
            # Inspect function parameters to determine preference
            if hasattr(self.llm_func, "__code__"):
                varnames = self.llm_func.__code__.co_varnames
                if "message" in varnames:
                    return "message"
                if "prompt" in varnames:
                    return "prompt"
        except (AttributeError, TypeError):
            pass

        # Default fallback
        return "prompt"

    @classmethod
    def can_handle(cls, llm_func: Callable) -> bool:
        """Check if this provider can handle any function as fallback."""
        return True  # This is the fallback provider
