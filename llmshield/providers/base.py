"""Base provider class for LLM API handling.

Description:
    This module defines the abstract base class for LLM providers. All provider
    implementations must inherit from this class and implement the required
    methods for parameter preparation and API compatibility checking.

Classes:
    BaseProvider: Abstract base class for LLM provider implementations

Author:
    LLMShield by brainpolo, 2025
"""

# Standard Library Imports
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class BaseLLMProvider(ABC):
    """Base class for LLM provider implementations.

    This defines the interface that all provider classes must implement
    to handle provider-specific parameter formatting and API quirks.
    """

    def __init__(self, llm_func: Callable):
        """Initialise the provider with the LLM function.

        Args:
            llm_func: The LLM function to wrap

        """
        self.llm_func = llm_func
        self.func_name = getattr(llm_func, "__name__", "")
        self.func_qualname = getattr(llm_func, "__qualname__", "")
        self.func_module = getattr(llm_func, "__module__", "")

    @abstractmethod
    def prepare_single_message_params(
        self, cloaked_text: str, input_param: str, stream: bool, **kwargs
    ) -> tuple[dict[str, Any], bool]:
        """Prepare parameters for single message calls.

        Args:
            cloaked_text: The cloaked message content
            input_param: Original parameter name ('message' or 'prompt')
            stream: Whether streaming is requested
            **kwargs: Additional parameters

        Returns:
            Tuple of (prepared_params, updated_stream_flag)

        """
        raise NotImplementedError

    @abstractmethod
    def prepare_multi_message_params(
        self, cloaked_messages: list[dict], stream: bool, **kwargs
    ) -> tuple[dict[str, Any], bool]:
        """Prepare parameters for multi-message calls.

        Args:
            cloaked_messages: List of cloaked message dictionaries
            stream: Whether streaming is requested
            **kwargs: Additional parameters

        Returns:
            Tuple of (prepared_params, updated_stream_flag)

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_handle(cls, llm_func: Callable) -> bool:
        """Check if this provider can handle the given LLM function.

        Args:
            llm_func: The LLM function to check

        Returns:
            True if this provider can handle the function

        """
        raise NotImplementedError
