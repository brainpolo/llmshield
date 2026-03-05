"""Base provider class for LLM API handling.

Description:
    This module defines the abstract base class for LLM providers. All provider
    implementations must inherit from this class and implement the required
    methods for parameter preparation and API compatibility checking.

Classes:
    BaseProvider: Abstract base class for LLM provider implementations

Author:
    LLMShield by brainpolo, 2025-2026
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

    def prepare_single_message_params(
        self,
        cloaked_text: str,
        input_param: str,
        stream: bool,
        **kwargs,
    ) -> tuple[dict[str, Any], bool]:
        """Prepare parameters for single message calls.

        Default converts to messages format. Subclasses may
        override for provider-specific formatting.

        Args:
            cloaked_text: The cloaked message content
            input_param: Original parameter name
            stream: Whether streaming is requested
            **kwargs: Additional parameters

        Returns:
            Tuple of (prepared_params, updated_stream_flag)

        """
        prepared_kwargs = kwargs.copy()
        prepared_kwargs.pop(input_param, None)
        prepared_kwargs["messages"] = [
            {"role": "user", "content": cloaked_text}
        ]
        prepared_kwargs["stream"] = stream
        return prepared_kwargs, stream

    def prepare_multi_message_params(
        self,
        cloaked_messages: list[dict],
        stream: bool,
        **kwargs,
    ) -> tuple[dict[str, Any], bool]:
        """Prepare parameters for multi-message calls.

        Default passes messages through. Subclasses may
        override for provider-specific formatting.

        Args:
            cloaked_messages: List of cloaked message dicts
            stream: Whether streaming is requested
            **kwargs: Additional parameters

        Returns:
            Tuple of (prepared_params, updated_stream_flag)

        """
        prepared_kwargs = kwargs.copy()
        prepared_kwargs["messages"] = cloaked_messages
        prepared_kwargs["stream"] = stream
        return prepared_kwargs, stream

    def execute_single_message(
        self,
        cloaked_text: str,
        input_param: str,
        stream: bool,
        **kwargs,
    ) -> tuple[Any, bool]:
        """Execute a single-message LLM call.

        Prepares parameters and invokes the LLM function.
        Subclasses may override to use non-standard invocation
        patterns (e.g., stateful chat builders).

        Args:
            cloaked_text: The cloaked message content
            input_param: Original parameter name
            stream: Whether streaming is requested
            **kwargs: Additional parameters

        Returns:
            Tuple of (llm_response, actual_stream_flag)

        """
        prepared_params, actual_stream = self.prepare_single_message_params(
            cloaked_text, input_param, stream, **kwargs
        )
        return self.llm_func(**prepared_params), actual_stream

    def execute_multi_message(
        self,
        cloaked_messages: list[dict],
        stream: bool,
        **kwargs,
    ) -> tuple[Any, bool]:
        """Execute a multi-message LLM call.

        Prepares parameters and invokes the LLM function.
        Subclasses may override to use non-standard invocation
        patterns (e.g., stateful chat builders).

        Args:
            cloaked_messages: List of cloaked message dicts
            stream: Whether streaming is requested
            **kwargs: Additional parameters

        Returns:
            Tuple of (llm_response, actual_stream_flag)

        """
        prepared_params, actual_stream = self.prepare_multi_message_params(
            cloaked_messages, stream, **kwargs
        )
        return self.llm_func(**prepared_params), actual_stream

    def execute_raw(self, **kwargs) -> Any:
        """Execute LLM call without parameter preparation.

        Used for the no-cloaking fast path where kwargs are
        passed directly to the LLM function. Subclasses may
        override for non-standard invocation patterns.

        Args:
            **kwargs: Parameters to pass to the LLM function

        Returns:
            The raw LLM response

        """
        return self.llm_func(**kwargs)

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
