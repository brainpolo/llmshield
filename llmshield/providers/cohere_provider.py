"""Cohere provider for handling Cohere API specifics.

Description:
    This module provides specialised handling for the Cohere V2
    SDK (cohere package). It manages streaming via the separate
    chat_stream() method and handles parameter conversion.

    The user should pass client.chat as the llm_func.

Classes:
    CohereProvider: Specialised provider for Cohere integration

Author:
    LLMShield by brainpolo, 2025-2026
"""

# Standard Library Imports
from collections.abc import Callable, Generator
from typing import Any

# Optional Third-party Import (cohere)
try:
    import cohere  # noqa: F401

    _HAS_COHERE = True
except ImportError:  # pragma: no cover
    _HAS_COHERE = False

# Local Imports
from .base import BaseLLMProvider

_MAX_CLOSURE_DEPTH = 5


class CohereProvider(BaseLLMProvider):
    """Provider for Cohere V2 API.

    Handles the Cohere chat API which is OpenAI-compatible
    for inputs but has a different response structure:
    response.message.content[0].text instead of
    response.choices[0].message.content.

    Streaming uses a separate chat_stream() method.

    The llm_func should be client.chat.
    """

    def __init__(self, llm_func: Callable):
        """Initialise the Cohere provider.

        Caches the chat_stream function for streaming support.
        The Cohere SDK wraps methods with decorators that
        strip __self__, so we walk the closure chain to find
        the parent client object.

        Args:
            llm_func: The client.chat function

        """
        super().__init__(llm_func)
        self._chat_stream = self._find_stream_func()

    def _find_stream_func(self) -> Callable | None:
        """Find the chat_stream sibling on the parent client.

        Tries __self__ first (standard bound methods), then
        walks the closure chain for decorated methods where
        __self__ has been stripped by SDK wrappers.

        Returns:
            Stream function or None if unavailable

        """
        client = getattr(self.llm_func, "__self__", None)
        if client is not None:
            return self._get_chat_stream(client)

        return self._walk_closures(self.llm_func)

    @staticmethod
    def _get_chat_stream(client: Any) -> Callable | None:
        """Extract chat_stream from a client object."""
        func = getattr(client, "chat_stream", None)
        if func is not None and callable(func):
            return func
        return None

    def _walk_closures(
        self,
        func: Any,
        depth: int = 0,
    ) -> Callable | None:
        """Walk closure cells to find a bound method's client.

        Inspects only the function's own closure chain (not
        the GC heap), bounded to 5 levels deep.

        """
        if depth >= _MAX_CLOSURE_DEPTH:
            return None
        closure = getattr(func, "__closure__", None)
        if not closure:
            return None
        for cell in closure:
            try:
                val = cell.cell_contents
            except ValueError:  # pragma: no cover
                continue
            client = getattr(val, "__self__", None)
            if client is not None:
                result = self._get_chat_stream(client)
                if result is not None:
                    return result
            if callable(val):
                result = self._walk_closures(val, depth + 1)
                if result is not None:
                    return result
        return None

    def prepare_single_message_params(
        self,
        cloaked_text: str,
        input_param: str,
        stream: bool,
        **kwargs,
    ) -> tuple[dict[str, Any], bool]:
        """Prepare parameters for Cohere single message calls."""
        prepared_kwargs = kwargs.copy()
        prepared_kwargs.pop(input_param, None)
        prepared_kwargs.pop("stream", None)
        prepared_kwargs["messages"] = [
            {"role": "user", "content": cloaked_text}
        ]
        return prepared_kwargs, stream

    def prepare_multi_message_params(
        self,
        cloaked_messages: list[dict],
        stream: bool,
        **kwargs,
    ) -> tuple[dict[str, Any], bool]:
        """Prepare parameters for Cohere multi-message calls."""
        prepared_kwargs = kwargs.copy()
        prepared_kwargs.pop("stream", None)
        prepared_kwargs["messages"] = cloaked_messages
        return prepared_kwargs, stream

    def execute_multi_message(
        self,
        cloaked_messages: list[dict],
        stream: bool,
        **kwargs,
    ) -> tuple[Any, bool]:
        """Execute multi-message Cohere call.

        Overrides base to handle streaming via the separate
        chat_stream() method instead of stream=True param.
        """
        prepared_kwargs, _ = self.prepare_multi_message_params(
            cloaked_messages, stream, **kwargs
        )

        if stream:
            stream_func = self._get_stream_func()
            if stream_func is not None:
                return (
                    self._stream(stream_func, prepared_kwargs),
                    True,
                )

        return self.llm_func(**prepared_kwargs), False

    def execute_single_message(
        self,
        cloaked_text: str,
        input_param: str,
        stream: bool,
        **kwargs,
    ) -> tuple[Any, bool]:
        """Execute single-message Cohere call."""
        prepared_kwargs, _ = self.prepare_single_message_params(
            cloaked_text, input_param, stream, **kwargs
        )

        if stream:
            stream_func = self._get_stream_func()
            if stream_func is not None:
                return (
                    self._stream(stream_func, prepared_kwargs),
                    True,
                )

        return self.llm_func(**prepared_kwargs), False

    def execute_raw(self, **kwargs) -> Any:
        """Execute Cohere call without cloaking.

        Strips stream/prompt/message kwargs that Cohere's
        chat() method does not accept directly.
        """
        stream = kwargs.pop("stream", False)
        prompt = kwargs.pop("prompt", None)
        message = kwargs.pop("message", None)
        messages = kwargs.pop("messages", None)

        if messages is None:
            text = prompt or message or ""
            messages = [{"role": "user", "content": str(text)}]

        kwargs["messages"] = messages

        if stream:
            stream_func = self._get_stream_func()
            if stream_func is not None:
                return self._stream(stream_func, kwargs)

        return self.llm_func(**kwargs)

    def _get_stream_func(self) -> Callable | None:
        """Return the cached chat_stream function."""
        return self._chat_stream

    @staticmethod
    def _stream(
        stream_func: Callable,
        call_kwargs: dict,
    ) -> Generator[str, None, None]:
        """Wrap Cohere streaming to yield string chunks."""
        for event in stream_func(**call_kwargs):
            if getattr(event, "type", None) == "content-delta" and hasattr(
                event, "delta"
            ):
                msg = getattr(event.delta, "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if content is not None:
                        text = getattr(content, "text", None)
                        if text:
                            yield text

    @classmethod
    def can_handle(cls, llm_func: Callable) -> bool:
        """Check if this is a Cohere API function."""
        if not _HAS_COHERE:
            return False
        func_module = getattr(llm_func, "__module__", "")
        return func_module == "cohere" or func_module.startswith("cohere.")
