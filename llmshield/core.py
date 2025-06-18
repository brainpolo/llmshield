"""Core module for llmshield.

This module provides the main LLMShield class for protecting sensitive information
in Large Language Model (LLM) interactions. It handles cloaking of sensitive entities
in prompts before sending to LLMs, and uncloaking of responses to restore the
original information.

Key features:
- Entity detection and protection (names, emails, numbers, etc.)
- Configurable delimiters for entity placeholders
- Direct LLM function integration
- Zero dependencies

Example:
    >>> shield = LLMShield()
    >>> (
    ...     safe_prompt,
    ...     entities,
    ... ) = shield.cloak(
    ...     "Hi, I'm John (john@example.com)"
    ... )
    >>> response = shield.uncloak(
    ...     llm_response,
    ...     entities,
    ... )

"""

# Python imports
from collections.abc import Callable, Generator
from typing import Any

from .cloak_prompt import _cloak_prompt  # type: ignore
from .lru_cache import LRUCache
from .uncloak_response import _uncloak_response  # type: ignore
from .uncloak_stream_response import uncloak_stream_response

# Local imports
from .utils import (
    Message,
    PydanticLike,
    ask_helper,  # type: ignore
    conversation_hash,
    is_valid_delimiter,
    is_valid_stream_response,
)

DEFAULT_START_DELIMITER = "<"
DEFAULT_END_DELIMITER = ">"


class LLMShield:
    """Main class for LLMShield - protects sensitive information in LLM interactions.

    Example:
        >>> from llmshield import (
        ...     LLMShield,
        ... )
        >>> shield = LLMShield()
        >>> (
        ...     cloaked_prompt,
        ...     entity_map,
        ... ) = shield.cloak(
        ...     "Hi, I'm John Doe (john.doe@example.com)"
        ... )
        >>> print(
        ...     cloaked_prompt
        ... )
        "Hi, I'm <PERSON_0> (<EMAIL_1>)"
        >>> llm_response = get_llm_response(
        ...     cloaked_prompt
        ... )  # Your LLM call
        >>> original = shield.uncloak(
        ...     llm_response,
        ...     entity_map,
        ... )

    """

    def __init__(
        self,
        start_delimiter: str = DEFAULT_START_DELIMITER,
        end_delimiter: str = DEFAULT_END_DELIMITER,
        llm_func: (
            Callable[[str], str] | Callable[[str], Generator[str, None, None]] | None
        ) = None,
        max_cache_size: int = 128,  # TODO: Is this value enough? Should it be reduced or increased?
    ) -> None:
        """Initialise LLMShield.

        Args:
            start_delimiter: Character(s) to wrap entity placeholders (default: '<')
            end_delimiter: Character(s) to wrap entity placeholders (default: '>')
            llm_func: Optional function that calls your LLM (enables direct usage)

        """
        if not is_valid_delimiter(start_delimiter):
            msg = "Invalid start delimiter"
            raise ValueError(msg)
        if not is_valid_delimiter(end_delimiter):
            msg = "Invalid end delimiter"
            raise ValueError(msg)
        if llm_func and not callable(llm_func):
            msg = "llm_func must be a callable"
            raise ValueError(msg)

        self.start_delimiter = start_delimiter
        self.end_delimiter = end_delimiter

        self.llm_func = llm_func

        self._last_entity_map = None
        self._cache: LRUCache[int, dict[str, str]] = LRUCache(
            max_cache_size
        )  # TODO: Determine the correct typing of the cache

    def cloak(
        self, prompt: str, entity_map_param: dict[str, str] | None = None
    ) -> tuple[str, dict[str, str]]:
        """Cloak sensitive information in the prompt.

        Args:
            prompt: The original prompt containing sensitive information.

        Returns:
            Tuple of (cloaked_prompt, entity_mapping)

        """
        cloaked, entity_map = _cloak_prompt(
            prompt=prompt,
            start_delimiter=self.start_delimiter,
            end_delimiter=self.end_delimiter,
            entity_map=entity_map_param,
        )
        self._last_entity_map = entity_map
        return cloaked, entity_map

    def uncloak(
        self,
        response: str | list[Any] | dict[str, Any] | PydanticLike,
        entity_map: dict[str, str] | None = None,
    ) -> str | list[Any] | dict[str, Any] | PydanticLike:
        """Restore original entities in the LLM response. It supports strings and
        structured outputs consisting of any combination of strings, lists, and
        dictionaries.

        For uncloaking stream responses, use the `stream_uncloak` method instead.

        Limitations:
            - Does not support tool calls.

        Args:
            response: The LLM response containing placeholders. Supports both
            strings and structured outputs (dicts).
            entity_map: Mapping of placeholders to original values
                        (if empty, uses mapping from last cloak call)

        Returns:
            Response with original entities restored

        Raises:
            TypeError: If response parameters of invalid type.
            ValueError: If no entity mapping is provided and no previous cloak call.s

        """
        # Validate inputs
        if not response:
            msg = "Response cannot be empty"
            raise ValueError(msg)

        if not isinstance(response, str | list | dict | PydanticLike):  # type: ignore
            msg = (
                "Response must be in [str, list, dict] or a Pydantic model, "
                f"but got: {type(response)}!"
            )
            raise TypeError(
                msg,
            )

        if entity_map is None:
            if self._last_entity_map is None:
                msg = "No entity mapping provided or stored from previous cloak!"
                raise ValueError(msg)
            entity_map = self._last_entity_map

        if isinstance(response, PydanticLike):
            model_class = response.__class__
            uncloaked_dict = _uncloak_response(response.model_dump(), entity_map)
            return model_class.model_validate(uncloaked_dict)

        return _uncloak_response(response, entity_map)

    def stream_uncloak(
        self,
        response_stream: Generator[str, None, None],
        entity_map: dict[str, str] | None = None,
    ) -> Generator[str, None, None]:
        """
        Restore original entities in the LLM response if the response comes in
        the form of a stream.

        The function processes the response stream in the form of chunks,
        attempting to yield either uncloaked chunks or the remaining buffer
        content in which there was no uncloaking done yet.

        For non-stream responses, use the `uncloak` method instead.

        Limitations:
            - Only supports a response from a single LLM function call.

        Args:
            response_stream: Iterator yielding cloaked LLM response chunks
            entity_map: Mapping of placeholders to original values.
                        By default, it is None, which means it will use the
                        last cloak call's entity map.

        Yields:
            str: Uncloaked response chunks
        """
        # Validate the inputs
        if not response_stream:
            msg = "Response stream cannot be empty"
            raise ValueError(msg)

        if not is_valid_stream_response(response_stream):
            msg = (
                "Response stream must be an iterable (excluding str, bytes, dict), "
                f"but got: {type(response_stream)}!"
            )
            raise TypeError(
                msg,
            )

        if entity_map is None:
            if self._last_entity_map is None:
                msg = "No entity mapping provided or stored from previous cloak!"
                raise ValueError(msg)
            entity_map = self._last_entity_map

        return uncloak_stream_response(
            response_stream,
            entity_map=entity_map,
            start_delimiter=self.start_delimiter,
            end_delimiter=self.end_delimiter,
        )

    def ask(
        self, stream: bool = False, messages: list[Message] | None = None, **kwargs
    ) -> str | Generator[str, None, None]:
        """Complete end-to-end LLM interaction with automatic protection.

        NOTE: If you are using a structured output, ensure that your keys
        do not contain PII and that any keys that may contain PII are either
        string, lists, or dicts. Other types like int, float, are unable to be
        cloaked and will be returned as is.

        Args:
            prompt/message: Original prompt with sensitive information. This
                    will be cloaked and passed to your LLM function. Do not pass
                    both, and do not use any other parameter names as they are
                    unrecognised by the shield.
            stream: Whether the LLM Function is a stream or not. If True, returns
                    a generator that yields incremental responses
                    following the OpenAI Realtime Streaming API. If False, returns
                    the complete response as a string.
                    By default, this is False.
            messages: List of message dictionaries for multi-turn conversations.
            They must come in the form of a list of dictionaries,
            where each dictionary has keys like "role" and "content".
            **kwargs: Additional arguments to pass to your LLM function, such as:
                    - model: The model to use (e.g., "gpt-4")
                    - system_prompt: System instructions
                    - temperature: Sampling temperature
                    - max_tokens: Maximum tokens in response
                    etc.
        ! The arguments do not have to be in any specific order!

        Returns:
            str: Uncloaked LLM response with original entities restored.

            Generator[str, None, None]: If stream is True, returns a generator
            that yields incremental responses, following the OpenAI Realtime
            Streaming API.

        ! Regardless of the specific implementation of the LLM Function,
        whenever the stream parameter is true, the function will return an generator. !

        Raises:
            ValueError: If no LLM function was provided during initialization,
                       if prompt is invalid, or if both prompt and message are provided

        """
        # * 1. Validate inputs
        if self.llm_func is None:
            msg = (
                "No LLM function provided. Either provide llm_func in constructor "
                "or use cloak/uncloak separately."
            )
            raise ValueError(
                msg,
            )

        if not (("prompt" in kwargs) or ("message" in kwargs) or (messages is not None)):
            msg = "Either 'prompt', 'message' or the messages parameter must be provided!"
            raise ValueError(msg)

        if "prompt" in kwargs and "message" in kwargs:
            msg = (
                "Do not provide both 'prompt' and 'message'. Use only 'prompt' "
                "parameter - it will be passed to your LLM function."
            )
            raise ValueError(
                msg,
            )

        if messages is not None and ("prompt" in kwargs or "message" in kwargs):
            msg = (
                "Do not provide both 'prompt', 'message' and 'messages'. Use only either prompt"
                "/message"
                " or messages parameter - it will be passed to your LLM function."
            )
            raise ValueError(
                msg,
            )

        if messages is None and ("message" in kwargs or "prompt" in kwargs):  # type: ignore
            return ask_helper(
                shield=self,
                stream=stream,
                **kwargs,
            )

        # * 2. Set up the initial history and hash the conversation
        # except for the last message
        history = messages[:-1]
        latest_message = messages[-1]
        history_key = conversation_hash(history)

        # * 3. Check the cache for an existing entity map for this conversation history
        entity_map = self._cache.get(history_key)
        if entity_map is None:
            # * Cache Miss: Build the entity map by processing the entire history
            entity_map = {}
            for message in history:
                _, entity_map = self.cloak(message["content"], entity_map)
                # Each message is placed in the cache paired to their entity map
                self._cache.put(conversation_hash(message), entity_map)

        # * 4. Cloak the last message using the existing entity map
        cloaked_latest_content, final_entity_map = self.cloak(
            latest_message["content"], entity_map_param=entity_map.copy()
        )

        # 5. Reconstruct the full, cloaked message list to send to the LLM
        cloaked_messages = []
        for msg in history:
            cloaked_content, _ = self.cloak(msg["content"], entity_map_param=final_entity_map)
            cloaked_messages.append({"role": msg["role"], "content": cloaked_content})  # type: ignore
        cloaked_messages.append({"role": latest_message["role"], "content": cloaked_latest_content})  # type: ignore

        # 6. Call the LLM with the protected payload
        llm_response = self.llm_func(messages=cloaked_messages, stream=stream, **kwargs)

        # 7. Uncloak the response
        if stream:
            uncloaked_response = self.stream_uncloak(llm_response, final_entity_map)
        else:
            uncloaked_response = self.uncloak(llm_response, final_entity_map)

        # 8. Update the history with the latest message and the uncloaked response
        next_history = history + [
            latest_message,
            {"role": "assistant", "content": uncloaked_response},
        ]

        new_key = conversation_hash(next_history)
        self._cache.put(new_key, final_entity_map)

        return uncloaked_response
