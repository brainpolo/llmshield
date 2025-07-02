"""Core module for PII protection in LLM interactions.

Description:
    This module provides the main LLMShield class for protecting sensitive
    information in Large Language Model (LLM) interactions. It handles
    cloaking of sensitive entities in prompts before sending to LLMs, and
    uncloaking of responses to restore the original information.

Classes:
    LLMShield: Main class orchestrating entity detection, cloaking, and
        uncloaking

Key Features:
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

Author:
    LLMShield by brainpolo, 2025

"""

# Standard Library Imports
from collections.abc import Callable, Generator
from typing import Any

# Local imports
from .cloak_prompt import cloak_prompt
from .detection_utils import (
    extract_response_content,
    is_chatcompletion_like,
)
from .entity_detector import EntityConfig
from .error_handling import (
    validate_delimiters,
    validate_entity_map,
    validate_prompt_input,
)
from .exceptions import ValidationError
from .lru_cache import LRUCache
from .providers import get_provider
from .uncloak_response import _uncloak_response
from .uncloak_stream_response import uncloak_stream_response
from .utils import (
    Message,
    PydanticLike,
    ask_helper,
    conversation_hash,
    is_valid_delimiter,
    is_valid_stream_response,
)

DEFAULT_START_DELIMITER = "<"
DEFAULT_END_DELIMITER = ">"


class LLMShield:
    """Main class for LLMShield protecting sensitive information in LLMs.

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
            Callable[[str], str]
            | Callable[[str], Generator[str, None, None]]
            | None
        ) = None,
        max_cache_size: int = 1_000,
        entity_config: EntityConfig | None = None,
    ) -> None:
        """Initialise LLMShield with selective entity protection.

        Args:
            start_delimiter: Character(s) to wrap entity placeholders
                (default: '<')
            end_delimiter: Character(s) to wrap entity placeholders
                (default: '>')
            llm_func: Optional function that calls your LLM (enables direct
                usage)
            max_cache_size: Maximum number of items to cache in the LRUCache
                (default: 1_000)
            entity_config: Configuration for selective entity detection.
                If None, all entity types are enabled.

        """
        # Validate delimiters
        try:
            validate_delimiters(start_delimiter, end_delimiter)
        except ValidationError as e:
            raise ValidationError(f"Invalid delimiters: {e}") from e

        # Additional delimiter validation
        if not is_valid_delimiter(start_delimiter):
            raise ValidationError(
                f"Invalid start delimiter: '{start_delimiter}'"
            )
        if not is_valid_delimiter(end_delimiter):
            raise ValidationError(f"Invalid end delimiter: '{end_delimiter}'")

        # Validate LLM function
        if llm_func is not None and not callable(llm_func):
            raise ValidationError("llm_func must be a callable")

        self.start_delimiter = start_delimiter
        self.end_delimiter = end_delimiter
        self.entity_config = entity_config

        self.llm_func = llm_func

        self._last_entity_map = None
        self._cache: LRUCache[int, dict[str, str]] = LRUCache(max_cache_size)

    def _build_cloaked_messages(
        self,
        history: list[Message],
        latest_message: Message,
        cloaked_latest_content: str | None,
        final_entity_map: dict[str, str],
    ) -> list[Message]:
        """Build cloaked message list for LLM.

        Args:
            history: Previous messages
            latest_message: Current message
            cloaked_latest_content: Already cloaked content for latest message
            final_entity_map: Entity mapping

        Returns:
            List of cloaked messages

        """
        cloaked_messages = []

        # Process history messages
        for msg in history:
            cloaked_msg = self._cloak_message(msg, final_entity_map)
            cloaked_messages.append(cloaked_msg)

        # Process latest message
        final_msg = {
            "role": latest_message["role"],
            "content": cloaked_latest_content,
        }
        # Preserve other fields
        for key in latest_message:
            if key not in ("role", "content"):
                if key == "tool_calls" and latest_message[key] is not None:
                    final_msg[key] = self._cloak_tool_calls(
                        latest_message[key], final_entity_map
                    )
                else:
                    final_msg[key] = latest_message[key]
        cloaked_messages.append(final_msg)  # type: ignore

        return cloaked_messages

    def _cloak_message(
        self, msg: Message, entity_map: dict[str, str]
    ) -> Message:
        """Cloak a single message.

        Args:
            msg: Message to cloak
            entity_map: Entity mapping

        Returns:
            Cloaked message

        """
        # Handle None content for tool calls
        # Handle list content for Anthropic tool results
        if msg["content"] is None or isinstance(msg["content"], list):
            cloaked_content = msg["content"]
        else:
            cloaked_content, _ = self.cloak(
                msg["content"], entity_map_param=entity_map
            )

        cloaked_msg = {"role": msg["role"], "content": cloaked_content}

        # Preserve other fields
        for key in msg:
            if key not in ("role", "content"):
                if key == "tool_calls" and msg[key] is not None:
                    cloaked_msg[key] = self._cloak_tool_calls(
                        msg[key], entity_map
                    )
                else:
                    cloaked_msg[key] = msg[key]

        return cloaked_msg  # type: ignore

    def _cloak_tool_calls(
        self,
        tool_calls: list[Any],
        entity_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Cloak PII in tool call arguments.

        Args:
            tool_calls: List of tool call objects
            entity_map: Entity mapping for cloaking

        Returns:
            List of cloaked tool calls

        """
        cloaked_tool_calls = []
        for tool_call in tool_calls:
            # Handle dict-like objects and actual dicts
            if isinstance(tool_call, dict):
                cloaked_tc = dict(tool_call)
            else:
                # For Mock or other objects, copy attributes
                cloaked_tc = {
                    "id": getattr(tool_call, "id", None),
                    "type": getattr(tool_call, "type", None),
                    "function": {
                        "name": getattr(tool_call.function, "name", None),
                        "arguments": getattr(
                            tool_call.function, "arguments", None
                        ),
                    }
                    if hasattr(tool_call, "function")
                    else None,
                }
            if (
                "function" in cloaked_tc
                and cloaked_tc["function"]
                and "arguments" in cloaked_tc["function"]
            ):
                # Cloak the arguments JSON string
                args_str = cloaked_tc["function"]["arguments"]
                cloaked_args, _ = self.cloak(
                    args_str, entity_map_param=entity_map
                )
                cloaked_tc["function"] = dict(cloaked_tc["function"])
                cloaked_tc["function"]["arguments"] = cloaked_args
            cloaked_tool_calls.append(cloaked_tc)
        return cloaked_tool_calls

    def cloak(
        self,
        prompt: str | None,
        entity_map_param: dict[str, str] | None = None,
    ) -> tuple[str | None, dict[str, str]]:
        """Cloak sensitive information in the prompt.

        Args:
            prompt: The original prompt containing sensitive information.
            entity_map_param: Optional existing entity map to maintain
                consistency.

        Returns:
            Tuple of (cloaked_prompt, entity_mapping)

        """
        # Handle None content (e.g., for tool calls)
        if prompt is None:
            return None, entity_map_param or {}

        cloaked, entity_map = cloak_prompt(
            prompt=prompt,
            start_delimiter=self.start_delimiter,
            end_delimiter=self.end_delimiter,
            entity_map=entity_map_param,
            entity_config=self.entity_config,
        )
        self._last_entity_map = entity_map
        return cloaked, entity_map

    def uncloak(
        self,
        response: str | list[Any] | dict[str, Any] | PydanticLike,
        entity_map: dict[str, str] | None = None,
    ) -> str | list[Any] | dict[str, Any] | PydanticLike:
        """Restore original entities in the LLM response.

        It supports strings and structured outputs consisting of any
        combination
        of strings, lists, and dictionaries.

        For uncloaking stream responses, use the `stream_uncloak` method
        instead.

        Args:
            response: The LLM response containing placeholders. Supports both
                strings and structured outputs (dicts).
            entity_map: Mapping of placeholders to original values
                        (if empty, uses mapping from last cloak call)

        Returns:
            Response with original entities restored

        Raises:
            TypeError: If response parameters of invalid type.
            ValueError: If no entity mapping is provided and no previous
                cloak call.

        """
        # Validate inputs
        if not response:
            raise ValidationError("Response cannot be empty")

        # Check if response is valid type or ChatCompletion-like
        valid_types = (str, list, dict, PydanticLike)
        if not isinstance(
            response, valid_types
        ) and not is_chatcompletion_like(response):  # type: ignore
            raise TypeError(
                f"Response must be in [str, list, dict] or a Pydantic model, "
                f"but got: {type(response)}!"
            )

        # Validate entity map
        try:
            entity_map = validate_entity_map(
                entity_map, self._last_entity_map, "Uncloak"
            )
        except ValidationError as e:
            raise ValidationError(f"Uncloak failed: {e}") from e

        if isinstance(response, PydanticLike):
            model_class = response.__class__
            uncloaked_dict = _uncloak_response(
                response.model_dump(), entity_map
            )
            return model_class.model_validate(uncloaked_dict)

        return _uncloak_response(response, entity_map)

    def stream_uncloak(
        self,
        response_stream: Generator[str, None, None],
        entity_map: dict[str, str] | None = None,
    ) -> Generator[str, None, None]:
        """Restore original entities in streaming LLM responses.

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
                "Response stream must be an iterable (excluding str, bytes, "
                "dict), "
                f"but got: {type(response_stream)}!"
            )
            raise TypeError(
                msg,
            )

        if entity_map is None:
            if self._last_entity_map is None:
                msg = (
                    "No entity mapping provided or stored from previous cloak!"
                )
                raise ValueError(msg)
            entity_map = self._last_entity_map

        return uncloak_stream_response(
            response_stream,
            entity_map=entity_map,
            start_delimiter=self.start_delimiter,
            end_delimiter=self.end_delimiter,
        )

    @classmethod
    def disable_locations(
        cls,
        start_delimiter: str = DEFAULT_START_DELIMITER,
        end_delimiter: str = DEFAULT_END_DELIMITER,
        llm_func: (
            Callable[[str], str]
            | Callable[[str], Generator[str, None, None]]
            | None
        ) = None,
        max_cache_size: int = 1_000,
    ) -> "LLMShield":
        """Create LLMShield with location-based entities disabled.

        Disables: PLACE, IP_ADDRESS, URL detection.
        """
        return cls(
            start_delimiter=start_delimiter,
            end_delimiter=end_delimiter,
            llm_func=llm_func,
            max_cache_size=max_cache_size,
            entity_config=EntityConfig.disable_locations(),
        )

    @classmethod
    def disable_persons(
        cls,
        start_delimiter: str = DEFAULT_START_DELIMITER,
        end_delimiter: str = DEFAULT_END_DELIMITER,
        llm_func: (
            Callable[[str], str]
            | Callable[[str], Generator[str, None, None]]
            | None
        ) = None,
        max_cache_size: int = 1_000,
    ) -> "LLMShield":
        """Create LLMShield with person entities disabled.

        Disables: PERSON detection.
        """
        return cls(
            start_delimiter=start_delimiter,
            end_delimiter=end_delimiter,
            llm_func=llm_func,
            max_cache_size=max_cache_size,
            entity_config=EntityConfig.disable_persons(),
        )

    @classmethod
    def disable_contacts(
        cls,
        start_delimiter: str = DEFAULT_START_DELIMITER,
        end_delimiter: str = DEFAULT_END_DELIMITER,
        llm_func: (
            Callable[[str], str]
            | Callable[[str], Generator[str, None, None]]
            | None
        ) = None,
        max_cache_size: int = 1_000,
    ) -> "LLMShield":
        """Create LLMShield with contact information disabled.

        Disables: EMAIL, PHONE detection.
        """
        return cls(
            start_delimiter=start_delimiter,
            end_delimiter=end_delimiter,
            llm_func=llm_func,
            max_cache_size=max_cache_size,
            entity_config=EntityConfig.disable_contacts(),
        )

    @classmethod
    def only_financial(
        cls,
        start_delimiter: str = DEFAULT_START_DELIMITER,
        end_delimiter: str = DEFAULT_END_DELIMITER,
        llm_func: (
            Callable[[str], str]
            | Callable[[str], Generator[str, None, None]]
            | None
        ) = None,
        max_cache_size: int = 1_000,
    ) -> "LLMShield":
        """Create LLMShield with only financial entities enabled.

        Enables: CREDIT_CARD detection only.
        """
        return cls(
            start_delimiter=start_delimiter,
            end_delimiter=end_delimiter,
            llm_func=llm_func,
            max_cache_size=max_cache_size,
            entity_config=EntityConfig.only_financial(),
        )

    def ask(
        self,
        stream: bool = False,
        messages: list[Message] | None = None,
        **kwargs,
    ) -> str | Generator[str, None, None]:
        """Complete end-to-end LLM interaction with automatic protection.

        NOTE: If you are using a structured output, ensure that your keys
        do not contain PII and that any keys that may contain PII are either
        string, lists, or dicts. Other types like int, float, are unable to be
        cloaked and will be returned as is.

        Args:
            prompt/message: Original prompt with sensitive information. This
                    will be cloaked and passed to your LLM function. Do not
                    pass
                    both, and do not use any other parameter names as they are
                    unrecognised by the shield.
            stream: Whether the LLM Function is a stream or not. If True,
                    returns
                    a generator that yields incremental responses
                    following the OpenAI Realtime Streaming API. If False,
                    returns
                    the complete response as a string.
                    By default, this is False.
            messages: List of message dictionaries for multi-turn
                    conversations.
            They must come in the form of a list of dictionaries,
            where each dictionary has keys like "role" and "content".
            **kwargs: Additional arguments to pass to your LLM function,
                    such as:
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
        whenever the stream parameter is true, the function will return an
        generator. !

        Raises:
            ValueError: If no LLM function was provided during initialization,
                       if prompt is invalid, or if both prompt and message
                       are provided

        """
        # * 1. Validate inputs
        if self.llm_func is None:
            raise ValidationError(
                "No LLM function provided. Either provide llm_func in "
                "constructor or use cloak/uncloak separately."
            )

        # Extract prompt/message from kwargs for validation
        prompt = kwargs.get("prompt")
        message = kwargs.get("message")

        # Validate using our validation utility
        try:
            validate_prompt_input(prompt, message, messages)
        except ValidationError as e:
            # Re-raise with more context
            raise ValidationError(f"Invalid input to ask(): {e}") from e

        if messages is None and ("message" in kwargs or "prompt" in kwargs):
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

        # * 3. Check the cache for an existing entity map for this
        # conversation history
        entity_map = self._cache.get(history_key)
        if entity_map is None:
            # * Cache Miss: Build the entity map by processing the entire
            # history
            entity_map = {}
            for message in history:
                # Skip cloaking for list content (Anthropic tool results)
                if isinstance(message.get("content"), list):
                    continue
                _, entity_map = self.cloak(message["content"], entity_map)
                # Each message is placed in the cache paired to their entity
                # map
                self._cache.put(conversation_hash(message), entity_map)

        # * 4. Cloak the last message using the existing entity map
        # Handle None content for tool calls
        # Handle list content for Anthropic tool results
        if latest_message["content"] is None or isinstance(
            latest_message["content"], list
        ):
            cloaked_latest_content = latest_message["content"]
            final_entity_map = entity_map.copy()
        else:
            cloaked_latest_content, final_entity_map = self.cloak(
                latest_message["content"], entity_map_param=entity_map.copy()
            )

        # 5. Reconstruct the full, cloaked message list to send to the LLM
        cloaked_messages = self._build_cloaked_messages(
            history, latest_message, cloaked_latest_content, final_entity_map
        )

        # 6. Call the LLM with the protected payload - with automatic
        # provider detection
        # Get the appropriate provider for this LLM function
        provider = get_provider(self.llm_func)

        # Let the provider prepare the parameters
        prepared_params, actual_stream = provider.prepare_multi_message_params(
            cloaked_messages, stream, **kwargs
        )

        # Call the LLM function
        llm_response = self.llm_func(**prepared_params)

        # 7. Uncloak the response
        if actual_stream:
            uncloaked_response = self.stream_uncloak(
                llm_response, final_entity_map
            )
        else:
            uncloaked_response = self.uncloak(llm_response, final_entity_map)

        # 8. Update the history with the latest message and the uncloaked
        # response
        response_content = extract_response_content(uncloaked_response)

        next_history = history + [
            latest_message,
            {"role": "assistant", "content": response_content},
        ]

        new_key = conversation_hash(next_history)
        self._cache.put(new_key, final_entity_map)

        return uncloaked_response
