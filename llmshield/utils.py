"""Module for utility functions for the llmshield library."""

# Python imports
import collections.abc
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any, BinaryIO, Protocol, runtime_checkable

# Local imports
from llmshield.entity_detector import EntityType


@runtime_checkable
class PydanticLike(Protocol):
    """A protocol for types that behave like Pydantic models.

    This is to provide type-safety for the uncloak function, which can accept
    either a string, list, dict, or a Pydantic model for LLM responses which
    return structured outputs.

    NOTE: This is not essential for the library, but it is used to provide
    type-safety for the uncloak function.

    Pydantic models have the following methods:
    - model_dump() -> dict
    - model_validate(data: dict) -> Any
    """

    def model_dump(self) -> dict: ...

    """
    Dump the Pydantic model into a dictionary.

    @return: The dictionary representation of the Pydantic model.
    """

    @classmethod
    def model_validate(cls, data: dict) -> Any: ...

    """
    Validate a dictionary against the Pydantic model.

    @param data: The dictionary to validate.

    @return: The validated data.
    """


def split_fragments(text: str) -> list[str]:
    """Split the text into fragments based on the following rules:
    - Split on sentence boundaries (punctuation / new line)
    - Remove any empty fragments.
    """
    fragments = re.split(r"[.!?]+\s+|\n+", text)
    return [f.strip() for f in fragments if f.strip()]


def is_valid_delimiter(delimiter: str) -> bool:
    """
    Validate a delimiter based on the following rules:
    - Must be a string.
    - Must be at least 1 character long.

    @param delimiter: The delimiter to validate.

    @return: True if the delimiter is valid, False otherwise.
    """
    return isinstance(delimiter, str) and len(delimiter) > 0


def wrap_entity(
    entity_type: EntityType,
    suffix: int,
    start_delimiter: str,
    end_delimiter: str,
) -> str:
    """Wrap an entity in a start and end delimiter.

    The wrapper works as follows:
    - The value will be wrapped with START_DELIMETER and END_DELIMETER.
    - The suffix will be appended to the entity.

    @param entity: The entity to wrap.
    @param start_delimiter: The start delimiter.
    @param end_delimiter: The end delimiter.

    @return: The wrapped entity.
    """
    return f"{start_delimiter}{entity_type.name}_{suffix}{end_delimiter}"


def normalise_spaces(text: str) -> str:
    """Normalise spaces in the text by replacing multiple spaces with a single space."""
    return re.sub(r"\s+", " ", text).strip()


def is_valid_stream_response(obj: object) -> bool:
    """
    Return True if obj is an iterable suitable for streaming
    (not str, bytes, bytearray, or any mapping).
    """
    # Exclude string-like and mapping types
    excluded_types = (str, bytes, bytearray, collections.abc.Mapping)
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, excluded_types)


# Type alias that follows the OpenAI API input format for the creation of model responses.
# This includes strings, lists of strings, dictionaries, Pydantic-like objects,
# file paths, file-like objects, raw binary data, and tuples of filename and content.
type Input = (
    str  # Single string
    | list[str]  # List of strings
    | "PydanticLike"
    | Path  # File paths
    | BinaryIO  # File-like objects (open files)
    | bytes  # Raw binary data
    | tuple[str, bytes]  # (filename, content) pairs
)


def _should_cloak_input(input_data: Input) -> bool:
    """Determine if the input should be cloaked.

    Only string and list[str] inputs are cloaked.

    Args:
        input_data: The input to check

    Returns:
        bool: True if input should be cloaked, False otherwise

    """
    return isinstance(input_data, str | list)


def ask_helper(shield, stream: bool, **kwargs) -> str | Generator[str, None, None]:
    """Helper function to handle the ask method of LLMShield.

    This function checks if the input should be cloaked and handles both
    streaming and non-streaming cases.

    Args:
        shield: The LLMShield instance.
        stream: Whether to stream the response.
        **kwargs: Additional keyword arguments to pass to the LLM function.

    Returns:
        str | Generator[str, None, None]: The response from the LLM.
    """
    # * 1. Get the input text and determine parameter name
    input_param = "message" if "message" in kwargs else "prompt"
    input_text = kwargs[input_param]
    if _should_cloak_input(input_data=input_text):
        # * 2. Cloak the input text
        cloaked_text, entity_map = shield.cloak(input_text)
        # * 3. Pass cloaked text under the correct parameter name for LLM function
        func_preferred_param = (
            "message" if "message" in shield.llm_func.__code__.co_varnames else "prompt"
        )
        # Remove the original parameter and add under the LLM's preferred name
        del kwargs[input_param]
        kwargs[func_preferred_param] = cloaked_text
        kwargs["stream"] = stream  # Ensure stream is passed to the LLM function
        # * 4. Get response from LLM
        llm_response = shield.llm_func(**kwargs)
        # * 5. Uncloak and return
        if stream:
            if not is_valid_stream_response(llm_response):
                # LLM didn't return a valid stream, treat as non-streaming
                return iter([shield.uncloak(llm_response, entity_map)])
            return shield.stream_uncloak(llm_response, entity_map)
        # Non-streaming: uncloak complete response
        return shield.uncloak(llm_response, entity_map)
    return shield.llm_func(**kwargs)  # No cloaking needed, call LLM directly


# Typedef for a hashable type used for conversation keys.
type Hash = int
type Message = dict[str, str]


def conversation_hash(obj: Message | list[Message]) -> Hash:
    """
    Generate a stable, hashable key for a message or a list of messages.
    If a single message is provided, hash its role and content.
    If a list of messages is provided, hash the set of (role, content) pairs.
    """
    if isinstance(obj, dict):
        # Single message
        return hash((obj.get("role", ""), obj.get("content", "")))

    # List of messages
    return hash(frozenset((msg.get("role", ""), msg.get("content", "")) for msg in obj))
