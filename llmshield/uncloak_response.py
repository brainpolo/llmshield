"""Response uncloaking module.

Description:
    This module handles the restoration of original sensitive data in LLM
    responses by replacing placeholders with their original values. It
    supports various response formats including strings, lists,
    dictionaries, and Pydantic models.

Functions:
    uncloak_response: Restore original entities in LLM response

Note:
    This module is intended for internal use only. Users should interact
    with the LLMShield class rather than calling these functions directly.

Author:
    LLMShield by brainpolo, 2025-2026

"""

# Standard Library Imports
import contextlib
import copy
import json
from typing import Any

# Local Imports
from llmshield.detection_utils import (
    is_anthropic_message_like,
    is_chatcompletion_like,
    is_cohere_response_like,
    is_google_response_like,
    is_xai_response_like,
)
from llmshield.utils import PydanticLike


def _uncloak_response(
    response: Any,
    entity_map: dict[str, str],
) -> str | list[Any] | dict[str, Any] | PydanticLike:
    """Securely uncloak LLM response by replacing placeholders.

    Replaces validated placeholders with their original values.
    Includes strict validation and safety checks for placeholder format and
    content.

    ! Do not call this function directly, use `LLMShield.uncloak()` instead.
    ! This
    ! is because this function is not type-safe.

    Args:
        response: The LLM response containing placeholders (e.g., [EMAIL_0],
            [PHONE_1]).
            Supports both strings and structured outputs (dicts). However, note
            that
            keys in dicts will NOT be uncloaked for integrity of the data
            structure,
            nor will non string values in dicts be uncloaked.
        entity_map: Mapping of placeholders to their original values

    Returns:
        Uncloaked response with original values restored

    """
    if not entity_map:
        return response

    # Handle basic types
    result = _uncloak_basic_types(response, entity_map)
    if result is not None:
        return result

    # Handle complex types
    return _uncloak_complex_types(response, entity_map)


def _uncloak_basic_types(response: Any, entity_map: dict[str, str]) -> Any:
    """Handle uncloaking for basic types (str, list, dict)."""
    if isinstance(response, str):
        result = response
        for placeholder, original in entity_map.items():
            result = result.replace(placeholder, original)
        # Handle JSON with unicode-escaped delimiters
        # (e.g. Cohere returns \u003c instead of <)
        if result == response:
            try:
                parsed = json.loads(response)
            except ValueError:
                parsed = None
            if isinstance(parsed, (dict, list)):
                uncloaked = _uncloak_response(parsed, entity_map)
                if uncloaked != parsed:
                    return json.dumps(uncloaked)
        return result

    if isinstance(response, list):
        return [_uncloak_response(item, entity_map) for item in response]

    if isinstance(response, dict):
        return {
            key: _uncloak_response(value, entity_map)
            for key, value in response.items()
        }

    return None


def _uncloak_complex_types(  # noqa: PLR0911
    response: Any, entity_map: dict[str, str]
) -> Any:
    """Handle uncloaking for complex types (Pydantic, ChatCompletion, etc)."""
    # Module-based detectors first: these are definitive and must
    # precede duck-typing checks. SDK models with extra='allow'
    # can acquire unexpected attributes from API responses, causing
    # duck-typing detectors to false-positive.
    if is_cohere_response_like(response):
        return _uncloak_cohere_response(response, entity_map)

    if is_xai_response_like(response):
        return _uncloak_xai_response(response, entity_map)

    # Duck-typing detectors: check LLM response types before
    # Pydantic, since some (e.g. ParsedChatCompletion) satisfy
    # both protocols
    if is_chatcompletion_like(response):
        return _uncloak_chatcompletion(response, entity_map)

    if is_anthropic_message_like(response):
        return _uncloak_anthropic_message(response, entity_map)

    if is_google_response_like(response):
        return _uncloak_google_response(response, entity_map)

    if isinstance(response, PydanticLike):
        return _uncloak_response(response.model_dump(), entity_map)

    # Return the response if not a recognized type
    return response


def _uncloak_function_tool_calls(
    tool_calls: Any, entity_map: dict[str, str]
) -> None:
    """Uncloak tool call arguments via the function.arguments pattern.

    Iterates over tool calls and replaces placeholders in each
    tool_call.function.arguments string with original values.

    Args:
        tool_calls: Iterable of tool call objects, each expected
            to have a function.arguments attribute
        entity_map: Mapping of placeholders to original values

    Returns:
        None. Modifies tool call objects in place.

    """
    if not tool_calls or not hasattr(tool_calls, "__iter__"):
        return
    for tc in tool_calls:
        func = getattr(tc, "function", None)
        if func and hasattr(func, "arguments"):
            func.arguments = _uncloak_response(func.arguments, entity_map)


# skipcq: PY-R1000
def _uncloak_chatcompletion(response: Any, entity_map: dict[str, str]) -> Any:
    """Handle uncloaking for ChatCompletion objects."""
    response_copy = copy.deepcopy(response)

    if hasattr(response_copy, "choices"):
        for choice in response_copy.choices:
            if hasattr(choice, "message") and hasattr(
                choice.message, "content"
            ):
                if choice.message.content is not None:  # ? None in tool-calls
                    choice.message.content = _uncloak_response(
                        choice.message.content, entity_map
                    )
            # Handle streaming delta content
            elif (
                hasattr(choice, "delta")
                and hasattr(choice.delta, "content")
                and choice.delta.content is not None  # ? None in tool-calls
            ):
                choice.delta.content = _uncloak_response(
                    choice.delta.content, entity_map
                )

            # Handle tool calls
            if (
                hasattr(choice, "message")
                and hasattr(choice.message, "tool_calls")
                and choice.message.tool_calls
            ):
                _uncloak_function_tool_calls(
                    choice.message.tool_calls, entity_map
                )

    return response_copy


def _uncloak_anthropic_message(
    response: Any, entity_map: dict[str, str]
) -> Any:
    """Handle uncloaking for Anthropic Message objects."""
    response_copy = copy.deepcopy(response)

    try:
        content = response_copy.content

        # Handle simple string content
        if isinstance(content, str):
            response_copy.content = _uncloak_response(content, entity_map)

        # Handle content blocks (list format)
        elif isinstance(content, list):
            for block in content:
                # Handle dict-style blocks
                if isinstance(block, dict):
                    if block.get("type") == "text" and "text" in block:
                        block["text"] = _uncloak_response(
                            block["text"], entity_map
                        )
                    elif block.get("type") == "tool_use" and "input" in block:
                        # Uncloak tool use input parameters
                        block["input"] = _uncloak_response(
                            block["input"], entity_map
                        )

                # Handle object-style blocks
                elif hasattr(block, "type"):
                    if getattr(block, "type", None) == "text" and hasattr(
                        block, "text"
                    ):
                        block.text = _uncloak_response(block.text, entity_map)
                    elif getattr(
                        block, "type", None
                    ) == "tool_use" and hasattr(block, "input"):
                        # Uncloak tool use input parameters
                        block.input = _uncloak_response(
                            block.input, entity_map
                        )

    except AttributeError:
        pass  # If content structure is unexpected, leave unchanged

    return response_copy


def _uncloak_xai_response(response: Any, entity_map: dict[str, str]) -> Any:
    """Handle uncloaking for xAI SDK response objects.

    The real xai_sdk.chat.Response has read-only properties, so
    we copy attributes into a mutable wrapper instead.
    """

    class _MutableResponse:
        """Mutable wrapper mirroring xAI Response attributes."""

    _MutableResponse.__module__ = getattr(
        type(response), "__module__", __name__
    )
    wrapper = _MutableResponse()

    # Copy all accessible attributes
    for attr in dir(response):
        if attr.startswith("_"):
            continue
        with contextlib.suppress(AttributeError, TypeError):
            setattr(wrapper, attr, getattr(response, attr))

    # Uncloak text content
    if hasattr(wrapper, "content") and wrapper.content is not None:
        wrapper.content = _uncloak_response(  # skipcq: PYL-W0201
            wrapper.content, entity_map
        )

    # Uncloak tool call arguments
    _uncloak_function_tool_calls(
        getattr(wrapper, "tool_calls", None), entity_map
    )

    return wrapper


def _uncloak_google_response(response: Any, entity_map: dict[str, str]) -> Any:
    """Handle uncloaking for Google GenAI response objects."""
    response_copy = copy.deepcopy(response)

    if (
        not hasattr(response_copy, "candidates")
        or not response_copy.candidates
    ):
        return response_copy

    for candidate in response_copy.candidates:
        if not hasattr(candidate, "content"):
            continue
        if not hasattr(candidate.content, "parts"):
            continue

        for part in candidate.content.parts:
            # Uncloak text content
            if hasattr(part, "text") and part.text is not None:
                part.text = _uncloak_response(part.text, entity_map)

            # Uncloak function call arguments
            fc = getattr(part, "function_call", None)
            if fc and hasattr(fc, "args") and fc.args:
                fc.args = _uncloak_response(fc.args, entity_map)

    return response_copy


def _uncloak_cohere_response(response: Any, entity_map: dict[str, str]) -> Any:
    """Handle uncloaking for Cohere V2 response objects."""
    response_copy = copy.deepcopy(response)

    msg = getattr(response_copy, "message", None)
    if msg is None:
        return response_copy

    # Uncloak text content blocks
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        for block in content:
            if (
                getattr(block, "type", None) == "text"
                and hasattr(block, "text")
                and block.text is not None
            ):
                block.text = _uncloak_response(block.text, entity_map)
    elif isinstance(content, str):
        msg.content = _uncloak_response(content, entity_map)

    # Uncloak tool call arguments
    _uncloak_function_tool_calls(getattr(msg, "tool_calls", None), entity_map)

    return response_copy
