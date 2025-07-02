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
    LLMShield by brainpolo, 2025

"""

# Standard Library Imports
import copy
from typing import Any

# Local Imports
from llmshield.detection_utils import (
    is_anthropic_message_like,
    is_chatcompletion_like,
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
        for placeholder, original in entity_map.items():
            response = response.replace(placeholder, original)
        return response

    if isinstance(response, list):
        return [_uncloak_response(item, entity_map) for item in response]

    if isinstance(response, dict):
        return {
            key: _uncloak_response(value, entity_map)
            for key, value in response.items()
        }

    return None


def _uncloak_complex_types(response: Any, entity_map: dict[str, str]) -> Any:
    """Handle uncloaking for complex types (Pydantic, ChatCompletion, etc)."""
    if isinstance(response, PydanticLike):
        return _uncloak_response(response.model_dump(), entity_map)

    # Handle OpenAI ChatCompletion objects
    if is_chatcompletion_like(response):
        return _uncloak_chatcompletion(response, entity_map)

    # Handle Anthropic Message objects
    if is_anthropic_message_like(response):
        return _uncloak_anthropic_message(response, entity_map)

    # Return the response if not a recognized type
    return response


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
                # Handle list vs Mock object
                tool_calls = choice.message.tool_calls
                if hasattr(tool_calls, "__iter__"):
                    for tool_call in tool_calls:
                        if hasattr(tool_call, "function") and hasattr(
                            tool_call.function, "arguments"
                        ):
                            # Uncloak the arguments string
                            tool_call.function.arguments = _uncloak_response(
                                tool_call.function.arguments, entity_map
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
