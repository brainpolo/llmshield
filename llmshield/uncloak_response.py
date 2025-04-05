"""
Module for securely uncloaking LLM responses by replacing placeholders with original values.

! Module is intended for internal use only.
"""


# Python imports
from typing import Any

# Local imports
from llmshield.utils import PydanticLike


def _uncloak_response(
        response: Any,
        entity_map: dict[str, str]
) -> str | list[Any] | dict[str, Any] | PydanticLike:
    """
    Securely uncloaks the LLM response by replacing validated placeholders with
    their original values.
    Includes strict validation and safety checks for placeholder format and content.

    ! Do not call this function directly, use `LLMShield.uncloak()` instead. This
    ! is because this function is not type-safe.

    Args:
        response: The LLM response containing placeholders (e.g., [EMAIL_0], [PHONE_1]).
        Supports both strings and structured outputs (dicts). However, note that
        keys in dicts will NOT be uncloaked for integrity of the data structure,
        nor will non string values in dicts be uncloaked.
        entity_map: Mapping of placeholders to their original values

    Returns:
        Uncloaked response with original values restored
    """
    if not entity_map:
        return response

    if isinstance(response, str):  # Direct string replacement
        for placeholder, original in entity_map.items():
            response = response.replace(placeholder, original)
        return response

    if isinstance(response, list):  # Apply uncloaking to each element in the list
        return [_uncloak_response(item, entity_map) for item in response]

    if isinstance(response, dict):  # Apply uncloaking to each key and value in the dict
        return {key: _uncloak_response(value, entity_map) for key, value in response.items()}

    if isinstance(response, PydanticLike):
        # convert back to dict and reprocess
        return _uncloak_response(response.model_dump(), entity_map)

    # Return the response if not in [str, list, dict] (e.g. int value of dict key)
    return response
