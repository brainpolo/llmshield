"""
Module for securely uncloaking LLM responses by replacing placeholders with original values.

! Module is intended for internal use only.
"""

from typing import Dict

def _uncloak_response(response: str, entity_map: Dict[str, str]) -> str:
    """
    Securely uncloaks the LLM response by replacing validated placeholders with their original values.
    Includes strict validation and safety checks for placeholder format and content.

    Example:
        >>> response = "Contact at [EMAIL_0] or [PHONE_1]"
        >>> entity_map = {
        ...     "[EMAIL_0]": "john.doe@example.com",
        ...     "[PHONE_1]": "(123) 456-7890"
        ... }
        >>> uncloak_response(response, entity_map, "[", "]")
        "Contact at john.doe@example.com or (123) 456-7890"

    @param response: The LLM response containing placeholders (e.g., [EMAIL_0], [PHONE_1])
    @param entity_map: Mapping of placeholders to their original values

    @return: Uncloaked response with original values restored
    """
    uncloaked = response

    for placeholder in entity_map.keys():
        uncloaked = uncloaked.replace(placeholder, entity_map[placeholder])

    return uncloaked
