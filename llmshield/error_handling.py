"""Error handling and validation utilities.

Description:
    This module provides utilities for input validation, safe operations,
    and error handling patterns used throughout LLMShield.

Functions:
    validate_prompt_input: Validate prompt/message parameters
    validate_delimiters: Validate delimiter parameters
    validate_entity_map: Validate entity mapping dictionaries
    safe_resource_load: Safely load package resources

Author:
    LLMShield by brainpolo, 2025
"""

import logging
from importlib import resources
from typing import Any

from .exceptions import ResourceLoadError, ValidationError

# Configure logger
logger = logging.getLogger(__name__)

# Maximum allowed prompt length (1 million characters)
MAX_PROMPT_LENGTH = 1_000_000


def validate_prompt_input(
    prompt: str | None = None,
    message: str | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> None:
    """Validate prompt/message input parameters.

    Args:
        prompt: Single prompt string
        message: Single message string
        messages: List of message dictionaries

    Raises:
        ValidationError: If validation fails

    """
    # Check mutual exclusivity
    provided_count = sum(x is not None for x in [prompt, message, messages])

    if provided_count == 0:
        raise ValidationError(
            "Must provide either 'prompt', 'message', or 'messages'"
        )

    if provided_count > 1:
        _validate_multiple_inputs(prompt, message, messages)

    # Validate individual inputs
    if prompt is not None:
        _validate_string_input(prompt, "Prompt")

    if message is not None:
        _validate_string_input(message, "Message")

    if messages is not None:
        _validate_messages_list(messages)


def _validate_multiple_inputs(
    prompt: str | None,
    message: str | None,
    messages: list[dict[str, Any]] | None,
) -> None:
    """Validate when multiple inputs are provided."""
    if prompt is not None and message is not None:
        raise ValidationError(
            "Do not provide both 'prompt' and 'message'. Use only "
            "'prompt' parameter - it will be passed to your LLM function."
        )
    elif messages is not None and (prompt is not None or message is not None):
        raise ValidationError(
            "Do not provide both 'prompt', 'message' and 'messages'. Use "
            "only either prompt/message or messages parameter - it will "
            "be passed to your LLM function."
        )
    else:
        raise ValidationError(
            "Only one of 'prompt', 'message', or 'messages' can be provided"
        )


def _validate_string_input(value: str, name: str) -> None:
    """Validate string input parameters."""
    if not isinstance(value, str):
        raise ValidationError(
            f"{name} must be string, got {type(value).__name__}"
        )
    if len(value) > MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"{name} too long: {len(value)} > {MAX_PROMPT_LENGTH}"
        )


def _validate_messages_list(messages: list[dict[str, Any]]) -> None:
    """Validate messages list parameter."""
    if not isinstance(messages, list):
        raise ValidationError(
            f"Messages must be list, got {type(messages).__name__}"
        )
    if not messages:
        raise ValidationError("Messages list cannot be empty")

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValidationError(
                f"Message {i} must be dict, got {type(msg).__name__}"
            )
        if "content" not in msg:
            raise ValidationError(f"Message {i} missing 'content' field")
        # Allow None content for tool calls (OpenAI format)
        # Allow list content for Anthropic tool results
        if msg["content"] is not None and not isinstance(
            msg["content"], str | list
        ):
            raise ValidationError(
                f"Message {i} content must be string, list, or None, got "
                f"{type(msg['content']).__name__}"
            )


def validate_delimiters(start: str, end: str) -> None:
    """Validate delimiter parameters.

    Args:
        start: Start delimiter
        end: End delimiter

    Raises:
        ValidationError: If delimiters are invalid

    """
    if not start:
        raise ValidationError("Start delimiter cannot be empty")
    if not end:
        raise ValidationError("End delimiter cannot be empty")
    if not isinstance(start, str):
        raise ValidationError(
            f"Start delimiter must be string, got {type(start).__name__}"
        )
    if not isinstance(end, str):
        raise ValidationError(
            f"End delimiter must be string, got {type(end).__name__}"
        )
    if start == end:
        raise ValidationError(
            f"Start and end delimiters cannot be identical: '{start}'"
        )
    # Check for common problematic delimiters
    if len(start) > 10 or len(end) > 10:  # noqa: PLR2004
        raise ValidationError("Delimiters should be short (max 10 characters)")


def validate_entity_map(
    entity_map: dict[str, str] | None,
    last_entity_map: dict[str, str] | None,
    operation: str = "Operation",
) -> dict[str, str]:
    """Validate and return appropriate entity map.

    Args:
        entity_map: Provided entity mapping
        last_entity_map: Cached entity mapping from previous operation
        operation: Name of the operation for error messages

    Returns:
        Valid entity map to use

    Raises:
        ValidationError: If no valid entity map is available

    """
    if entity_map is not None:
        if not isinstance(entity_map, dict):
            raise ValidationError(
                f"Entity map must be dict, got {type(entity_map).__name__}"
            )
        # Validate all keys and values are strings
        for key, value in entity_map.items():
            if not isinstance(key, str):
                raise ValidationError(
                    f"Entity map key must be string, got "
                    f"{type(key).__name__}: {key}"
                )
            if not isinstance(value, str):
                raise ValidationError(
                    f"Entity map value must be string, got "
                    f"{type(value).__name__}: {value}"
                )
        return entity_map

    if last_entity_map is None:
        raise ValidationError(
            f"{operation}: No entity mapping provided or stored from "
            "previous cloak operation"
        )

    return last_entity_map


def safe_resource_load(
    package: str, resource_name: str, operation_name: str = "Loading resource"
) -> list[str]:
    """Safely load package resource with error handling.

    Args:
        package: Package containing the resource
        resource_name: Name of the resource file
        operation_name: Description of the operation for error messages

    Returns:
        List of non-empty, stripped lines from the resource

    Raises:
        ResourceLoadError: If resource cannot be loaded

    """
    try:
        with (
            resources.files(package)
            .joinpath(resource_name)
            .open("r", encoding="utf-8")
        ) as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    except FileNotFoundError:
        raise ResourceLoadError(
            f"{operation_name} failed: Resource not found - "
            f"{package}/{resource_name}"
        ) from None
    except Exception as e:
        logger.error(
            f"{operation_name} failed for {package}/{resource_name}: {e}"
        )
        raise ResourceLoadError(
            f"{operation_name} failed: {package}/{resource_name} - {e}"
        ) from e
