"""Utility functions for entity and response detection.

Description:
    This module provides common detection functions to reduce code
    duplication across the library, particularly for ChatCompletion-like
    object detection and entity map validation.

Functions:
    is_chatcompletion_like: Check if object appears to be a ChatCompletion
    extract_chatcompletion_content: Extract content from ChatCompletion

Author:
    LLMShield by brainpolo, 2025
"""

from typing import Any


def is_chatcompletion_like(obj: Any) -> bool:
    """Check if object appears to be a ChatCompletion response.

    ChatCompletion-like objects have both 'choices' and 'model' attributes.
    The choices list may be empty in some cases.

    Args:
        obj: Object to check

    Returns:
        True if object appears to be a ChatCompletion, False otherwise

    """
    return hasattr(obj, "choices") and hasattr(obj, "model")


def is_anthropic_message_like(obj: Any) -> bool:
    """Check if object appears to be an Anthropic Message response.

    Anthropic Message objects have 'content', 'model', and 'role' attributes.

    Args:
        obj: Object to check

    Returns:
        True if object appears to be an Anthropic Message, False otherwise

    """
    return (
        hasattr(obj, "content")
        and hasattr(obj, "model")
        and hasattr(obj, "role")
    )


def extract_chatcompletion_content(obj: Any) -> str | None:
    """Extract content from a ChatCompletion-like object.

    Safely extracts the content from the first choice's message.
    Handles both regular message content and streaming delta content.

    Args:
        obj: ChatCompletion-like object

    Returns:
        Content string if found, None otherwise

    """
    if not is_chatcompletion_like(obj):
        return None

    try:
        choice = obj.choices[0]

        # Try regular message content first
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            return choice.message.content

        # Try streaming delta content
        if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
            return choice.delta.content

        return None
    except (IndexError, AttributeError):
        return None


def extract_anthropic_content(obj: Any) -> str | None:
    """Extract content from an Anthropic Message-like object.

    Safely extracts the text content from Anthropic message objects.
    Handles both simple text content and content blocks.

    Args:
        obj: Anthropic Message-like object

    Returns:
        Content string if found, None otherwise

    """
    if not is_anthropic_message_like(obj):
        return None

    try:
        content = obj.content

        # Handle simple string content
        if isinstance(content, str):
            return content

        # Handle content blocks (list format)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif hasattr(block, "type") and block.type == "text":
                    text_parts.append(getattr(block, "text", ""))

            return " ".join(text_parts) if text_parts else None

        return None
    except AttributeError:
        return None


def extract_response_content(response: Any) -> str | Any:
    """Extract content string from various response objects.

    This function handles extraction from different LLM response formats,
    including OpenAI ChatCompletion, Anthropic Message, and plain text.
    Used primarily for conversation history tracking.

    Args:
        response: The response object from an LLM

    Returns:
        Content string if found, empty string for known types with None
        content, or the original response if type is not recognized

    """
    # Try OpenAI ChatCompletion format first
    response_content = extract_chatcompletion_content(response)

    # Try Anthropic format if OpenAI extraction failed
    if response_content is None:
        response_content = extract_anthropic_content(response)

    if response_content is None:
        if is_chatcompletion_like(response) or is_anthropic_message_like(
            response
        ):
            # Known response type but content is None (e.g., tool calls)
            response_content = ""
        else:
            # Not a known response type - use as-is
            response_content = response

    return response_content
