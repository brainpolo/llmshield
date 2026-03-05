"""Utility functions for entity and response detection.

Description:
    This module provides detection and content extraction functions for
    all supported LLM provider response formats. Each provider has a
    detector (is_*_like) and an extractor (extract_*_content) pair.

Functions:
    is_chatcompletion_like: Check if object is a ChatCompletion
    extract_chatcompletion_content: Extract content from ChatCompletion
    is_anthropic_message_like: Check if object is an Anthropic Message
    extract_anthropic_content: Extract content from Anthropic Message
    is_google_response_like: Check if object is a Google GenAI response
    extract_google_content: Extract content from Google response
    is_cohere_response_like: Check if object is a Cohere response
    extract_cohere_content: Extract content from Cohere response
    is_xai_response_like: Check if object is an xAI response
    extract_xai_content: Extract content from xAI response
    extract_response_content: Universal content extractor

Author:
    LLMShield by brainpolo, 2025-2026
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


def is_xai_response_like(obj: Any) -> bool:
    """Check if object appears to be an xAI SDK response.

    Checks for xai_sdk type first, then falls back to duck-typing:
    xAI responses have 'content' and 'usage' attributes but
    lack 'choices' (OpenAI) and 'model' (Anthropic/OpenAI).

    Args:
        obj: Object to check

    Returns:
        True if object appears to be an xAI response

    """
    obj_module = getattr(type(obj), "__module__", "")
    if "xai_sdk" in obj_module:
        return True
    return (
        hasattr(obj, "content")
        and hasattr(obj, "usage")
        and not hasattr(obj, "choices")
        and not hasattr(obj, "role")
    )


def extract_xai_content(obj: Any) -> str | None:
    """Extract content from an xAI SDK response.

    Args:
        obj: xAI response object

    Returns:
        Content string if found, None otherwise

    """
    if not is_xai_response_like(obj):
        return None
    try:
        return obj.content
    except AttributeError:
        return None


def is_google_response_like(obj: Any) -> bool:
    """Check if object appears to be a Google GenAI response.

    Google GenerateContentResponse objects have 'candidates'
    and 'usage_metadata' attributes but lack 'choices' (OpenAI).

    Args:
        obj: Object to check

    Returns:
        True if object appears to be a Google response

    """
    return (
        hasattr(obj, "candidates")
        and hasattr(obj, "usage_metadata")
        and not hasattr(obj, "choices")
    )


def extract_google_content(obj: Any) -> str | None:
    """Extract content from a Google GenAI response.

    Uses the .text convenience property which returns
    concatenated text from the first candidate's parts.

    Args:
        obj: Google GenerateContentResponse object

    Returns:
        Content string if found, None otherwise

    """
    if not is_google_response_like(obj):
        return None
    try:
        return obj.text
    except (AttributeError, IndexError, ValueError):
        return None


def is_cohere_response_like(obj: Any) -> bool:
    """Check if object appears to be a Cohere V2 response.

    Checks for cohere module first, then falls back to duck-typing:
    Cohere responses have 'message' and 'finish_reason' attributes
    but lack 'choices' (OpenAI) and 'candidates' (Google).

    Args:
        obj: Object to check

    Returns:
        True if object appears to be a Cohere response

    """
    obj_module = getattr(type(obj), "__module__", "")
    if obj_module == "cohere" or obj_module.startswith("cohere."):
        return True
    return (
        hasattr(obj, "message")
        and hasattr(obj, "finish_reason")
        and not hasattr(obj, "choices")
        and not hasattr(obj, "candidates")
        and not hasattr(obj, "role")
    )


def extract_cohere_content(obj: Any) -> str | None:
    """Extract content from a Cohere V2 response.

    Cohere responses have message.content as a list of
    content blocks with type and text attributes.

    Args:
        obj: Cohere response object

    Returns:
        Content string if found, None otherwise

    """
    if not is_cohere_response_like(obj):
        return None
    try:
        msg = obj.message
        if msg is None:
            return None
        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list) and content:
            text_parts = []
            for block in content:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
            return " ".join(text_parts) if text_parts else None
        return None
    except AttributeError:
        return None


def extract_response_content(response: Any) -> str | Any:
    """Extract content string from various response objects.

    This function handles extraction from different LLM response
    formats, including OpenAI ChatCompletion, Anthropic Message,
    xAI SDK, and plain text. Used primarily for conversation
    history tracking.

    Args:
        response: The response object from an LLM

    Returns:
        Content string if found, empty string for known types
        with None content, or the original response if type is
        not recognized

    """
    # Module-based detectors first (definitive, no false positives)
    response_content = extract_cohere_content(response)

    if response_content is None:
        response_content = extract_xai_content(response)

    # Duck-typing detectors
    if response_content is None:
        response_content = extract_chatcompletion_content(response)

    if response_content is None:
        response_content = extract_anthropic_content(response)

    if response_content is None:
        response_content = extract_google_content(response)

    if response_content is None:
        if (
            is_chatcompletion_like(response)
            or is_anthropic_message_like(response)
            or is_xai_response_like(response)
            or is_google_response_like(response)
            or is_cohere_response_like(response)
        ):
            response_content = ""
        else:
            response_content = response

    return response_content
