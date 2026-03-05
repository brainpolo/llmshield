"""Shared test utilities and fixtures.

Description:
    Common factories, constants, and helpers used across the test
    suite to reduce duplication and improve maintainability.

Author:
    LLMShield by brainpolo, 2025-2026
"""

from unittest.mock import Mock


def make_capture_llm():
    """Create a capture LLM that records messages sent to it.

    Returns:
        Tuple of (llm_func, captured_list).

    """
    captured = []

    def capture(**kwargs):
        captured.append(kwargs.get("messages", []))
        return "OK"

    return capture, captured


def make_mock_func(
    name="create",
    qualname="client.chat.completions.create",
    module="openai.client",
):
    """Create a mock function with standard attributes.

    Args:
        name: Function __name__
        qualname: Function __qualname__
        module: Function __module__

    Returns:
        Mock function with the specified attributes.

    """
    func = Mock()
    func.__name__ = name
    func.__qualname__ = qualname
    func.__module__ = module
    return func


STANDARD_ENTITY_MAP = {
    "<PERSON_0>": "John Doe",
    "<EMAIL_0>": "john@example.com",
    "<PLACE_0>": "New York",
}


def make_anthropic_msg(
    content="Hello world",
    model="claude-3-5-haiku-20241022",
    role="assistant",
    **extra,
):
    """Create a Mock Anthropic Message object.

    Args:
        content: Message content (str or list of blocks)
        model: Model name
        role: Message role
        **extra: Additional attributes to set

    Returns:
        Mock with Anthropic-like attributes.

    """
    msg = Mock()
    msg.content = content
    msg.model = model
    msg.role = role
    for key, value in extra.items():
        setattr(msg, key, value)
    return msg
