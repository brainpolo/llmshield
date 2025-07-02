"""Zero-dependency PII protection for LLM applications.

Description:
    llmshield is a lightweight Python library that automatically detects and
    protects personally identifiable information (PII) in prompts sent to
    language models. It replaces sensitive data with placeholders before
    processing and seamlessly restores the original information in responses.

Classes:
    LLMShield: Main interface for prompt cloaking and response uncloaking
    EntityConfig: Configuration for selective entity detection
    EntityType: Enumeration of supported entity types

Functions:
    create_shield: Factory function to create configured LLMShield instances

Examples:
    Basic usage:
    >>> from llmshield import (
    ...     LLMShield,
    ... )
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

    Direct usage with LLM:
    >>> def my_llm(
    ...     prompt: str,
    ... ) -> str:
    ...     # Your LLM API call here
    ...     return response
    >>> shield = LLMShield(
    ...     llm_func=my_llm
    ... )
    >>> response = shield.ask(
    ...     prompt="Hi, I'm John (john@example.com)"
    ... )

Author:
    LLMShield by brainpolo, 2025

"""

from .core import LLMShield
from .entity_detector import EntityConfig, EntityType

__all__ = ["LLMShield", "EntityConfig", "EntityType"]


def create_shield(**kwargs) -> LLMShield:  # noqa: ANN003
    """Create a new LLMShield instance with the given configuration.

    Args:
        **kwargs: Arguments to pass to LLMShield constructor
            - start_delimiter: Character(s) to wrap entities (default: '<')
            - end_delimiter: Character(s) to wrap entities (default: '>')
            - llm_func: Optional function to call LLM

    Returns:
        LLMShield instance

    """
    return LLMShield(**kwargs)
