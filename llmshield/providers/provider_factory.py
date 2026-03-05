"""Provider factory for detecting and creating appropriate providers.

Description:
    This module implements the factory pattern for LLM provider detection
    and instantiation. It maintains a registry of available providers and
    automatically selects the most appropriate one based on the LLM function
    characteristics.

Functions:
    get_provider: Automatically selects appropriate provider for LLM function
    register_provider: Registers new providers in the factory

Author:
    LLMShield by brainpolo, 2025-2026
"""

# Standard Library Imports
from collections.abc import Callable

# Local Imports
from .anthropic_provider import AnthropicProvider
from .base import BaseLLMProvider
from .cohere_provider import CohereProvider
from .default_provider import DefaultProvider
from .google_provider import GoogleProvider
from .openai_provider import OpenAIProvider
from .xai_provider import XAIProvider

# Registry of available providers in priority order
# Earlier providers are checked first
# Non-OpenAI providers must precede OpenAI (OpenAI's
# broad can_handle would match their SDK functions)
PROVIDER_REGISTRY = [
    AnthropicProvider,
    GoogleProvider,
    XAIProvider,
    CohereProvider,
    OpenAIProvider,
    DefaultProvider,  # Must be last as it's the fallback
]


def get_provider(llm_func: Callable) -> BaseLLMProvider:
    """Get the appropriate provider for the given LLM function.

    Args:
        llm_func: The LLM function to analyze

    Returns:
        An instance of the appropriate provider

    Raises:
        RuntimeError: If no provider can handle the function (should never
                     happen
                     due to DefaultProvider fallback)

    """
    for provider_class in PROVIDER_REGISTRY:
        if provider_class.can_handle(llm_func):
            return provider_class(llm_func)

    # This should never happen since DefaultProvider can handle anything
    raise RuntimeError(f"No provider found for function: {llm_func}")


def register_provider(
    provider_class: type[BaseLLMProvider], priority: int = 0
) -> None:
    """Register a new provider in the registry.

    Args:
        provider_class: The provider class to register
        priority: Where to insert in the registry (0 = highest priority)
                 -1 means insert before DefaultProvider

    """
    if priority == -1:
        # Insert before DefaultProvider (which should be last)
        PROVIDER_REGISTRY.insert(-1, provider_class)
    else:
        PROVIDER_REGISTRY.insert(priority, provider_class)
